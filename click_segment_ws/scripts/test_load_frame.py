from click_segment_core.load import load_scene_camera, load_scene_rgbd, depth_to_pointcloud, pick_point_interactive, project_points_to_mask, remove_dominant_plane, cluster_points, get_clicked_cluster, load_ground_truth_mask, compute_iou
import open3d as o3d
import numpy as np
import json

scene_path = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/data/ycbv/test/000054"
frame_id = 1029
cam_K, cam_R_w2c, cam_t_w2c, depth_scale = load_scene_camera(scene_path, frame_id)
rgb, depth = load_scene_rgbd(scene_path, depth_scale, frame_id)

pcd = depth_to_pointcloud(depth, cam_K, rgb=rgb)

object_points_pcd, plane_points_pcd, plane_model = remove_dominant_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000)
object_clusters, cluster_labels = cluster_points(object_points_pcd, eps=0.02, min_points=20)

geometries_to_draw = []
for cluster in object_clusters:
    random_color = np.random.rand(3)
    cluster.paint_uniform_color(random_color)
    geometries_to_draw.append(cluster)

plane_points_pcd.paint_uniform_color([0.2, 0.2, 0.2])
#Add the plane points to the geometries to draw 
geometries_to_draw.append(plane_points_pcd)
#Visualize the point cloud with clusters and the plane
o3d.visualization.draw_geometries(geometries_to_draw)

clicked_index = pick_point_interactive(object_points_pcd) 
final_clusters = []
for i in range(len(clicked_index[0])):
    clicked_cluster = get_clicked_cluster(object_points_pcd, cluster_labels, clicked_index[0][i])
    if clicked_cluster is not None:
        clicked_cluster.paint_uniform_color(np.random.rand(3))
        final_clusters.append(clicked_cluster)

if len(final_clusters) > 0:
    with open(f"{scene_path}/scene_gt.json", "r") as f:
        gt_data = json.load(f)
    frame_gt = gt_data[str(frame_id)]
    available_obj_ids = [obj["obj_id"] for obj in frame_gt]
    print(f"Ground Truth objects present in this frame: {available_obj_ids}")

    for idx, cluster in enumerate(final_clusters):
        print(f"\nSelected Cluster {idx + 1}:")
        cluster_points_3d = np.asarray(cluster.points)
        predicted_mask = project_points_to_mask(cluster_points_3d, cam_K, depth.shape)
        contributing_objects = []
        for obj_id in available_obj_ids:
            try:
                gt_mask = load_ground_truth_mask(scene_path, frame_id, obj_id)
                gt_mask_bool = gt_mask > 0
                overlap_pixels = np.logical_and(predicted_mask, gt_mask_bool).sum()
                predicted_pixels = predicted_mask.sum()
                
                if predicted_pixels > 0:
                    composition_percentage = (overlap_pixels / predicted_pixels) * 100
                    if composition_percentage > 5.0:
                        current_iou = compute_iou(predicted_mask, gt_mask_bool)
                        contributing_objects.append({
                            "id": obj_id,
                            "share": composition_percentage,
                            "iou": current_iou * 100
                        })
            except Exception:
                continue
        if len(contributing_objects) > 0:
            print(f"This 3D cluster is a COMBINATION of the following items:")
            # Sort by the biggest share contributor first
            contributing_objects.sort(key=lambda x: x["share"], reverse=True)
            for obj in contributing_objects:
                print(f" Object ID {obj['id']}: {obj['share']:.1f}% (Individual IoU: {obj['iou']:.1f}%)")
            print(f" Final Segmentation Score: {contributing_objects[0]['iou']:.1f}% IoU")
        else:
            print(" No valid objects found in the ground truth.")
else:
    print("\nEvaluation skipping: No target clusters were picked during interaction.")


# Visualize the original point cloud and the clicked clusters
o3d.visualization.draw_geometries([pcd] + final_clusters)
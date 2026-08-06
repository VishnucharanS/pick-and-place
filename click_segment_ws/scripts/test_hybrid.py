import open3d as o3d
import numpy as np
import json
from click_segment_core.hybrid_segment import fuse_sam_and_dbscan, remove_radius_outliers
from click_segment_core.load import compute_iou, depth_to_pointcloud, load_ground_truth_mask, load_scene_camera, load_scene_rgbd, pick_point_interactive, project_points_to_mask, remove_dominant_plane
from click_segment_core.sam_segment import project_mask_to_points, project_points_to_pixels, test_sam2_predictor

scene_path = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/data/ycbv/test/000050"
frame_id = 1246

cam_K, cam_R_w2c, cam_t_w2c, depth_scale = load_scene_camera(scene_path, frame_id)
rgb, depth = load_scene_rgbd(scene_path, depth_scale, frame_id)
pcd1  = depth_to_pointcloud(depth, cam_K, rgb=rgb)
pcd, plane_points_pcd, plane_model = remove_dominant_plane(
    pcd1, distance_threshold=0.01, ransac_n=3, num_iterations=1000
)
input_points = pick_point_interactive(pcd)
input_point_mask_pixel = project_points_to_pixels(np.asarray(pcd.points)[input_points[0]], cam_K, depth.shape)  # Project the selected 3D points to 2D pixel coordinates
sam_combined_mask, sam_individual_masks = test_sam2_predictor(scene_path, frame_id, input_point_mask_pixel) 

seg_clusters = []  
for i in range(len(input_point_mask_pixel)):
    points_3d = project_mask_to_points(sam_individual_masks[i], cam_K, depth)
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(points_3d)
    seg_clusters.append(cluster_pcd)

fused_clusters = []
for i in range(len(seg_clusters)):
    fused_clusters.append(fuse_sam_and_dbscan(np.asarray(seg_clusters[i].points), np.asarray(pcd.points), radius=0.02))
final_clusters = []
for i in range(len(fused_clusters)):
    filtered_pcd = remove_radius_outliers(fused_clusters[i], radius=0.01)
    filtered_pcd.paint_uniform_color(np.random.rand(3))
    if len(filtered_pcd.points) > 100:
        final_clusters.append(filtered_pcd)

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



o3d.visualization.draw_geometries([pcd1] + final_clusters)  # Visualize the original point cloud and the selected clusters


import open3d as o3d
import json
import numpy as np
import torch
from click_segment_core.load import compute_iou, load_ground_truth_mask, load_scene_camera, load_scene_rgbd, depth_to_pointcloud, pick_point_interactive, project_points_to_mask, remove_dominant_plane
from click_segment_core.pointnet import load_pointnet2_model, prepare_pcd_for_pointnet2, crop_cloud_around_click

scene_path = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/data/ycbv/test/000050"
frame_id = 1246
checkpoint_path = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth"

cam_K, cam_R_w2c, cam_t_w2c, depth_scale = load_scene_camera(scene_path, frame_id)
rgb, depth = load_scene_rgbd(scene_path, depth_scale, frame_id)
pcd1 = depth_to_pointcloud(depth, cam_K, rgb=rgb)
pcd, plane_points_pcd, plane_model = remove_dominant_plane(
    pcd1, distance_threshold=0.01, ransac_n=3, num_iterations=1000
)

input_points = pick_point_interactive(pcd)

if len(input_points) > 0 and len(input_points[0]) > 0:
    model = load_pointnet2_model(checkpoint_path)
    final_clusters = []
    segmented_indices = []
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    for click_idx in input_points[0]:
        clicked_coord = np.asarray(pcd.points)[click_idx]
        cropped_pcd = crop_cloud_around_click(pcd, clicked_coord, radius=0.08)
        xyz_tensor, cls_tensor, centroid, max_distance = prepare_pcd_for_pointnet2(cropped_pcd, target_class_idx=0
                                                                                   )
        
        if xyz_tensor is not None:
            with torch.no_grad():
                logits, _ = model(xyz_tensor, cls_tensor)
                predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            normalized_points = xyz_tensor.squeeze(0).transpose(1, 0).cpu().numpy()
            
            tensor_pcd = o3d.geometry.PointCloud()
            tensor_pcd.points = o3d.utility.Vector3dVector(normalized_points[:, :3])
            kdtree = o3d.geometry.KDTreeFlann(tensor_pcd)
            
            high_res_points = np.asarray(cropped_pcd.points, dtype=np.float32).copy()
            high_res_points -= centroid
            if max_distance > 0:
                high_res_points /= max_distance
                
            high_res_predictions = np.zeros(len(high_res_points), dtype=np.int32)
            for i, pt in enumerate(high_res_points):
                [_, idx, _] = kdtree.search_knn_vector_3d(pt, 1)
                high_res_predictions[i] = predictions[idx[0]]
                
            unique_parts = np.unique(high_res_predictions)
            foreground_part_id = unique_parts[np.argmax([np.sum(high_res_predictions == p) for p in unique_parts])]
            
            foreground_sub_indices = np.where(high_res_predictions == foreground_part_id)[0]
            foreground_pcd = cropped_pcd.select_by_index(foreground_sub_indices)
            foreground_pcd.paint_uniform_color(np.random.rand(3))
            final_clusters.append(foreground_pcd)
            
            for pt in np.asarray(foreground_pcd.points):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                segmented_indices.append(idx[0])
                
    segmented_indices = np.unique(segmented_indices)
    background_indices = np.setdiff1d(np.arange(len(pcd.points)), segmented_indices)
    
    background_pcd = pcd.select_by_index(background_indices)
    # background_pcd.paint_uniform_color([0.2, 0.2, 0.2])
    # plane_points_pcd.paint_uniform_color([0.15, 0.15, 0.15])
    
    geometries_to_draw = [background_pcd, plane_points_pcd] + final_clusters
    
    if len(final_clusters) > 0:
        with open(f"{scene_path}/scene_gt.json", "r") as f:
            gt_data = json.load(f)
        frame_gt = gt_data[str(frame_id)]
        available_obj_ids = [obj["obj_id"] for obj in frame_gt]
        print(f"Ground Truth objects present in this frame: {available_obj_ids}")
        
        for idx, cluster in enumerate(final_clusters):
            print(f"\nSelected PointNet++ Cluster {idx + 1}:")
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
                contributing_objects.sort(key=lambda x: x["share"], reverse=True)
                for obj in contributing_objects:
                    print(f" Object ID {obj['id']}: {obj['share']:.1f}% (Individual IoU: {obj['iou']:.1f}%)")
                print(f" Final Segmentation Score: {contributing_objects[0]['iou']:.1f}% IoU")
            else:
                print(" No valid objects found in the ground truth.")
                
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="High Contrast PointNet++ 3D Segmentation")
else:
    print("\nEvaluation skipping: No target clusters were picked during interaction.")

import os
import json
import torch
import traceback
import numpy as np
import open3d as o3d
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import cv2

cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda x: 1

from click_segment_core.load import (
    load_scene_camera, load_scene_rgbd, depth_to_pointcloud, 
    remove_dominant_plane, cluster_points, project_points_to_mask, 
    compute_iou, load_ground_truth_mask
)
from click_segment_core.sam_segment import project_mask_to_points, project_points_to_pixels, test_sam2_predictor
from click_segment_core.pointnet import load_pointnet2_model, prepare_pcd_for_pointnet2, crop_cloud_around_click
from click_segment_core.hybrid_segment import fuse_sam_and_dbscan, remove_radius_outliers

def run_batch_evaluation():
    base_test_dir = Path("/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/data/ycbv/test")
    checkpoint_path = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing PointNet++ Backbone on Device: {device}")
    model = load_pointnet2_model(checkpoint_path).to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("--> PyTorch CUDA execution kernels successfully optimized.")
    scene_folders = sorted([f for f in base_test_dir.iterdir() if f.is_dir() and f.name.isdigit()])
    all_results = []
    print(f"--> Comprehensive evaluation mode activated. Parsing through all available frames...")

    for scene_path in scene_folders:
        rgb_dir = scene_path / "rgb"
        if not rgb_dir.exists():
            continue
            
        frame_files = sorted([f.stem for f in rgb_dir.glob("*.png")])
        
        with open(scene_path / "scene_gt.json", "r") as f:
            gt_data = json.load(f)
            
        for frame_str in frame_files:
            frame_id = int(frame_str)
            gt_key = str(frame_id)
            if gt_key not in gt_data:
                continue
                
            frame_gt = gt_data[gt_key]
            available_obj_ids = [obj["obj_id"] for obj in frame_gt]
            
            try:
                cam_K, _, _, depth_scale = load_scene_camera(str(scene_path), frame_id)
                rgb, depth = load_scene_rgbd(str(scene_path), depth_scale, frame_id)
                pcd_full = depth_to_pointcloud(depth, cam_K, rgb=rgb)
                pcd_no_plane, _, _ = remove_dominant_plane(pcd_full, 0.01, 3, 1000)
                
                scene_pts = np.asarray(pcd_no_plane.points)
                if len(scene_pts) == 0:
                    continue
                    
                fx, fy = cam_K[0, 0], cam_K[1, 1]
                cx_opt, cy_opt = cam_K[0, 2], cam_K[1, 2]
                
                proj_x = (scene_pts[:, 0] * fx / scene_pts[:, 2] + cx_opt).astype(int)
                proj_y = (scene_pts[:, 1] * fy / scene_pts[:, 2] + cy_opt).astype(int)
                
                for inst_idx, gt_obj in enumerate(frame_gt):
                    obj_id = gt_obj["obj_id"]
                    gt_mask = load_ground_truth_mask(str(scene_path), frame_id, obj_id)
                    if gt_mask is None or gt_mask.sum() == 0:
                        continue
                    gt_mask_bool = gt_mask > 0
                    
                    y_idx, x_idx = np.where(gt_mask_bool)
                    cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
                    centroid_pixel = np.array([[cx, cy]])
                    
                    pt_dists = (proj_x - cx)**2 + (proj_y - cy)**2
                    best_pt_idx = np.argmin(pt_dists)
                    clicked_coord = scene_pts[best_pt_idx]
                    
                    frame_scores = {
                        "Scene": scene_path.name, "Frame": frame_id, "Obj_ID": obj_id,
                        "Geometric": 0.0, "PointNet++": 0.0, "SAM": 0.0, "Hybrid": 0.0
                    }
                    
                    # 1. RUN GEOMETRIC PIPELINE
                    individual_objects, _ = cluster_points(pcd_no_plane, eps=0.025, min_points=30)
                    best_geo = None
                    min_dist = float('inf')
                    for cluster in individual_objects:
                        dists = np.linalg.norm(np.asarray(cluster.points) - clicked_coord, axis=1)
                        if dists.min() < min_dist:
                            min_dist = dists.min()
                            best_geo = cluster
                    if best_geo is not None and min_dist < 0.05:
                        geo_m = project_points_to_mask(np.asarray(best_geo.points), cam_K, depth.shape)
                        best_iou = 0.0
                        for ref_id in available_obj_ids:
                            try:
                                ref_mask = load_ground_truth_mask(str(scene_path), frame_id, ref_id) > 0
                                overlap = np.logical_and(geo_m, ref_mask).sum()
                                if geo_m.sum() > 0 and (overlap / geo_m.sum()) * 100 > 5.0:
                                    best_iou = max(best_iou, compute_iou(geo_m, ref_mask) * 100)
                            except Exception: continue
                        frame_scores["Geometric"] = best_iou
                        
                    # 2. RUN SAM PIPELINE
                    with torch.inference_mode():
                        _, sam_masks = test_sam2_predictor(str(scene_path), frame_id, centroid_pixel)
                    if len(sam_masks) > 0:
                        target_sam_mask = sam_masks[0]
                        sam_m = target_sam_mask > 0
                        best_iou = 0.0
                        for ref_id in available_obj_ids:
                            try:
                                ref_mask = load_ground_truth_mask(str(scene_path), frame_id, ref_id) > 0
                                overlap = np.logical_and(sam_m, ref_mask).sum()
                                if sam_m.sum() > 0 and (overlap / sam_m.sum()) * 100 > 5.0:
                                    best_iou = max(best_iou, compute_iou(sam_m, ref_mask) * 100)
                            except Exception: continue
                        frame_scores["SAM"] = best_iou
                        
                    # 3. RUN POINTNET++ PIPELINE
                    cropped_pcd = crop_cloud_around_click(pcd_no_plane, clicked_coord, radius=0.08)
                    if len(cropped_pcd.points) >= 32:
                        xyz_t, cls_t, cent, max_d = prepare_pcd_for_pointnet2(cropped_pcd, target_class_idx=0)
                        if xyz_t is not None:
                            xyz_t, cls_t = xyz_t.to(device), cls_t.to(device)
                            with torch.no_grad():
                                logits, _ = model(xyz_t, cls_t)
                                predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                                
                            tensor_pcd = o3d.geometry.PointCloud()
                            tensor_pcd.points = o3d.utility.Vector3dVector(xyz_t.squeeze(0).transpose(1, 0).cpu().numpy()[:, :3])
                            kdtree = o3d.geometry.KDTreeFlann(tensor_pcd)
                            high_res_points = np.asarray(cropped_pcd.points, dtype=np.float32).copy() - cent
                            if max_d > 0: high_res_points /= max_d
                            
                            high_res_preds = np.zeros(len(high_res_points), dtype=np.int32)
                            for i, pt in enumerate(high_res_points):
                                [_, idx, _] = kdtree.search_knn_vector_3d(pt, 1)
                                # Extract the scalar integer index directly out of the list container
                                high_res_preds[i] = predictions[idx[0]]

                                
                            unique_parts = np.unique(high_res_preds)
                            if len(unique_parts) > 0:
                                f_id = unique_parts[np.argmax([np.sum(high_res_preds == p) for p in unique_parts])]
                                # Unpack the flat 1D array from the np.where tuple container
                                fg_indices = np.where(high_res_preds == f_id)[0]
                                fg_pcd = cropped_pcd.select_by_index(fg_indices)

                                pn_m = project_points_to_mask(np.asarray(fg_pcd.points), cam_K, depth.shape)
                                best_iou = 0.0
                                for ref_id in available_obj_ids:
                                    try:
                                        ref_mask = load_ground_truth_mask(str(scene_path), frame_id, ref_id) > 0
                                        overlap = np.logical_and(pn_m, ref_mask).sum()
                                        if pn_m.sum() > 0 and (overlap / pn_m.sum()) * 100 > 5.0:
                                            best_iou = max(best_iou, compute_iou(pn_m, ref_mask) * 100)
                                    except Exception: continue
                                frame_scores["PointNet++"] = best_iou
                                
                    # 4. RUN HYBRID PIPELINE
                    if len(sam_masks) > 0:
                        sam_pts = project_mask_to_points(target_sam_mask, cam_K, depth)
                        if len(sam_pts) > 0:
                            fused_pts = fuse_sam_and_dbscan(sam_pts, np.asarray(pcd_no_plane.points), radius=0.02)
                            if len(fused_pts) > 0:
                                filtered_pcd = remove_radius_outliers(fused_pts, radius=0.01)
                                hyb_m = project_points_to_mask(np.asarray(filtered_pcd.points), cam_K, depth.shape)
                                best_iou = 0.0
                                for ref_id in available_obj_ids:
                                    try:
                                        ref_mask = load_ground_truth_mask(str(scene_path), frame_id, ref_id) > 0
                                        overlap = np.logical_and(hyb_m, ref_mask).sum()
                                        if hyb_m.sum() > 0 and (overlap / hyb_m.sum()) * 100 > 5.0:
                                            best_iou = max(best_iou, compute_iou(hyb_m, ref_mask) * 100)
                                    except Exception: 
                                        continue
                                frame_scores["Hybrid"] = best_iou
                                all_results.append(frame_scores)
                                print(f"[{scene_path.name} | Frame {frame_id:04d}] Obj {obj_id:02d} Scores -> Geo: {frame_scores['Geometric']:.1f}% | SAM: {frame_scores['SAM']:.1f}% | PN++: {frame_scores['PointNet++']:.1f}% | Hybrid: {frame_scores['Hybrid']:.1f}%")
            except Exception as e:
                print(f"\n[CRASH ENCOUNTERED] Scene {scene_path.name} | Frame {frame_id} failed with exception:")
                traceback.print_exc()
                continue

        if len(all_results) > 0:
            df_raw = pd.DataFrame(all_results)
            df_scene_summary = df_raw.groupby("Scene", as_index=False)[["Geometric", "PointNet++", "SAM", "Hybrid"]].mean()
            global_mean = df_raw[["Geometric", "PointNet++", "SAM", "Hybrid"]].mean().to_dict()
            global_mean["Scene"] = "OVERALL MEAN IoU"
            df_summary = pd.concat([df_scene_summary, pd.DataFrame([global_mean])], ignore_index=True)
            output_md = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/metrics_summary.md"
            with open(output_md, "w") as f:
                f.write(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False, floatfmt=".1f"))
            print("\n" + "="*80)
            print("                 COMPLETED DATASET BENCHMARK METRIC SUMMARY")
            print("="*80)
            print(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False, floatfmt=".1f"))
            print("="*80)

class openmd:
    def enter(self):
        pass
    def exit(self, *args):
        pass

if __name__ == "__main__":
    run_batch_evaluation()
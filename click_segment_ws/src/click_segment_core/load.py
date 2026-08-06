import json
import cv2
import numpy as np
import open3d as o3d

def load_scene_camera(scene_path, frame_id):
    with open(f"{scene_path}/scene_camera.json", "r") as f:
        camera_data = json.load(f)
    frame_data = camera_data[str(frame_id)]
    cam_K = np.reshape(frame_data["cam_K"], (3, 3))
    cam_R_w2c = np.reshape(frame_data["cam_R_w2c"], (3, 3))
    cam_t_w2c = np.array(frame_data["cam_t_w2c"])
    depth_scale = frame_data["depth_scale"]
    return cam_K, cam_R_w2c, cam_t_w2c, depth_scale

def load_scene_rgbd(scene_path, depth_scale, frame_id):
    rgb = cv2.imread(f"{scene_path}/rgb/{frame_id:06d}.png", cv2.IMREAD_COLOR)
    rgb =cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(f"{scene_path}/depth/{frame_id:06d}.png", cv2.IMREAD_UNCHANGED)*depth_scale/ 1000.0 
    depth = depth.astype(np.float32)
    return rgb, depth

def depth_to_pointcloud(depth, cam_K, rgb=None):
    cam_K_o3d = o3d.camera.PinholeCameraIntrinsic()
    cam_K_o3d.set_intrinsics(depth.shape[1], depth.shape[0], cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])

    if rgb is not None:
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(depth)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_K_o3d)
    else:
        depth_o3d = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, cam_K_o3d)

    return pcd

def remove_dominant_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, indices = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    plane_points_pcd = pcd.select_by_index(indices)
    object_points_pcd = pcd.select_by_index(indices, invert=True)
    return object_points_pcd, plane_points_pcd, plane_model

def cluster_points(pcd, eps=0.01, min_points=20):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        clusters.append(cluster_pcd)#
    print(len(clusters), "clusters found")
    return clusters, labels


def get_clicked_cluster(pcd, labels, clicked_point_index):
    clicked_label = labels[clicked_point_index]
    if clicked_label == -1:
        return None
    cluster_indices = np.where(labels == clicked_label)[0]
    cluster_pcd = pcd.select_by_index(cluster_indices)
    return cluster_pcd

def pick_point_interactive(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked_points = []
    picked_points.append(vis.get_picked_points())
    return picked_points

def project_points_to_mask(points_3d, cam_K, image_shape):
    u = np.round((cam_K[0, 0] * points_3d[:, 0] / points_3d[:, 2]) + cam_K[0, 2]).astype(int)
    v = np.round((cam_K[1, 1] * points_3d[:, 1] / points_3d[:, 2]) + cam_K[1, 2]).astype(int)
    valid_indices = (u >= 0) & (u < image_shape[1]) &  (v >= 0) & (v < image_shape[0])
    u = u[valid_indices]
    v = v[valid_indices]
    mask = np.zeros(image_shape, dtype=bool)
    mask[v, u] = True   
    return mask

def compute_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    if union.sum() == 0:
        return 1.0  # If both masks are empty, we consider IoU to be 1
    iou = intersection.sum() / union.sum()
    return iou

def load_ground_truth_mask(scene_path, frame_id, obj_id):
    with open(f"{scene_path}/scene_gt.json", "r") as f:
        gt_data = json.load(f)
    frame_gt = gt_data[str(frame_id)]
    obj_index = None
    for i, obj in enumerate(frame_gt):
        if obj["obj_id"] == obj_id:
            obj_index = i
            break
    if obj_index is None:
        raise ValueError(f"Object ID {obj_id} not found in frame {frame_id}")
    mask_path = f"{scene_path}/mask_visib/{frame_id:06d}_{obj_index:06d}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask

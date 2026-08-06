import sys
import torch
import numpy as np
import open3d as o3d

repo_root = "/home/vishnucharan/Projects/point_cloud_segmentation/click_segment_ws/Pointnet_Pointnet2_pytorch"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.pointnet2_part_seg_msg import get_model

def load_pointnet2_model(checkpoint_path, num_classes=16, num_parts=50):
    model = get_model(num_parts, normal_channel=True)
    checkpoint = torch.load(
        checkpoint_path, 
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def crop_cloud_around_click(o3d_pcd, click_coord, radius=0.15):
    points = np.asarray(o3d_pcd.points)
    distances = np.linalg.norm(points - click_coord, axis=1)
    indices = np.where(distances <= radius)[0]
    return o3d_pcd.select_by_index(indices)

def prepare_pcd_for_pointnet2(o3d_pcd, target_class_idx=0, num_points=2048, num_classes=16):
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    
    points = np.asarray(o3d_pcd.points, dtype=np.float32)
    normals = np.asarray(o3d_pcd.normals, dtype=np.float32)
    current_size = len(points)
    
    if current_size == 0:
        return None, None, None, None
        
    if current_size > num_points:
        indices = np.random.choice(current_size, num_points, replace=False)
        points = points[indices]
        normals = normals[indices]
    elif current_size < num_points:
        indices = np.random.choice(current_size, num_points - current_size, replace=True)
        points = np.vstack((points, points[indices]))
        normals = np.vstack((normals, normals[indices]))
        
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_distance > 0:
        points /= max_distance
        
    xyz_normal_features = np.hstack((points, normals))
    xyz_tensor = torch.from_numpy(xyz_normal_features).transpose(1, 0).unsqueeze(0)
    
    cls_label = np.zeros((1, num_classes), dtype=np.float32)
    cls_label[0, target_class_idx] = 1.0
    cls_tensor = torch.from_numpy(cls_label)
    
    if torch.cuda.is_available():
        xyz_tensor = xyz_tensor.cuda()
        cls_tensor = cls_tensor.cuda()
        
    return xyz_tensor, cls_tensor, centroid, max_distance

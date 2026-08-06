import numpy as np
import open3d as o3d

def fuse_sam_and_dbscan(sam_points, dbscan_points, radius):
    sam_pcd = o3d.geometry.PointCloud()
    sam_pcd.points = o3d.utility.Vector3dVector(sam_points)
    dbscan_pcd = o3d.geometry.PointCloud()
    dbscan_pcd.points = o3d.utility.Vector3dVector(dbscan_points)
    dbscan_kdtree = o3d.geometry.KDTreeFlann(dbscan_pcd)
    fused_points = []
    for point in sam_points:
        [k, idx, _] = dbscan_kdtree.search_radius_vector_3d(point, radius)
        if k > 0:
            fused_points.append(point)
    return np.array(fused_points)

def remove_radius_outliers(points, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_radius_outlier(nb_points=40, radius=radius)
    return cl 
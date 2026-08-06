import cv2
import numpy as np
import glob
import os
import torch
import open3d as o3d
from abc import ABC, abstractmethod

# ==========================================
# CONFIGURATION
# ==========================================
RGB_DIR = "percipio_captures1/rgb_images/"
DEPTH_DIR = "percipio_captures1/depth_npy/" # Ensure this points to your .npy files

pallet_corners = []

def click_event(event, x, y, flags, param):
    """Handles mouse clicks to define the 4 corners of the pallet ROI."""
    global pallet_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pallet_corners) < 4:
            pallet_corners.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pallet_corners) > 0:
            pallet_corners.pop()

# ==========================================
# 1. MODEL INTERFACES 
# ==========================================
class BaseSegmenter(ABC):
    @abstractmethod
    def load_model(self): pass

    @abstractmethod
    def predict(self, rgb_image, box_prompt, point_prompts): pass

class SAM2Segmenter(BaseSegmenter):
    def __init__(self, checkpoint_path, config_path):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.predictor = None

    def load_model(self):
        print("[Model] Loading SAM 2 Predictor...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(self.config_path, self.checkpoint_path)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("[Model] SAM 2 Ready.")

    def predict(self, rgb_image, box_prompt, point_prompts):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_image)
            labels = np.ones(len(point_prompts), dtype=np.int32)
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(point_prompts),
                point_labels=labels,
                box=np.array(box_prompt),
                multimask_output=False 
            )
        return masks[0], scores[0]

# ==========================================
# 2. DATA INGESTION ENGINE
# ==========================================
def get_projected_points(npy_path, img_w, img_h):
    """
    Loads raw .npy files using the user's exact math and safely projects to 2D.
    """
    if not os.path.exists(npy_path):
        return None, None, None, None, None
        
    raw_npy = np.load(npy_path)
    
    # Smart loader: Tries structured array first, falls back to normal array
    try:
        x = raw_npy['x'].flatten()
        y = raw_npy['y'].flatten()
        z = raw_npy['z'].flatten()
    except (IndexError, ValueError):
        raw_flat = raw_npy.reshape(-1, raw_npy.shape[-1])
        x, y, z = raw_flat[:, 0], raw_flat[:, 1], raw_flat[:, 2]

    # Filter invalid points BEFORE casting to prevent NaN crashes
    valid_mask = (z > 0.1) & (~np.isnan(z)) & (~np.isnan(x)) & (~np.isnan(y))
    x_v, y_v, z_v = x[valid_mask], y[valid_mask], z[valid_mask]
    
    if len(z_v) == 0:
        return None, None, None, None, None

    # User's exact projection logic
    u, v = x_v / z_v, y_v / z_v
    max_abs_u, max_abs_v = np.percentile(np.abs(u), 99.5), np.percentile(np.abs(v), 99.5)
    
    # Prevent divide by zero on completely flat data
    if max_abs_u == 0: max_abs_u = 1e-5
    if max_abs_v == 0: max_abs_v = 1e-5
    
    u_norm = np.clip((u / max_abs_u * (img_w / 2) + img_w / 2).astype(np.int32), 0, img_w - 1)
    v_norm = np.clip((v / max_abs_v * (img_h / 2) + img_h / 2).astype(np.int32), 0, img_h - 1)
    
    return x_v, y_v, z_v, u_norm, v_norm

# ==========================================
# 3. 3D HYBRID TARGET LOCATOR (RANSAC + DBSCAN)
# ==========================================
class Hybrid3DLocator:
    def __init__(self, pad_ratio=0.15):
        self.pad_ratio = pad_ratio

    def find_target(self, img_shape, x, y, z, u, v):
        if len(z) < 10: return None
        
        # 1. Isolate the highest points (Top 15cm)
        peak_z = np.percentile(z, 1.0)
        top_mask = z <= (peak_z + 0.15)
        tx, ty, tz = x[top_mask], y[top_mask], z[top_mask]
        tu, tv = u[top_mask], v[top_mask]
        
        if len(tz) < 5: return None

        # 2. Cluster points physically touching in 3D space
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack((tx, ty, tz)))
        
        labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=3, print_progress=False))
        if len(labels) == 0 or labels.max() < 0: return None

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        sorted_labels = unique_labels[np.argsort(-counts)]

        # 3. Analyze clusters via RANSAC
        for lbl in sorted_labels:
            cluster_idx = np.where(labels == lbl)[0]
            if len(cluster_idx) < 5: continue

            cluster_pcd = pcd.select_by_index(cluster_idx)
            
            # Extract dominant plane mathematically
            plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
            A, B, C, D = plane_model
            
            normal = np.array([A, B, C])
            normal /= np.linalg.norm(normal)
            
            # 4. Normal Validation: Only accept faces pointing UP
            if abs(normal[2]) > 0.75: 
                inlier_u = tu[cluster_idx][inliers]
                inlier_v = tv[cluster_idx][inliers]

                # Convert purely geometric points to a solid 2D mask
                geom_mask = np.zeros(img_shape, dtype=np.uint8)
                geom_mask[inlier_v, inlier_u] = 255

                # Fill tape holes to make it a solid polygon
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
                geom_mask = cv2.morphologyEx(geom_mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(geom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                
                best_cnt = max(contours, key=cv2.contourArea)
                geom_rot_rect = cv2.minAreaRect(best_cnt)
                
                solid_poly = np.zeros(img_shape, dtype=np.uint8)
                cv2.drawContours(solid_poly, [best_cnt], -1, 255, -1)
                dist_map = cv2.distanceTransform(solid_poly, cv2.DIST_L2, 5)
                _, _, _, geom_centroid = cv2.minMaxLoc(dist_map)
                
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                pad_w, pad_h = int(bw * self.pad_ratio), int(bh * self.pad_ratio)
                
                return {
                    "geom_rot_rect": geom_rot_rect,
                    "geom_centroid": geom_centroid,
                    "sam_bbox": (bx, by, bw, bh),
                    "sam_point": [geom_centroid], 
                    "sam_pad": (pad_w, pad_h)
                }
                
        return None

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    global pallet_corners
    
    segmenter = SAM2Segmenter("sam2/checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml")
    segmenter.load_model()
    locator = Hybrid3DLocator(pad_ratio=0.15) 

    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")) + glob.glob(os.path.join(RGB_DIR, "*.jpg")))
    if len(rgb_files) < 3:
        print("Error: Need at least 3 RGB images.")
        return

    # Image 1 is the Empty Pallet
    pallet_img_path = rgb_files[1] 
    img_bgr_pallet = cv2.imread(pallet_img_path)
    img_h, img_w = img_bgr_pallet.shape[:2]

    print("\n[Calibration] Please draw the pallet region on the EMPTY PALLET image...")
    cv2.namedWindow("0. Session Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("0. Session Calibration", int(img_w*0.6), int(img_h*0.6))
    cv2.setMouseCallback("0. Session Calibration", click_event)
    
    while True:
        clone = img_bgr_pallet.copy()
        for i, pt in enumerate(pallet_corners):
            cv2.circle(clone, pt, 5, (0, 255, 0), -1)
            if i > 0: cv2.line(clone, pallet_corners[i-1], pt, (0, 255, 0), 2)
                
        if len(pallet_corners) == 4:
            cv2.line(clone, pallet_corners[3], pallet_corners[0], (0, 255, 0), 2)
            cv2.putText(clone, "Press ENTER to confirm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("0. Session Calibration", clone)
        k = cv2.waitKey(15) & 0xFF
        if k == 13 and len(pallet_corners) == 4: break
        elif k in [ord('c'), ord('C')]: pallet_corners = []
            
    cv2.destroyWindow("0. Session Calibration")
    
    user_region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    poly_pts = np.array(pallet_corners, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(user_region_mask, [poly_pts], 255)

    print("\n[Math] Extracting Pallet Base Plane via RANSAC...")
    # Exact Name Matching for Timestamp Sync
    base_name = os.path.basename(pallet_img_path)
    name_no_ext = os.path.splitext(base_name)[0]
    depth_name = name_no_ext.replace("rgb", "depth") + ".npy"
    depth_path_pallet = os.path.join(DEPTH_DIR, depth_name)
    
    x_v, y_v, z_v, u_norm, v_norm = get_projected_points(depth_path_pallet, img_w, img_h)
    if x_v is None:
        print(f"Error: Could not process {depth_path_pallet}")
        return
        
    inside_poly = user_region_mask[v_norm, u_norm] == 255
    px, py, pz = x_v[inside_poly], y_v[inside_poly], z_v[inside_poly]
    
    if len(pz) < 10:
        print("Error: Not enough valid points inside the drawn polygon.")
        return

    # Extract Pallet Plane Equation
    pallet_pcd = o3d.geometry.PointCloud()
    pallet_pcd.points = o3d.utility.Vector3dVector(np.column_stack((px, py, pz)))
    pallet_plane, _ = pallet_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    print(f" -> Pallet Plane Equation: {pallet_plane[0]:.3f}x + {pallet_plane[1]:.3f}y + {pallet_plane[2]:.3f}z + {pallet_plane[3]:.3f} = 0")

    # Start BATCH PROCESSING Loop (Frame 2 onwards)
    cv2.namedWindow("1. Pure RANSAC Geometry", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. SAM 2 Verification", cv2.WINDOW_NORMAL)

    for idx in range(2, len(rgb_files)):
        rgb_path = rgb_files[idx]
        depth_name = os.path.splitext(os.path.basename(rgb_path))[0].replace("rgb", "depth") + ".npy"
        depth_path = os.path.join(DEPTH_DIR, depth_name)
        
        print(f"\nProcessing Image [{idx+1}/{len(rgb_files)}]: {os.path.basename(rgb_path)}")
        img_bgr = cv2.imread(rgb_path)
        
        bx, by, bz, bu, bv = get_projected_points(depth_path, img_w, img_h)
        if bx is None: continue
            
        # Apply Polygon Mask
        inside = user_region_mask[bv, bu] == 255
        bx, by, bz = bx[inside], by[inside], bz[inside]
        bu, bv = bu[inside], bv[inside]
        
        # Apply Background Subtraction (Remove Floor/Pallet)
        A, B, C, D = pallet_plane
        distances = np.abs(A * bx + B * by + C * bz + D) / np.sqrt(A**2 + B**2 + C**2)
        above_floor = distances > 0.03 # 3cm strict threshold
        
        bx, by, bz = bx[above_floor], by[above_floor], bz[above_floor]
        bu, bv = bu[above_floor], bv[above_floor]
        
        target = locator.find_target((img_h, img_w), bx, by, bz, bu, bv)
        if not target:
            print(" -> Failed to locate target.")
            continue
            
        # ----------------------------------------------------
        # WINDOW 1: RANSAC Geometry
        # ----------------------------------------------------
        geom_img = img_bgr.copy()
        cv2.polylines(geom_img, [poly_pts], True, (0, 255, 0), 2)
        box_points_geom = np.int32(cv2.boxPoints(target["geom_rot_rect"]))
        cv2.drawContours(geom_img, [box_points_geom], 0, (0, 255, 255), 2)
        
        gc_x, gc_y = target["geom_centroid"]
        cv2.circle(geom_img, (gc_x, gc_y), 5, (0, 0, 255), -1)

        # ----------------------------------------------------
        # WINDOW 2: SAM 2 AI Mask
        # ----------------------------------------------------
        x, y, w, h = target["sam_bbox"]
        pad_w, pad_h = target["sam_pad"]
        
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
        
        cropped_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        local_box = [x - x1, y - y1, x + w - x1, y + h - y1]
        local_points = [[px - x1, py - y1] for (px, py) in target["sam_point"]]

        local_mask, _ = segmenter.predict(cropped_rgb, local_box, local_points)

        sam_img = img_bgr.copy()
        global_sam_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        global_sam_mask[y1:y2, x1:x2] = (local_mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(global_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            sam_cnt = max(contours, key=cv2.contourArea)
            sam_rot_rect = cv2.minAreaRect(sam_cnt)
            box_points_sam = np.int32(cv2.boxPoints(sam_rot_rect))
            cv2.drawContours(sam_img, [box_points_sam], 0, (255, 0, 255), 2)
            
            dist_map = cv2.distanceTransform(global_sam_mask, cv2.DIST_L2, 5)
            _, _, _, sam_centroid = cv2.minMaxLoc(dist_map)
            cv2.circle(sam_img, sam_centroid, 5, (255, 255, 255), -1)
            cv2.circle(sam_img, sam_centroid, 3, (0, 0, 255), -1)

        overlay = sam_img.copy()
        overlay[global_sam_mask == 255] = [0, 255, 0] 
        sam_final_output = cv2.addWeighted(sam_img, 0.6, overlay, 0.4, 0)

        cv2.imshow("1. Pure RANSAC Geometry", geom_img)
        cv2.imshow("2. SAM 2 Verification", sam_final_output)
        
        if cv2.waitKey(0) & 0xFF == ord('q'): break

if __name__ == "__main__":
    main()
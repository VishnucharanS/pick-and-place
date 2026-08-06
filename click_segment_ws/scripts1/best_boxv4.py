import cv2
import numpy as np
import glob
import os
import torch
import open3d as o3d
from abc import ABC, abstractmethod
import math

# ==========================================
# CONFIGURATION
# ==========================================
RGB_DIR = "percipio_captures1/rgb_images/"
DEPTH_DIR = "percipio_captures1/depth_npy/" # Folder where your generated .pcd files reside

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
                multimask_output=False # We only need the single best mask now
            )
            
        return masks[0], scores[0]

# ==========================================
# 2. HYBRID 3D LOCATOR (Scoring Matrix)
# ==========================================
class Hybrid3DLocator:
    def __init__(self, pad_ratio=0.15):
        self.pad_ratio = pad_ratio

    def project_to_2d(self, x, y, z, img_w, img_h):
        """Standard pinhole camera model to map 3D PCD points to 2D RGB Pixels."""
        fx = fy = max(img_h, img_w) * 0.8
        cx, cy = img_w / 2.0, img_h / 2.0
        
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy
        
        u_norm = np.clip(np.round(u).astype(np.int32), 0, img_w - 1)
        v_norm = np.clip(np.round(v).astype(np.int32), 0, img_h - 1)
        return u_norm, v_norm

    def align_plane_to_camera(self, plane_eq):
        """Ensures the plane normal always points UP towards the camera (origin at 0,0,0)."""
        A, B, C, D = plane_eq
        # In optical depth coordinates where Z > 0, a vector pointing to the camera has C < 0.
        if C > 0:
            return (-A, -B, -C, -D)
        return (A, B, C, D)

    def get_distance_to_plane(self, x, y, z, plane_eq):
        """Calculates exact SIGNED orthogonal height from the pallet plane (positive = above pallet)."""
        A, B, C, D = plane_eq
        num = A*x + B*y + C*z + D # Removed np.abs() to preserve directional sign
        den = np.sqrt(A**2 + B**2 + C**2)
        return num / den

    def find_target(self, pcd, pallet_plane, user_region_mask, img_shape):
        img_h, img_w = img_shape
        pts = np.asarray(pcd.points)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # 1. Map to 2D and Apply User Polygon
        u, v = self.project_to_2d(x, y, z, img_w, img_h)
        inside_poly = user_region_mask[v, u] == 255
        
        x, y, z = x[inside_poly], y[inside_poly], z[inside_poly]
        u, v = u[inside_poly], v[inside_poly]
        if len(z) < 10: return None
        
        # 2. Signed 3D Background Subtraction (Vanish the pallet and floor)
        heights_above_pallet = self.get_distance_to_plane(x, y, z, pallet_plane)
        print(f"  [Debug] Scene Signed Height Range -> Min: {heights_above_pallet.min():.3f}m | Max: {heights_above_pallet.max():.3f}m")
        
        above_pallet = heights_above_pallet > 0.03 # Positive distance only (>3cm above wood)
        
        bx, by, bz = x[above_pallet], y[above_pallet], z[above_pallet]
        bu, bv = u[above_pallet], v[above_pallet]
        b_heights = heights_above_pallet[above_pallet]
        if len(bz) < 10: return None

        # 3. Tape-Resistant DBSCAN Clustering
        # eps=0.08 (8cm) radius allows the algorithm to jump across black tape holes
        box_pcd = o3d.geometry.PointCloud()
        box_pcd.points = o3d.utility.Vector3dVector(np.column_stack((bx, by, bz)))
        labels = np.array(box_pcd.cluster_dbscan(eps=0.08, min_points=10, print_progress=False))
        
        if len(labels) == 0 or labels.max() < 0: return None
        unique_labels = np.unique(labels[labels >= 0])
        
        # Pallet Normal Vector
        pA, pB, pC, _ = pallet_plane
        pallet_normal = np.array([pA, pB, pC])
        pallet_normal = pallet_normal / np.linalg.norm(pallet_normal)

        best_score = -1
        best_cluster_data = None
        
        print("\n  --- Target Scoring Matrix ---")
        
        # 4. The Weighted Scoring System
        for lbl in unique_labels:
            cluster_idx = np.where(labels == lbl)[0]
            if len(cluster_idx) < 15: continue # Ignore tiny noise
            
            cluster_pcd = box_pcd.select_by_index(cluster_idx)
            
            # Graspability Check: Run RANSAC on the cluster itself
            plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=500)
            cA, cB, cC, _ = plane_model
            cluster_normal = np.array([cA, cB, cC])
            cluster_normal = cluster_normal / np.linalg.norm(cluster_normal)
            
            # Calculate Tilt relative to pallet floor
            tilt_dot = np.abs(np.dot(pallet_normal, cluster_normal))
            tilt_deg = math.degrees(math.acos(np.clip(tilt_dot, 0.0, 1.0)))
            
            # Wall Killer: Reject side walls immediately (> 45 degrees tilt)
            if tilt_deg > 45:
                print(f"  [Cluster {lbl}] REJECTED: Side-wall detected (Tilt: {tilt_deg:.1f} deg)")
                continue
                
            # Metric 1: Height (90th percentile to ignore extreme noise points)
            cluster_heights = b_heights[cluster_idx[inliers]]
            avg_height = np.percentile(cluster_heights, 90)
            norm_height = min(avg_height / 0.5, 1.0) # Assume max box height is ~0.5m
            
            # Metric 2: Flatness (Ratio of inliers to total cluster points)
            flatness = len(inliers) / len(cluster_idx)
            
            # Metric 3: Area (Based on point count due to ToF sparsity)
            norm_area = min(len(inliers) / 200.0, 1.0) 
            
            # WEIGHTED FORMULA
            score = (0.5 * norm_height) + (0.3 * flatness) + (0.2 * norm_area)
            
            print(f"  [Cluster {lbl}] Pts: {len(inliers):>3} | Height: {avg_height:.3f}m | Flat: {flatness:.2f} | Tilt: {tilt_deg:>4.1f}° | Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                # Store the exact INLIERS of the RANSAC plane (guarantees flatness)
                best_cluster_data = {
                    "u": bu[cluster_idx[inliers]],
                    "v": bv[cluster_idx[inliers]]
                }

        if not best_cluster_data:
            return None
            
        print(f"  => WINNER: Score {best_score:.3f}")

        # 5. Prompt Generation from best inliers
        inlier_u = best_cluster_data["u"]
        inlier_v = best_cluster_data["v"]
        
        geom_mask = np.zeros(img_shape, dtype=np.uint8)
        geom_mask[inlier_v, inlier_u] = 255
        
        # Heal point cloud sparsity strictly for 2D bounding box generation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        geom_mask = cv2.morphologyEx(geom_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(geom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        best_cnt = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        geom_rot_rect = cv2.minAreaRect(best_cnt)
        
        # Solid geometric center
        solid_poly = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(solid_poly, [best_cnt], -1, 255, -1)
        dist_map = cv2.distanceTransform(solid_poly, cv2.DIST_L2, 5)
        _, _, _, geom_centroid = cv2.minMaxLoc(dist_map)
        
        pad_w, pad_h = int(bw * self.pad_ratio), int(bh * self.pad_ratio)
        
        return {
            "geom_rot_rect": geom_rot_rect,
            "geom_centroid": geom_centroid,
            "sam_bbox": (bx, by, bw, bh),
            "sam_pad": (pad_w, pad_h)
        }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    global pallet_corners
    
    segmenter = SAM2Segmenter("sam2/checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml")
    segmenter.load_model()
    locator = Hybrid3DLocator(pad_ratio=0.15) 

    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    if len(rgb_files) < 3:
        print("Error: Need images in directory.")
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
    
    timestamp = os.path.splitext(os.path.basename(pallet_img_path))[0].replace("rgb_", "")
    pcd_path_pallet = os.path.join(DEPTH_DIR, f"depth_{timestamp}.pcd")
    
    if not os.path.exists(pcd_path_pallet):
        print(f"Error: Could not find matching PCD file: {pcd_path_pallet}")
        return
        
    pallet_pcd = o3d.io.read_point_cloud(pcd_path_pallet)
    pallet_pts = np.asarray(pallet_pcd.points)
    px, py, pz = pallet_pts[:, 0], pallet_pts[:, 1], pallet_pts[:, 2]
    
    pu, pv = locator.project_to_2d(px, py, pz, img_w, img_h)
    inside_poly = user_region_mask[pv, pu] == 255
    
    pallet_roi_pcd = o3d.geometry.PointCloud()
    pallet_roi_pcd.points = o3d.utility.Vector3dVector(pallet_pts[inside_poly])
    
    pallet_plane, _ = pallet_roi_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    
    # Align normal to point UP towards the camera
    pallet_plane = locator.align_plane_to_camera(pallet_plane)
    
    print(f" -> Aligned Pallet Equation: {pallet_plane[0]:.3f}x + {pallet_plane[1]:.3f}y + {pallet_plane[2]:.3f}z + {pallet_plane[3]:.3f} = 0")
    print(f" -> Aligned Normal Vector: ({pallet_plane[0]:.3f}, {pallet_plane[1]:.3f}, {pallet_plane[2]:.3f}) [Pointing UP to camera]")

    cv2.namedWindow("1. RANSAC 3D Geometry", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. AI Safe Pick Extraction", cv2.WINDOW_NORMAL)
    cw, ch = int(img_w * 0.5), int(img_h * 0.5)
    cv2.resizeWindow("1. RANSAC 3D Geometry", cw, ch)
    cv2.resizeWindow("2. AI Safe Pick Extraction", cw, ch)

    for idx in range(2, len(rgb_files)):
        rgb_path = rgb_files[idx]
        timestamp = os.path.splitext(os.path.basename(rgb_path))[0].replace("rgb_", "")
        pcd_path = os.path.join(DEPTH_DIR, f"depth_{timestamp}.pcd")
        
        print(f"\n==============================================")
        print(f"Processing [{idx+1}/{len(rgb_files)}]: rgb_{timestamp}.png")
        
        img_bgr = cv2.imread(rgb_path)
        
        if not os.path.exists(pcd_path):
            print(f" -> Missing PCD file, skipping...")
            continue
            
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # Execute 3D Smart Scoring Pipeline
        target = locator.find_target(pcd, pallet_plane, user_region_mask, (img_h, img_w))
        if not target:
            print(" -> Failed to locate a valid geometric target.")
            continue
            
        # ----------------------------------------------------
        # WINDOW 1: Pure RANSAC 3D Geometry Visualization
        # ----------------------------------------------------
        geom_img = img_bgr.copy()
        cv2.polylines(geom_img, [poly_pts], True, (0, 255, 0), 2)
        
        box_points_geom = np.int32(cv2.boxPoints(target["geom_rot_rect"]))
        cv2.drawContours(geom_img, [box_points_geom], 0, (0, 255, 255), 2)
        
        gc_x, gc_y = target["geom_centroid"]
        cv2.circle(geom_img, (gc_x, gc_y), 6, (255, 255, 255), -1)
        cv2.circle(geom_img, (gc_x, gc_y), 3, (0, 0, 255), -1)

        # ----------------------------------------------------
        # WINDOW 2: SAM 2 AI Verification & Safe Pick Point
        # ----------------------------------------------------
        x, y, w, h = target["sam_bbox"]
        pad_w, pad_h = target["sam_pad"]
        
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
        
        cropped_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        local_box = [x - x1, y - y1, x + w - x1, y + h - y1]
        
        # Hand off exactly ONE geometric center point
        local_points = [[gc_x - x1, gc_y - y1]]

        local_mask, _ = segmenter.predict(cropped_rgb, local_box, local_points)

        global_sam_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        global_sam_mask[y1:y2, x1:x2] = (local_mask * 255).astype(np.uint8)
        
        sam_img = img_bgr.copy()
        contours, _ = cv2.findContours(global_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            sam_cnt = max(contours, key=cv2.contourArea)
            sam_rot_rect = cv2.minAreaRect(sam_cnt)
            box_points_sam = np.int32(cv2.boxPoints(sam_rot_rect))
            cv2.drawContours(sam_img, [box_points_sam], 0, (255, 0, 255), 2)
            
            # The Ultimate Safe Centroid (Distance Transform strictly on AI edges)
            solid_sam_poly = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(solid_sam_poly, [sam_cnt], -1, 255, -1)
            dist_map = cv2.distanceTransform(solid_sam_poly, cv2.DIST_L2, 5)
            _, _, _, safe_ai_centroid = cv2.minMaxLoc(dist_map)
            
            cv2.circle(sam_img, safe_ai_centroid, 8, (255, 255, 255), -1)
            cv2.circle(sam_img, safe_ai_centroid, 4, (0, 0, 255), -1)
            cv2.putText(sam_img, "Safe Pick Center", (safe_ai_centroid[0] + 15, safe_ai_centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        overlay = sam_img.copy()
        overlay[global_sam_mask == 255] = [0, 255, 0] 
        sam_final_output = cv2.addWeighted(sam_img, 0.6, overlay, 0.4, 0)

        cv2.imshow("1. RANSAC 3D Geometry", geom_img)
        cv2.imshow("2. AI Safe Pick Extraction", sam_final_output)
        
        print(" -> Press ANY KEY for next image, or 'q' to Quit.")
        if cv2.waitKey(0) & 0xFF == ord('q'): break

if __name__ == "__main__":
    main()
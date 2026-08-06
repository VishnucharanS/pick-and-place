import cv2
import numpy as np
import glob
import os
import torch
from abc import ABC, abstractmethod

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_INDEX = 11

RGB_DIR = "percipio_captures/color/"
DEPTH_DIR = "percipio_captures/points/"

# Global variables for the interactive clicker
pallet_corners = []

def click_event(event, x, y, flags, param):
    """Handles mouse clicks to define the 4 corners of the pallet."""
    global pallet_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pallet_corners) < 4:
            pallet_corners.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pallet_corners) > 0:
            pallet_corners.pop()

# ==========================================
# 1. MODEL INTERFACES (EASY TO SWAP LATER)
# ==========================================
class BaseSegmenter(ABC):
    """Abstract base class for any segmentation model."""
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, rgb_image, box_prompt, point_prompt):
        """
        rgb_image: Cropped BGR/RGB image array
        box_prompt: [x1, y1, x2, y2] relative to cropped image
        point_prompt: [cx, cy] relative to cropped image
        Returns: binary mask (2D numpy array), confidence score
        """
        pass

class SAM2Segmenter(BaseSegmenter):
    """Implementation for SAM 2."""
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

    def predict(self, rgb_image, box_prompt, point_prompt):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_image)
            # Enable multimask to let SAM guess sub-parts vs whole objects
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([point_prompt]),
                point_labels=np.array([1]), # 1 indicates a positive point prompt
                box=np.array(box_prompt),
                multimask_output=True 
            )
            
        # FIX: Force SAM to pick the "Whole Object" mask instead of the "Sub-part" (tape) mask
        # By finding the mask with the largest pixel area.
        largest_mask_idx = np.argmax([np.sum(m) for m in masks])
        
        return masks[largest_mask_idx], scores[largest_mask_idx]

# ==========================================
# 2. TARGET LOCATOR (GEOMETRY & MATH)
# ==========================================
class TargetLocator:
    """Handles 3D depth processing, projection, and 2D prompt generation."""
    def __init__(self, z_tolerance=0.05, pad_ratio=0.25):
        self.z_tolerance = z_tolerance  # How deep into the top surface to look (meters)
        self.pad_ratio = pad_ratio      # How much to expand the box for the model's context

    def find_target(self, img_shape, x_raw, y_raw, z_raw, user_region_mask):
        h, w = img_shape
        
        # 1. Filter valid points
        valid_mask = (z_raw > 0.1) & (~np.isnan(z_raw))
        x_val, y_val, z_val = x_raw[valid_mask], y_raw[valid_mask], z_raw[valid_mask]
        
        if len(z_val) == 0:
            print("TargetLocator Error: No valid depth points.")
            return None
            
        # 2. Project 3D points to 2D image coordinates
        u, v = x_val / z_val, y_val / z_val
        max_abs_u, max_abs_v = np.percentile(np.abs(u), 99.5), np.percentile(np.abs(v), 99.5)
        
        u_norm = np.clip((u / max_abs_u * (w / 2) + w / 2).astype(np.int32), 0, w - 1)
        v_norm = np.clip((v / max_abs_v * (h / 2) + h / 2).astype(np.int32), 0, h - 1)

        # 3. Find points strictly inside the user's polygon
        inside_poly_mask = user_region_mask[v_norm, u_norm] == 255
        if not np.any(inside_poly_mask):
            print("TargetLocator Error: No 3D points found inside the selected region.")
            return None

        # 4. Find the Peak Z (Top of the highest box)
        z_poly = z_val[inside_poly_mask]
        peak_z = np.percentile(z_poly, 1.0)
        
        # 5. Isolate top layer points (allowing overhangs)
        top_layer_pts_mask = (z_val >= peak_z - 0.01) & (z_val <= peak_z + self.z_tolerance)
        top_surface_mask = np.zeros((h, w), dtype=np.uint8)
        top_surface_mask[v_norm[top_layer_pts_mask], u_norm[top_layer_pts_mask]] = 255
        
        # Morphological Opening: Breaks thin bridges between adjacent boxes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        top_surface_mask = cv2.morphologyEx(top_surface_mask, cv2.MORPH_OPEN, kernel)
        
        # 6. Get tightest box around the largest continuous top surface
        contours, _ = cv2.findContours(top_surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 100:
            return None
            
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        if bw == 0 or bh == 0: 
            return None # Failsafe against empty boxes
        
        # 7. Smart Prompting: Distance Transform ensures point is deep inside the specific box
        single_box_mask = np.zeros_like(top_surface_mask)
        cv2.drawContours(single_box_mask, [best_cnt], -1, 255, -1)
        
        dist_map = cv2.distanceTransform(single_box_mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        cx, cy = max_loc
        
        # 8. Dynamic Padding
        pad_w = int(bw * self.pad_ratio)
        pad_h = int(bh * self.pad_ratio)
        
        return {
            "bbox": (x, y, bw, bh),
            "center": (cx, cy),
            "padding": (pad_w, pad_h),
            "raw_mask": top_surface_mask
        }

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    global pallet_corners
    
    # Initialize our modular components
    segmenter = SAM2Segmenter("sam2/checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml")
    segmenter.load_model()
    
    # Using 5cm tolerance for cardboard box variance
    locator = TargetLocator(z_tolerance=0.05, pad_ratio=0.25) 

    # Load Data
    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    if not rgb_files or TARGET_INDEX >= len(rgb_files):
        print("Error: Invalid dataset index.")
        return

    rgb_path = rgb_files[TARGET_INDEX]
    depth_path = os.path.join(DEPTH_DIR, f"{os.path.splitext(os.path.basename(rgb_path))[0]}.npy")
    
    img_bgr = cv2.imread(rgb_path)
    img_h, img_w = img_bgr.shape[:2]
    
    raw_npy = np.load(depth_path)
    x_raw = raw_npy['x'].flatten()
    y_raw = raw_npy['y'].flatten()
    z_raw = raw_npy['z'].flatten()

    # Step 1: Interactive Session Calibration
    print("\n[Step 1] Interactive Session Calibration...")
    cv2.namedWindow("1. Session Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1. Session Calibration", int(img_w*0.6), int(img_h*0.6))
    cv2.setMouseCallback("1. Session Calibration", click_event)
    
    while True:
        clone = img_bgr.copy()
        for i, pt in enumerate(pallet_corners):
            cv2.circle(clone, pt, 5, (0, 255, 0), -1)
            if i > 0: cv2.line(clone, pallet_corners[i-1], pt, (0, 255, 0), 2)
                
        if len(pallet_corners) == 4:
            cv2.line(clone, pallet_corners[3], pallet_corners[0], (0, 255, 0), 2)
            cv2.putText(clone, "Press ENTER to confirm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # FIX: Restored multi-line UI instructions
            cv2.putText(clone, f"Points: {len(pallet_corners)}/4", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(clone, "L-Click: Add | R-Click: Undo | 'c': Clear", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        cv2.imshow("1. Session Calibration", clone)
        k = cv2.waitKey(15) & 0xFF
        if k == 13 and len(pallet_corners) == 4: break
        elif k in [ord('c'), ord('C')]: pallet_corners = []
            
    cv2.destroyWindow("1. Session Calibration")
    
    user_region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    poly_pts = np.array(pallet_corners, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(user_region_mask, [poly_pts], 255)

    # Step 2: Locate Target Mathematically
    print("[Step 2] Processing Depth Data...")
    target = locator.find_target((img_h, img_w), x_raw, y_raw, z_raw, user_region_mask)
    if not target:
        print("Failed to locate a valid target.")
        return
        
    x, y, w, h = target["bbox"]
    cx, cy = target["center"]
    pad_w, pad_h = target["padding"]

    # Step 3: Prepare Cropped Prompt Image
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
    
    # Crash-prevention safety check!
    if x2 <= x1 or y2 <= y1:
        print("Error: Target coordinates are invalid. Bounding box is empty.")
        return
        
    cropped_bgr = img_bgr[y1:y2, x1:x2]
    if cropped_bgr.size == 0:
        print("Error: Cropped region has size 0.")
        return
        
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    # Calculate prompts relative to the cropped image
    local_box = [x - x1, y - y1, x + w - x1, y + h - y1]
    local_point = [cx - x1, cy - y1]

    # Step 4: Execute Segmentation via Model Interface
    print("[Step 3] Running Segmentation Model...")
    local_mask, confidence = segmenter.predict(cropped_rgb, local_box, local_point)
    print(f" -> Target successfully segmented! Model Confidence: {confidence:.2f}")

    # Map local mask back to global image
    global_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    global_mask[y1:y2, x1:x2] = (local_mask * 255).astype(np.uint8)

    # ==========================================
    # VISUALIZATION
    # ==========================================
    # Draw geometric prompts
    debug_img = img_bgr.copy()
    cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)     # Strict Box (Blue)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 2)   # Padded Context Box (Purple)
    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)              # Center Point (Red)

    # Draw segmented mask overlay
    overlay = img_bgr.copy()
    overlay[global_mask == 255] = [0, 255, 0] # Green mask
    
    mask_contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, mask_contours, -1, (255, 255, 255), 2)
    final_output = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    
    # Final Robot Pick Point
    y_indices, x_indices = np.where(global_mask == 255)
    if len(x_indices) > 0 and len(y_indices) > 0:
        final_cx, final_cy = int(np.median(x_indices)), int(np.median(y_indices))
        cv2.circle(final_output, (final_cx, final_cy), 8, (255, 255, 255), -1)
        cv2.circle(final_output, (final_cx, final_cy), 4, (0, 0, 255), -1)

    cv2.namedWindow("1. Mathematical Target Location", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Final Segmented Output", cv2.WINDOW_NORMAL)
    
    cw, ch = int(img_w * 0.5), int(img_h * 0.5)
    cv2.resizeWindow("1. Mathematical Target Location", cw, ch)
    cv2.resizeWindow("2. Final Segmented Output", cw, ch)

    cv2.imshow("1. Mathematical Target Location", debug_img)
    cv2.imshow("2. Final Segmented Output", final_output)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
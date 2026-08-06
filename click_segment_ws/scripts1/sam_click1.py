import os
import cv2
import torch
import numpy as np
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Initialize SAM 2 Predictor
checkpoint = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 2. Load the latest captured image
DATASET_DIR = "calibrated_box_dataset/undistorted_images/*.png"
image_files = sorted(glob.glob(DATASET_DIR))

if not image_files:
    print("Error: No images found in 'calibrated_box_dataset/undistorted_images/'.")
    exit()

target_image_path = image_files[2]
print(f"Loading image for intelligent area filtering: {target_image_path}")

img_bgr = cv2.imread(target_image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, c = img_bgr.shape

# Embed the image into GPU memory
predictor.set_image(img_rgb)

# Global canvas variables
display_frame = img_bgr.copy()
hard_projection = np.zeros_like(img_bgr)

def mouse_callback(event, x, y, flags, param):
    """Triggered on click to calculate and select masks while ignoring QR subparts."""
    global display_frame, hard_projection
    
    if event == cv2.EVENT_LBUTTONDOWN:
        single_point = np.array([[x, y]], dtype=np.float32)
        single_label = np.array([1], dtype=np.int32)
        
        # 3. Model Prediction
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(
                point_coords=single_point, 
                point_labels=single_label,
                multimask_output=True
            )
            
        # Format tensor safely to numpy array
        if hasattr(masks, 'cpu'):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)
            
        masks_np = np.squeeze(masks_np) # Ensure shape is (3, H, W)
        
        # 4. Integrate your Area Filtering Logic
        # Calculate pixel count areas for Candidate 1, Candidate 2, and Candidate 3
        mask_areas = [masks_np[m].sum() for m in range(3)]
        
        # Identify the largest candidate index (typically Candidate 2 or 3)
        largest_mask_idx = np.argmax(mask_areas)
        
        # Explicit target choice: Check Candidate 2 (Index 1) first
        cand_2_area = mask_areas[1]
        total_pixels = h * w
        
        # If Candidate 2 isolates a tiny QR patch (less than 1.5% of total frame),
        # automatically fall back to the largest available structural candidate.
        if cand_2_area < (total_pixels * 0.015):
            print(f"-> Target component at ({x}, {y}) is too small ({cand_2_area} px). Overriding QR code.")
            best_mask_bool = masks_np[largest_mask_idx].astype(bool)
            selected_tier = largest_mask_idx + 1
        else:
            # Otherwise, use Candidate 2 because its object boundaries are pristine
            best_mask_bool = masks_np[1].astype(bool)
            selected_tier = 2

        print(f"Click at ({x}, {y}) | Using Candidate {selected_tier} | Score: {scores[selected_tier-1]:.4f}")
        
        # 5. Render Selected Mask Layouts
        random_color = np.random.randint(50, 255, size=3).tolist()
        
        # Update the blended overlay window
        overlay = display_frame.copy()
        overlay[best_mask_bool] = random_color
        cv2.addWeighted(display_frame, 0.6, overlay, 0.4, 0, dst=display_frame)
        
        # Update the hard silhouette projection window (No background)
        hard_projection[best_mask_bool] = random_color
        
        # Draw click location tracker dot
        cv2.circle(display_frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(display_frame, (x, y), 5, (0, 0, 0), 1)

def main():
    global display_frame, hard_projection
    
    win_overlay = "Interactive Area-Filtered Visualizer"
    win_hard = "Hard Silhouette Projection"
    
    cv2.namedWindow(win_overlay, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_hard, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_overlay, mouse_callback)
    
    sw, sh = int(w * 0.6), int(h * 0.6)
    cv2.resizeWindow(win_overlay, sw, sh)
    cv2.resizeWindow(win_hard, sw, sh)
    cv2.moveWindow(win_overlay, 100, 100)
    cv2.moveWindow(win_hard, 150 + sw, 100)
    
    print("\n--- Intelligent Filtering Active ---")
    print("Click boxes freely. Small QR codes/stickers will automatically upscale.")
    print("Press 'c' to clear screen.")
    print("Press 'q' to exit.")
    
    while True:
        cv2.imshow(win_overlay, display_frame)
        cv2.imshow(win_hard, hard_projection)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            display_frame = img_bgr.copy()
            hard_projection = np.zeros_like(img_bgr)
            print("Canvas cleared.")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

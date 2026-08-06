import os
import cv2
import torch
import numpy as np
import glob
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Initialize SAM 2 Models
checkpoint = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, checkpoint)

# Generator for the overall scene detection
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,          
    pred_iou_thresh=0.88,         
    stability_score_thresh=0.95,  
    min_mask_region_area=100      
)

# Predictor to break down the individual 3 native candidates
predictor = SAM2ImagePredictor(sam2_model)

# 2. Load the latest captured image
DATASET_DIR = "percipio_captures1/rgb_images/*.png"
image_files = sorted(glob.glob(DATASET_DIR))

if not image_files:
    print("Error: No images found in 'calibrated_box_dataset/undistorted_images/'.")
    exit()

target_image_path = image_files[18]
print(f"Generating 6-window structural matrix for: {target_image_path}")

img_bgr = cv2.imread(target_image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, c = img_bgr.shape

# 3. Compute the automatic scene baseline
predictor.set_image(img_rgb)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks_data = mask_generator.generate(img_rgb)

# 4. Initialize canvas buffers
blended_overlay = np.zeros_like(img_bgr)
hard_projection = np.zeros_like(img_bgr)

# Discrete layers for the strict Candidate 1, 2, and 3 comparison windows
candidate_layers = [np.zeros((h, w), dtype=bool) for _ in range(3)]

np.random.seed(42)

for mask in masks_data:
    binary_mask = mask['segmentation']
    area_ratio = mask['area'] / (h * w)
    
    # Drop large environment background planes from windows 2 and 3
    if area_ratio > 0.40:
        continue
        
    random_color = np.random.randint(0, 255, size=3).tolist()
    blended_overlay[binary_mask] = random_color
    hard_projection[binary_mask] = random_color

    # Extract the centroid of this specific mask to query its native 3 scale options
    # This reveals the exact candidates SAM 2 chooses between for this object
    y_indices, x_indices = np.where(binary_mask)
    if len(x_indices) > 0 and len(y_indices) > 0:
        cx = int(np.median(x_indices))
        cy = int(np.median(y_indices))
        
        input_point = np.array([[cx, cy]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            raw_masks, scores, _ = predictor.predict(
                point_coords=input_point, 
                point_labels=input_label,
                multimask_output=True
            )
        
        raw_masks_np = raw_masks.cpu().numpy() if hasattr(raw_masks, 'cpu') else np.array(raw_masks)
        
        # Accumulate the true unblended structure levels across the objects
        for scale_idx in range(3):
            if scores[scale_idx] > 0.85:
                candidate_layers[scale_idx] |= raw_masks_np[scale_idx].astype(bool)

# Build the blended final output for window 2
blended_output = img_bgr.copy()
cv2.addWeighted(img_bgr, 0.5, blended_overlay, 0.5, 0, dst=blended_output)

# 5. Define Window Names
windows = {
    "raw": "1. Raw Undistorted Image",
    "blended": "2. Blended Instance Masks",
    "hard": "3. Hard Projection (Masks Only)",
    "cand1": "4. Candidate 1 (Subparts - Blue)",
    "cand2": "5. Candidate 2 (Objects - Green)",
    "cand3": "6. Candidate 3 (Groups - Red)"
}

# Create resizable windows
for name in windows.values():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

# 6. Calculate desktop layout positions (2 rows x 3 columns grid)
scale = 0.45
sw = int(w * scale)
sh = int(h * scale)

for name in windows.values():
    cv2.resizeWindow(name, sw, sh)

# Row 1 positions (Top row: The scene analysis windows)
cv2.moveWindow(windows["raw"], 30, 50)
cv2.moveWindow(windows["blended"], 40 + sw, 50)
cv2.moveWindow(windows["hard"], 50 + (sw * 2), 50)

# Row 2 positions (Bottom row: The raw candidate breakdown windows)
cv2.moveWindow(windows["cand1"], 30, 80 + sh)
cv2.moveWindow(windows["cand2"], 40 + sw, 80 + sh)
cv2.moveWindow(windows["cand3"], 50 + (sw * 2), 80 + sh)

print("\n--- Spawning 6-Window Analysis Panel ---")
print("Press any key in any window to exit.")

# Build color displays for candidates
cand_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
cand_displays = []
for idx in range(3):
    disp = img_bgr.copy()
    layer_overlay = np.zeros_like(img_bgr)
    layer_overlay[candidate_layers[idx]] = cand_colors[idx]
    cv2.addWeighted(img_bgr, 0.6, layer_overlay, 0.4, 0, dst=disp)
    cand_displays.append(disp)

while True:
    cv2.imshow(windows["raw"], img_bgr)
    cv2.imshow(windows["blended"], blended_output)
    cv2.imshow(windows["hard"], hard_projection)
    cv2.imshow(windows["cand1"], cand_displays[0])
    cv2.imshow(windows["cand2"], cand_displays[1])
    cv2.imshow(windows["cand3"], cand_displays[2])
    
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()

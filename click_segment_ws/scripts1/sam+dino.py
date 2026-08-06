import os
import cv2
import torch
import numpy as np
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# 1. INITIALIZE MODELS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")

try:
    dinov3_model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14').to(device)
except Exception:
    dinov3_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
dinov3_model.eval()

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(sam2_cfg, sam2_checkpoint)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# ==========================================
# 2. LOAD IMAGE
# ==========================================
DATASET_DIR = "percipio_captures1/rgb_images/*.png"
image_files = sorted(glob.glob(DATASET_DIR))
target_image_path = image_files[18] if len(image_files) > 18 else image_files[-1]

img_bgr = cv2.imread(target_image_path)
h, w, _ = img_bgr.shape
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

patch_size = 14
h_patch, w_patch = h // patch_size, w // patch_size
img_resized = cv2.resize(img_rgb, (w_patch * patch_size, h_patch * patch_size))

img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
img_tensor = (img_tensor - mean) / std

# Extract Dense Embeddings
with torch.inference_mode():
    features_dict = dinov3_model.forward_features(img_tensor)
    patch_tokens = features_dict['x_norm_patchtokens'].squeeze(0).cpu().numpy()

norms = np.maximum(np.linalg.norm(patch_tokens, axis=1, keepdims=True), 1e-8)
tokens_norm = patch_tokens / norms

# Define window names
win_sam = "1. Final Output: SAM 2 Precision Contours"
win_dino = "2. DINO Step 3 & 4: Colored Semantic Islands"

# ==========================================
# 3. INTERACTIVE CLICK & SEGMENTATION FUNCTION
# ==========================================
def segment_material(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\n--- Selected Material at Pixel: X={x}, Y={y} ---")
        
        # Map pixel click back to DINO patch grid coordinates
        patch_x = min(x // patch_size, w_patch - 1)
        patch_y = min(y // patch_size, h_patch - 1)
        click_idx = patch_y * w_patch + patch_x
        
        # Extract clicked material feature vector
        target_material_vector = tokens_norm[click_idx : click_idx + 1]
        
        # Compute material similarity map (Step 3)
        sim_map = np.dot(tokens_norm, target_material_vector.T).squeeze()
        sim_map = np.nan_to_num(sim_map, nan=0.0)
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
        sim_grid = sim_map.reshape(h_patch, w_patch)
        
        # Threshold (> 75% match) and run Connected Components (Step 4)
        binary_semantic_zones = (sim_grid > 0.75).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_semantic_zones, connectivity=8)
        
        # ---------------------------------------------------------
        # BUILD WINDOW 2: DINO COLORED ISLANDS VISUALIZATION
        # ---------------------------------------------------------
        clean_labels = labels.copy()
        prompts = []
        
        # Filter out noise islands (< 2 patches) and collect valid prompts
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 2:
                clean_labels[clean_labels == i] = 0  # Zero out tiny noise
            else:
                cx_patch, cy_patch = centroids[i]
                prompts.append([int(cx_patch * patch_size + 7), int(cy_patch * patch_size + 7)])
        
        if not prompts:
            prompts.append([x, y])
            
        # Generate a distinct random color for each island ID (ID 0 is black background)
        np.random.seed(101) # Seed for consistent, high-contrast island colors
        island_palette = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
        island_palette[0] = [0, 0, 0]  # Ensure background stays dark
        
        # Map the 2D label grid to our RGB color palette
        island_patch_img = island_palette[clean_labels]
        
        # Upscale to full image resolution using nearest-neighbor to show DINO's exact patch bounds
        island_full_img = cv2.resize(island_patch_img, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Blend the colored islands over the raw image
        dino_island_display = cv2.addWeighted(img_bgr, 0.35, island_full_img, 0.65, 0)
        
        # ---------------------------------------------------------
        # BUILD WINDOW 1: SAM 2 PRECISION CONTOURS
        # ---------------------------------------------------------
        sam2_predictor.set_image(img_rgb)
        sam_overlay = img_bgr.copy()
        np.random.seed(42)
        
        for pt in prompts:
            input_point = np.array([[pt[0], pt[1]]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, _, _ = sam2_predictor.predict(
                    point_coords=input_point, point_labels=input_label, multimask_output=False
                )
            
            best_mask = masks[0].astype(bool)
            color = np.random.randint(50, 255, size=3).tolist()
            sam_overlay[best_mask] = color
            
            # Draw crisp contour on SAM output
            contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sam_overlay, contours, -1, (255, 255, 255), 2)
            
            # Draw center bullseyes on BOTH windows so you can see the 1-to-1 mapping!
            for disp in [sam_overlay, dino_island_display]:
                cv2.circle(disp, (pt[0], pt[1]), 5, (0, 0, 255), -1)
                cv2.circle(disp, (pt[0], pt[1]), 7, (255, 255, 255), 1)
            
        final_sam_output = cv2.addWeighted(img_bgr, 0.45, sam_overlay, 0.55, 0)
        
        # Update both displays
        cv2.imshow(win_sam, final_sam_output)
        cv2.imshow(win_dino, dino_island_display)
        print(f"Found {len(prompts)} distinct islands. Displays updated!")

# ==========================================
# 4. RUN INTERACTIVE WINDOWS
# ==========================================
for win in [win_sam, win_dino]:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(w * 0.45), int(h * 0.45))

# Position side-by-side on your desktop
cv2.moveWindow(win_sam, 30, 50)
cv2.moveWindow(win_dino, 50 + int(w * 0.45), 50)

# Set mouse callback on BOTH windows so you can click on either one!
cv2.setMouseCallback(win_sam, segment_material)
cv2.setMouseCallback(win_dino, segment_material)

# Show raw frame initially
cv2.imshow(win_sam, img_bgr)
cv2.imshow(win_dino, img_bgr)

print("\n--- 2-Window Analysis Panel Ready ---")
print("Instruction: Left-click on any cardboard box in EITHER window.")
print("Window 2 will show how DINO clusters the material into distinct colored islands.")
print("Window 1 will show how SAM 2 uses those island centers to draw sharp contours.")
print("Press any key in any window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
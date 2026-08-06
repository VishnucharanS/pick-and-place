import os
import cv2
import torch
import numpy as np
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Initialize DINOv2 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading DINOv2 model on {device}...")
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
dinov2_model.eval()

# 2. Load the latest captured image
DATASET_DIR = "percipio_captures1/rgb_images/*.png"
image_files = sorted(glob.glob(DATASET_DIR))

if not image_files:
    print("Error: No images found in dataset directory.")
    exit()

target_image_path = image_files[6] if len(image_files) > 18 else image_files[-1]
print(f"Generating 6-window DINOv2 semantic matrix for: {target_image_path}")

img_bgr = cv2.imread(target_image_path)
h, w, c = img_bgr.shape
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. Preprocess for DINOv2 (Dimensions must be divisible by patch size 14)
patch_size = 14
h_patch, w_patch = h // patch_size, w // patch_size
img_resized = cv2.resize(img_rgb, (w_patch * patch_size, h_patch * patch_size))

# Prepare tensor with ImageNet standard normalization
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
img_tensor = (img_tensor - mean) / std

# 4. Extract Dense Patch Tokens
with torch.inference_mode():
    features_dict = dinov2_model.forward_features(img_tensor)
    # Shape: [N_patches, 768]
    patch_tokens = features_dict['x_norm_patchtokens'].squeeze(0).cpu().numpy()

# Normalize tokens for similarity calculations
tokens_norm = patch_tokens / np.linalg.norm(patch_tokens, axis=1, keepdims=True)

# ---------------------------------------------------------
# WINDOW 2: PCA RGB Semantic Feature Map
# ---------------------------------------------------------
pca = PCA(n_components=3)
pca_features = pca.fit_transform(patch_tokens)
# Min-Max scale to [0, 255] for RGB display
pca_min, pca_max = pca_features.min(axis=0), pca_features.max(axis=0)
pca_rgb = ((pca_features - pca_min) / (pca_max - pca_min) * 255).astype(np.uint8)
pca_rgb_patch = pca_rgb.reshape(h_patch, w_patch, 3)
# Bicubic interpolation smooths out the 14x14 blocky ViT grid
pca_bgr_full = cv2.cvtColor(cv2.resize(pca_rgb_patch, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------
# WINDOW 3 & 4: Unsupervised K-Means Semantic Segmentation
# ---------------------------------------------------------
num_clusters = 6  # E.g., Cardboard boxes, Pallet wood, Carpet floor, Cables, Metal rig, Shadows
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(patch_tokens).reshape(h_patch, w_patch)

# Generate distinct color palette for materials
np.random.seed(15)
cluster_palette = np.random.randint(50, 255, size=(num_clusters, 3), dtype=np.uint8)
cluster_img_patch = cluster_palette[cluster_labels]
cluster_img_full = cv2.resize(cluster_img_patch, (w, h), interpolation=cv2.INTER_NEAREST)

# Blended overlay
blended_clusters = cv2.addWeighted(img_bgr, 0.4, cluster_img_full, 0.6, 0)

# ---------------------------------------------------------
# WINDOW 5: Foreground Salience Map (Distance from Carpet Floor)
# ---------------------------------------------------------
# Sample top and bottom edge patches as the background baseline (carpet)
bg_patches = np.vstack([tokens_norm[: w_patch * 2], tokens_norm[-w_patch * 2 :]])
bg_baseline = np.median(bg_patches, axis=0, keepdims=True)
bg_baseline /= np.linalg.norm(bg_baseline)

# Compute cosine distance from floor background
sim_to_bg = np.dot(tokens_norm, bg_baseline.T).squeeze()
fg_score = 1.0 - sim_to_bg  # Higher score = less like the floor
fg_score = (fg_score - fg_score.min()) / (fg_score.max() - fg_score.min())
fg_patch = (fg_score * 255).astype(np.uint8).reshape(h_patch, w_patch)
fg_full = cv2.applyColorMap(cv2.resize(fg_patch, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO)

# ---------------------------------------------------------
# WINDOW 6: Material Similarity Heatmap (Center Object Tracking)
# ---------------------------------------------------------
# Take the feature vector from the exact center patch of the image (top of pallet/boxes)
center_idx = (h_patch // 2) * w_patch + (w_patch // 2)
target_feature = tokens_norm[center_idx : center_idx + 1]

# Find all regions in the room sharing this exact same material/texture
sim_to_target = np.dot(tokens_norm, target_feature.T).squeeze()
sim_rescaled = (sim_to_target - sim_to_target.min()) / (sim_to_target.max() - sim_to_target.min())
sim_patch = (sim_rescaled * 255).astype(np.uint8).reshape(h_patch, w_patch)
sim_full = cv2.applyColorMap(cv2.resize(sim_patch, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET)

# ---------------------------------------------------------
# Layout & Display Configuration
# ---------------------------------------------------------
windows = {
    "raw": "1. Raw Undistorted Image",
    "pca": "2. PCA RGB Feature Map (Semantic Understanding)",
    "kmeans": "3. Unsupervised K-Means Clusters (6 Materials)",
    "blended": "4. Blended Material Overlay",
    "salience": "5. Foreground Salience (Carpet Floor Separation)",
    "similarity": "6. Material Similarity Heatmap (Target: Center Object)"
}

for name in windows.values():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

scale = 0.45
sw, sh = int(w * scale), int(h * scale)

for name in windows.values():
    cv2.resizeWindow(name, sw, sh)

# 2x3 Grid Positioning
cv2.moveWindow(windows["raw"], 30, 50)
cv2.moveWindow(windows["pca"], 40 + sw, 50)
cv2.moveWindow(windows["kmeans"], 50 + (sw * 2), 50)

cv2.moveWindow(windows["blended"], 30, 80 + sh)
cv2.moveWindow(windows["salience"], 40 + sw, 80 + sh)
cv2.moveWindow(windows["similarity"], 50 + (sw * 2), 80 + sh)

print("\n--- Spawning True DINOv2 6-Window Analysis Panel ---")
print("Press any key in any window to exit.")

while True:
    cv2.imshow(windows["raw"], img_bgr)
    cv2.imshow(windows["pca"], pca_bgr_full)
    cv2.imshow(windows["kmeans"], cluster_img_full)
    cv2.imshow(windows["blended"], blended_clusters)
    cv2.imshow(windows["salience"], fg_full)
    cv2.imshow(windows["similarity"], sim_full)
    
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
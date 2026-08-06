import numpy as np
import open3d as o3d
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
DEPTH_DIR = "percipio_captures1/depth_npy/" # Directory containing your .npy files

def robust_load_npy(npy_path):
    """
    Aggressively tries every known numpy shape/format to extract X, Y, Z.
    """
    try:
        # allow_pickle=True is sometimes needed if the array was saved as an object/dict
        raw = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"  [!] Failed to load file: {e}")
        return None, None, None

    print(f"  -> Raw Data Shape: {raw.shape}, Type: {raw.dtype}")

    x, y, z = None, None, None

    try:
        # 1. Check for Structured Array (e.g., raw['x'])
        if raw.dtype.names is not None and 'x' in raw.dtype.names:
            print("  -> Format matched: Structured Array ['x', 'y', 'z']")
            x = raw['x'].flatten()
            y = raw['y'].flatten()
            z = raw['z'].flatten()

        # 2. Check for 2D Depth Image (H, W) -> THIS IS YOUR PERCIPIO FORMAT
        elif raw.ndim == 2 and raw.shape[1] > 100:
            print(f"  -> Format matched: 2D Depth Image {raw.shape} ({raw.dtype})")
            H, W = raw.shape
            
            # Generic Pinhole Camera Intrinsics
            fx = fy = max(H, W) * 0.8 
            cx, cy = W / 2.0, H / 2.0
            
            # uint16 is usually in millimeters. Convert to meters.
            depth_scale = 0.001 if raw.dtype == np.uint16 else 1.0
            z_img = raw.astype(np.float32) * depth_scale
            
            # Generate U, V pixel coordinates
            v, u = np.indices((H, W))
            
            # Filter zero/invalid depths immediately
            valid = z_img > 0.05 # > 5cm
            z = z_img[valid]
            u = u[valid]
            v = v[valid]
            
            # Project to 3D Space
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

        # 3. Check for standard 3D Matrix (H, W, 3)
        elif raw.ndim == 3 and raw.shape[-1] >= 3:
            print("  -> Format matched: 3D Matrix (H, W, 3)")
            x = raw[..., 0].flatten()
            y = raw[..., 1].flatten()
            z = raw[..., 2].flatten()

        # 3. Check for standard 2D Matrix (N, 3)
        elif raw.ndim == 2 and raw.shape[-1] >= 3:
            print("  -> Format matched: 2D Matrix (N, 3)")
            x = raw[:, 0]
            y = raw[:, 1]
            z = raw[:, 2]

        # 4. Check if it was saved as a python dictionary inside a numpy object
        elif raw.dtype == object and isinstance(raw.item(), dict):
            print("  -> Format matched: Dictionary inside Numpy Object")
            d = raw.item()
            x = d['x'].flatten()
            y = d['y'].flatten()
            z = d['z'].flatten()

        else:
            print(f"  [!] Unknown format! Cannot parse array of shape {raw.shape} and dtype {raw.dtype}")
            return None, None, None

    except Exception as e:
        print(f"  [!] Error parsing structure: {e}")
        return None, None, None

    return x, y, z

def main():
    npy_files = sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy")))
    
    if not npy_files:
        print(f"Error: No .npy files found in {DEPTH_DIR}")
        return

    print(f"Found {len(npy_files)} .npy files. Starting conversion...\n")

    for npy_path in npy_files:
        filename = os.path.basename(npy_path)
        print(f"Processing: {filename}")

        # 1. Robust Extraction
        x, y, z = robust_load_npy(npy_path)
        
        if x is None or y is None or z is None:
            print(f"  [!] Skipping {filename} due to extraction failure.\n")
            continue

        # 2. Filter Invalid Points (NaNs and zero-depth)
        valid_mask = (~np.isnan(z)) & (~np.isnan(x)) & (~np.isnan(y)) & (z > 0.01)
        x_val = x[valid_mask]
        y_val = y[valid_mask]
        z_val = z[valid_mask]
        
        print(f"  -> Total points: {len(x)}, Valid points after filtering: {len(x_val)}")

        if len(x_val) == 0:
            print(f"  [!] Skipping {filename} (0 valid points). Are units correct? Max Z: {np.nanmax(z)}\n")
            continue

        # 3. Build Open3D Point Cloud
        pts = np.column_stack((x_val, y_val, z_val))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # 4. Save to disk in the same directory
        output_filename = filename.replace(".npy", ".pcd")
        output_path = os.path.join(DEPTH_DIR, output_filename)
        
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"  [SUCCESS] Saved to {output_path}\n")

if __name__ == "__main__":
    main()
import numpy as np
import cv2
import glob
import os

# 1. Define configuration settings
# For a 9x12 grid of squares, there are 8x11 internal corner intersections
CHECKERBOARD_CORNERS = (8, 11)  
SQUARE_SIZE_MM = 20.0  # Change this to your actual square size in millimeters

# Define the mathematical termination criteria for sub-pixel corner accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 2. Prepare 3D object points in real-world coordinates (x, y, z=0)
objp = np.zeros((CHECKERBOARD_CORNERS[0] * CHECKERBOARD_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_CORNERS[0], 0:CHECKERBOARD_CORNERS[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# Arrays to store tracking data from all valid calibration images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Load your captured calibration images (adjust extension if using .jpg)
images = glob.glob('captured_images/*.png')

if len(images) == 0:
    print("Error: No images found in 'captured_images/'. Run your capture script first.")
    exit()

print(f"Processing {len(images)} images for calibration...")

gray_shape = None
valid_image_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_CORNERS, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        valid_image_count += 1

        # Refine 2D coordinates to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the discovered patterns
        cv2.drawChessboardCorners(img, CHECKERBOARD_CORNERS, corners2, ret)
        cv2.imshow('Checking Grid Detection', img)
        cv2.waitKey(100) # Displays each image briefly
    else:
        print(f"Warning: Checkerboard pattern not found in {fname}")

cv2.destroyAllWindows()

if valid_image_count < 10:
    print(f"\nWarning: Only {valid_image_count} images were usable. You should use at least 10-20 images for good calibration.")

# 3. Perform Camera Calibration
print("\nComputing calibration parameters...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

# 4. Display Results
print("\n=== Calibration Successful ===")
print(f"Reprojection Error: {ret:.4f} pixels (Ideally should be < 0.5)")

print("\nCamera Matrix (K):")
print(mtx)

print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
print(dist)

# 5. Optional: Save parameters to a file for later use
np.savez("camera_calibration.npz", mtx=mtx, dist=dist)
print("\nCalibration parameters saved securely to 'camera_calibration.npz'")

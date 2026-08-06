import os
import cv2
import time
import numpy as np

# Configuration
OUTPUT_DIR = "calibrated_box_dataset"
os.makedirs(os.path.join(OUTPUT_DIR, "raw_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "undistorted_images"), exist_ok=True)

# 1. Load Calibration Parameters
CALIB_FILE = "camera_calibration.npz"
if not os.path.exists(CALIB_FILE):
    print(f"Error: '{CALIB_FILE}' not found. Please run your calibration script first.")
    exit()

with np.load(CALIB_FILE) as data:
    mtx = data['mtx']      # Camera Matrix
    dist = data['dist']    # Distortion Coefficients

def mouse_callback(event, x, y, flags, param):
    """Saves both raw and undistorted frames instantly on click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        raw_frame = param['raw_frame']
        undist_frame = param['undist_frame']
        
        if raw_frame is None or undist_frame is None:
            return
            
        timestamp = int(time.time() * 1000)
        
        # Save raw frame
        raw_path = os.path.join(OUTPUT_DIR, "raw_images", f"box_raw_{timestamp}.png")
        cv2.imwrite(raw_path, raw_frame)
        
        # Save mathematically straight frame
        undist_path = os.path.join(OUTPUT_DIR, "undistorted_images", f"box_undist_{timestamp}.png")
        cv2.imwrite(undist_path, undist_frame)
        
        print(f"Captured! Saved raw and undistorted frames for timestamp: {timestamp}")

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Grab one frame to setup optimal camera matrices for current aspect ratio
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera failed.")
        return
    h, w = frame.shape[:2]
    
    # Optimize the camera matrix to minimize pixel loss after straightening curves
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    window_name = "Calibrated Feed - Left Click Box to Save"
    cv2.namedWindow(window_name)
    
    runtime_data = {'raw_frame': None, 'undist_frame': None}
    cv2.setMouseCallback(window_name, mouse_callback, runtime_data)

    print(f"\n--- Calibrated Recording Started ---")
    print(f"Saving outputs to: ./{OUTPUT_DIR}/")
    print("Left Click : Save raw and straight image of the box")
    print("Press 'q'  : Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Mirror for natural user interaction
        frame = cv2.flip(frame, 1)
        
        # Remove radial distortion from the live frame
        undistorted = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)
        
        # Crop the image boundaries slightly based on the optimal ROI calculation
        x_roi, y_roi, w_roi, h_roi = roi
        undistorted_cropped = undistorted[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        
        # Ensure cropped frame is valid before assigning
        if undistorted_cropped.size == 0:
            undistorted_cropped = undistorted

        # Share current state with callback buffers
        runtime_data['raw_frame'] = frame
        runtime_data['undist_frame'] = undistorted_cropped
        
        # Display the undistorted straight line version to the user
        cv2.imshow(window_name, undistorted_cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

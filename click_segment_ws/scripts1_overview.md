# Overview of the files in the scripts1 folder

This folder contains a mix of calibration utilities, data-capture scripts, point-cloud conversion helpers, and segmentation pipelines that use SAM 2 and DINO-based vision models.

## Summary table

| File | What it does | Model / approach used |
|---|---|---|
| [scripts1/best_box.py](scripts1/best_box.py) | Runs a depth-guided pallet/box detection pipeline. It lets the user draw a pallet region, finds a candidate box from 3D depth data, and sends SAM 2 prompts to generate a segmentation mask. | SAM 2 (sam2.1_hiera_base_plus) |
| [scripts1/best_box_all.py](scripts1/best_box_all.py) | A more complete hybrid pipeline that combines user ROI selection, 3D point-cloud clustering, RANSAC plane fitting, and SAM 2 segmentation. | SAM 2 (sam2.1_hiera_base_plus) + Open3D geometry processing |
| [scripts1/best_box_live_stream.py](scripts1/best_box_live_stream.py) | A ROS 2 live-stream version of the box-picking pipeline. It subscribes to RGB and depth topics and performs the same target localization and SAM 2 segmentation online. | SAM 2 (sam2.1_hiera_base_plus) |
| [scripts1/best_boxv4.py](scripts1/best_boxv4.py) | A variant pipeline that uses a 3D scoring matrix and geometry-based prompts to choose a box and then pass it to SAM 2. | SAM 2 + Open3D + RANSAC/DBSCAN |
| [scripts1/best_boxv5.py](scripts1/best_boxv5.py) | Refines the hybrid 3D locator with better plane scoring and ROI filtering before handing off to SAM 2. | SAM 2 + Open3D geometry processing |
| [scripts1/best_boxv6.py](scripts1/best_boxv6.py) | A more robust depalletizing pipeline that adds seam detection, rectangularity checks, and automatic-mask fallback logic for better segmentation quality. | SAM 2 + Open3D + geometric validation |
| [scripts1/camera_calibration.py](scripts1/camera_calibration.py) | Calibrates the camera using a chessboard pattern. It finds checkerboard corners, estimates camera intrinsics, and saves calibration parameters to a .npz file. | No trained model; uses OpenCV camera calibration |
| [scripts1/camera_percipio.py](scripts1/camera_percipio.py) | A ROS 2 node that streams RGB and depth images and saves RGB/depth snapshots when the user clicks in the display window. | No trained model; ROS 2 + OpenCV |
| [scripts1/capture_images.py](scripts1/capture_images.py) | Captures raw webcam frames and undistorted versions using previously computed camera calibration parameters. | No trained model; OpenCV undistortion |
| [scripts1/dino_boxes.py](scripts1/dino_boxes.py) | Runs DINOv2 feature extraction on an image and visualizes semantic clusters, saliency, and material-similarity maps. | DINOv2 |
| [scripts1/key_capture_node.py](scripts1/key_capture_node.py) | A ROS 2 synchronization node that saves aligned color, depth, point-cloud, and camera-info data when the user presses a key. | No trained model; ROS 2 + message filters |
| [scripts1/numpy_to_pcd_converter.py](scripts1/numpy_to_pcd_converter.py) | Converts saved .npy depth arrays into .pcd point clouds using Open3D. | No trained model; Open3D point-cloud conversion |
| [scripts1/sam+dino.py](scripts1/sam+dino.py) | Combines DINO embeddings with SAM 2 segmentation. The user clicks a region of interest, the script clusters DINO features into semantic islands, and SAM 2 segments those regions. | DINOv3 (with DINOv2 fallback) + SAM 2 |
| [scripts1/sam_boxes.py](scripts1/sam_boxes.py) | Visualizes SAM 2 automatic masks and candidate-level segmentation outputs for a captured image. | SAM 2 automatic mask generator + predictor |
| [scripts1/sam_click1.py](scripts1/sam_click1.py) | Lets the user click an image and uses SAM 2 to generate segmentation masks while filtering out small or unwanted subparts like QR patches. | SAM 2 |

## Notes

- The main object-segmentation pipeline in this folder is centered around SAM 2.
- The DINO scripts are mostly used for semantic/feature analysis rather than direct box-picking.
- The calibration and capture scripts are support tools for preparing the dataset and camera setup.

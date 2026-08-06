import os
import cv2
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ==========================================
# CONFIGURATION
# ==========================================
SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# ROS 2 Topics (Update to match your exact setup)
RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'  # Recommended: aligned depth topic if available

# Global state for interactive clicker and buffers
pallet_corners = []
calibration_done = False
runtime_data = {'rgb_frame': None, 'depth_frame': None}

def click_event(event, x, y, flags, param):
    """Handles mouse clicks on the live preview to define the 4 corners of the pallet."""
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
    def predict(self, rgb_image, local_depth, box_prompt, point_prompts):
        pass

class SAM2Segmenter(BaseSegmenter):
    def __init__(self, checkpoint_path, config_path):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.predictor = None

    def load_model(self):
        print("[Model] Loading SAM 2 Predictor onto GPU...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_model = build_sam2(self.config_path, self.checkpoint_path)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("[Model] SAM 2 Ready.")

    def predict(self, rgb_image, local_depth, box_prompt, point_prompts):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_image)
            
            # Multi-Point Prompting (List of [cx, cy])
            labels = np.ones(len(point_prompts), dtype=np.int32)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(point_prompts),
                point_labels=labels,
                box=np.array(box_prompt),
                multimask_output=True 
            )
            
        best_mask = None
        best_score = -1
        
        for i, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            
            z_in_mask = local_depth[mask_bool]
            z_valid = z_in_mask[(z_in_mask > 0.1) & (~np.isnan(z_in_mask))]
            
            if len(z_valid) < 100: continue
            
            z_std = np.std(z_valid) 
            area = np.sum(mask_bool)
            
            # Reject mask if variance is too high (bleeding off the box)
            if z_std > 0.03:
                continue
                
            if area > best_score:
                best_score = area
                best_mask = mask
                
        if best_mask is None:
            best_mask = masks[np.argmax(scores)]
            
        return best_mask, max(scores)

# ==========================================
# 2. TARGET LOCATOR (ELEVATION SCORING)
# ==========================================
class TargetLocator:
    def __init__(self, z_tolerance=0.05, pad_ratio=0.25):
        self.z_tolerance = z_tolerance
        self.pad_ratio = pad_ratio

    def find_target(self, user_region_mask, full_depth_map):
        h, w = full_depth_map.shape
        
        # Filter valid depth points directly inside the user-defined polygon
        valid_mask = (full_depth_map > 0.1) & (~np.isnan(full_depth_map))
        inside_poly_mask = (user_region_mask == 255) & valid_mask
        
        z_poly = full_depth_map[inside_poly_mask]
        if len(z_poly) == 0: 
            return None
        
        # ELEVATION SCORING: Find the top layer (physically closest to camera / lowest z)
        peak_z = np.percentile(z_poly, 1.0)
        top_layer_pts_mask = inside_poly_mask & (full_depth_map >= peak_z - 0.01) & (full_depth_map <= peak_z + self.z_tolerance)
        
        top_layer_2d = np.zeros((h, w), dtype=np.uint8)
        top_layer_2d[top_layer_pts_mask] = 255
        
        # Gentler erosion to separate adjacent boxes without erasing small ones
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded = cv2.erode(top_layer_2d, kernel, iterations=2)
        
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return None
            
        best_core_cnt = None
        min_median_z = float('inf')
        
        # Pick the physically highest box among the top candidates
        for cnt in contours:
            if cv2.contourArea(cnt) < 400: 
                continue
                
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            
            valid_z = full_depth_map[(cnt_mask == 255) & (full_depth_map > 0.1)]
            if len(valid_z) == 0: continue
            
            median_z = np.median(valid_z)
            if median_z < min_median_z:
                min_median_z = median_z
                best_core_cnt = cnt
                
        if best_core_cnt is None: 
            return None
        
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(core_mask, [best_core_cnt], -1, 255, -1)
        single_box_mask = cv2.dilate(core_mask, kernel, iterations=2)
        single_box_mask = cv2.bitwise_and(single_box_mask, top_layer_2d) 
        
        restored_contours, _ = cv2.findContours(single_box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not restored_contours:
            return None
        best_cnt = max(restored_contours, key=cv2.contourArea)
        
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        rot_rect = cv2.minAreaRect(best_cnt)
        
        # Generate 5 Point Prompts using distance transform
        dist_map = cv2.distanceTransform(single_box_mask, cv2.DIST_L2, 5)
        _, _, _, center_loc = cv2.minMaxLoc(dist_map)
        cx, cy = center_loc
        
        point_prompts = [(cx, cy)]
        offsets = [(bw//4, bh//4), (-bw//4, bh//4), (bw//4, -bh//4), (-bw//4, -bh//4)]
        for dx, dy in offsets:
            px, py = cx + dx, cy + dy
            if 0 <= px < w and 0 <= py < h and single_box_mask[py, px] == 255:
                point_prompts.append((px, py))
        
        pad_w, pad_h = int(bw * self.pad_ratio), int(bh * self.pad_ratio)
        
        return {
            "bbox": (bx, by, bw, bh),
            "rotated_rect": rot_rect,
            "points": point_prompts,
            "padding": (pad_w, pad_h)
        }

# ==========================================
# 3. ROS 2 STREAMING NODE
# ==========================================
class PercipioSAM2LiveNode(Node):
    def __init__(self):
        super().__init__('percipio_sam2_live_node')
        self.bridge = CvBridge()
        
        self.rgb_sub = self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        
        print(f"\n--- ROS 2 SAM 2 Live Stream Node Started ---")
        print(f"Listening to Color: {RGB_TOPIC}")
        print(f"Listening to Depth: {DEPTH_TOPIC}")

    def rgb_callback(self, msg):
        # Bypass cv_bridge C++ bindings using direct NumPy byte buffer conversion
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
        # ROS 2 default color encoding is often rgb8 or bgr8
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        runtime_data['rgb_frame'] = cv2.flip(frame, 1)  # Mirror for natural workspace alignment

    def depth_callback(self, msg):
        # Bypass cv_bridge C++ bindings for 16-bit integer or 32-bit float depth streams
        if msg.encoding in ['16UC1', 'mono16']:
            depth_frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth_frame = depth_frame.astype(np.float32) / 1000.0  # Convert mm to meters
        elif msg.encoding == '32FC1':
            depth_frame = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            # Fallback assuming 16-bit unsigned integers (standard for Percipio depth)
            depth_frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth_frame = depth_frame.astype(np.float32) / 1000.0
            
        runtime_data['depth_frame'] = cv2.flip(depth_frame, 1)

# ==========================================
# 4. MAIN LIVE EXECUTION LOOP
# ==========================================
def main(args=None):
    global pallet_corners, calibration_done
    
    rclpy.init(args=args)
    node = PercipioSAM2LiveNode()
    
    # Initialize Models
    segmenter = SAM2Segmenter(SAM2_CHECKPOINT, SAM2_CONFIG)
    segmenter.load_model()
    locator = TargetLocator(z_tolerance=0.05, pad_ratio=0.25) 
    
    user_region_mask = None
    poly_pts = None

    print("\nWaiting for camera streams to start...")
    while rclpy.ok() and (runtime_data['rgb_frame'] is None or runtime_data['depth_frame'] is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        
    print("Streams detected! Starting interactive pipeline...")
    
    try:
        while rclpy.ok():
            # Process ROS callbacks without blocking the main rendering loop
            rclpy.spin_once(node, timeout_sec=0.005)
            
            img_bgr = runtime_data['rgb_frame']
            depth_m = runtime_data['depth_frame']
            
            if img_bgr is None or depth_m is None:
                continue
                
            img_h, img_w = img_bgr.shape[:2]

            if depth_m.shape[:2] != (img_h, img_w):
                depth_m = cv2.resize(depth_m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
            # --- PHASE 1: LIVE INTERACTIVE CALIBRATION ---
            if not calibration_done:
                cv2.namedWindow("1. Live Session Calibration", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("1. Live Session Calibration", int(img_w*0.7), int(img_h*0.7))
                cv2.setMouseCallback("1. Live Session Calibration", click_event)
                
                clone = img_bgr.copy()
                for i, pt in enumerate(pallet_corners):
                    cv2.circle(clone, pt, 6, (0, 255, 0), -1)
                    if i > 0: cv2.line(clone, pallet_corners[i-1], pt, (0, 255, 0), 2)
                        
                if len(pallet_corners) == 4:
                    cv2.line(clone, pallet_corners[3], pallet_corners[0], (0, 255, 0), 2)
                    cv2.putText(clone, "Press ENTER to confirm pallet zone", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(clone, f"Left-Click Corners: {len(pallet_corners)}/4 (Right-click to undo)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                cv2.imshow("1. Live Session Calibration", clone)
                k = cv2.waitKey(15) & 0xFF
                if k == 13 and len(pallet_corners) == 4: 
                    user_region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    poly_pts = np.array(pallet_corners, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(user_region_mask, [poly_pts], 255)
                    calibration_done = True
                    cv2.destroyWindow("1. Live Session Calibration")
                    
                    # Setup live visualization windows
                    cv2.namedWindow("1. Live Target Prompts", cv2.WINDOW_NORMAL)
                    cv2.namedWindow("2. Live SAM 2 Segmentation", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("1. Live Target Prompts", int(img_w*0.5), int(img_h*0.5))
                    cv2.resizeWindow("2. Live SAM 2 Segmentation", int(img_w*0.5), int(img_h*0.5))
                    print("\n[Live] Calibration confirmed! Running SAM 2 stream. Press 'r' to recalibrate, 'q' to quit.")
                elif k in [ord('c'), ord('C')]: 
                    pallet_corners = []
                continue

            # --- PHASE 2: LIVE TARGET LOCATOR & SAM 2 SEGMENTATION ---
            target = locator.find_target(user_region_mask, depth_m)
            
            debug_img = img_bgr.copy()
            final_output = img_bgr.copy()
            cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(final_output, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            if target:
                x, y, w, h = target["bbox"]
                pad_w, pad_h = target["padding"]
                point_prompts = target["points"]
                rot_rect = target["rotated_rect"]

                # Crop bounding area around target for efficient SAM inference
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
                
                cropped_bgr = img_bgr[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                local_depth = depth_m[y1:y2, x1:x2]
                
                local_box = [x - x1, y - y1, x + w - x1, y + h - y1]
                local_points = [[px - x1, py - y1] for (px, py) in point_prompts]

                # Run SAM 2
                local_mask, confidence = segmenter.predict(cropped_rgb, local_depth, local_box, local_points)
                global_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                global_mask[y1:y2, x1:x2] = (local_mask * 255).astype(np.uint8)

                # Draw Bounding Box & 5 Prompts
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                box_points = np.int32(cv2.boxPoints(rot_rect))
                cv2.drawContours(debug_img, [box_points], 0, (255, 0, 255), 2)
                for (px, py) in point_prompts:
                    cv2.circle(debug_img, (px, py), 5, (0, 0, 255), -1)              

                # Overlay Segmentation Mask
                overlay = img_bgr.copy()
                overlay[global_mask == 255] = [0, 255, 0] 
                mask_contours, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, mask_contours, -1, (255, 255, 255), 2)
                final_output = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
                
                # Draw Center of Mass
                y_indices, x_indices = np.where(global_mask == 255)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    final_cx, final_cy = int(np.median(x_indices)), int(np.median(y_indices))
                    cv2.circle(final_output, (final_cx, final_cy), 8, (255, 255, 255), -1)
                    cv2.circle(final_output, (final_cx, final_cy), 4, (0, 0, 255), -1)
            else:
                cv2.putText(debug_img, "No Valid Target Box Detected in Zone", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("1. Live Target Prompts", debug_img)
            cv2.imshow("2. Live SAM 2 Segmentation", final_output)
            
            # Keyboard controls during streaming
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Shutting down live stream...")
                break
            elif key in [ord('r'), ord('R'), ord('c'), ord('C')]:
                print("[Live] Resetting calibration zone...")
                pallet_corners = []
                calibration_done = False
                cv2.destroyAllWindows()

    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
import os
import cv2
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Configuration: Separate directories for each data type
OUTPUT_DIR = "percipio_captures1"
RGB_DIR = os.path.join(OUTPUT_DIR, "rgb_images")
DEPTH_PNG_DIR = os.path.join(OUTPUT_DIR, "depth_images")
DEPTH_NPY_DIR = os.path.join(OUTPUT_DIR, "depth_npy")

for directory in [RGB_DIR, DEPTH_PNG_DIR, DEPTH_NPY_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global runtime frame storage buffer
runtime_data = {'rgb_frame': None, 'depth_frame': None}

def mouse_callback(event, x, y, flags, param):
    """Saves RGB (.png), Depth (.png), and Depth (.npy) instantly on click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = runtime_data['rgb_frame']
        depth = runtime_data['depth_frame']
        
        if rgb is None or depth is None:
            print("Warning: Waiting for both RGB and Depth ROS 2 camera frames...")
            return
            
        timestamp = int(time.time() * 1000)
        
        # 1. Save RGB Frame (.png)
        rgb_path = os.path.join(RGB_DIR, f"rgb_{timestamp}.png")
        cv2.imwrite(rgb_path, rgb)
        
        # 2. Save Depth Visual (.png - preserves 16-bit/8-bit structure for viewing)
        depth_png_path = os.path.join(DEPTH_PNG_DIR, f"depth_{timestamp}.png")
        cv2.imwrite(depth_png_path, depth)

        # 3. Save Raw Depth Array (.npy - preserves exact raw numerical distance values)
        depth_npy_path = os.path.join(DEPTH_NPY_DIR, f"depth_{timestamp}.npy")
        np.save(depth_npy_path, depth)
        
        print(f"Captured! Saved RGB (png), Depth (png), and Depth (npy) for timestamp: {timestamp}")

class PercipioMultimodalCaptureNode(Node):
    def __init__(self):
        super().__init__('percipio_multimodal_capture_node')
        self.bridge = CvBridge()
        
        # 1. RGB Color Subscription
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Verify your exact color topic
            self.rgb_callback,
            10
        )

        # 2. Depth Subscription
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Change to your camera's depth topic (e.g., /camera/aligned_depth_to_color/image_raw)
            self.depth_callback,
            10
        )
        
        self.window_name = "Percipio RGB Stream - Click Box to Save All Formats"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        print(f"\n--- ROS 2 Multimodal Capture Node Started ---")
        print(f"Saving outputs to: ./{OUTPUT_DIR}/")
        print("Left Click : Save RGB (.png), Depth (.png), and Depth (.npy)")
        print("Press 'q'  : Quit node application")

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.flip(frame, 1)  # Mirror for workspace alignment
        runtime_data['rgb_frame'] = frame
        
        # Display the RGB stream in the GUI window
        cv2.imshow(self.window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Shutting down capture node...")
            rclpy.shutdown()

    def depth_callback(self, msg):
        # Use 'passthrough' so CvBridge keeps exact 16-bit integers or 32-bit floats from the sensor
        depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_frame = cv2.flip(depth_frame, 1)  # Mirror to match RGB alignment
        runtime_data['depth_frame'] = depth_frame

def main(args=None):
    rclpy.init(args=args)
    node = PercipioMultimodalCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
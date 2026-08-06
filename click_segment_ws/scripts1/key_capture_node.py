#!/usr/bin/env python3
import os
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge


class KeyCaptureNode(Node):
    def __init__(self):
        super().__init__('key_capture_node')

        self.declare_parameter('out_dir', 'percipio_captures')
        self.out_dir = self.get_parameter('out_dir').get_parameter_value().string_value
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'color'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'points'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'camera_info'), exist_ok=True)

        self.bridge = CvBridge()
        self.count = 0

        self.latest_color_info = None
        self.latest_depth_info = None
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                  lambda msg: setattr(self, 'latest_color_info', msg),
                                  qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info',
                                  lambda msg: setattr(self, 'latest_depth_info', msg),
                                  qos_profile_sensor_data)

        # Set up message filter subscribers
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw',
                                                 qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw',
                                                 qos_profile=qos_profile_sensor_data)
        points_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth_registered/points',
                                                  qos_profile=qos_profile_sensor_data)

        # Synchronize color, depth, AND the point cloud together
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, points_sub], queue_size=10, slop=0.08)
        self.ts.registerCallback(self.sync_callback)

        self.last_color = None
        self.last_depth = None
        self.last_points = None

        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
        self.get_logger().info(
            "Ready. Window 'preview' focused: [SPACE]=save  [q]=quit")

        self.create_timer(0.03, self.spin_gui)

    def sync_callback(self, color_msg, depth_msg, points_msg):
        self.last_color = color_msg
        self.last_depth = depth_msg
        self.last_points = points_msg

        color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        cv2.imshow('preview', color_img)

    def spin_gui(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self.save_snapshot()
        elif key == ord('q'):
            self.get_logger().info("Quitting.")
            rclpy.shutdown()

    def save_snapshot(self):
        if self.last_color is None or self.last_depth is None or self.last_points is None:
            self.get_logger().warn("No synced frameset yet, skip.")
            return

        stamp = self.last_color.header.stamp
        ts_str = f"{stamp.sec}_{stamp.nanosec:09d}"
        idx = f"{self.count:04d}"

        color_img = self.bridge.imgmsg_to_cv2(self.last_color, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding='passthrough')

        color_path = os.path.join(self.out_dir, 'color', f'{idx}_{ts_str}.png')
        depth_path = os.path.join(self.out_dir, 'depth', f'{idx}_{ts_str}.png')
        points_path = os.path.join(self.out_dir, 'points', f'{idx}_{ts_str}.npy')
        info_path = os.path.join(self.out_dir, 'camera_info', f'{idx}_{ts_str}.yaml')

        # Save image tracks
        cv2.imwrite(color_path, color_img)
        if depth_img.dtype == np.float32:
            np.save(depth_path.replace('.png', '.npy'), depth_img)
        else:
            cv2.imwrite(depth_path, depth_img)

        # Parse and save the point cloud data efficiently
        # Generates a numpy structured array containing fields: x, y, z, (and optionally rgb)
                # Read points into a structured numpy array natively supported by the API
        gen = pc2.read_points(self.last_points, skip_nans=False)
        pc_data = np.array(list(gen))
        np.save(points_path, pc_data)

        # Save camera info yaml
        info_dict = {}
        if self.latest_color_info is not None:
            info_dict['color'] = {
                'K': [float(x) for x in self.latest_color_info.k],
                'width': int(self.latest_color_info.width),
                'height': int(self.latest_color_info.height),
            }
        if self.latest_depth_info is not None:
            info_dict['depth'] = {
                'K': [float(x) for x in self.latest_depth_info.k],
                'width': int(self.latest_depth_info.width),
                'height': int(self.latest_depth_info.height),
            }
        with open(info_path, 'w') as f:
            yaml.safe_dump(info_dict, f)

        self.get_logger().info(f"Saved snapshot {idx} including PointCloud data.")
        self.count += 1


def main():
    rclpy.init()
    node = KeyCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == '__main__':
    main()

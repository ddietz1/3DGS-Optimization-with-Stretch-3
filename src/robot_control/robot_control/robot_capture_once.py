"""
Minimal robot-side capture trigger. NO navigation, NO camera movement --
just grabs one synced RGB+depth frame from wherever the D435 currently
points, saves it, and exits. Meant to be invoked over SSH by the GPU-side
orchestrator (gpu_transport_test.py) once per interval, to test the
transport pipeline before any real IK/motion is involved.

Mirrors the synced-capture pattern from your existing MoveJoints node
(message_filters ApproximateTimeSynchronizer over the same D435 topics),
stripped down to just the capture, no waypoint/navigation machinery.

Usage: python3 robot_capture_once.py
(run this on the robot -- it's what ROBOT_CAPTURE_SCRIPT in robot_loop.py
/ gpu_transport_test.py should point at)
"""

import os
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

CAPTURE_DIR = os.path.expanduser("~/stretch_user/captures/transport_test")
TIMEOUT_SEC = 15.0  # bumped up from 5.0 -- a freshly-spun-up node needs time for
# DDS discovery to match with the already-running camera driver's publishers,
# on top of SSH/python/rclpy startup overhead; 5s was likely cutting this close.

image_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

class SingleCapture(Node):
    def __init__(self):
        super().__init__("single_capture")
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        self.bridge = CvBridge()
        self.got_frame = False

        # Explicit qos_profile_sensor_data matches the QoS most camera
        # drivers actually publish with (BEST_EFFORT reliability). Without
        # this, a subscriber defaults to RELIABLE -- if the publisher is
        # BEST_EFFORT, the two are QoS-incompatible and DDS silently never
        # connects them at all, with no error printed anywhere. Confirm via
        # `ros2 topic info <topic> --verbose` before assuming this was the
        # actual cause here, rather than just trusting this fixed it.
        rgb_sub = message_filters.Subscriber(self, Image, "/camera/color/image_raw",
                                              qos_profile=image_QOS)
        depth_sub = message_filters.Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw",
                                                qos_profile=image_QOS)
        info_sub = message_filters.Subscriber(self, CameraInfo, "/camera/aligned_depth_to_color/camera_info",
                                               qos_profile=image_QOS)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.frame_cb)

    def frame_cb(self, rgb_msg, depth_msg, info_msg):
        if self.got_frame:
            return  # only want the first synced frame
        self.got_frame = True

        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Same naming suffix convention as the rest of the pipeline
        # (_d435_rgb.png / _d435_depth.npy) so it's directly compatible
        # with build_frame_to_depth_map on the GPU side if you want to
        # test depth features later too.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rgb_path = os.path.join(CAPTURE_DIR, f"capture_{stamp}_d435_rgb.png")
        depth_path = os.path.join(CAPTURE_DIR, f"capture_{stamp}_d435_depth.npy")
        cv2.imwrite(rgb_path, rgb_img)
        np.save(depth_path, depth_img)
        self.get_logger().info(f"Saved {rgb_path}")
        self.get_logger().info(f"Saved {depth_path}")


def main():
    rclpy.init()
    node = SingleCapture()

    start = time.time()
    while not node.got_frame:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start > TIMEOUT_SEC:
            node.get_logger().error(f"Timed out after {TIMEOUT_SEC}s waiting for a synced frame")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
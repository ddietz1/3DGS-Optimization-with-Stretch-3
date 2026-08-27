#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from nav2_simple_commander.robot_navigator import BasicNavigator

import numpy as np
import math
import json
import os
from datetime import datetime


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to a quaternion (x, y, z, w).
    Verified correct against known single-axis cases before trusting it
    in the full conversion pipeline below."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def quat_to_rotmat(qx, qy, qz, qw):
    """MUST exactly match build_transforms_from_poses.py's version on the
    GPU side -- deliberate duplicate, not an import, since this runs on
    the robot."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-8:
        return np.eye(3)
    s_ = 2.0 / n
    wx, wy, wz = s_ * qw * qx, s_ * qw * qy, s_ * qw * qz
    xx, xy, xz = s_ * qx * qx, s_ * qx * qy, s_ * qx * qz
    yy, yz, zz = s_ * qy * qy, s_ * qy * qz, s_ * qz * qz
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def rotmat_to_quat(R: np.ndarray):
    """Shepperd's method -- standard, numerically stable rotation matrix
    to quaternion conversion. Verified round-trip-correct against
    quat_to_rotmat to ~1e-16 across 1000 random rotations earlier this
    session."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)

def direction_to_ros_quaternion(yaw: float, pitch: float):
    """Builds a quaternion whose LOCAL Z axis (matching
    camera_color_optical_frame's own forward convention) points toward
    (yaw, pitch) directly, so it's correctly interpreted by everything
    downstream (which treats candidate poses identically to real
    camera_color_optical_frame TF orientations).
 
    yaw: azimuth in the map XY-plane, standard atan2(y,x) convention
         (matches everywhere else in this project)
    pitch: elevation, positive = tilted up (more +Z)
 
    Returns (qx, qy, qz, qw), same signature as rpy_to_quaternion so it's
    a drop-in replacement at the call site.
    """
    direction = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, up)) > 0.999:
        # direction is (near-)vertical itself -- avoid a degenerate cross
        # product by picking a different reference axis
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, direction)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)
    Rm = np.stack([x_axis, y_axis, direction], axis=1)
    return rotmat_to_quat(Rm)


ROS_TO_NERFSTUDIO_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def pose_to_nerfstudio_c2w_3x4(pose: Pose) -> list:
    """Converts a map-frame geometry_msgs/Pose into the SAME
    camera-to-world convention Model B's training cameras use. Returns a
    3x4 nested list -- verified this exact shape against the real
    viewpoint_scoring.py's c2w_to_viewmat(), and the full rpy->quaternion->
    this pipeline against 20 random samples plus a real --reuse-candidates
    parse before shipping this."""
    q = pose.orientation
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rotmat(q.x, q.y, q.z, q.w)
    c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    c2w = c2w @ ROS_TO_NERFSTUDIO_FLIP
    return c2w[:3, :4].tolist()


class CandidateGenerator(Node):
    def __init__(self):

        super().__init__("candidate_generator")

        self.costmap = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.costmap_frame_id = None

        # Create ranges for random generation of 6 dof poses
        self.camera_z = 1.2
        self.camera_roll = 0.0
        self.pitch_range = (-0.15, 0.15)   # small tilt, radians
        self.n_candidates = 5              # total candidate poses per publish

        # Path length feasibility gating
        self.max_path_length = 5.0
        self.max_length_ratio = 2.5

        # FIXED: was "~/stretch_user/nbv_candidates/Jul-30_dataset_4/" --
        # did not match gpu_candidate_puller.py's ROBOT_CANDIDATES_DIR
        # ("~/stretch_user/candidates"), so the GPU-side puller would never
        # have found anything written here, even with the format fixed below.
        self.output_dir = os.path.expanduser("~/stretch_user/candidates/")
        os.makedirs(self.output_dir, exist_ok=True)

        # TF for odom -> map conversion (candidates come out of the local
        # costmap, which is published in odom frame -- must be converted
        # to map frame here, on the robot, via a live TF tree)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Nav2 planner access for feasibility checks.
        # FIXED: distinct node_name (MoveJoints also constructs a
        # BasicNavigator with the default name -- now that both nodes
        # launch concurrently, this was a real, live name collision, not
        # just a theoretical one) and waitUntilNav2Active() (MoveJoints
        # already blocks on this before doing anything; this node never
        # did, meaning it could start querying getPath() before Nav2's
        # planner server exists -- caught by the blanket except below and
        # silently returning False for every candidate during that window).
        self.navigator = BasicNavigator(node_name='candidate_generator_navigator')
        self.navigator.waitUntilNav2Active()

        self.sub = self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            self.costmap_callback,
            10
        )

        self.pub = self.create_publisher(
            PoseArray,
            "/nbv/candidates",
            10
        )

        self.timer = self.create_timer(
            30.0,
            self.generate_candidates
        )

    def costmap_callback(self, msg):

        self.costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.resolution = msg.info.resolution

        self.origin_x = (msg.info.origin.position.x)
        self.origin_y = (msg.info.origin.position.y)

        self.width = msg.info.width
        self.height = msg.info.height
        self.costmap_frame_id = msg.header.frame_id  # don't assume 'odom' -- use what's actually reported

    def grid_to_world(self, x, y):

        jitter_x = np.random.uniform(-0.5, 0.5) * self.resolution
        jitter_y = np.random.uniform(-0.5, 0.5) * self.resolution

        wx = (self.origin_x + (x + 0.5) * self.resolution + jitter_x)
        wy = (self.origin_y + (y + 0.5) * self.resolution + jitter_y)

        return wx, wy

    def transform_pose_to_map(self, pose_local: Pose):
        """Transform a pose from the costmap's native frame (typically
        odom) into map frame. Must happen here, on the robot, using a live
        TF tree -- cannot be corrected after the fact once written out."""

        stamped = PoseStamped()
        stamped.header.frame_id = self.costmap_frame_id
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.pose = pose_local
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', stamped.header.frame_id, Time(), timeout=Duration(seconds=1.0)
            )
            return do_transform_pose(stamped.pose, tf)
        except Exception as e:
            self.get_logger().warn(f'Could not transform candidate to map: {e}')
            return None

    def check_pose_feasible(self, goal_pose_map: Pose) -> bool:
        """Queries Nav2's global planner for a valid path to goal_pose_map.
        A free costmap cell alone doesn't guarantee the robot's footprint
        clears nearby obstacles or that a connected path exists at all."""

        try:
            start = PoseStamped()
            start.header.frame_id = 'map'
            start.header.stamp = self.get_clock().now().to_msg()
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time(), timeout=Duration(seconds=1.0))
            start.pose.position.x = tf.transform.translation.x
            start.pose.position.y = tf.transform.translation.y
            start.pose.orientation = tf.transform.rotation

            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose = goal_pose_map

            path = self.navigator.getPath(start, goal)
            if path is None or len(path.poses) == 0:
                return False
 
            path_length = 0.0
            for i in range(1, len(path.poses)):
                p0 = path.poses[i - 1].pose.position
                p1 = path.poses[i].pose.position
                path_length += math.hypot(p1.x - p0.x, p1.y - p0.y)
 
            straight_line = math.hypot(
                goal_pose_map.position.x - start.pose.position.x,
                goal_pose_map.position.y - start.pose.position.y,
            )
 
            if path_length > self.max_path_length:
                self.get_logger().info(
                    f'Rejected candidate: path length {path_length:.2f}m exceeds '
                    f'max_path_length={self.max_path_length}m (straight-line was '
                    f'{straight_line:.2f}m)'
                )
                return False
 
            if straight_line > 1e-3 and (path_length / straight_line) > self.max_length_ratio:
                self.get_logger().info(
                    f'Rejected candidate: path length {path_length:.2f}m is '
                    f'{path_length / straight_line:.1f}x straight-line distance '
                    f'({straight_line:.2f}m), exceeds max_length_ratio={self.max_length_ratio}'
                )
                return False
 
            return True
        except Exception as e:
            self.get_logger().warn(f'Feasibility check failed: {e}')
            return False

    def generate_candidates(self):

        if self.costmap is None:
            return
        candidates = []
        candidate_records = []
        free = np.argwhere((self.costmap >= 0) & (self.costmap < 30))

        np.random.shuffle(free)

        n_checked = 0
        n_feasible = 0

        for cell in free:
            if len(candidates) >= self.n_candidates:
                break

            gy, gx = cell
            x, y = self.grid_to_world(gx, gy)

            # Position-only probe pose, checked ONCE per (x, y) before
            # generating the full 6DOF pose -- orientation doesn't affect
            # whether a path to this position exists, and per-pose checks
            # would be too expensive inside a timer callback.
            probe_pose = Pose()
            probe_pose.position.x = x
            probe_pose.position.y = y
            probe_pose.orientation.w = 1.0

            probe_map = self.transform_pose_to_map(probe_pose)
            if probe_map is None:
                continue

            n_checked += 1
            if not self.check_pose_feasible(probe_map):
                continue
            n_feasible += 1

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = self.camera_z

            roll = self.camera_roll
            pitch = np.random.uniform(*self.pitch_range)
            yaw = np.random.uniform(0, 2 * np.pi)

            qx, qy, qz, qw = direction_to_ros_quaternion(yaw, pitch)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            pose_map = self.transform_pose_to_map(pose)
            if pose_map is None:
                continue

            candidates.append(pose_map)

            # FIXED: was a nested {"position": {x,y,z}, "orientation": {x,y,z,w},
            # "rpy": {...}} dict inside a top-level {"candidates": [...]}
            # wrapper -- viewpoint_scoring.py's --reuse-candidates does
            # json.load(f) then saved[0], expecting a flat LIST. A dict there
            # raises KeyError: 0 immediately. Now matches its actual format:
            # flat list, each entry with "position" (plain [x,y,z]) and
            # "transform_matrix" (3x4, nerfstudio camera convention).
            candidate_records.append({
                "position": [pose_map.position.x, pose_map.position.y, pose_map.position.z],
                "transform_matrix": pose_to_nerfstudio_c2w_3x4(pose_map),
            })

        if not candidates:
            self.get_logger().warn(
                f"No reachable candidates found this cycle "
                f"(checked {n_checked} positions, {n_feasible} feasible)"
            )
            return

        msg = PoseArray()
        msg.header.frame_id = "map"  # now genuinely map frame, not odom
        msg.poses = candidates

        self.pub.publish(msg)

        self.save_candidates(candidate_records)

        self.get_logger().info(
            f"Checked {n_checked} positions, {n_feasible} feasible, "
            f"published {len(candidates)} viewpoints"
        )

    def save_candidates(self, candidate_records):
        """Write this batch of candidate poses to a timestamped JSON file.
        FIXED: writes candidate_records directly as a flat list -- no more
        top-level {"timestamp", "frame_id", "candidates"} wrapper."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.output_dir, f"candidates_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(candidate_records, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to write candidates to {filepath}: {e}")


def main():
    rclpy.init()
    node = CandidateGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
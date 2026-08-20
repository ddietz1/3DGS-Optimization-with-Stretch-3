"""Standalone test: compute a route between two graph nodes, then drive it.

Usage:
    python3 test_route_follow.py

This is disposable test code, not meant to be integrated as-is.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import ComputeRoute, FollowPath


class RouteFollowTest(Node):

    def __init__(self):
        super().__init__('route_follow_test')

        self.compute_route_client = ActionClient(
            self, ComputeRoute, 'compute_route')
        self.follow_path_client = ActionClient(
            self, FollowPath, 'follow_path')

        self.computed_path = None

    def wait_for_servers(self, timeout_sec=10.0):
        self.get_logger().info('Waiting for compute_route action server...')
        if not self.compute_route_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error('compute_route server not available')
            return False

        self.get_logger().info('Waiting for follow_path action server...')
        if not self.follow_path_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error('follow_path server not available')
            return False

        return True

    def compute_route(self, start_id, goal_id, use_start=False, timeout_sec=30.0):
        """Call ComputeRoute and block until we get a result with a path."""

        goal_msg = ComputeRoute.Goal()
        goal_msg.use_poses = False
        goal_msg.use_start = use_start
        goal_msg.start_id = start_id
        goal_msg.goal_id = goal_id

        self.get_logger().info(f'Requesting route: node {start_id} -> node {goal_id}')
        goal_future = self.compute_route_client.send_goal_async(goal_msg)

        start = self.get_clock().now()
        while not goal_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out sending compute_route goal')
                return None

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('compute_route goal was rejected')
            return None

        result_future = goal_handle.get_result_async()
        start = self.get_clock().now()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out waiting for compute_route result')
                return None

        result_wrapper = result_future.result()
        status = result_wrapper.status
        result = result_wrapper.result

        self.get_logger().info(f'compute_route finished with status: {status}')
        self.get_logger().info(f'Path has {len(result.path.poses)} poses')

        if len(result.path.poses) == 0:
            self.get_logger().error('Computed path is empty!')
            return None

        return result.path

    def follow_path(self, path, timeout_sec=60.0):
        """Send a Path to FollowPath and block until it finishes."""

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = ''
        goal_msg.goal_checker_id = ''

        self.get_logger().info(
            f'Sending FollowPath goal with {len(path.poses)} poses')
        goal_future = self.follow_path_client.send_goal_async(
            goal_msg, feedback_callback=self.follow_path_feedback_cb)

        start = self.get_clock().now()
        while not goal_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out sending follow_path goal')
                return False

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('follow_path goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        start = self.get_clock().now()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().error('Timed out waiting for follow_path result')
                return False

        result_wrapper = result_future.result()
        self.get_logger().info(f'follow_path finished with status: {result_wrapper.status}')
        return result_wrapper.status == 4  # GoalStatus.STATUS_SUCCEEDED

    def follow_path_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'FollowPath feedback: distance_to_goal={fb.distance_to_goal:.3f}, '
            f'speed={fb.speed:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RouteFollowTest()

    try:
        if not node.wait_for_servers():
            sys.exit(1)

        # Adjust these node IDs to match your actual graph
        start_id = 5
        goal_id = 6

        path = node.compute_route(start_id, goal_id, use_start=False)
        if path is None:
            node.get_logger().error('Failed to compute route, aborting test')
            sys.exit(1)

        node.get_logger().info('Route computed successfully. Sending to controller...')
        time.sleep(1.0)

        success = node.follow_path(path)
        if success:
            node.get_logger().info('SUCCESS: robot followed the path!')
        else:
            node.get_logger().error('FollowPath did not succeed')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
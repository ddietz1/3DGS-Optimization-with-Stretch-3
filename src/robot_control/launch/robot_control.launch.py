
import os
import launch_ros
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    map_yamlPath = os.path.join(
        get_package_share_directory('robot_control'),
        'config/maps',
        'Zoo_Map_Final.yaml'
    )
    yamlPath = os.path.join(
        get_package_share_directory('robot_control'),
        'config',
        'parameters.yaml'
    )

    JointsPath = os.path.join(
        get_package_share_directory('robot_control'),
        'config',
        'robot_joints.yaml'
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('stretch_core'),
            'launch', 'multi_camera.launch.py'
        )),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('stretch_nav2'),
            'launch', 'navigation.launch.py')),
        launch_arguments={'map': map_yamlPath, 'use_rviz': 'false'}.items()
    )

    move_joints_node = Node(
        package='robot_control',
        executable='move_joints',
        name='move_joints',
        output='screen',
        parameters=[yamlPath]
    )

    # candidate_generation_node = Node(
    #     package='candidate_generation',
    #     executable='generate_candidates',
    #     name='generate_candidates',
    #     output='screen'
    # )

    capture_next_view_process = ExecuteProcess(
        cmd=['python3', '/home/hello-robot/ament_ws/robot_capture_next_view.py',
             '--poll-interval', '10'],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='/home/hello-robot/stretch_user/maps/Zoo_Map_Final.yaml',
            description='Full path to map yaml file'
        ),
        # stretch_driver_launch,
        nav2_launch,
        camera_launch,
        move_joints_node,
        # candidate_generation_node,
        capture_next_view_process
    ])
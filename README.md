# 3DGS-Optimization-with-Stretch-3

* Derek Dietz
* Final Project
# Package List
This repository consists of several ROS packages
- `robot_control` - 
- `robot_control_interfaces` - 
- `candidate_generation` - 

## Summary
A system for optimizing the creation of a 3D Gaussian Splatting model of a scene using a Hello Robot Stretch 3.

## Overview
The system integrates:

* ROS2 nodes for controlling the onboard and gripper cameras of the robot, navigation to various waypoints, and capture of images with the mounted d435i RealSense camera.

* Python based scripts on a separate GPU for creation of a 3D Gaussian Splatting model using NerfStudio tools, as well as scripts for updating the model with new images, automatic seeding of gaussian primitives using the depth files from the camera, and scoring of generated reachable poses using Shannon Mutual information. 

This repository contains the ROS2 packages and python scripts that allow the user to automate the creation and optimization of a 3DGS model by setting waypoints on a set map frame.

### Python Scripts
* gpu_main_loop.py - Runs the entire pipeline, pulls and pushes data from Stretch 3 to GPU using scp protocol
* build_transforms_from_poses.py - Creates a standard frame for the 3DGS model, used as an alternative to creating the frame using COLMAP which is standard in NerfStudio
* gpu_candidate_puller.py - Used to pull generated, reachable candidate poses from the Stretch 3's stretch_user/candidates/ directory for scoring on the GPU
* viewpoint_scoring.py - Uses plain PyTorch to score possible poses by Shannon-Mutual Information gain. Based on the GauSS-MI paper cited below, this script initializaes a per-primitive reliability tensor that is used to determine the reliability of a given gaussian for scoring
* ns_train_patched.py - A recreation of the ns-train script that comes standard with NerfStudio. This script operates similarly but is capable of inserting new image/pose combinations to update the model rather than retraining from scratch

## Installation
1. Clone the repository into your ROS 2 workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/ddietz1/3DGS-Optimization-with-Stretch-3.git
```
2. Install Python dependencies
```bash
pip install transformers
```
3. Install Python dependencies
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```
4. Build the workspace
```bash
cd ~/ros2_ws
colcon build --packages-select bluerov_control
source install/setup.bash
```

## Running the system
```bash
ros2 launch robot_control robot_control.launch.py
```
This starts:
* The Stretch 3 to automatically start taking images at the list of waypoints set in parameters.yaml
* The candidate generation node to begin sampling the local costmap, determine pose reachability using the NAV2 navigator, and saving feasible poses in json files. 

For run the full pipeline, also run on the GPU:
```bash
CUDA_VISIBLE_DEVICES=1 python3 gpu_main_loop.py   --robot-run-name <run ID set in parameters.yaml>   --base-dir <directory to save this runs information and models>   --fx <Camera focal length x> --fy <Camera focal length y> --cx <Camera principal point x> --cy <Camera principal point y> --width <Image width> --height <Image height>   --pose-source direct --stop-after-round <However many rounds you would like to run the optimization pipeline>
```
This stars the main loop file that polls the captures directory on the Stretch 3 until a DONE is seen in the json file. After that it pulls all captures and creates the initial model. The script then scores the generated candidates and returns the top scoring candidate to the robot, commanding it to capture that pose using the test_d435_ik service in the robot_control package. 

## Package architecture
```
Retriever_Bot/
├── bluerov_control/          # Main ROS 2 package
|   ├── bridge_node.py        # MAVROS bridge
|   ├── camera_node.py        # Retreives camera info from ROV
│   ├── object_detection.py   # Vision pipeline & ring localization
│   ├── control.py            # Velocity-based servoing & gripper logic
|   ├── yolo_detect.py        # Runs image topics through yolo model
│   └── ...
├── bluerov_heading/          # ROS 2 package for heading control
|   ├── heading_node.py       # Determines current heading of the ROV
├── launch/                   # ROS2 launch files
├── resource/                 # Package resource files
├── package.xml               # ROS2 package manifest
├── setup.py                  # Python package setup
└── setup.cfg
```
# DEMO VIDEOS
## Spliced video of the robot running alongside the robot running in SIM collecting data
https://private-user-images.githubusercontent.com/107367597/546992272-88b9b63d-8e34-403e-ac9a-281c74fcaded.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA2Mjc3MzMsIm5iZiI6MTc3MDYyNzQzMywicGF0aCI6Ii8xMDczNjc1OTcvNTQ2OTkyMjcyLTg4YjliNjNkLThlMzQtNDAzZS1hYzlhLTI4MWM3NGZjYWRlZC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIwOVQwODU3MTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZjBmM2IxODYxZTM1ZGYyMGY3MGQyN2NiMzQzZTM5OWI2NTNkYmQzMjY1MWNkMGQxMjE3MWM2ZmM5ODAzNDAwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.GCWQS1hsusx8Jw-dexZiTYXbugxJihUEK93SpvoFZew

## Video showing a high scoring pose being slowly rendered better after a single round
https://private-user-images.githubusercontent.com/107367597/546992131-274ce723-63c9-45a2-945a-19d323abb639.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA2Mjc3OTUsIm5iZiI6MTc3MDYyNzQ5NSwicGF0aCI6Ii8xMDczNjc1OTcvNTQ2OTkyMTMxLTI3NGNlNzIzLTYzYzktNDVhMi05NDVhLTE5ZDMyM2FiYjYzOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIwOVQwODU4MTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zZTdkODhhMThhNThiNjRhZmZkNDhkZTgxMWE0MzdiMDYzZWYzYWU4YmFiZmY3YmVhY2VjMGZkMWEzMDU5NjU4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.tEr1M7rPtvg7_8GTZdAtZIOagLRV5WvAopuezxetguo

## Data
### Graph showing PSNR(Peak Signal to Noise ratio) for an optimized run versus PSNR for a run that uses the same number of random images and creates a model in one shot

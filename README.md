# 3DGS-Optimization-with-Stretch-3

## Summary
A system for optimizing the creation of a 3D Gaussian Splatting model of a scene using a Hello Robot Stretch 3. 

## Overview
The system integrates:

* ROS2 nodes for controlling the onboard camera of the robot, navigating to waypoints, sampling and validating next-best-view candidate poses against the local costmap/Nav2 planner, and capturing RGB-D images with the mounted D435 RealSense camera.
* Python scripts on a separate GPU for building and updating a 3D Gaussian Splatting model with NerfStudio, automatically seeding Gaussian primitives from the camera's depth data, and scoring reachable candidate poses by Shannon Mutual Information (an active-view-selection / next-best-view strategy).

Together, these let the pipeline automatically capture a scene, train an initial 3DGS model, and iteratively drive the robot to the most informative next viewpoint.

## Package List
This repository consists of three ROS2 packages plus a set of standalone Python scripts run on a GPU workstation.

- `robot_control` - Core motion/capture node (`motion_control.py`). Handles mode switching (navigation/position), waypoint following, Nav2-based navigation to a solved base pose, head pan/tilt IK for aiming the D435 at a requested pose, and capturing + logging RGB, depth, and map-frame pose for each frame. Exposes the `test_d435_ik` service used by the NBV loop to command a specific candidate pose.
- `robot_interfaces` - Custom ROS2 service definitions shared between packages (`Camera.srv`, `TestPoses.srv`).
- `candidate_generation` - Periodically samples the local costmap for free, Nav2-reachable cells, generates randomized 6-DOF candidate viewpoints around them, and writes them as timestamped JSON files to `~/stretch_user/candidates/` for the GPU-side scoring pipeline to pull.

## Python Scripts
Run on a separate GPU workstation (needs a CUDA-capable GPU, NerfStudio + gsplat installed):

* `gpu_main_loop.py` - Orchestrates the entire pipeline end to end: waits for the robot's initial waypoint capture, builds the seed dataset, trains the initial model, then repeatedly scores candidates, sends the top pose(s) back to the robot, waits for the resulting new capture, and resume-trains the model. Pulls/pushes data to and from the Stretch 3 over `scp`.
* `gpu_candidate_puller.py` - Runs continuously in the background (started automatically by `gpu_main_loop.py`); pulls newly-written candidate JSON files from the Stretch 3's `~/stretch_user/candidates/` directory to the GPU.
* `score_and_return_top_candidates.py` - Scores the currently pulled candidate pool against the latest trained checkpoint (via `viewpoint_scoring.py`), converts the top-N poses back into ROS-native (map-frame position + quaternion) form, and pushes them to the robot's `~/stretch_user/scored_candidates/`.
* `viewpoint_scoring.py` - Plain-PyTorch reimplementation of Shannon Mutual Information viewpoint scoring, based on the GauSS-MI approach (see Citations). Maintains a per-Gaussian-primitive "reliability" tensor updated from photometric residuals over training views, and scores candidate poses by the expected information gain of observing from there.
* `build_transforms_from_poses.py` - Builds a `transforms.json` dataset directly from the robot's own AMCL/map-frame poses (rather than deriving camera poses via COLMAP, which is NerfStudio's usual approach), including initial point-cloud seeding from depth.
* `ns_train_patched.py` - A patched drop-in replacement for NerfStudio's standard `ns-train` entrypoint that correctly resumes/updates an existing Gaussian Splatting model with newly incorporated images instead of retraining from scratch.
* `check_convergence.py` - Decides whether the active-view loop should keep going or stop, using two self-calibrating criteria (score relative to the first round's max, and diminishing returns over recent rounds) rather than a fixed score threshold.
* `robot_capture_next_view.py` - Runs on the Stretch 3, not the GPU. Watches `~/stretch_user/scored_candidates/` for newly-scored candidates and calls `robot_control`'s `test_d435_ik` service to drive to and capture the top-ranked reachable pose.

## Installation
1. Clone the repository into your ROS 2 workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/ddietz1/3DGS-Optimization-with-Stretch-3.git
```
2. Install ROS2 dependencies (tested on ROS2 Humble)
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```
3. Set up the GPU-side Python environment (used for `gpu_main_loop.py` and everything it calls). Tested against:
   - `torch==2.1.2` (cu118)
   - `nerfstudio==1.1.5`
   - `gsplat==1.4.0`
   - `numpy==1.24.4`, `scipy==1.10.1`, `plyfile==1.0.3`
```bash
conda create -n nerfstudio python=3.10
conda activate nerfstudio
pip install nerfstudio==1.1.5 gsplat==1.4.0 plyfile
```
4. Build the ROS2 workspace
```bash
cd ~/ros2_ws
colcon build --packages-select robot_control robot_interfaces candidate_generation
source install/setup.bash
```

## Running the system
On the robot:
```bash
ros2 launch robot_control robot_control.launch.py
```
This starts:
* `robot_control`'s motion node, which drives the robot through the waypoints listed in `parameters.yaml` and captures an image at each, tagging the run with the ID set there (used as `--robot-run-name` below).
* The `candidate_generation` node, which begins sampling the local costmap, checking pose reachability via the Nav2 planner, and writing feasible candidate poses to JSON files for the GPU to pull.

Separately, once initial waypoint capture is underway, run on the GPU:
```bash
CUDA_VISIBLE_DEVICES=<gpu id> python3 gpu_main_loop.py \
  --robot-run-name <run ID set in parameters.yaml> \
  --base-dir <directory to save this run's data and models> \
  --fx <camera focal length x> --fy <camera focal length y> \
  --cx <camera principal point x> --cy <camera principal point y> \
  --width <image width> --height <image height> \
  --pose-source direct \
  --stop-after-round <number of active-view rounds to run, or omit to run until convergence>
```
This polls the robot's capture directory until a `DONE` marker appears, pulls the initial waypoint captures, and trains the seed model. It then scores the current candidate pool and sends the top-scoring reachable candidate(s) back to the robot, which drives to and captures the winning pose via `robot_control`'s `test_d435_ik` service (through `robot_capture_next_view.py`, running on the robot). The GPU incorporates the new capture, resume-trains, and repeats until convergence or `--stop-after-round` is reached.

## Package architecture
```
3DGS-Optimization-with-Stretch-3/
├── robot_control/                   # Core motion + capture ROS2 package
│   ├── robot_control/
│   │   └── motion_control.py        # Nav2 driving, pan/tilt IK, capture + pose logging, test_d435_ik service
│   ├── launch/
│   │   └── robot_control.launch.py
│   └── config/
│       ├── parameters.yaml          # Waypoints, run ID, robot joint config
│       └── maps/                    # Saved Nav2 map(s)
├── robot_interfaces/                 # Shared ROS2 service definitions
│   └── srv/
│       ├── Camera.srv
│       └── TestPoses.srv
├── candidate_generation/             # Next-best-view candidate sampling
│   └── candidate_generation/
│       └── candidate_generator.py
├── gpu_main_loop.py                  # GPU-side pipeline orchestrator
├── gpu_candidate_puller.py
├── score_and_return_top_candidates.py
├── viewpoint_scoring.py
├── build_transforms_from_poses.py
├── ns_train_patched.py
├── check_convergence.py
├── robot_capture_next_view.py        # Runs on the robot
└── ...
```

## Citations
This project's Shannon-MI viewpoint scoring is a plain-PyTorch reimplementation based on the GauSS-MI approach.
> Y. Xie, Y. Cai, Y. Zhang, L. Yang, and J. Pan, "GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction," *arXiv preprint arXiv:2504.21067*, 2025.

<details>
<summary>BibTeX</summary>

```bibtex
@article{xie2025gaussmi,
  title={GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction},
  author={Xie, Yuhan and Cai, Yixi and Zhang, Yinqiang and Yang, Lei and Pan, Jia},
  journal={arXiv preprint arXiv:2504.21067},
  year={2025}
}
```
</details>

# Demo Videos

## Updated rendering of a high scoring pose
https://github.com/user-attachments/assets/c097206c-a71c-4437-8668-94fcb764495a

## Flythrough of the Initial versus Optimized Final Model
https://github.com/user-attachments/assets/4734ea2d-501a-4d96-9bd4-d07b42582b60



## Data
### PSNR (Peak Signal to Noise Ratio) for an optimized run versus a run that uses the same number of random images
<img width="1050" height="750" alt="held_out_view_quality" src="https://github.com/user-attachments/assets/43f5f3b0-30d4-4a24-b66c-631ccc834009" />

This graph shows the mean PSNR values for 3 poses not contained in the dataset during the run. The mean PSNR is consistently higher through the course of the run when the SMI scoring mechanism is used to determine the Next Best View versus random selection, lending credence to the scoring mechanism as a useful method for model optimization. 
## Limitations and Future Work
The initial model contains more prominent streaks and floater gaussian artifacts that it would using COLMAP(NerfStudio's default for building the transforms.json). This is due to the inherent uncertainty in the XY position of the robot when it captures the initial images for the model. To counteract this, the capture_frame() method used in the robot_control package checks the AMCL pose covariance and ensures that the uncertainty in XY position is below a threshold, but the uncertainty still causes small irregularities when creating the transforms.json directly from the camera poses. 

The Stretch 3 has two cameras(a fixed height, mounted d435 and a d405 mounted on the gripper arm). Due to time limitations, this system only uses the mounted d435 camera. The generated candidate poses are thus fixed to a certain height which limits the kinds of poses that are scored. It entirely possible that higher scoring poses exist in the initial model that are at different heights than the mounted camera can reach and these poses are ignored by the current system. Future work in this project would include allow for full 6 DOF pose freedom for generated candidate poses and enable the system to capture potentially higher scoring poses using the gripper mounted camera.

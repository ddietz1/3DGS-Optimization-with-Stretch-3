"""
Scores a candidates file (pulled by gpu_candidate_puller.py) against a
trained model via viewpoint_scoring.py, extracts the top N, converts them
back into ROS-native (map-frame position + quaternion) form, and pushes
the result up to the robot, the return half of the loop.

Usage:
  python3 score_and_return_top_candidates.py \
    --load-config outputs/exp_run1/model_b_direct/model_b/splatfacto/<ts>/config.yml \
    --candidates-file candidates_incoming/candidates_20260807_120000_000000.json \
    --top-n 10
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

ROBOT_HOST = "hello-robot@10.106.29.84"
ROBOT_SCORED_DIR = "~/stretch_user/scored_candidates"

ROS_TO_NERFSTUDIO_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def rotmat_to_quat(R: np.ndarray):
    """Shepperd's method."""
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


def nerfstudio_c2w_3x4_to_ros_pose(c2w_3x4: list) -> dict:
    c2w4 = np.eye(4)
    c2w4[:3, :4] = np.array(c2w_3x4, dtype=np.float64)
    c2w_ros = c2w4 @ ROS_TO_NERFSTUDIO_FLIP

    qx, qy, qz, qw = rotmat_to_quat(c2w_ros[:3, :3])
    px, py, pz = c2w_ros[:3, 3].tolist()

    return {
        "position": {"x": px, "y": py, "z": pz},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }


def run_scoring(load_config: str, candidates_file: str, out_dir: str) -> Path:
    cmd = ["python3", str(Path(__file__).parent / "viewpoint_scoring.py"),
           "--load-config", load_config,
           "--reuse-candidates", candidates_file,
           "--out-dir", out_dir,
           "--auto-confirm"]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"viewpoint_scoring.py failed (exit {result.returncode})")
    return Path(out_dir) / "candidate_scores.json"


def send_to_robot(payload: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    local_path = Path(f"scored_{stamp}.json")
    local_path.write_text(json.dumps(payload, indent=2))

    subprocess.run(["ssh", ROBOT_HOST, f"mkdir -p {ROBOT_SCORED_DIR}"])
    cmd = ["scp", str(local_path), f"{ROBOT_HOST}:{ROBOT_SCORED_DIR}/{local_path.name}"]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"scp to robot failed (exit {result.returncode})")
    return local_path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", required=True)
    ap.add_argument("--candidates-file", required=True,
                     help="A file pulled by gpu_candidate_puller.py, or "
                          "gpu_candidate_puller.latest_local_candidates_file()")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out-dir", default="scoring_live")
    args = ap.parse_args()

    scores_path = run_scoring(args.load_config, args.candidates_file, args.out_dir)
    scored = json.loads(scores_path.read_text())  # already sorted best-first

    top_n = scored[:args.top_n]
    print(f"\nTop {len(top_n)} candidates (of {len(scored)} scored):")
    for i, entry in enumerate(top_n):
        print(f"  #{i}: score={entry['score']:.4f}  pos={entry['position']}")

    payload = {
        "model_checkpoint": args.load_config,
        "candidates_source": args.candidates_file,
        "timestamp": datetime.now().isoformat(),
        "top_candidates": [
            {
                "rank": i,
                "score": entry["score"],
                **nerfstudio_c2w_3x4_to_ros_pose(entry["transform_matrix"]),
            }
            for i, entry in enumerate(top_n)
        ],
    }

    sent_name = send_to_robot(payload)
    print(f"\nSent top {len(top_n)} candidates to robot as {ROBOT_SCORED_DIR}/{sent_name}")


if __name__ == "__main__":
    main()
"""
GPU-side: continuously pulls newly-written candidate JSON files from the
robot's ~/stretch_user/candidates/ (written by CandidateGenerator, see
candidate_generator_json_patch.py) down to this workstation.

Same SSH/scp transport as gpu_transport_test.py's pull_captures(), but
tracks what's already been pulled (via a remote `ls` diff) rather than
re-copying everything every poll -- candidates arrive more frequently and
in smaller files than image captures, so avoiding redundant transfers
matters more here.

Runs standalone and continuously, independent of training/scoring status.
Per your pipeline description, candidates can arrive early (while the
robot is still traveling to initial waypoints) well before there's a
trained model to score them against -- this script's only job is getting
files from robot to GPU reliably; score_and_return_top_candidates.py
decides when scoring actually happens.

Usage (run continuously, e.g. under tmux/nohup/systemd):
  python3 gpu_candidate_puller.py --poll-interval 10

Usage (single poll, for testing):
  python3 gpu_candidate_puller.py --once
"""

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

ROBOT_HOST = "hello-robot@10.106.29.84"
ROBOT_CANDIDATES_DIR = "~/stretch_user/candidates"
LOCAL_CANDIDATES_DIR = Path("candidates_incoming")


def list_remote_candidates() -> list:
    cmd = ["ssh", ROBOT_HOST, f"ls {ROBOT_CANDIDATES_DIR} 2>/dev/null || true"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: could not list remote candidates dir: {result.stderr}")
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def pull_one(filename: str) -> bool:
    LOCAL_CANDIDATES_DIR.mkdir(exist_ok=True)
    cmd = ["scp", f"{ROBOT_HOST}:{ROBOT_CANDIDATES_DIR}/{filename}",
           str(LOCAL_CANDIDATES_DIR / filename)]
    result = subprocess.run(cmd)
    return result.returncode == 0


def latest_local_candidates_file() -> Optional[Path]:
    """Returns the most recently pulled candidates file -- what
    score_and_return_top_candidates.py scores against by default."""
    if not LOCAL_CANDIDATES_DIR.exists():
        return None
    files = sorted(LOCAL_CANDIDATES_DIR.glob("candidates_*.json"),
                    key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-interval", type=float, default=10.0)
    ap.add_argument("--once", action="store_true",
                     help="Poll and pull once, then exit (for testing/smoke-checks)")
    args = ap.parse_args()

    already_pulled = set(p.name for p in LOCAL_CANDIDATES_DIR.glob("*.json")) \
        if LOCAL_CANDIDATES_DIR.exists() else set()

    while True:
        remote_files = list_remote_candidates()
        new_files = [f for f in remote_files if f.endswith(".json") and f not in already_pulled]

        for f in new_files:
            print(f"[{datetime.now().isoformat()}] Pulling new candidate file: {f}")
            if pull_one(f):
                already_pulled.add(f)
                print(f"  -> saved to {LOCAL_CANDIDATES_DIR / f}")
            else:
                print(f"  -> WARNING: pull failed for {f}, will retry next poll")

        if not new_files:
            print(f"[{datetime.now().isoformat()}] No new candidate files.")

        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
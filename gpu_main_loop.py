"""
Single command to start the entire GPU-side system.

Starts gpu_candidate_puller.py running continuously in the background, then:
  PHASE 1 (bootstrap, once): wait for the robot's initial waypoint capture
    to finish, pull it, build the direct-pose dataset, train the initial
    model.
  PHASE 2 (loop, indefinite): each round -- score the latest pulled
    candidates against the current checkpoint, send the top N back to the
    robot, wait for the robot to send back a new capture, incorporate it,
    resume-train, check convergence, repeat.

Checkpointed like run_full_pipeline.py -- state.json tracks whether
bootstrap is done and which round you're on, so a crash mid-loop resumes
rather than restarting everything.

Usage:
  python3 gpu_main_loop.py \
    --robot-run-name holdout_run-08-10-1 \
    --base-dir outputs/live_run1 \
    --fx 304.02 --fy 302.69 --cx 212.16 --cy 123.31 --width 424 --height 240

Usage (dry run -- print planned commands without executing, same as
run_full_pipeline.py):
  python3 gpu_main_loop.py ... --dry-run

Usage (stop after N loop rounds, for a controlled smoke test rather than
letting it run to convergence -- same idea as the --stop-after ask from
earlier, now actually implemented):
  python3 gpu_main_loop.py ... --stop-after-round 1
"""

import argparse
import atexit
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_transforms_from_poses import add_frame_to_dataset  # noqa: E402

ROBOT_HOST = "hello-robot@10.106.29.84"


class State:
    def __init__(self, base_dir: Path):
        self.path = base_dir / "loop_state.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "bootstrap_done": False,
            "last_completed_round": -1,
            "seen_capture_files": [],
        }

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))


def run_cmd(cmd: list, log_path: Path, dry_run: bool) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    print(f"  (log -> {log_path})")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n\n=== {datetime.now().isoformat()} ===\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode


def find_latest_config(experiment_dir: Path, experiment_name: str) -> Path:
    """Same convention confirmed against real training output in
    run_full_pipeline.py: <experiment_dir>/<experiment_name>/splatfacto/<timestamp>/config.yml"""
    candidates = sorted(experiment_dir.glob(f"{experiment_name}/splatfacto/*/config.yml"),
                         key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No config.yml found under {experiment_dir}/{experiment_name}/splatfacto/*/")
    return candidates[-1]


def frozen_dataparser_flags():
    return ["nerfstudio-data", "--orientation-method", "none",
            "--center-method", "none", "--auto-scale-poses", "False"]


def start_candidate_puller(base_dir: Path, dry_run: bool):
    """Starts gpu_candidate_puller.py as a background process for the
    lifetime of this script -- registered to terminate on exit so it
    doesn't keep running orphaned after a crash or Ctrl+C."""
    if dry_run:
        print("[dry run] Would start gpu_candidate_puller.py in the background")
        return None

    log_path = base_dir / "logs" / "candidate_puller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    # .resolve() is required here, not optional -- Path(__file__).parent can
    # stay a RELATIVE path (e.g. '.') when this script is invoked as
    # `python3 gpu_main_loop.py`, and combining that with cwd=str(base_dir)
    # below breaks the lookup: the relative script path gets resolved
    # against the NEW cwd (base_dir), which doesn't contain
    # gpu_candidate_puller.py at all. Confirmed this was the actual cause
    # of the repeated "can't open file" errors in the log.
    puller_script = (Path(__file__).parent / "gpu_candidate_puller.py").resolve()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(puller_script), "--poll-interval", "10"],
        stdout=log_f, stderr=subprocess.STDOUT, cwd=str(base_dir),
    )
    print(f"Started gpu_candidate_puller.py in background (pid {proc.pid}, log -> {log_path})")

    def _cleanup():
        if proc.poll() is None:
            print(f"Terminating candidate puller (pid {proc.pid})...")
            proc.terminate()
    atexit.register(_cleanup)
    return proc


def check_robot_done_marker(robot_run_name: str) -> bool:
    remote_dir = f"~/stretch_user/captures/{robot_run_name}"
    cmd = ["ssh", ROBOT_HOST, f"test -f {remote_dir}/DONE && echo YES || echo NO"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and "YES" in result.stdout


def pull_initial_captures(robot_run_name: str, local_dir: Path, dry_run: bool):
    """Uses the same scp-glob pattern already proven working in
    gpu_transport_test.py's pull_captures() -- avoids requiring rsync,
    which conflicts with colmap's lz4-c dependency in this conda env (do
    not force that install; it risks silently breaking colmap).

    IMPORTANT: uses -r + the remote '*' glob rather than scp'ing the
    directory itself -- scp doesn't support rsync's trailing-slash
    "contents of" convention, so `scp -r host:dir/ local/` would nest an
    extra subdirectory level (host:dir/dir/*.png inside local/), breaking
    build_transforms_from_poses.py's non-recursive `captures_dir.glob(
    "*_map_pose.json")`. The '*' glob expands on the remote shell, so
    files land flat in local_dir as intended.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"~/stretch_user/captures/{robot_run_name}"
    cmd = ["scp", "-r", f"{ROBOT_HOST}:{remote_dir}/*", str(local_dir) + "/"]
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"scp of initial captures failed (exit {result.returncode})")


def list_remote_capture_basenames(robot_run_name: str) -> set:
    """Lists complete (rgb+depth+map_pose all present) capture basenames
    currently on the robot -- used to detect a NEW loop capture, the same
    diff-against-seen approach gpu_candidate_puller.py uses for candidates."""
    remote_dir = f"~/stretch_user/captures/{robot_run_name}"
    cmd = ["ssh", ROBOT_HOST, f"ls {remote_dir} 2>/dev/null || true"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return set()
    files = set(line.strip() for line in result.stdout.splitlines() if line.strip())
    basenames = set()
    for f in files:
        if f.endswith("_map_pose.json"):
            base = f[: -len("_map_pose.json")]
            if f"{base}_rgb.png" in files and f"{base}_depth.npy" in files:
                basenames.add(base)
    return basenames


def wait_for_new_captures(robot_run_name: str, seen: set, poll_interval: float,
                           dry_run: bool, settle_polls: int = 2) -> list:
    """Blocks until at least one new capture appears, then waits for the
    new-capture set to stop growing for `settle_polls` consecutive polls
    before returning ALL of them.

    Necessary now that a single NBV stop can produce multiple captures
    (a multi-shot pan/tilt burst), not just one -- the previous version
    returned the first new file it saw, which would silently strand the
    rest as unincorporated 'new' captures that then get wrongly picked up
    on a LATER round instead of the round they actually belong to.

    This is a debounce, not a guarantee -- if a burst is slower than
    settle_polls * poll_interval, this can still return early. Increase
    settle_polls (or poll_interval) if you see truncated batches."""
    if dry_run:
        print("[dry run] Would poll robot for new captures (skipping the actual wait)")
        return ["DRY_RUN_PLACEHOLDER"]

    print(f"Waiting for new captures on the robot (not in {len(seen)} already-seen)...")
    stable_count = 0
    last_new = set()
    while True:
        current = list_remote_capture_basenames(robot_run_name)
        new = current - seen
        if new and new == last_new:
            stable_count += 1
        else:
            stable_count = 0
        last_new = new
        if new and stable_count >= settle_polls:
            chosen = sorted(new)
            print(f"New captures detected (settled after {settle_polls} stable polls): {chosen}")
            return chosen
        time.sleep(poll_interval)


def pull_capture(robot_run_name: str, base_name: str, local_dir: Path, dry_run: bool):
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"~/stretch_user/captures/{robot_run_name}"
    if dry_run:
        print(f"[dry run] Would pull {base_name}_{{rgb.png,depth.npy,map_pose.json}}")
        return
    for suffix in ("_rgb.png", "_depth.npy", "_map_pose.json"):
        cmd = ["scp", f"{ROBOT_HOST}:{remote_dir}/{base_name}{suffix}",
               str(local_dir / f"{base_name}{suffix}")]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull {base_name}{suffix}")


def merge_candidate_files(candidates_dir: Path, out_path: Path) -> int:
    """Merges every candidates_*.json file pulled so far into one combined
    list -- CandidateGenerator only writes ~5 candidates per 30s cycle, so
    scoring only the latest file (the old behavior) throws away everything
    from earlier cycles while the robot is stationary. All of it stays
    spatially valid until the robot actually moves, so it's all worth
    scoring together. Must genuinely re-score the full merged set every
    round, even entries seen before -- the model itself changes between
    rounds (resume-trained), so old scores would be stale against the
    current checkpoint; there's no valid way to cache them."""
    merged = []
    files = sorted(candidates_dir.glob("candidates_*.json"))
    for f in files:
        try:
            merged.extend(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: skipping unreadable candidate file {f}: {e}")
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(files)} candidate files -> {len(merged)} total candidates -> {out_path}")
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-run-name", required=True,
                     help="Matches MoveJoints' 'run_num' ROS param -- determines "
                          "which ~/stretch_user/captures/<name>/ dir to watch/pull")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--total-budget", type=int, default=150000)
    ap.add_argument("--baseline-iterations", type=int, default=20000)
    ap.add_argument("--resume-iterations", type=int, default=10000)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--poll-interval", type=float, default=10.0)
    ap.add_argument("--pose-source", choices=["direct", "colmap"], default="direct",
                     help="direct: build_transforms_from_poses.py using robot AMCL poses "
                          "(existing, default behavior). colmap: ns-process-data + "
                          "colmap_map_align.py fit/bulk-apply instead. NOTE: only affects "
                          "the BOOTSTRAP dataset -- the loop's incorporation step "
                          "(add_frame_to_dataset) is direct-pose-only right now and will "
                          "raise a clear error if reached under --pose-source colmap, "
                          "rather than silently mis-incorporating a new capture into a "
                          "COLMAP-frame dataset. Use --stop-after-round 0 with this until "
                          "COLMAP-compatible incorporation is wired in.")
    ap.add_argument("--stop-after-round", type=int, default=None,
                     help="Exit cleanly after this many loop rounds, for a "
                          "controlled smoke test instead of running indefinitely")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    state = State(base_dir)
    # Deterministic from pose_source alone -- must be correct even when the
    # bootstrap block below is SKIPPED on a resumed run (bootstrap_done
    # already True), not just when bootstrap actually executes this time.
    dataset_dir = (base_dir / "dataset" if args.pose_source == "direct"
                   else base_dir / "colmap_raw" / "transforms_map_frame.json")
    experiment_name = "live_model"

    start_candidate_puller(base_dir, args.dry_run)

    # --- PHASE 1: bootstrap (once) ---
    if not state.data["bootstrap_done"]:
        print("=== Bootstrap: waiting for robot's initial capture to finish ===")
        while not args.dry_run and not check_robot_done_marker(args.robot_run_name):
            print(f"  DONE marker not present yet, polling again in {args.poll_interval}s...")
            time.sleep(args.poll_interval)
        if args.dry_run:
            print("[dry run] Would poll for DONE marker, then proceed")

        initial_captures_dir = base_dir / "captures_initial"
        pull_initial_captures(args.robot_run_name, initial_captures_dir, args.dry_run)

        if args.pose_source == "direct":
            rc = run_cmd(
                [sys.executable, str(Path(__file__).parent / "build_transforms_from_poses.py"),
                 "--captures-dir", str(initial_captures_dir), "--out-dir", str(dataset_dir),
                 "--fx", str(args.fx), "--fy", str(args.fy), "--cx", str(args.cx),
                 "--cy", str(args.cy), "--width", str(args.width), "--height", str(args.height),
                 "--build-pointcloud"],
                base_dir / "logs" / "build_dataset.log", args.dry_run,
            )
            if rc != 0:
                raise RuntimeError("build_transforms_from_poses.py failed")
        else:  # colmap
            colmap_raw_dir = base_dir / "colmap_raw"
            rc = run_cmd(
                ["ns-process-data", "images", "--data", str(initial_captures_dir),
                 "--output-dir", str(colmap_raw_dir), "--no-gpu"],
                base_dir / "logs" / "colmap_process.log", args.dry_run,
            )
            if rc != 0:
                raise RuntimeError("ns-process-data failed")

            alignment_path = base_dir / "alignment.json"
            rc = run_cmd(
                [sys.executable, str(Path(__file__).parent / "colmap_map_align.py"), "fit",
                 "--colmap-transforms", str(colmap_raw_dir / "transforms.json"),
                 "--map-pose-dir", str(initial_captures_dir), "--out", str(alignment_path)],
                base_dir / "logs" / "colmap_align_fit.log", args.dry_run,
            )
            if rc != 0:
                raise RuntimeError("colmap_map_align.py fit failed")

            map_frame_transforms = colmap_raw_dir / "transforms_map_frame.json"
            rc = run_cmd(
                [sys.executable, str(Path(__file__).parent / "colmap_map_align.py"), "bulk-apply",
                 "--colmap-transforms", str(colmap_raw_dir / "transforms.json"),
                 "--alignment", str(alignment_path), "--out", str(map_frame_transforms)],
                base_dir / "logs" / "colmap_align_bulk_apply.log", args.dry_run,
            )
            if rc != 0:
                raise RuntimeError("colmap_map_align.py bulk-apply failed")

        rc = run_cmd(
            [sys.executable, str(Path(__file__).parent / "ns_train_patched.py"), "splatfacto",
             "--data", str(dataset_dir), "--experiment-name", experiment_name,
             "--output-dir", str(base_dir), "--max-num-iterations", str(args.baseline_iterations),
             "--pipeline.model.stop-split-at", str(args.total_budget),
             "--optimizers.means.scheduler.max-steps", str(args.total_budget),
             "--vis", "viewer", "--viewer.quit-on-train-completion", "True"] + frozen_dataparser_flags(),
            base_dir / "logs" / "train_initial.log", args.dry_run,
        )
        if rc != 0:
            raise RuntimeError("Initial training failed")

        if not args.dry_run:
            state.data["bootstrap_done"] = True
            state.data["seen_capture_files"] = list(
                p.name.replace("_map_pose.json", "")
                for p in initial_captures_dir.glob("*_map_pose.json")
            )
            state.save()
        print("=== Bootstrap complete ===\n")

    # --- PHASE 2: continuous loop ---
    round_i = state.data["last_completed_round"] + 1
    while True:
        if args.stop_after_round is not None and round_i > args.stop_after_round:
            print(f"Reached --stop-after-round {args.stop_after_round}, exiting cleanly.")
            break

        print(f"\n=== Round {round_i} ===")
        round_dir = base_dir / f"round_{round_i:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            ckpt_config = Path("DRY_RUN_PLACEHOLDER/config.yml")
        else:
            ckpt_config = find_latest_config(base_dir, experiment_name)

        candidates_dir = base_dir / "candidates_incoming"
        if args.dry_run:
            candidates_file = Path("DRY_RUN_PLACEHOLDER/candidates.json")
        else:
            candidate_files = sorted(candidates_dir.glob("candidates_*.json")) \
                if candidates_dir.exists() else []
            if not candidate_files:
                print(f"No candidate files pulled yet -- waiting {args.poll_interval}s...")
                time.sleep(args.poll_interval)
                continue
            candidates_file = round_dir / "merged_candidates.json"
            merge_candidate_files(candidates_dir, candidates_file)

        rc = run_cmd(
            [sys.executable, str(Path(__file__).parent / "score_and_return_top_candidates.py"),
             "--load-config", str(ckpt_config), "--candidates-file", str(candidates_file),
             "--top-n", str(args.top_n), "--out-dir", str(round_dir / "scoring")],
            round_dir / "score_and_return.log", args.dry_run,
        )
        if rc != 0:
            raise RuntimeError(f"score_and_return_top_candidates.py failed in round {round_i}")

        scores_path = round_dir / "scoring" / "candidate_scores.json"
        history_path = base_dir / "convergence_history.json"
        rc = run_cmd(
            [sys.executable, str(Path(__file__).parent / "check_convergence.py"),
             "--candidate-scores", str(scores_path), "--history-file", str(history_path)],
            round_dir / "convergence.log", args.dry_run,
        )
        if rc == 1:
            print(f"\n=== CONVERGED at round {round_i} -- stopping loop ===")
            break
        elif rc not in (0, 1) and not args.dry_run:
            raise RuntimeError(f"check_convergence.py errored unexpectedly (exit {rc})")

        seen = set(state.data["seen_capture_files"])
        new_capture_bases = wait_for_new_captures(args.robot_run_name, seen, args.poll_interval, args.dry_run)

        loop_captures_dir = base_dir / "captures_loop"
        for base_name in new_capture_bases:
            pull_capture(args.robot_run_name, base_name, loop_captures_dir, args.dry_run)

        if not args.dry_run:
            if args.pose_source == "colmap":
                raise RuntimeError(
                    "Reached loop incorporation under --pose-source colmap, which "
                    "isn't supported yet -- add_frame_to_dataset only knows how to "
                    "append a direct-pose frame, and doing that into a COLMAP-frame "
                    "dataset would silently mix conventions. Use --stop-after-round 0 "
                    "with --pose-source colmap until COLMAP-compatible incorporation "
                    "(full incremental registration, or alignment-transform application "
                    "to the raw capture pose) is built."
                )
            for base_name in new_capture_bases:
                add_frame_to_dataset(
                    dataset_dir=str(dataset_dir),
                    map_pose_path=str(loop_captures_dir / f"{base_name}_map_pose.json"),
                )

        rc = run_cmd(
            [sys.executable, str(Path(__file__).parent / "ns_train_patched.py"), "splatfacto",
             "--data", str(dataset_dir), "--experiment-name", experiment_name,
             "--output-dir", str(base_dir),
             "--load-dir", str(ckpt_config.parent / "nerfstudio_models"),
             "--max-num-iterations", str(args.resume_iterations),
             "--pipeline.model.stop-split-at", str(args.total_budget),
             "--optimizers.means.scheduler.max-steps", str(args.total_budget),
             "--vis", "viewer", "--viewer.quit-on-train-completion", "True"] + frozen_dataparser_flags(),
            round_dir / "resume_train.log", args.dry_run,
        )
        if rc != 0:
            raise RuntimeError(f"Resume training failed in round {round_i}")

        if not args.dry_run:
            state.data["seen_capture_files"].extend(new_capture_bases)
            state.data["last_completed_round"] = round_i
            state.save()

        round_i += 1


if __name__ == "__main__":
    main()
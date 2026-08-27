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

Usage (stop after N loop rounds, for a controlled smoke test rather than
letting it run to convergence -- same idea as the --stop-after ask from
earlier, now actually implemented):
  python3 gpu_main_loop.py ... --stop-after-round 1

Usage (hold out 3 bootstrap poses from training entirely, and get an
automatic PSNR-vs-round generalization check for them, logged to
holdout_poses_history.json every round -- see select_holdout_poses()):
  python3 gpu_main_loop.py ... --holdout-poses 3
"""

import argparse
import atexit
import json
import math
import re
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
            "points_at_current_checkpoint": None,
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


def get_checkpoint_step(ckpt_dir: Path) -> int:
    """Parses the current cumulative step count from the latest checkpoint
    filename (e.g. step-000032999.ckpt -> 32999) -- needed to compute a
    densification ceiling that's always ahead of where training actually
    is, rather than a fixed number that silently caps out once enough
    rounds accumulate past it."""
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    latest = ckpts[-1].stem  # e.g. "step-000032999"
    return int(latest.split("-")[-1])


def count_ply_points(ply_path: Path) -> int:
    """Total point count currently in the dataset's point cloud, used as
    a baseline to determine which points are genuinely NEW since a given
    checkpoint was trained."""
    if not ply_path.exists():
        return 0
    from plyfile import PlyData
    return len(PlyData.read(str(ply_path))['vertex'].data)


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
    gpu_transport_test.py's pull_captures() -- avoids requiring rsync.
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
    currently on the robot -- used to detect a NEW loop capture."""
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
    """
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


def load_captured_positions(base_dir: Path, capture_basenames: list) -> list:
    """Reads the real, achieved camera position (map frame, from
    motion_control.py's capture_frame()"""
    positions = []
    for base_name in capture_basenames:
        for subdir in ("captures_initial", "captures_loop"):
            pose_path = base_dir / subdir / f"{base_name}_map_pose.json"
            if pose_path.exists():
                positions.append(json.loads(pose_path.read_text())["position"])
                break
    return positions


def merge_candidate_files(candidates_dir: Path, out_path: Path,
                           captured_positions: list = None,
                           exclusion_radius: float = 0.15) -> int:
    """Merges every candidates_*.json file pulled so far into one combined
    list.

    EXCLUDES candidates within exclusion_radius of any already-captured real
    camera position. Without this, a candidate near a region that's already
    been photographed stays in the pool and can keep re-winning if its
    scored uncertainty hasn't fully resolved yet.
    """
    merged = []
    files = sorted(candidates_dir.glob("candidates_*.json"))
    for f in files:
        try:
            merged.extend(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: skipping unreadable candidate file {f}: {e}")

    n_excluded = 0
    if captured_positions:
        kept = []
        for c in merged:
            if any(math.dist(c["position"], cp) <= exclusion_radius
                   for cp in captured_positions):
                n_excluded += 1
            else:
                kept.append(c)
        merged = kept

    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(files)} candidate files -> {len(merged)} candidates "
          f"({n_excluded} excluded as already-captured within {exclusion_radius}m) -> {out_path}")
    return len(merged)


def select_holdout_poses(captures_dir: Path, n: int) -> list:
    """Picks n bootstrap-capture basenames spread across n DISTINCT
    waypoints, not just any n captures -- each waypoint is a genuinely
    different robot base position, while the ~21 poses captured at a given
    waypoint are just pan/tilt variations from the same spot, so picking
    naively (e.g. the first n files alphabetically) could easily hold out
    poses that are all near-duplicates of each other rather than a spatially
    useful validation set.

    Deterministic (evenly-spaced waypoint indices, first pose by filename
    at each chosen waypoint) rather than random, so a resumed bootstrap
    (same captures_initial contents) reselects the identical set without
    needing to persist a random seed."""
    if n <= 0:
        return []

    by_waypoint = {}
    for p in sorted(captures_dir.glob("*_map_pose.json")):
        base = p.name.replace("_map_pose.json", "")
        m = re.match(r"^wp(\d+)_", base)
        if m:
            by_waypoint.setdefault(int(m.group(1)), []).append(base)

    waypoints = sorted(by_waypoint)
    if not waypoints:
        print("[holdout] WARNING: no 'wp<N>_...'-prefixed captures found in "
              f"{captures_dir} -- can't select a spatially-spread holdout "
              "set, disabling holdout for this run.")
        return []
    if n > len(waypoints):
        print(f"[holdout] WARNING: requested {n} holdout poses but only "
              f"{len(waypoints)} distinct waypoints exist -- capping to {len(waypoints)}.")
        n = len(waypoints)

    if n == 1:
        indices = [len(waypoints) // 2]
    else:
        indices = sorted(set(round(i * (len(waypoints) - 1) / (n - 1)) for i in range(n)))

    return [sorted(by_waypoint[waypoints[idx]])[0] for idx in indices]


def render_holdout_psnr(holdout_poses: list, base_dir: Path, dataset_dir: Path,
                         ckpt_config: Path, tag: str) -> dict:
    """Renders every held-out pose against ckpt_config and returns
    {basename: psnr}. Reuses render_pose.py exactly as used for the manual
    before/after validation this replaces -- --psnr-out-json gives back a
    structured result instead of scraping it out of stdout. Ground-truth
    images/poses are read from captures_initial, since holdout poses are
    always drawn from the bootstrap set."""
    if not holdout_poses:
        return {}

    results = {}
    out_dir = base_dir / "holdout_renders" / tag
    intrinsics_json = dataset_dir / "transforms.json"

    for base_name in holdout_poses:
        pose_json = base_dir / "captures_initial" / f"{base_name}_map_pose.json"
        ref_image = base_dir / "captures_initial" / f"{base_name}_rgb.png"
        psnr_json = out_dir / f"{base_name}.json"
        cmd = [sys.executable, str(Path(__file__).parent / "render_pose.py"),
               "--load-config", str(ckpt_config), "--pose-json", str(pose_json),
               "--intrinsics-json", str(intrinsics_json),
               "--output", str(out_dir / f"{base_name}.png"),
               "--reference-image", str(ref_image),
               "--psnr-out-json", str(psnr_json)]
        rc = run_cmd(cmd, base_dir / "logs" / "holdout_render.log", dry_run=False)
        if rc != 0:
            print(f"[holdout] WARNING: render failed for {base_name} against "
                  f"'{tag}' -- skipping (see logs/holdout_render.log)")
            continue
        try:
            results[base_name] = json.loads(psnr_json.read_text())["psnr"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"[holdout] WARNING: couldn't read PSNR result for {base_name}: {e}")
    return results


def log_holdout_history(base_dir: Path, round_label, per_pose_psnr: dict):
    """Appends one entry to holdout_poses_history.json, keyed by
    round_label ('bootstrap' or an int round number). If an entry for this
    round_label already exists -- a crash-resumed round re-running from
    scratch -- REPLACES it rather than appending a duplicate, the same
    staleness class of bug found in convergence_history.json."""
    if not per_pose_psnr:
        return
    history_path = base_dir / "holdout_poses_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [e for e in history if e["round"] != round_label]
    mean_psnr = sum(per_pose_psnr.values()) / len(per_pose_psnr)
    history.append({
        "round": round_label,
        "timestamp": datetime.now().isoformat(),
        "per_pose_psnr": per_pose_psnr,
        "mean_psnr": mean_psnr,
    })
    history.sort(key=lambda e: -1 if e["round"] == "bootstrap" else e["round"])
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[holdout] round={round_label}: mean PSNR={mean_psnr:.2f} dB "
          f"across {len(per_pose_psnr)} held-out pose(s) -> {history_path}")


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
    ap.add_argument("--budget-margin", type=int, default=10000,
                     help="Densification/scheduler ceiling is now computed dynamically "
                          "each round as current_step + this_call's_iterations + margin, "
                          "rather than a fixed --total-budget -- a fixed ceiling, however "
                          "large, silently caps densification once enough rounds "
                          "accumulate past it (confirmed as the cause of a real render "
                          "regression: default budget=50000 capped exactly at round 4 "
                          "given baseline=40000, resume=3000/round). This margin is the "
                          "safety buffer beyond exactly matching, per training call.")
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
    ap.add_argument("--full-retrain-every", type=int, default=0,
                     help="Every Nth round, train FRESH from scratch on the full "
                          "accumulated dataset instead of resuming from the "
                          "checkpoint. This is the only way accumulated "
                          "depth-seeded points in sparse_pc.ply actually reach "
                          "a live model -- confirmed via nerfstudio source that "
                          "populate_modules() correctly reads the current point "
                          "cloud on a fresh start, but --load-dir resume's "
                          "_load_checkpoint() runs afterward and unconditionally "
                          "overwrites gauss_params with the checkpoint's own "
                          "saved (older, smaller) Gaussian set, discarding it. "
                          "0 disables this (every round resumes, matching prior "
                          "behavior). Round 0 is never a full retrain regardless "
                          "of this value -- redundant with bootstrap, which is "
                          "already a fresh train.")
    ap.add_argument("--full-retrain-iterations", type=int, default=None,
                     help="Iterations for a full-retrain round. Defaults to "
                          "--baseline-iterations if not set, matching the "
                          "budget the original bootstrap used -- a fresh "
                          "train from scratch needs comparable budget to "
                          "reach good quality, not the much shorter "
                          "--resume-iterations used for incremental rounds.")
    ap.add_argument("--stop-after-round", type=int, default=None,
                     help="Exit cleanly after this many loop rounds, for a "
                          "controlled smoke test instead of running indefinitely")
    ap.add_argument("--capture-exclusion-radius", type=float, default=0.15,
                     help="Candidates within this many meters of an already-captured "
                          "real camera position are dropped from the scoring pool each "
                          "round (see merge_candidate_files()). STARTING GUESS, not "
                          "measured: derived from the 0.05m base-position-convergence "
                          "threshold added in motion_control.py's test_d435_ik, plus "
                          "rough slack for residual yaw error and standoff/pan-tilt "
                          "geometry -- re-tune from real candidate-vs-achieved-capture-"
                          "position gaps once the position fix has been run for real.")
    ap.add_argument("--holdout-poses", type=int, default=0,
                     help="Hold out this many bootstrap captures (spread "
                          "across distinct waypoints) from training entirely "
                          "-- never added to the dataset, ever. Every round "
                          "(plus once right after bootstrap), each held-out "
                          "pose is rendered against the current checkpoint "
                          "and scored by PSNR against its real photo, logged "
                          "to holdout_poses_history.json. This is a standing "
                          "generalization check: does reconstruction quality "
                          "improve somewhere OTHER than whatever pose was "
                          "just captured and trained on this round -- as "
                          "opposed to convergence_history.json's score, which "
                          "only reflects the just-visited pose. direct-pose-"
                          "source only. 0 disables (default).")
    ap.add_argument("--random-baseline", action="store_true",
                     help="Ablation mode: passed straight through to "
                          "score_and_return_top_candidates.py -- send a random "
                          "--top-n candidates to the robot each round instead "
                          "of the --top-n highest-scoring ones. Scoring itself "
                          "is unaffected, only which candidates get SENT.")
    ap.add_argument("--random-seed", type=int, default=None,
                     help="Optional seed for --random-baseline's sampling, "
                          "passed straight through. Unseeded (default) draws "
                          "fresh each round.")
    ap.add_argument("--skip-convergence-check", action="store_true",
                     help="check_convergence.py still runs and still logs to "
                          "convergence_history.json every round, but a CONVERGED "
                          "result no longer stops the loop. For runs (e.g. "
                          "--random-baseline ablations) where you want a fixed "
                          "round count to match another run, rather than "
                          "whatever round the top-candidate-score plateau check "
                          "happens to fire on.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    state = State(base_dir)
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

        holdout_poses = []
        if args.holdout_poses > 0:
            if args.pose_source != "direct":
                print("[holdout] WARNING: --holdout-poses only supports "
                      "--pose-source direct right now -- disabling holdout "
                      "for this run.")
            elif args.dry_run:
                print(f"[dry run] Would select {args.holdout_poses} holdout pose(s)")
            else:
                holdout_poses = select_holdout_poses(initial_captures_dir, args.holdout_poses)
                (base_dir / "holdout_poses.json").write_text(json.dumps(holdout_poses, indent=2))
                print(f"[holdout] Holding out {len(holdout_poses)} pose(s), "
                      f"never added to training: {holdout_poses}")

        if args.pose_source == "direct":
            build_cmd = [sys.executable, str(Path(__file__).parent / "build_transforms_from_poses.py"),
                 "--captures-dir", str(initial_captures_dir), "--out-dir", str(dataset_dir),
                 "--fx", str(args.fx), "--fy", str(args.fy), "--cx", str(args.cx),
                 "--cy", str(args.cy), "--width", str(args.width), "--height", str(args.height),
                 "--build-pointcloud"]
            if holdout_poses:
                build_cmd += ["--exclude-basenames-file", str(base_dir / "holdout_poses.json")]
            rc = run_cmd(
                build_cmd,
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

        # Starting from step 0, so this round's ceiling is simply this
        # call's own iteration count plus margin
        bootstrap_budget = args.baseline_iterations + args.budget_margin
        rc = run_cmd(
            [sys.executable, str(Path(__file__).parent / "ns_train_patched.py"), "splatfacto",
             "--data", str(dataset_dir), "--experiment-name", experiment_name,
             "--output-dir", str(base_dir), "--max-num-iterations", str(args.baseline_iterations),
             "--pipeline.model.stop-split-at", str(bootstrap_budget),
             "--optimizers.means.scheduler.max-steps", str(bootstrap_budget),
             "--vis", "viewer", "--viewer.quit-on-train-completion", "True"] + frozen_dataparser_flags(),
            base_dir / "logs" / "train_initial.log", args.dry_run,
        )
        if rc != 0:
            raise RuntimeError("Initial training failed")

        if not args.dry_run and holdout_poses:
            bootstrap_config = find_latest_config(base_dir, experiment_name)
            per_pose = render_holdout_psnr(holdout_poses, base_dir, dataset_dir,
                                            bootstrap_config, tag="bootstrap")
            log_holdout_history(base_dir, "bootstrap", per_pose)

        if not args.dry_run:
            state.data["bootstrap_done"] = True
            state.data["seen_capture_files"] = list(
                p.name.replace("_map_pose.json", "")
                for p in initial_captures_dir.glob("*_map_pose.json")
            )
            if args.pose_source == "direct":
                state.data["points_at_current_checkpoint"] = count_ply_points(dataset_dir / "sparse_pc.ply")
            state.save()
        print("=== Bootstrap complete ===\n")

    # --- PHASE 2: continuous loop ---
    # Reload rather than reuse the bootstrap block's local variable
    holdout_poses_file = base_dir / "holdout_poses.json"
    holdout_poses = json.loads(holdout_poses_file.read_text()) if holdout_poses_file.exists() else []

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
            captured_positions = load_captured_positions(base_dir, state.data["seen_capture_files"])
            merge_candidate_files(candidates_dir, candidates_file,
                                   captured_positions=captured_positions,
                                   exclusion_radius=args.capture_exclusion_radius)

        score_cmd = [sys.executable, str(Path(__file__).parent / "score_and_return_top_candidates.py"),
             "--load-config", str(ckpt_config), "--candidates-file", str(candidates_file),
             "--top-n", str(args.top_n), "--out-dir", str(round_dir / "scoring")]
        if args.random_baseline:
            score_cmd += ["--random-baseline"]
            if args.random_seed is not None:
                score_cmd += ["--random-seed", str(args.random_seed)]
        rc = run_cmd(
            score_cmd,
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
            if args.skip_convergence_check:
                print(f"\n=== Round {round_i}: check_convergence.py says CONVERGED, but "
                      f"--skip-convergence-check is set -- continuing anyway (still logged "
                      f"to convergence_history.json) ===")
            else:
                print(f"\n=== CONVERGED at round {round_i} -- stopping loop ===")
                break
        elif rc not in (0, 1) and not args.dry_run:
            raise RuntimeError(f"check_convergence.py errored unexpectedly (exit {rc})")

        seen = set(state.data["seen_capture_files"])
        new_capture_bases = wait_for_new_captures(args.robot_run_name, seen, args.poll_interval, args.dry_run)

        if not args.dry_run:
            # Records exactly which captures belong to THIS round, plus which
            # checkpoint was current before this round's training
            (round_dir / "incorporated_captures.json").write_text(json.dumps({
                "round": round_i,
                "capture_basenames": new_capture_bases,
                "pre_round_checkpoint_config": str(ckpt_config),
            }, indent=2))

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

        is_full_retrain = (args.full_retrain_every > 0 and round_i > 0
                           and round_i % args.full_retrain_every == 0)

        if args.dry_run:
            round_budget = "DRY_RUN_PLACEHOLDER"
            train_iterations = "DRY_RUN_PLACEHOLDER"
        elif is_full_retrain:
            train_iterations = args.full_retrain_iterations or args.baseline_iterations
            # Fresh start, step 0 -- same principle as bootstrap's own budget,
            # NOT get_checkpoint_step() against the old checkpoint (that
            # checkpoint's step count is irrelevant to a run starting over).
            round_budget = train_iterations + args.budget_margin
            print(f"Round {round_i}: FULL RETRAIN from scratch on the full "
                  f"accumulated dataset (this is the round that actually picks "
                  f"up accumulated depth-seeded points), {train_iterations} "
                  f"iterations, densification ceiling={round_budget}")
        else:
            train_iterations = args.resume_iterations
            current_step = get_checkpoint_step(ckpt_config.parent / "nerfstudio_models")
            round_budget = current_step + args.resume_iterations + args.budget_margin
            print(f"Round {round_i}: incremental resume, current_step={current_step}, "
                  f"densification ceiling this round={round_budget}")

        train_cmd = [sys.executable, str(Path(__file__).parent / "ns_train_patched.py"), "splatfacto",
                     "--data", str(dataset_dir), "--experiment-name", experiment_name,
                     "--output-dir", str(base_dir),
                     "--max-num-iterations", str(train_iterations),
                     "--pipeline.model.stop-split-at", str(round_budget),
                     "--optimizers.means.scheduler.max-steps", str(round_budget),
                     "--vis", "viewer", "--viewer.quit-on-train-completion", "True"]
        if not is_full_retrain:
            train_cmd += ["--load-dir", str(ckpt_config.parent / "nerfstudio_models")]
            if args.pose_source == "direct" and not args.dry_run:
                baseline = state.data.get("points_at_current_checkpoint")
                if baseline is not None:
                    train_cmd += ["--injected-points-baseline", str(baseline)]
        train_cmd += frozen_dataparser_flags()

        rc = run_cmd(
            train_cmd,
            round_dir / "resume_train.log", args.dry_run,
        )
        if rc != 0:
            raise RuntimeError(f"Resume training failed in round {round_i}")

        if not args.dry_run and holdout_poses:
            round_config = find_latest_config(base_dir, experiment_name)
            per_pose = render_holdout_psnr(holdout_poses, base_dir, dataset_dir,
                                            round_config, tag=f"round_{round_i:04d}")
            log_holdout_history(base_dir, round_i, per_pose)

        if not args.dry_run:
            state.data["seen_capture_files"].extend(new_capture_bases)
            state.data["last_completed_round"] = round_i
            if args.pose_source == "direct":
                # Whatever training path was used, the resulting checkpoint
                # now reflects points up through the CURRENT total -- either
                # because a full retrain read them all fresh, or because
                # resume just injected everything past the old baseline.
                state.data["points_at_current_checkpoint"] = count_ply_points(dataset_dir / "sparse_pc.ply")
            state.save()

        round_i += 1


if __name__ == "__main__":
    main()
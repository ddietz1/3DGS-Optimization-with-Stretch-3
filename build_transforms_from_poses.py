"""
Builds a nerfstudio-ready dataset directly from map_pose.json + rgb + depth
captures, bypassing COLMAP entirely.

REQUIRES known camera intrinsics -- grab once via:
  ros2 topic echo /camera/color/camera_info --once

Usage:
  python3 build_transforms_from_poses.py \
    --captures-dir ~/Final_Project/captures/all_combined/ \
    --out-dir ~/Final_Project/direct_dataset/ \
    --fx 304.02 --fy 302.69 --cx 212.16 --cy 123.31 --width 424 --height 240 \
    --build-pointcloud
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from viewpoint_scoring import load_raw_depth  # reuse the already-validated mm/m auto-detection


def quat_to_rotmat(qx, qy, qz, qw):
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


ROS_TO_NERFSTUDIO_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def map_pose_to_nerfstudio_c2w(map_pose_path: Path) -> np.ndarray:
    with open(map_pose_path) as f:
        record = json.load(f)
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rotmat(*record["orientation"])
    c2w[:3, 3] = record["position"]
    return c2w @ ROS_TO_NERFSTUDIO_FLIP


def unproject_depth_to_points(depth_m: np.ndarray, rgb: np.ndarray, c2w: np.ndarray,
                               fx, fy, cx, cy, stride=8, max_depth=6.0):
    """Unprojects a depth map into world-frame 3D points, colored from rgb."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth_m[ys, xs]
    valid = (zs > 0) & (zs < max_depth)

    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    x_cam = (xs - cx) / fx * zs
    y_cam = (ys - cy) / fy * zs
    z_cam = zs
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(zs)], axis=1)

    # Unprojected points are in the ORIGINAL (pre-flip) optical convention,
    # so undo the flip on c2w for this specific step.
    c2w_preflip = c2w @ np.linalg.inv(ROS_TO_NERFSTUDIO_FLIP)
    points_world = (c2w_preflip @ points_cam.T).T[:, :3]

    colors = rgb[ys, xs]
    return points_world, colors


def append_points_to_ply(ply_path: Path, new_points: np.ndarray, new_colors: np.ndarray) -> int:
    """Appends points to the dataset's point cloud, preserving the ASCII
    PLY format the bootstrap itself writes below.
    
    """
    from plyfile import PlyData, PlyElement
    import os

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertices = np.empty(len(new_points), dtype=dtype)
    new_vertices['x'], new_vertices['y'], new_vertices['z'] = new_points.T
    new_vertices['red'], new_vertices['green'], new_vertices['blue'] = new_colors.T

    if ply_path.exists():
        existing = PlyData.read(str(ply_path))['vertex'].data
        combined = np.concatenate([existing, new_vertices])
    else:
        combined = new_vertices

    tmp_path = ply_path.with_suffix(ply_path.suffix + ".tmp")
    PlyData([PlyElement.describe(combined, 'vertex')], text=True).write(str(tmp_path))
    os.replace(str(tmp_path), str(ply_path))  # atomic on the same filesystem
    return len(combined)


def add_frame_to_dataset(dataset_dir: Path, map_pose_path: Path, session_label: str = None):
    """Adds one new capture to an existing direct-poses dataset, safely.

    """
    dataset_dir = Path(dataset_dir)
    map_pose_path = Path(map_pose_path)
    base = map_pose_path.name.replace("_map_pose.json", "")
    rgb_path = map_pose_path.with_name(f"{base}_rgb.png")

    if session_label is None:
        session_label = map_pose_path.parent.name

    safe_name = f"{session_label}_{rgb_path.name}"
    dest_path = dataset_dir / "images" / safe_name
    if dest_path.exists():
        raise FileExistsError(
            f"{dest_path} already exists -- refusing to silently overwrite. "
            f"This usually means the same capture was added twice, or the "
            f"session_label collided with an existing one."
        )

    with open(dataset_dir / "transforms.json") as f:
        meta = json.load(f)

    existing_paths = {fr["file_path"] for fr in meta["frames"]}
    target_file_path = f"images/{safe_name}"
    if target_file_path in existing_paths:
        raise FileExistsError(f"{target_file_path} is already a frame in transforms.json")

    c2w = map_pose_to_nerfstudio_c2w(map_pose_path)
    shutil.copy(rgb_path, dest_path)
    meta["frames"].append({"file_path": target_file_path, "transform_matrix": c2w.tolist()})

    depth_path = map_pose_path.with_name(f"{base}_depth.npy")
    if depth_path.exists():
        from PIL import Image
        depth_m = load_raw_depth(depth_path, target_hw=(meta["h"], meta["w"]))
        rgb_img = np.array(Image.open(rgb_path).convert("RGB").resize((meta["w"], meta["h"])))
        points_world, colors = unproject_depth_to_points(
            depth_m, rgb_img, c2w,
            meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"],
        )
        if len(points_world) > 0:
            ply_path = dataset_dir / "sparse_pc.ply"
            total = append_points_to_ply(ply_path, points_world, colors)
            meta["ply_file_path"] = "sparse_pc.ply"  # set even if this dataset had none before
            print(f"Added {len(points_world)} depth-seeded points ({total} total in point cloud)")
    else:
        print(f"WARNING: no depth file at {depth_path} -- incorporating "
              f"{target_file_path} WITHOUT point-cloud seeding")

    with open(dataset_dir / "transforms.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Added {target_file_path} to {dataset_dir / 'transforms.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--build-pointcloud", action="store_true")
    ap.add_argument("--pointcloud-stride", type=int, default=8)
    ap.add_argument("--exclude-basenames-file", type=Path, default=None,
                     help="Path to a JSON file containing a list of capture "
                          "basenames (e.g. 'wp3_d435_pose1_d435') to skip "
                          "entirely -- excluded from BOTH the transforms.json "
                          "frame list and point-cloud seeding, since a frame "
                          "excluded only from the former still leaks its "
                          "geometry into the model via the latter. Used to "
                          "carve out a genuinely held-out validation set at "
                          "bootstrap time (see gpu_main_loop.py's "
                          "--holdout-poses).")
    args = ap.parse_args()

    captures_dir = Path(args.captures_dir)
    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    excluded_basenames = set()
    if args.exclude_basenames_file is not None:
        excluded_basenames = set(json.loads(args.exclude_basenames_file.read_text()))
        print(f"Excluding {len(excluded_basenames)} basename(s) from "
              f"{args.exclude_basenames_file}: {sorted(excluded_basenames)}")

    map_pose_files = sorted(captures_dir.glob("*_map_pose.json"))
    print(f"Found {len(map_pose_files)} map_pose.json files")

    frames = []
    all_points, all_colors = [], []
    n_excluded = 0

    for map_pose_path in map_pose_files:
        base = map_pose_path.name.replace("_map_pose.json", "")
        if base in excluded_basenames:
            n_excluded += 1
            continue
        rgb_path = captures_dir / f"{base}_rgb.png"
        depth_path = captures_dir / f"{base}_depth.npy"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"  WARNING: missing rgb/depth for {base}, skipping")
            continue

        c2w = map_pose_to_nerfstudio_c2w(map_pose_path)
        shutil.copy(rgb_path, images_out / rgb_path.name)

        frames.append({
            "file_path": f"images/{rgb_path.name}",
            "transform_matrix": c2w.tolist(),
        })

        if args.build_pointcloud:
            from PIL import Image
            depth_m = load_raw_depth(depth_path, target_hw=(args.height, args.width))
            rgb = np.array(Image.open(rgb_path).convert("RGB").resize((args.width, args.height)))
            pts, cols = unproject_depth_to_points(
                depth_m, rgb, c2w, args.fx, args.fy, args.cx, args.cy,
                stride=args.pointcloud_stride)
            all_points.append(pts)
            all_colors.append(cols)

    meta = {
        "fl_x": args.fx, "fl_y": args.fy, "cx": args.cx, "cy": args.cy,
        "w": args.width, "h": args.height,
        "camera_model": "OPENCV",
        "frames": frames,
    }
    if args.build_pointcloud:

        meta["ply_file_path"] = "sparse_pc.ply"
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(meta, f, indent=2)
    excluded_note = f" ({n_excluded} excluded)" if n_excluded else ""
    print(f"Wrote {len(frames)} frames to {out_dir / 'transforms.json'}{excluded_note}")

    if args.build_pointcloud and all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        print(f"Writing point cloud: {len(points)} points")
        ply_path = out_dir / "sparse_pc.ply"
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        print(f"Saved {ply_path}")

    print("\nDone. Train directly with:")
    print(f"  ns-train splatfacto --data {out_dir}")
    print("(no ns-process-data, no align_frames.py needed for this dataset)")


if __name__ == "__main__":
    main()
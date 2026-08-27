"""
Render a single image from a trained nerfstudio Splatfacto model, given a
camera pose defined in a JSON file.

Two pose JSON formats are accepted:
  1. build_transforms_from_poses.py frame format:
       {"file_path": "...", "transform_matrix": [[...4x4...]]}
  2. Raw capture format (map_pose.json style):
       {"position": [x,y,z], "orientation": [qx,qy,qz,qw], ...}

Usage:
  python3 render_pose.py \
    --load-config outputs/direct_dataset/splatfacto/2026-08-03_120000/config.yml \
    --pose-json wp3_d435_pose1_map_pose.json \
    --intrinsics-json /path/to/transforms.json \
    --output render_before.png

  # With a ground-truth/reference image for quick quantitative comparison:
  python3 render_pose.py \
    --load-config .../config.yml \
    --pose-json wp3_d435_pose1_map_pose.json \
    --intrinsics-json /path/to/transforms.json \
    --output render_after.png \
    --reference-image wp3_d435_pose1_rgb.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType


ROS_TO_NERFSTUDIO_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


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


def load_pose_as_c2w(pose_json_path: Path) -> np.ndarray:
    """Loads a pose JSON in any of three formats and returns a 4x4 c2w
    matrix in the same convention used at training time (post ROS->
    nerfstudio flip):
      1. build_transforms_from_poses.py frame format:
         {"transform_matrix": [[...]]}  (4x4 or 3x4)
      2. Raw capture format (map_pose.json style):
         {"position": [x,y,z], "orientation": [qx,qy,qz,qw]}
      3. score_and_return_top_candidates.py's scored output format:
         {"position": {"x":,"y":,"z":}, "orientation": {"x":,"y":,"z":,"w":}}
         -- nested dicts instead of flat arrays. If your file is a full
         scored-candidates list rather than a single entry, extract one
         entry's dict first (e.g. via jq or a one-line json.load + index)."""
    with open(pose_json_path) as f:
        record = json.load(f)

    if "transform_matrix" in record:
        tm = np.array(record["transform_matrix"], dtype=np.float64)
        if tm.shape == (3, 4):
            tm = np.vstack([tm, [0, 0, 0, 1]])
        return tm

    if "position" in record and "orientation" in record:
        pos, ori = record["position"], record["orientation"]
        # Normalize both nested-dict and flat-array forms to flat lists
        if isinstance(pos, dict):
            pos = [pos["x"], pos["y"], pos["z"]]
        if isinstance(ori, dict):
            ori = [ori["x"], ori["y"], ori["z"], ori["w"]]

        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_rotmat(*ori)
        c2w[:3, 3] = pos
        return c2w @ ROS_TO_NERFSTUDIO_FLIP

    raise ValueError(
        f"{pose_json_path} has neither 'transform_matrix' nor "
        f"'position'/'orientation' -- unrecognized pose format"
    )


def load_dataparser_transform(load_config: Path):
    """Looks for dataparser_transforms.json next to the model config and
    returns (transform_4x4, scale). Returns (identity, 1.0) if not found,
    with a loud warning either way about consistency across training runs."""
    candidate = load_config.parent / "dataparser_transforms.json"
    if not candidate.exists():
        print(f"[WARN] No dataparser_transforms.json found at {candidate}. "
              f"Assuming identity transform / scale=1.0. If your training "
              f"used auto-orientation, this render will be WRONG.")
        return np.eye(4), 1.0

    with open(candidate) as f:
        data = json.load(f)

    transform = np.array(data["transform"], dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0, 0, 0, 1]])
    scale = float(data.get("scale", 1.0))

    is_identity = np.allclose(transform, np.eye(4), atol=1e-6)
    is_unit_scale = abs(scale - 1.0) < 1e-6
    if not (is_identity and is_unit_scale):
        print(f"[WARN] dataparser_transforms.json is NOT identity/scale=1 "
              f"(scale={scale:.6f}). This model was auto-oriented/scaled. "
              f"This transform is being applied to your raw pose below -- "
              f"but if you retrain after adding an image, re-check this "
              f"file, since auto-orientation can recompute it and silently "
              f"shift the whole scene relative to your checkpoint.")
    else:
        print("[OK] dataparser_transforms.json is identity, scale=1.0 -- "
              "raw poses map directly into model space, no correction needed.")

    return transform, scale


def apply_dataparser_transform(raw_c2w: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    """Applies the same auto-orient/center + scale transform nerfstudio
    applied to training poses, so a new raw pose lands in the same space
    the model was actually trained in."""
    new_c2w = transform @ raw_c2w
    new_c2w[:3, 3] *= scale
    return new_c2w


def simple_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", required=True, type=Path,
                     help="Path to the trained model's config.yml")
    ap.add_argument("--pose-json", required=True, type=Path,
                     help="JSON file with the camera pose to render")
    ap.add_argument("--intrinsics-json", required=True, type=Path,
                     help="transforms.json to pull fl_x/fl_y/cx/cy/w/h from "
                          "(use the same one training used, for consistency)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reference-image", type=Path, default=None,
                     help="Optional ground-truth image for this pose, to "
                          "compute PSNR against the render")
    ap.add_argument("--psnr-out-json", type=Path, default=None,
                     help="If set (requires --reference-image), also write "
                          "the PSNR result as structured JSON to this path "
                          "-- for callers (e.g. gpu_main_loop.py's holdout "
                          "logging) that need the number back programmatically "
                          "instead of parsing it out of stdout")
    ap.add_argument("--skip-dataparser-transform", action="store_true",
                     help="Skip applying dataparser_transforms.json (only "
                          "correct if you already know your model was "
                          "trained with orientation-method=none, "
                          "center-method=none, auto-scale-poses=False)")
    args = ap.parse_args()

    with open(args.intrinsics_json) as f:
        meta = json.load(f)
    fx, fy = meta["fl_x"], meta["fl_y"]
    cx, cy = meta["cx"], meta["cy"]
    w, h = int(meta["w"]), int(meta["h"])

    raw_c2w = load_pose_as_c2w(args.pose_json)

    if args.skip_dataparser_transform:
        final_c2w = raw_c2w
    else:
        transform, scale = load_dataparser_transform(args.load_config)
        final_c2w = apply_dataparser_transform(raw_c2w, transform, scale)

    print("Loading pipeline (this can take a bit for a trained splat)...")
    config, pipeline, _, _ = eval_setup(args.load_config, test_mode="inference")

    c2w_tensor = torch.tensor(final_c2w[:3, :4], dtype=torch.float32).unsqueeze(0)

    camera = Cameras(
        camera_to_worlds=c2w_tensor,
        fx=torch.tensor([[fx]], dtype=torch.float32),
        fy=torch.tensor([[fy]], dtype=torch.float32),
        cx=torch.tensor([[cx]], dtype=torch.float32),
        cy=torch.tensor([[cy]], dtype=torch.float32),
        width=torch.tensor([[w]], dtype=torch.int64),
        height=torch.tensor([[h]], dtype=torch.int64),
        camera_type=CameraType.PERSPECTIVE,
    )
    camera = camera.to(pipeline.device)

    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)

    rgb = outputs["rgb"].detach().cpu().numpy()
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(args.output)
    print(f"Saved render to {args.output}")

    if args.reference_image is not None:
        ref = np.array(Image.open(args.reference_image).convert("RGB").resize((w, h)))
        psnr = simple_psnr(rgb_uint8, ref)
        print(f"PSNR vs {args.reference_image}: {psnr:.2f} dB "
              f"(higher = closer match; compare this number between your "
              f"before/after renders)")
        if args.psnr_out_json is not None:
            args.psnr_out_json.parent.mkdir(parents=True, exist_ok=True)
            args.psnr_out_json.write_text(json.dumps({
                "psnr": psnr,
                "pose_json": str(args.pose_json),
                "load_config": str(args.load_config),
                "reference_image": str(args.reference_image),
                "output": str(args.output),
            }, indent=2))
    elif args.psnr_out_json is not None:
        print("[WARN] --psnr-out-json given without --reference-image -- "
              "nothing to write, no PSNR was computed")


if __name__ == "__main__":
    main()
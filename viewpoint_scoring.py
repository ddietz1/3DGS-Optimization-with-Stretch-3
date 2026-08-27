"""
Standalone viewpoint-scoring harness -- v3: real GauSS-MI Shannon mutual
information, reimplemented in plain PyTorch (no CUDA build required).

"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from nerfstudio.utils.eval_utils import eval_setup
except ImportError as e:
    raise SystemExit(
        "Could not import nerfstudio.utils.eval_utils.eval_setup -- run this "
        "inside the same conda env you used for ns-train."
    ) from e

try:
    from gsplat.rendering import rasterization
except ImportError as e:
    raise SystemExit("gsplat not found in this env.") from e


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_L = 1.0
LAMBDA_T = 1.0
MIN_LOSS = 1e-3
TOUCH_THRESHOLD = 3          # min "touch count" before a Gaussian's update counts
DOMINANT_T_THRESHOLD = 0.8   # only let near-front (dominant) contributors update reliability


# ----------------------------------------------------------------------
# 1. Load pipeline / Gaussians / training cameras (unchanged from v2)
# ----------------------------------------------------------------------

def load_pipeline(config_path: str):
    config, pipeline, checkpoint_path, step = eval_setup(Path(config_path))
    print(f"Loaded pipeline from {config_path} (checkpoint step {step})")
    pipeline.eval()
    return pipeline


def get_gaussians(pipeline):
    model = pipeline.model
    means = model.means.detach()
    quats = model.quats.detach()
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    scales = torch.exp(model.scales.detach())
    opacities = torch.sigmoid(model.opacities.detach())
    if opacities.dim() > 1:
        opacities = opacities.squeeze(-1)

    SH_C0 = 0.28209479177387814
    features_dc = model.features_dc.detach()
    colors = torch.clamp(0.5 + SH_C0 * features_dc, 0.0, 1.0)

    print(f"Pulled {means.shape[0]} Gaussians directly from the trained model.")
    return {
        "means": means.to(DEVICE),
        "quats": quats.to(DEVICE),
        "scales": scales.to(DEVICE),
        "opacities": opacities.to(DEVICE),
        "colors": colors.to(DEVICE),
    }


def get_training_cameras(pipeline):
    cameras = pipeline.datamanager.train_dataset.cameras
    print(f"Found {cameras.size} training cameras.")
    return cameras


# ----------------------------------------------------------------------
# 2. Reliability model
# ----------------------------------------------------------------------

def init_reliability(n_gaussians: int) -> torch.Tensor:
    """(N,4) tensor, one value per direction bin (+x,+y,-x,-y), init 0.5 --
    matches GaussianModel's self.init_reliability = 0.5."""
    return torch.full((n_gaussians, 4), 0.5, device=DEVICE, dtype=torch.float32)


def reliability_to_logodds(r: torch.Tensor) -> torch.Tensor:
    r = r.clamp(1e-4, 1 - 1e-4)
    return torch.log(r / (1 - r))


def logodds_to_reliability(logodds: torch.Tensor) -> torch.Tensor:
    r = torch.sigmoid(logodds)
    return r.clamp(1e-4, 1 - 1e-4)


def compute_unreliable_coeff(means: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    """Exact port of computeUnreliableCoeff from forward.cu.
    means: (N,3), cam_pos: (3,). Returns (N,4) coefficient -- a soft
    quadrant-interpolation weight based on the direction from each Gaussian
    to the camera, in the horizontal (x,y) plane only (their CUDA code
    doesn't use the z/elevation component either -- purely azimuthal)."""
    direction = cam_pos.unsqueeze(0) - means  # (N,3), gaussian -> camera
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    dir_abs = direction.abs()

    cos_theta = dir_abs[:, 0]  # |dir.x|
    sin_theta = dir_abs[:, 1]  # |dir.y|

    coeff = torch.zeros((means.shape[0], 4), device=DEVICE, dtype=torch.float32)
    pos_x = direction[:, 0] >= 0
    pos_y = direction[:, 1] >= 0
    coeff[:, 0] = torch.where(pos_x, cos_theta, torch.zeros_like(cos_theta))
    coeff[:, 1] = torch.where(pos_y, sin_theta, torch.zeros_like(sin_theta))
    coeff[:, 2] = torch.where(~pos_x, cos_theta, torch.zeros_like(cos_theta))
    coeff[:, 3] = torch.where(~pos_y, sin_theta, torch.zeros_like(sin_theta))
    return coeff


def compute_unreliable1(reliability: torch.Tensor, means: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    """Exact port of: unreliable1[idx] = sum(unreliabilities[idx] * coeff).
    NOTE: their variable is literally called `unreliabilities` (i.e. this
    already represents 1-reliability going in) in the CUDA code, but
    GaussianModel stores `_reliability` (reliability, not unreliability) and
    exposes get_unreliability() = 1 - reliability for exactly this purpose.
    We do the same here explicitly."""
    unreliability = 1.0 - reliability  # (N,4)
    coeff = compute_unreliable_coeff(means, cam_pos)  # (N,4)
    return (unreliability * coeff).sum(dim=1)  # (N,)


# ----------------------------------------------------------------------
# 3. Convention conversion
# ----------------------------------------------------------------------

def c2w_to_viewmat(c2w_3x4: torch.Tensor) -> torch.Tensor:
    c2w4 = torch.eye(4, device=c2w_3x4.device, dtype=c2w_3x4.dtype)
    c2w4[:3, :4] = c2w_3x4
    flip = torch.diag(torch.tensor(
        [1.0, -1.0, -1.0, 1.0], device=c2w_3x4.device, dtype=c2w_3x4.dtype
    ))
    c2w_cv = c2w4 @ flip
    return torch.linalg.inv(c2w_cv)


def render_our_own(gaussians, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int,
                    colors_override: torch.Tensor = None):
    """viewmats: (C,4,4), Ks: (C,3,3). colors_override, if given, replaces
    gaussians['colors'] -- lets us render an arbitrary per-Gaussian scalar
    (like unreliable1) through the same alpha-compositing math used for RGB,
    exactly mirroring how their kernel composites MI alongside color."""
    C = viewmats.shape[0]
    N = gaussians["means"].shape[0]
    colors = colors_override if colors_override is not None else gaussians["colors"]

    renders, alphas, meta = rasterization(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=gaussians["scales"],
        opacities=gaussians["opacities"],
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
    )

    if "camera_ids" in meta and "gaussian_ids" in meta:
        raw_radii = meta["radii"]
        if raw_radii.dim() > 1:
            raw_radii = raw_radii.max(dim=-1).values
        dense_radii = torch.zeros(C, N, device=DEVICE, dtype=raw_radii.dtype)
        dense_radii[meta["camera_ids"], meta["gaussian_ids"]] = raw_radii
        radii = dense_radii
    else:
        radii = meta["radii"]
        if radii.dim() == 3:
            radii = radii.max(dim=-1).values

    return renders, alphas, radii


# ----------------------------------------------------------------------
# 4. Reliability UPDATE pass over training views
# ----------------------------------------------------------------------

def update_reliability_from_view(gaussians, reliability, cam_single, real_image, w, h):
    """Approximate port of map_backend.py::update_reliability.

    Their version: render loss_image, then a SECOND custom-CUDA render pass
    scatters loss_p * alpha * T * coeff onto every Gaussian that dominantly
    touched each pixel, via atomicAdd inside the rasterizer -- an exact
    per-(pixel,Gaussian) attribution using the real alpha-compositing
    weights.

    APPROXIMATION HERE: gsplat's public API doesn't expose per-Gaussian
    alpha*T contribution directly, so instead we attribute a view's average
    photometric loss to every Gaussian that's visibly dominant in that view
    (radii > 0), weighted by that Gaussian's own opacity (as a stand-in for
    its typical alpha*T contribution) and its directional coefficient
    (exact match for the directional routing). This is coarser than their
    per-pixel scatter -- every touched Gaussian in a view gets a share of
    that view's *average* loss, rather than the loss at the exact pixels it
    actually influenced. Good enough to validate the mechanism end-to-end;
    not a bit-exact match to their CUDA kernel.
    """
    c2w = cam_single.camera_to_worlds[0].to(DEVICE)
    viewmat = c2w_to_viewmat(c2w).unsqueeze(0)
    cam_pos = c2w[:3, 3]
    fx, fy = float(cam_single.fx[0]), float(cam_single.fy[0])
    cx, cy = float(cam_single.cx[0]), float(cam_single.cy[0])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      dtype=torch.float32, device=DEVICE).unsqueeze(0)

    render, _, radii = render_our_own(gaussians, viewmat, K, w, h)
    render = render[0].clamp(0, 1)  # (H,W,3)

    real = real_image.to(DEVICE)
    if real.dim() == 3 and real.shape[0] in (1, 3):
        real = real.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
    real = real[..., :3]

    photometric_loss = (render - real).abs().mean()  # scalar, per-view average
    loss_val = torch.clamp(photometric_loss * LAMBDA_L, min=MIN_LOSS)
    loss_prime = -torch.log(loss_val)  # matches L' = -log(lambda * L)

    visible = radii[0] > 0  # (N,)
    n_touched = visible.float()  # single-view touch count contribution
    coeff = compute_unreliable_coeff(gaussians["means"], cam_pos)  # (N,4)

    # loss_p * alpha * T -> approximated as loss_prime * opacity, gated to
    # only Gaussians that are dominant contributors somewhere in this view.
    dominant = visible & (gaussians["opacities"] > DOMINANT_T_THRESHOLD * 0 + 0.1)
    # (opacity > small threshold is our stand-in for "T > 0.8 dominant contributor";

    attributed = torch.zeros_like(coeff)
    attributed[dominant] = loss_prime * LAMBDA_T * coeff[dominant]

    return attributed, n_touched, dominant


def run_reliability_update_pass(gaussians, reliability, pipeline, cameras, max_views=None):
    """One full pass over training views, folding each into the persistent
    reliability via a log-odds Bayesian update -- matches
    self.gaussians.input_logodds(logodds_zt)."""
    n_train = cameras.size if max_views is None else min(max_views, cameras.size)
    print(f"Running reliability update over {n_train} training views...")

    logodds = reliability_to_logodds(reliability)

    for i in range(n_train):
        cam_single = cameras[i:i + 1].to(pipeline.device)
        real_image = pipeline.datamanager.train_dataset[i]["image"]
        if not torch.is_tensor(real_image):
            real_image = torch.from_numpy(np.array(real_image)).float() / 255.0
        w, h = int(cam_single.width[0]), int(cam_single.height[0])

        attributed, n_touched, dominant = update_reliability_from_view(
            gaussians, reliability, cam_single, real_image, w, h
        )

        touch_ok = dominant  # per-Gaussian bool: eligible to update this round
        logodds_delta = torch.clamp(attributed, min=-10.0, max=10.0)
        logodds_delta[~touch_ok] = 0.0
        logodds = logodds + logodds_delta

    reliability = logodds_to_reliability(logodds)
    print(f"Reliability update done. Mean reliability: {reliability.mean().item():.3f} "
          f"(1.0 = fully reliable, matches init 0.5 where under-observed)")
    return reliability


def prune_floaters(gaussians, cam_positions: np.ndarray, scale_percentile=99.9,
                    opacity_min=0.02, max_radius_multiplier=6.0):
    """Removes likely floater/background-artifact Gaussians before scoring.

    DEFAULTS CHANGED after seeing this eat real wall/ceiling geometry in a
    room-scale scene -- large, flat, continuous surfaces legitimately need
    a handful of LARGE-scale Gaussians and are naturally FAR from where the
    camera stood. Neither "large scale" nor "spatially distant" reliably
    means floater once you're doing room-scale capture rather than an
    isolated small object -- these were tuned for the latter. Treat these
    thresholds as something to verify against your own sanity-check render
    every time you change them, not as safe defaults.
    """
    means = gaussians["means"]
    scales = gaussians["scales"]
    opacities = gaussians["opacities"]

    max_scale = scales.max(dim=1).values  # (N,) largest axis per Gaussian
    scale_thresh = torch.quantile(max_scale, scale_percentile / 100.0)
    scale_ok = max_scale <= scale_thresh

    opacity_ok = opacities >= opacity_min

    center = torch.tensor(cam_positions.mean(axis=0), device=DEVICE, dtype=torch.float32)
    cam_radius = float(np.linalg.norm(cam_positions - cam_positions.mean(axis=0), axis=1).mean())
    dist_from_center = (means - center).norm(dim=1)
    spatial_ok = dist_from_center <= (cam_radius * max_radius_multiplier)

    keep = scale_ok & opacity_ok & spatial_ok
    n_before = means.shape[0]
    n_after = int(keep.sum().item())
    print(f"Pruned {n_before - n_after} / {n_before} Gaussians as likely floaters "
          f"({n_after} remaining) -- scale_ok={int(scale_ok.sum())}, "
          f"opacity_ok={int(opacity_ok.sum())}, spatial_ok={int(spatial_ok.sum())}")

    return {k: v[keep] for k, v in gaussians.items()}


# ----------------------------------------------------------------------
# 5. Sanity check
# ----------------------------------------------------------------------

def sanity_check(pipeline, gaussians, cameras, check_index: int, out_path: str, use_eval: bool = False):
    from PIL import Image

    device = pipeline.device
    if use_eval:
        dataset = pipeline.datamanager.eval_dataset
        cameras = dataset.cameras
        print(f"[sanity_check] Using EVAL-set image index {check_index} "
              f"(genuinely held out from training -- a real generalization check, "
              f"not the training-image convention check).")
    else:
        dataset = pipeline.datamanager.train_dataset
        print(f"[sanity_check] Using TRAINING-set image index {check_index} "
              f"(this is expected to match near-perfectly -- it validates "
              f"pose/rendering CONVENTION correctness, not generalization).")

    cam_single = cameras[check_index:check_index + 1].to(device)

    with torch.no_grad():
        ns_outputs = pipeline.model.get_outputs_for_camera(cam_single)
    ns_render = ns_outputs["rgb"].clamp(0, 1).cpu().numpy()

    c2w = cam_single.camera_to_worlds[0].to(DEVICE)
    viewmat = c2w_to_viewmat(c2w).unsqueeze(0)
    fx, fy = float(cam_single.fx[0]), float(cam_single.fy[0])
    cx, cy = float(cam_single.cx[0]), float(cam_single.cy[0])
    w, h = int(cam_single.width[0]), int(cam_single.height[0])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
    our_render, _, _ = render_our_own(gaussians, viewmat, K, w, h)
    our_render = our_render[0].clamp(0, 1).cpu().numpy()

    real = dataset[check_index]["image"]
    if torch.is_tensor(real):
        real = real.cpu().numpy()
    real = (real * 255).astype(np.uint8) if real.max() <= 1.0 else real.astype(np.uint8)

    ns_render_u8 = (ns_render * 255).astype(np.uint8)
    our_render_u8 = (our_render * 255).astype(np.uint8)
    h_max = max(real.shape[0], ns_render_u8.shape[0], our_render_u8.shape[0])

    def pad(img):
        pad_amt = h_max - img.shape[0]
        return np.pad(img, ((0, pad_amt), (0, 0), (0, 0))) if pad_amt else img

    triple = np.concatenate([pad(real), pad(ns_render_u8), pad(our_render_u8)], axis=1)
    Image.fromarray(triple).save(out_path)
    print(f"\nSanity check saved to {out_path}")
    print("Panels: [real image] [nerfstudio's render] [our gsplat render]")


# ----------------------------------------------------------------------
# 6. Candidate generation
# ----------------------------------------------------------------------

def fibonacci_sphere(n, radius, center, min_elev_deg=-10.0):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x, z = np.cos(theta) * r, np.sin(theta) * r
        if np.degrees(np.arcsin(y)) < min_elev_deg:
            continue
        points.append(center + radius * np.array([x, y, z]))
    return np.array(points, dtype=np.float32)


def look_at_c2w(cam_pos, target, up=np.array([0, 1, 0])):
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, fwd)
    R = np.stack([right, true_up, -fwd], axis=1)
    c2w = np.zeros((3, 4), dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos
    return c2w


def generate_candidates(scene_center, radius, n_candidates):
    cam_positions = fibonacci_sphere(n_candidates, radius, scene_center)
    return [look_at_c2w(pos, scene_center) for pos in cam_positions]


# ----------------------------------------------------------------------
# 7. Scoring
# ----------------------------------------------------------------------

def score_candidate_mi(gaussians, reliability, c2w_np, template_intrinsics):
    """One candidate at a time -- matches their own FSM, which calls
    compute_GauSS_MI per candidate in a loop, not batched."""
    fx, fy, cx, cy, w, h = template_intrinsics
    c2w = torch.tensor(c2w_np, dtype=torch.float32, device=DEVICE)
    cam_pos = c2w[:3, 3]
    viewmat = c2w_to_viewmat(c2w).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      dtype=torch.float32, device=DEVICE).unsqueeze(0)

    unreliable1 = compute_unreliable1(reliability, gaussians["means"], cam_pos)  # (N,)
    # Render as a 3-channel "color"
    mi_colors = unreliable1.unsqueeze(-1).repeat(1, 3)
    render, _, _ = render_our_own(gaussians, viewmat, K, w, h, colors_override=mi_colors)
    mi_map = render[0, :, :, 0]  # (H,W) -- MI channel, alpha-composited exactly like color
    return float(mi_map.sum().item())


def render_and_save_candidates(gaussians, c2ws, labels, template_intrinsics, out_dir: Path):
    from PIL import Image

    fx, fy, cx, cy, w, h = template_intrinsics
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
    render_dir = out_dir / "candidate_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    for label, c2w in zip(labels, c2ws):
        viewmat = c2w_to_viewmat(torch.tensor(c2w, dtype=torch.float32, device=DEVICE)).unsqueeze(0)
        render, _, _ = render_our_own(gaussians, viewmat, K, w, h)
        img = (render[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(render_dir / f"{label}.png")
    print(f"Saved {len(labels)} candidate renders to {render_dir}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", required=True)
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--sanity-check-index", type=int, default=0)
    ap.add_argument("--sanity-check-on-eval", action="store_true",
                     help="Run the sanity check against a held-out EVAL-set image "
                          "instead of a training image -- a genuine generalization "
                          "spot-check, as opposed to the default's convention check "
                          "(which is EXPECTED to look near-perfect, by design).")
    ap.add_argument("--out-dir", default="./viewpoint_scoring_out")
    ap.add_argument("--render-top-bottom-n", type=int, default=5)
    ap.add_argument("--max-update-views", type=int, default=None,
                     help="Cap on training views used for the reliability update pass (default: all)")
    ap.add_argument("--enable-pruning", action="store_true",
                     help="Attempt floater pruning (off by default -- scale/spatial-based "
                          "pruning has repeatedly eaten real geometry on room-scale scenes; "
                          "see comments on prune_floaters(). Only useful for cosmetic "
                          "clarity in spot-check renders, not required for correct scoring.")
    ap.add_argument("--scale-percentile", type=float, default=99.9)
    ap.add_argument("--opacity-min", type=float, default=0.02)
    ap.add_argument("--max-radius-multiplier", type=float, default=6.0)
    ap.add_argument("--reuse-candidates", default=None,
                     help="Path to a previous run's candidate_scores.json. If given, "
                          "re-scores those EXACT saved poses (position AND orientation) "
                          "against this checkpoint instead of generating a new fibonacci-"
                          "sphere candidate set -- required for a fair before/after "
                          "comparison, since scene_center/radius can drift slightly once "
                          "training cameras change (e.g. after adding a holdout image).")
    ap.add_argument("--auto-confirm", action="store_true",
                     help="Skip the interactive 'press Enter to continue' prompt after "
                          "the sanity check -- required for unattended/autonomous runs. "
                          "The sanity_check.png is still saved for later review; this "
                          "just doesn't block waiting for a human to look at it first.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(args.load_config)
    gaussians = get_gaussians(pipeline)
    cameras = get_training_cameras(pipeline)

    cam_positions_np = cameras.camera_to_worlds[:, :3, 3].cpu().numpy()
    if args.enable_pruning:
        gaussians = prune_floaters(gaussians, cam_positions_np,
                                    scale_percentile=args.scale_percentile,
                                    opacity_min=args.opacity_min,
                                    max_radius_multiplier=args.max_radius_multiplier)
    else:
        print("Pruning off by default (pass --enable-pruning to try it, but see the "
              "known issue with room-scale scenes in prune_floaters()'s docstring).")

    sanity_check(pipeline, gaussians, cameras, args.sanity_check_index,
                 str(out_dir / "sanity_check.png"), use_eval=args.sanity_check_on_eval)
    if args.auto_confirm:
        print("(--auto-confirm set, skipping interactive prompt -- review "
              "sanity_check.png after the run completes)")
    else:
        input("\nCheck sanity_check.png now. Press Enter to continue, or Ctrl+C to stop...")

    reliability = init_reliability(gaussians["means"].shape[0])
    reliability = run_reliability_update_pass(gaussians, reliability, pipeline, cameras,
                                               max_views=args.max_update_views)

    template_intrinsics = (
        float(cameras.fx[0]), float(cameras.fy[0]),
        float(cameras.cx[0]), float(cameras.cy[0]),
        int(cameras.width[0]), int(cameras.height[0]),
    )

    if args.reuse_candidates is not None:
        with open(args.reuse_candidates) as f:
            saved = json.load(f)
        if "transform_matrix" not in saved[0]:
            raise SystemExit(
                f"{args.reuse_candidates} has no 'transform_matrix' field -- it was "
                f"saved before this flag existed. Re-run scoring once WITHOUT "
                f"--reuse-candidates first (which now saves transform_matrix), then "
                f"use that new file for --reuse-candidates on subsequent runs."
            )
        candidates = [np.array(r["transform_matrix"], dtype=np.float32) for r in saved]
        print(f"Reusing {len(candidates)} exact candidate poses from {args.reuse_candidates} "
              f"(skipping candidate generation)")
    else:
        # Derive the candidate region from TRAINING CAMERA positions, not the
        # Gaussian point cloud's bounding extent
        cam_positions = cameras.camera_to_worlds[:, :3, 3].cpu().numpy()
        scene_center = cam_positions.mean(axis=0)
        if args.radius is None:
            cam_radii = np.linalg.norm(cam_positions - scene_center, axis=1)
            args.radius = float(cam_radii.mean())
        print(f"Candidate sphere: center={scene_center}, radius={args.radius:.3f} "
              f"(derived from {len(cam_positions)} training camera positions)")

        candidates = generate_candidates(scene_center, args.radius, args.n_candidates)

    print(f"Scoring {len(candidates)} candidates via Shannon MI (one at a time)...")
    scores = [score_candidate_mi(gaussians, reliability, c2w, template_intrinsics)
              for c2w in candidates]

    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    results = [{"index": i, "position": c[:3, 3].tolist(), "transform_matrix": c.tolist(), "score": s}
               for i, (c, s) in enumerate(ranked)]
    with open(out_dir / "candidate_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTop 5 candidates:")
    for r in results[:5]:
        print(f"  score={r['score']:.4f}  pos={r['position']}")
    print("\nBottom 5 candidates:")
    for r in results[-5:]:
        print(f"  score={r['score']:.4f}  pos={r['position']}")
    print(f"\nFull ranking written to {out_dir / 'candidate_scores.json'}")

    n_render = min(args.render_top_bottom_n, len(ranked))
    top_c2ws = [c for c, s in ranked[:n_render]]
    bottom_c2ws = [c for c, s in ranked[-n_render:]]
    labels = [f"top_{i:02d}_score_{s:.3f}" for i, (c, s) in enumerate(ranked[:n_render])] + \
             [f"bottom_{i:02d}_score_{s:.3f}" for i, (c, s) in enumerate(ranked[-n_render:])]
    render_and_save_candidates(gaussians, top_c2ws + bottom_c2ws, labels, template_intrinsics, out_dir)


# ----------------------------------------------------------------------
# 8. Depth confidence
# ----------------------------------------------------------------------

def build_frame_to_depth_map(source_rgb_dir, raw_captures_dir):
    """Reconstructs ns-process-data's frame_XXXXX <-> original-file mapping,
    then derives the matching raw depth .npy path for each.

    Handles two capture layouts, tried in order per file:
    - Newer (flat): depth file has the SAME name as the RGB file, just
      "_rgb.png" -> "_depth.npy", sitting directly in raw_captures_dir with
      no subfolder nesting (e.g. captures_d435/<run>/full_data/).
    - Older (per-waypoint subfolders): source_rgb_dir filenames were
      prefixed "{waypoint_folder}_{orig}" when flattened for ns-process-data,
      and the real depth file lives in raw_captures_dir/{folder}/{orig
      with _rgb->_depth}.

    ASSUMPTIONS -- verify against your own capture layout before trusting:
    - source_rgb_dir contains the exact files originally handed to
      ns-process-data.
    - ns-process-data assigns frame_00001, frame_00002, ... in ALPHABETICAL
      order of source_rgb_dir's filenames (confirmed via nerfstudio source:
      process_data_utils.py sorts glob results before renaming).

    Returns: {full_meta frame index -> depth .npy Path}. Since full_meta is
    ALSO sorted alphabetically by file_path (see load_full_transforms), and
    frame_00001 < frame_00002 alphabetically too, position i in this sorted
    source list lines up with position i in the sorted full_meta frames list.
    """
    source_rgb_dir = Path(source_rgb_dir)
    raw_captures_dir = Path(raw_captures_dir)
    original_files_sorted = sorted(source_rgb_dir.glob("*.png"))

    mapping = {}
    for i, orig_path in enumerate(original_files_sorted):
        name = orig_path.name

        # Newer (flat) layout: same name, suffix swapped, no subfolder.
        candidate_flat = raw_captures_dir / name.replace("_rgb.png", "_depth.npy")

        # Older (per-waypoint subfolder) layout.
        candidate_nested = None
        if "_" in name:
            folder, rest = name.split("_", 1)
            candidate_nested = raw_captures_dir / folder / rest.replace("_rgb.png", "_depth.npy")

        if candidate_flat.exists():
            mapping[i] = candidate_flat
        elif candidate_nested is not None and candidate_nested.exists():
            mapping[i] = candidate_nested
        else:
            mapping[i] = candidate_flat  # best guess, for the missing-count/debug output

    n_missing = sum(1 for p in mapping.values() if not p.exists())
    print(f"Built frame->depth mapping: {len(mapping)} entries, "
          f"{n_missing} missing files on disk -- spot check a few before trusting this.")
    return mapping


def load_raw_depth(depth_path, target_hw=None, assume_mm_uint16="auto"):
    """Loads a raw depth .npy file."""
    arr = np.load(depth_path)
    if assume_mm_uint16 == "auto":
        is_mm = arr.dtype == np.uint16
    else:
        is_mm = assume_mm_uint16
    depth_m = arr.astype(np.float32) * (0.001 if is_mm else 1.0)

    if target_hw is not None and depth_m.shape[:2] != target_hw:
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)
        depth_t = torch.nn.functional.interpolate(depth_t, size=target_hw, mode="nearest")
        depth_m = depth_t[0, 0].numpy()
    return depth_m


def init_depth_confidence(n_gaussians: int) -> torch.Tensor:
    """(N,4) directional bins, same structure as appearance reliability --
    reuses compute_unreliable_coeff for the same directional routing."""
    return torch.full((n_gaussians, 4), 0.5, device=DEVICE, dtype=torch.float32)


def estimate_depth_scale_factor(gaussians, cameras, frame_to_depth_map, frame_indices, max_views=5):
    """Estimates the scale factor between the reconstruction's own coordinate
    units (arbitrary/uncalibrated, since auto_scale_poses=False) and real
    metric sensor depth.
    """
    ratios = []
    n_check = min(max_views, cameras.size)
    for i in range(n_check):
        orig_idx = frame_indices[i]
        depth_path = frame_to_depth_map.get(orig_idx)
        if depth_path is None or not depth_path.exists():
            continue
        cam_single = cameras[i:i + 1].to(DEVICE)
        w, h = int(cam_single.width[0]), int(cam_single.height[0])
        real_depth_m = load_raw_depth(depth_path, target_hw=(h, w))

        c2w = cam_single.camera_to_worlds[0].to(DEVICE)
        viewmat = c2w_to_viewmat(c2w).unsqueeze(0)
        cam_pos = c2w[:3, 3]
        fx, fy = float(cam_single.fx[0]), float(cam_single.fy[0])
        cx, cy = float(cam_single.cx[0]), float(cam_single.cy[0])
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                          dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dists = (gaussians["means"] - cam_pos).norm(dim=1)
        depth_colors = dists.unsqueeze(-1).repeat(1, 3)
        rendered_depth, _, _ = render_our_own(gaussians, viewmat, K, w, h, colors_override=depth_colors)
        rendered_depth_map = rendered_depth[0, :, :, 0].cpu().numpy()

        valid = (real_depth_m > 0) & (rendered_depth_map > 1e-6)
        if valid.sum() < 100:
            continue
        ratio = np.median(real_depth_m[valid] / rendered_depth_map[valid])
        ratios.append(ratio)

    if not ratios:
        print("WARNING: could not estimate depth scale factor (no valid sample views) -- defaulting to 1.0")
        return 1.0
    scale = float(np.median(ratios))
    print(f"Estimated depth scale factor (reconstruction units -> meters): {scale:.4f} "
          f"(from {len(ratios)} sample views)")
    return scale


def update_depth_confidence_from_view(gaussians, cam_single, real_depth_m, w, h,
                                       depth_scale_factor: float = 1.0,
                                       lambda_d=1.0, min_depth_loss=1e-3, dominant_opacity=0.1):
    """Depth analogue of update_reliability_from_view.

    """
    c2w = cam_single.camera_to_worlds[0].to(DEVICE)
    viewmat = c2w_to_viewmat(c2w).unsqueeze(0)
    cam_pos = c2w[:3, 3]
    fx, fy = float(cam_single.fx[0]), float(cam_single.fy[0])
    cx, cy = float(cam_single.cx[0]), float(cam_single.cy[0])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      dtype=torch.float32, device=DEVICE).unsqueeze(0)

    dists = (gaussians["means"] - cam_pos).norm(dim=1)
    depth_colors = dists.unsqueeze(-1).repeat(1, 3)
    rendered_depth, _, radii = render_our_own(gaussians, viewmat, K, w, h, colors_override=depth_colors)
    rendered_depth_map = rendered_depth[0, :, :, 0] * depth_scale_factor  # correct for reconstruction-vs-metric unit mismatch

    real_depth_t = torch.from_numpy(real_depth_m).to(DEVICE).float()
    valid_pixel_mask = real_depth_t > 0
    depth_residual_map = torch.zeros_like(rendered_depth_map)
    depth_residual_map[valid_pixel_mask] = (
        rendered_depth_map[valid_pixel_mask] - real_depth_t[valid_pixel_mask]
    ).abs()

    # Project every Gaussian's mean into this camera's pixel space directly.
    means = gaussians["means"]
    means_h = torch.cat([means, torch.ones(means.shape[0], 1, device=DEVICE)], dim=1)  # (N,4)
    means_cam = (viewmat[0] @ means_h.T).T  # (N,4), camera-space
    z = means_cam[:, 2]
    in_front = z > 1e-4
    u = (fx * means_cam[:, 0] / z.clamp(min=1e-4) + cx)
    v = (fy * means_cam[:, 1] / z.clamp(min=1e-4) + cy)
    in_frame = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    ui = u.clamp(0, w - 1).long()
    vi = v.clamp(0, h - 1).long()

    per_gaussian_residual = torch.zeros(means.shape[0], device=DEVICE)
    per_gaussian_has_real_depth = torch.zeros(means.shape[0], dtype=torch.bool, device=DEVICE)
    idx = torch.nonzero(in_frame, as_tuple=True)[0]
    per_gaussian_residual[idx] = depth_residual_map[vi[idx], ui[idx]]
    per_gaussian_has_real_depth[idx] = valid_pixel_mask[vi[idx], ui[idx]]

    loss_val = torch.clamp(per_gaussian_residual * lambda_d, min=min_depth_loss)
    loss_prime = -torch.log(loss_val)  # (N,) -- now genuinely per-Gaussian, not a broadcast scalar

    visible = radii[0] > 0
    dominant = visible & (gaussians["opacities"] > dominant_opacity) & in_frame & per_gaussian_has_real_depth
    coeff = compute_unreliable_coeff(gaussians["means"], cam_pos)

    attributed = torch.zeros_like(coeff)
    attributed[dominant] = loss_prime[dominant].unsqueeze(-1) * coeff[dominant]
    return attributed, dominant


def run_depth_confidence_update_pass(gaussians, depth_confidence, pipeline, cameras,
                                      frame_to_depth_map, frame_indices, max_views=None,
                                      depth_scale_factor: float = 1.0):
    """frame_indices: the ORIGINAL full-dataset index (matching
    frame_to_depth_map's keys) for each camera in `cameras`, in the SAME
    order -- required since a round's training Cameras object is indexed
    0..n_revealed-1, not by original dataset index."""
    n_train = cameras.size if max_views is None else min(max_views, cameras.size)
    print(f"Running depth confidence update over {n_train} training views...")
    logodds = reliability_to_logodds(depth_confidence)

    for i in range(n_train):
        orig_idx = frame_indices[i]
        depth_path = frame_to_depth_map.get(orig_idx)
        if depth_path is None or not depth_path.exists():
            continue

        cam_single = cameras[i:i + 1].to(pipeline.device)
        w, h = int(cam_single.width[0]), int(cam_single.height[0])
        real_depth_m = load_raw_depth(depth_path, target_hw=(h, w))

        attributed, dominant = update_depth_confidence_from_view(
            gaussians, cam_single, real_depth_m, w, h, depth_scale_factor=depth_scale_factor)
        logodds_delta = torch.clamp(attributed, min=-10.0, max=10.0)
        logodds_delta[~dominant] = 0.0
        logodds = logodds + logodds_delta

    depth_confidence = logodds_to_reliability(logodds)
    print(f"Depth confidence update done. Mean: {depth_confidence.mean().item():.3f}")
    return depth_confidence


def render_metric_depth(gaussians, viewmat: torch.Tensor, K: torch.Tensor, w: int, h: int,
                         depth_scale_factor: float = 1.0):
    """Renders the actual reconstructed depth (distance from camera to each
    Gaussian, alpha-composited -- same math as splatfacto's own depth
    output), scaled to real metric units via depth_scale_factor. No
    confidence/reliability tracking involved -- just the geometry."""
    cam_pos = torch.linalg.inv(viewmat[0])[:3, 3]
    dists = (gaussians["means"] - cam_pos).norm(dim=1)
    depth_colors = dists.unsqueeze(-1).repeat(1, 3)
    render, _, _ = render_our_own(gaussians, viewmat, K, w, h, colors_override=depth_colors)
    return render[0, :, :, 0].cpu().numpy() * depth_scale_factor


def render_depth_confidence_scalar(gaussians, depth_confidence, viewmat, K, w, h):
    """Renders per-Gaussian depth unreliability as a raw (H,W) float scalar
    map, no normalization or colormap applied -- lets the caller choose a
    shared normalization range across multiple renders for an honest
    side-by-side comparison, rather than each render being independently
    stretched to its own value range (which can make two genuinely
    different underlying uncertainty levels look visually identical)."""
    unreliable_avg = (1.0 - depth_confidence).mean(dim=1)  # (N,) view-independent
    colors = unreliable_avg.unsqueeze(-1).repeat(1, 3)
    render, _, _ = render_our_own(gaussians, viewmat, K, w, h, colors_override=colors)
    return render[0, :, :, 0].cpu().numpy()


def render_depth_confidence_heatmap(gaussians, depth_confidence, viewmat, K, w, h, colormap="jet"):
    """Renders per-Gaussian depth confidence as a heatmap.

    """
    import matplotlib.cm as cm

    unreliable_avg = (1.0 - depth_confidence).mean(dim=1)  # (N,) view-independent
    colors = unreliable_avg.unsqueeze(-1).repeat(1, 3)
    render, _, _ = render_our_own(gaussians, viewmat, K, w, h, colors_override=colors)
    scalar_map = render[0, :, :, 0].cpu().numpy()

    lo, hi = np.percentile(scalar_map, [1, 99])
    print(f"Heatmap raw value range: min={scalar_map.min():.4f} max={scalar_map.max():.4f} "
          f"1st-99th pctile=[{lo:.4f}, {hi:.4f}] std={scalar_map.std():.4f}")
    normalized = np.clip((scalar_map - lo) / max(hi - lo, 1e-6), 0, 1)

    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(normalized)
    return (heatmap_rgba[..., :3] * 255).astype(np.uint8)


def build_basename_to_index(full_meta: dict) -> dict:
    """Maps each frame's filename to its index in full_meta['frames'] --
    needed to match a round's own training cameras (indexed 0..n_revealed-1)
    back to the original full-dataset index that depth/canonical-camera
    lookups are keyed by."""
    return {Path(fr["file_path"]).name: i for i, fr in enumerate(full_meta["frames"])}


def get_original_indices_for_training_set(pipeline, basename_to_index: dict) -> list:
    """Maps each of the model's own actual training cameras back to its
    original full-dataset index, by matching filenames. Needed because
    pipeline.datamanager.train_dataset.cameras is indexed 0..n_revealed-1
    for whatever subset THIS round trained on, not by original index."""
    image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
    indices = []
    for path in image_filenames:
        name = Path(path).name
        if name not in basename_to_index:
            raise KeyError(f"Training image {name} not found in full dataset -- "
                            f"basename matching assumption may be wrong.")
        indices.append(basename_to_index[name])
    return indices


if __name__ == "__main__":
    main()
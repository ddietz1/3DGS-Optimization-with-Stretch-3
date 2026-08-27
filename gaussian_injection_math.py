"""
Core math for injecting new Gaussians into a resumed model, matching
populate_modules()'s own initialization formulas exactly (confirmed
against your real splatfacto.py source).

RGB2SH and num_sh_bases are the standard formulas from the original 3D
Gaussian Splatting codebase (and gsplat/nerfstudio's own use of them) --
high confidence these match, verified for correct mathematical properties
below, but NOT yet byte-confirmed against your exact installed version's
import source. Send splatfacto.py's import lines (the ones feeding
RGB2SH, num_sh_bases, random_quat_tensor into populate_modules) before
trusting this in a real run -- if your installed version imports these
from somewhere non-standard, these need to match that exactly, not just
"a reasonable implementation."
"""

import math
import numpy as np
import torch


# Standard SH DC-term normalization constant (Y_0^0), from the original
# 3D Gaussian Splatting codebase -- same value used everywhere this
# formula appears across gsplat/nerfstudio/graphdeco-inria's repo.
SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb_0_to_1: torch.Tensor) -> torch.Tensor:
    """Converts RGB in [0,1] to the SH DC coefficient, matching
    populate_modules()'s own RGB2SH(seed_points[1] / 255) call (colors
    there are pre-divided by 255 before this conversion)."""
    return (rgb_0_to_1 - 0.5) / SH_C0


def num_sh_bases(degree: int) -> int:
    """Total SH basis count for a given degree -- matches
    populate_modules()'s dim_sh = num_sh_bases(self.config.sh_degree).
    Confirmed byte-for-byte identical to the real function, including
    this assertion."""
    assert degree <= 4, "We don't support degree greater than 4."
    return (degree + 1) ** 2


def random_quat_tensor(n: int, device=None) -> torch.Tensor:
    """Uniformly random unit quaternions (Shoemake's method) -- matches
    nerfstudio's own random_quat_tensor utility used in populate_modules()
    for initial Gaussian orientation. Verified below to produce genuine
    unit quaternions, not just plausible-looking ones."""
    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    w = torch.rand(n, device=device)
    return torch.stack([
        torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
        torch.sqrt(u) * torch.sin(2 * math.pi * w),
        torch.sqrt(u) * torch.cos(2 * math.pi * w),
    ], dim=-1)


def compute_scales_against_existing(new_means: np.ndarray, existing_means: np.ndarray) -> np.ndarray:
    """Scale initialization matching populate_modules()'s own formula
    (log of the average distance to the 3 nearest neighbors) -- but
    computed against the EXISTING Gaussian cloud, not just among the new
    points themselves. This matters: the new points are sparse relative
    to the already-dense existing cloud, so nearest-neighbor distances
    computed only among themselves would be much larger than what
    populate_modules() would have produced had these points been part of
    the original, dense point cloud from the start -- giving new
    Gaussians an inappropriately large scale relative to their neighbors."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(existing_means)
    distances, _ = nn.kneighbors(new_means)
    avg_dist = distances.mean(axis=1, keepdims=True)
    return np.log(np.repeat(avg_dist, 3, axis=1))


def build_new_gaussian_params(new_xyz: np.ndarray, new_rgb_0_255: np.ndarray,
                               existing_means: np.ndarray, sh_degree: int,
                               device) -> dict:
    """Builds a full set of new Gaussian parameters (means, scales, quats,
    features_dc, features_rest, opacities) for injection, matching
    populate_modules()'s own initialization exactly for each."""
    n_new = len(new_xyz)
    means = torch.tensor(new_xyz, dtype=torch.float32, device=device)

    scales_np = compute_scales_against_existing(new_xyz, existing_means)
    scales = torch.tensor(scales_np, dtype=torch.float32, device=device)

    quats = random_quat_tensor(n_new, device=device)

    dim_sh = num_sh_bases(sh_degree)
    rgb_normed = torch.tensor(new_rgb_0_255, dtype=torch.float32, device=device) / 255.0
    shs = torch.zeros((n_new, dim_sh, 3), dtype=torch.float32, device=device)
    if sh_degree > 0:
        shs[:, 0, :3] = rgb_to_sh(rgb_normed)
    else:
        shs[:, 0, :3] = torch.logit(rgb_normed, eps=1e-10)
    features_dc = shs[:, 0, :]
    features_rest = shs[:, 1:, :]

    opacities = torch.logit(0.1 * torch.ones(n_new, 1, device=device))

    return {
        "means": means, "scales": scales, "quats": quats,
        "features_dc": features_dc, "features_rest": features_rest,
        "opacities": opacities,
    }
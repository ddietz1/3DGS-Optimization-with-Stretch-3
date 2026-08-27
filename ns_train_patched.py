"""
Drop-in replacement for the `ns-train` command that allows for retraining from a model checkpoint.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_INJECTED_POINTS_BASELINE = None
if "--injected-points-baseline" in sys.argv:
    _idx = sys.argv.index("--injected-points-baseline")
    _INJECTED_POINTS_BASELINE = int(sys.argv[_idx + 1])
    del sys.argv[_idx:_idx + 2]

_DATASET_DIR = None
if "--data" in sys.argv:
    _idx = sys.argv.index("--data")
    _DATASET_DIR = sys.argv[_idx + 1]


from nerfstudio.engine.trainer import Trainer

_original_load_checkpoint = Trainer._load_checkpoint


def _inject_new_gaussians(trainer: Trainer, dataset_dir_str: str, baseline_count: int):
    """Reads sparse_pc.ply, finds points beyond baseline_count, and injects them as real new
    Gaussian parameters + correctly-extended optimizer state."""
    import numpy as np
    import torch
    from plyfile import PlyData
    from gsplat.strategy.ops import _update_param_with_optimizer
    from gaussian_injection_math import build_new_gaussian_params

    ply_path = Path(dataset_dir_str) / "sparse_pc.ply"
    if not ply_path.exists():
        print(f"[ns_train_patched] No point cloud at {ply_path} -- skipping Gaussian injection")
        return

    try:
        all_points = PlyData.read(str(ply_path))["vertex"].data
    except Exception as e:
        print(f"[ns_train_patched] ERROR: could not read {ply_path} ({e}) -- "
              f"SKIPPING Gaussian injection for this round. Training will "
              f"continue normally otherwise, but this round's accumulated "
              f"depth points will NOT reach the model.")
        return
    total_now = len(all_points)
    if total_now <= baseline_count:
        print(f"[ns_train_patched] No new points since baseline ({baseline_count}), "
              f"skipping injection")
        return

    new_points = all_points[baseline_count:total_now]
    new_xyz = np.stack([new_points["x"], new_points["y"], new_points["z"]], axis=1).astype(np.float32)
    new_rgb = np.stack([new_points["red"], new_points["green"], new_points["blue"]], axis=1).astype(np.float32)

    model = trainer.pipeline.model
    device = model.gauss_params["means"].device
    existing_means = model.gauss_params["means"].detach().cpu().numpy()
    sh_degree = model.config.sh_degree

    new_values = build_new_gaussian_params(new_xyz, new_rgb, existing_means, sh_degree, device)
    n_new = len(new_xyz)

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, new_values[name]]))

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(
        param_fn, optimizer_fn, model.gauss_params, trainer.optimizers.optimizers,
        names=list(new_values.keys()),
    )

    print(f"[ns_train_patched] Injected {n_new} new Gaussians from depth-seeded points "
          f"(point cloud grew from {baseline_count} to {total_now}). New total: "
          f"{model.gauss_params['means'].shape[0]} Gaussians.")


def _patched_load_checkpoint(self):
    model = self.pipeline.model
    has_gauss_params = hasattr(model, "gauss_params")
    old_params = dict(model.gauss_params) if has_gauss_params else {}

    _original_load_checkpoint(self)  # runs the normal load

    if not has_gauss_params:
        return  # not a splatfacto-style model -- nothing to fix

    new_params = model.gauss_params
    rebound = []
    for name, old_param in old_params.items():
        if name not in self.optimizers.optimizers:
            continue
        optimizer = self.optimizers.optimizers[name]
        new_param = new_params[name]
        if old_param is new_param:
            continue  # shapes matched and nothing was actually replaced -- nothing to fix
        for group in optimizer.param_groups:
            if old_param in optimizer.state:
                param_state = optimizer.state.pop(old_param)
                optimizer.state[new_param] = param_state
            group["params"] = [new_param]
        rebound.append(name)

    if rebound:
        print(f"[ns_train_patched] Rebound optimizer state after checkpoint "
              f"load for: {rebound} -- resumed training should now actually "
              f"update these parameters (see file docstring for why this "
              f"was needed).")

    if _INJECTED_POINTS_BASELINE is not None and _DATASET_DIR is not None:
        _inject_new_gaussians(self, _DATASET_DIR, _INJECTED_POINTS_BASELINE)


Trainer._load_checkpoint = _patched_load_checkpoint

from nerfstudio.scripts.train import entrypoint

if __name__ == "__main__":
    entrypoint()
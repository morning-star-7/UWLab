# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Interactive World Simulator's ``main.py`` (unchanged) with RPM Lab runtime fixes. Used by ``train_iws.py``:

    python iws_main.py <iws repo> <hydra overrides...>

Fix: IWS's ``Attention`` restricts scaled-dot-product attention to the flash kernel on every GPU whose compute
capability is x.0 (meant for the A100; it also catches the H100) and lists no fallback. Flash attention takes fp16/bf16
only and IWS trains in fp32, so the dynamics model (stage 2/3) fails there with "No available kernel"; the A6000 (8.6)
takes IWS's other branch and never saw it. The memory-efficient and math kernels are added as fallbacks: flash is
still used whenever it applies.

Evaluation hooks (``train_iws.py --eval``), off unless their environment variable is set:

* ``IWS_EVAL_ACTIONS=hold`` replaces every action of a validation rollout with its first one (the arm told to stay
  where it starts). A model that follows its actions scores well below its true-action rollout; one that ignores them
  scores the same.
* ``IWS_HORIZON_METRICS=<file.json>`` also measures the validation rollouts per frame: PSNR (on [0, 1] images: IWS's
  own uses data_range 2.0, 6 dB higher) and SSIM over ranges of the rollout horizon, the same for a baseline that
  repeats the first frame (nothing moves), logged as ``<namespace>/h<from>-<to>/...`` and written to the file.
"""

import json
import math
import os
import runpy
import sys

from torch.nn.attention import SDPBackend, sdpa_kernel

FALLBACKS = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)


def sdpa_kernel_with_fallbacks(backends, *args, **kwargs):
    backends = [backends] if isinstance(backends, SDPBackend) else list(backends)
    return sdpa_kernel(backends + [b for b in FALLBACKS if b not in backends], *args, **kwargs)


HORIZON_RANGES = ((1, 10), (11, 50), (51, 100), (101, 200), (201, 400))
"""Rollout frames (1 = the first predicted frame; frame 0 is the given one) measured together: 20 fps."""


def horizon_metrics(outputs: list) -> dict:
    """Per-frame PSNR and per-range PSNR / SSIM of validation rollouts, and of repeating the first frame."""
    import torch
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    pred = torch.cat([p for p, _ in outputs], 1).float().clamp(0, 1)  # (T, B, C, H, W)
    gt = torch.cat([g for _, g in outputs], 1).float().clamp(0, 1)
    still = gt[:1].expand_as(gt)

    def psnr(mse):
        return 10 * math.log10(1.0 / max(float(mse), 1e-10))

    result = {"frames": gt.shape[0], "videos": gt.shape[1], "per_frame_psnr": {}, "ranges": {}}
    for name, x in (("model", pred), ("first_frame", still)):
        mse = ((x - gt) ** 2).mean(dim=(1, 2, 3, 4))  # (T,)
        result["per_frame_psnr"][name] = [round(psnr(m), 3) for m in mse]
        for lo, hi in HORIZON_RANGES:
            if lo >= gt.shape[0]:
                continue
            sl = slice(lo, min(hi, gt.shape[0] - 1) + 1)
            key = f"h{lo}-{min(hi, gt.shape[0] - 1)}"
            frames = x[sl].flatten(0, 1)
            result["ranges"].setdefault(key, {})[name] = {
                "psnr": round(psnr(mse[sl].mean()), 3),
                "ssim": round(float(ssim(frames, gt[sl].flatten(0, 1), data_range=1.0)), 4),
            }
    return result


def add_eval_hooks():
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel

    if os.environ.get("IWS_EVAL_ACTIONS", "true") == "hold":
        validation_step = LatentWorldModel.validation_step

        def hold_actions(self, batch, *args, **kwargs):
            batch = dict(batch)
            batch["action"] = batch["action"][:, :1].expand_as(batch["action"]).clone()
            return validation_step(self, batch, *args, **kwargs)

        LatentWorldModel.validation_step = hold_actions

    path = os.environ.get("IWS_HORIZON_METRICS")
    if path:
        epoch_end = LatentWorldModel.on_validation_epoch_end

        def with_horizon_metrics(self, namespace: str = "validation"):
            if self.validation_step_outputs:
                result = horizon_metrics(self.validation_step_outputs)
                with open(path, "w") as f:
                    json.dump(result, f, indent=1)
                print(json.dumps(result["ranges"], indent=1), flush=True)
                for key, by_name in result["ranges"].items():
                    for name, values in by_name.items():
                        for metric, value in values.items():
                            self.log(f"{namespace}/{key}/{name}_{metric}", value, on_step=False, on_epoch=True)
            return epoch_end(self, namespace)

        LatentWorldModel.on_validation_epoch_end = with_horizon_metrics


def main():
    iws = os.path.abspath(sys.argv[1])
    sys.argv = [os.path.join(iws, "main.py"), *sys.argv[2:]]
    sys.path.insert(0, iws)  # as when main.py is run directly

    import interactive_world_sim.algorithms.models.attention as attention

    attention.sdpa_kernel = sdpa_kernel_with_fallbacks
    add_eval_hooks()
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

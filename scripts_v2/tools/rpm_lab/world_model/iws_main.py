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

Fix: IWS's ``main.py`` repoints ``<project>/latest-run`` at each new run (unlink, then symlink). Two jobs starting
together race on it and the loser dies with FileExistsError (a 2026-11-16 eval job did); the link is a convenience, so
the loser leaves it to the winner.

``IWS_NORMALIZER_DATA=<dataset dir>`` (``train_iws.py --normalizer_data``) takes the training set's action normalizer
from that dataset's train split instead of its own. IWS refits the normalizer to the training set at every new stage,
overriding the loaded checkpoint's: a stage trained on a subset (e.g. stage 3 on the real episodes alone, after a
co-trained stage 2) would otherwise rescale the actions the dynamics model learned (here by 9-42 % per joint).

``IWS_REAL_SHARE=<0..1>`` (``train_iws.py --real_share``) sets the co-training ratio at sampling time: every training
sample is real with this probability and sim otherwise (drawn with replacement, uniformly over the windows of its
domain, from the dataset's per-step ``data/domain``). Without it IWS shuffles all windows, so the real share is the
real frames' share of the dataset.

Evaluation hooks (``train_iws.py --eval``), off unless their environment variable is set:

* ``IWS_EVAL_ACTIONS=hold`` replaces every action of a validation rollout with its first one (the arm told to stay
  where it starts). A model that follows its actions scores well below its true-action rollout; one that ignores them
  scores the same.
* ``IWS_HORIZON_METRICS=<file.json>`` also measures the validation rollouts per frame: PSNR (on [0, 1] images: IWS's
  own uses data_range 2.0, 6 dB higher) and SSIM over ranges of the rollout horizon, the same for a baseline that
  repeats the first frame (nothing moves), logged as ``<namespace>/h<from>-<to>/...`` and written to the file.
* ``IWS_EVAL_START_EVERY=<frames>`` (with ``IWS_ROLLOUT_DIR``, ``IWS_EVAL_HORIZON``) rolls out from several points of
  every validation episode and writes each rollout as a video: see ``add_multi_start_eval``.
"""

import copy
import glob
import json
import math
import os
import pathlib
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

    if add_multi_start_eval():  # writes its own per-horizon metrics
        return
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


MIN_ROLLOUT = 20
"""A multi-start rollout begins only where at least this many frames of the episode are left (1 s)."""


def rollout_starts(episode_ends, every: int) -> list:
    """(episode, first frame in the episode) of every rollout: frame 0, then every ``every`` frames."""
    starts, begin = [], 0
    for episode, end in enumerate(int(e) for e in episode_ends):
        length = end - begin
        starts += [(episode, s) for s in range(0, max(1, length - MIN_ROLLOUT + 1), every)]
        begin = end
    return starts


def write_rollout(path: str, pred, real):
    """One rollout as a video, prediction left and real right (near lossless, full chroma: the fish is 11 px)."""
    import av
    import numpy as np

    frames = np.concatenate([pred, real], axis=2)  # (T, H, 2 W, 3) uint8
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "8", "preset": "medium"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
        container.mux(stream.encode())


def add_multi_start_eval():
    """``IWS_EVAL_START_EVERY=<frames>``: validation rolls out from several points of each episode, not only its start.

    IWS's validation set is one item per episode (its first ``val_horizon`` frames), which gives one rollout per
    episode: weak statistics. Here an item is (episode, start frame), ``val_horizon`` frames from there (the last frame
    repeated past the episode's end, as IWS does). Every rollout is written to ``IWS_ROLLOUT_DIR`` as
    ``ep<episode>_s<start>.mp4`` (its real frames only, no padding) with an ``index.json``; PSNR by horizon is
    accumulated per rollout (padding excluded) and written to ``IWS_HORIZON_METRICS``. IWS's own end-of-validation
    logging (one wandb video per rollout, FVD over all of them in memory) is skipped.
    """
    every = int(os.environ.get("IWS_EVAL_START_EVERY", "0"))
    if every <= 0:
        return False
    import numpy as np
    import torch
    from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
    from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset

    out_dir = os.environ["IWS_ROLLOUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.environ.get("IWS_HORIZON_METRICS")
    dataset_len, dataset_getitem = RealAlohaDataset.__len__, RealAlohaDataset.__getitem__

    def starts_of(self):
        if getattr(self, "_rollout_starts_every", None) != every:
            self._rollout_starts = rollout_starts(self.replay_buffer.episode_ends[:], every)
            self._rollout_starts_every = every
        return self._rollout_starts

    def length(self):
        return len(starts_of(self)) if self.is_val else dataset_len(self)

    def getitem(self, idx):
        if not self.is_val:
            return dataset_getitem(self, idx)
        if self.skip_frame != 1:
            raise RuntimeError("multi-start evaluation assumes dataset.skip_frame = 1")
        episode, start = starts_of(self)[idx]
        ends = self.replay_buffer.episode_ends
        first = (int(ends[episode - 1]) if episode > 0 else 0) + start
        last = min(int(ends[episode]), first + self.val_horizon)
        sample = {}
        for key in self.sampler.keys:
            x = self.replay_buffer[key][first:last]
            if x.shape[0] < self.val_horizon:
                x = np.concatenate([x, np.repeat(x[-1:], self.val_horizon - x.shape[0], axis=0)], axis=0)
            sample[key] = x
            sample[f"{key}_final"] = x[-1]
        sample["is_early_stop"] = False
        sample["rel_stop_idx"] = self.val_horizon - 1
        data = self._sample_to_data(sample)
        data["rollout"] = torch.tensor([episode, start, last - first])
        return data

    RealAlohaDataset.__len__, RealAlohaDataset.__getitem__ = length, getitem

    validation_step = LatentWorldModel.validation_step  # already wrapped by IWS_EVAL_ACTIONS=hold, if set
    index, sums = [], {}

    def to_uint8(x):  # (T, C, H, W) in [0, 1]
        return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()

    def step(self, batch, *args, **kwargs):
        if not index:  # the checkpoint's normalizer must be in use (the training set's), not the validation set's
            stats = self.normalizer["action"].params_dict["input_stats"]
            low, high = ([round(float(v), 3) for v in stats[k].flatten()] for k in ("min", "max"))
            print(f"Action normalizer in use: min {low}, max {high}", flush=True)
        before = len(self.validation_step_outputs)
        result = validation_step(self, batch, *args, **kwargs)
        for pred, real in self.validation_step_outputs[before:]:  # (T, B, C, H, W) on the CPU
            if pred.shape[2] != 3:
                raise RuntimeError("multi-start evaluation writes one camera per model")
            for b, (episode, start, frames) in enumerate(batch["rollout"].tolist()):
                p, r = pred[:frames, b].float().clamp(0, 1), real[:frames, b].float().clamp(0, 1)
                name = f"ep{episode:03d}_s{start:04d}.mp4"
                write_rollout(os.path.join(out_dir, name), to_uint8(p), to_uint8(r))
                index.append({"file": name, "episode": episode, "start": start, "frames": frames})
                for who, x in (("model", p), ("first_frame", r[:1].expand_as(r))):
                    mse = ((x - r) ** 2).mean(dim=(1, 2, 3)).double()
                    total, count = sums.setdefault(who, (torch.zeros(self.val_horizon_frames), torch.zeros(self.val_horizon_frames)))
                    total[:frames] += mse
                    count[:frames] += 1
        del self.validation_step_outputs[before:]  # nothing kept in memory: IWS's epoch-end logging has nothing to do
        return result

    def epoch_end(self, namespace: str = "validation"):
        index.sort(key=lambda r: (r["episode"], r["start"]))
        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump({"start_every": every, "fps": 20, "layout": "prediction | real", "rollouts": index}, f, indent=1)
        result = {"rollouts": len(index), "start_every": every, "per_frame_psnr": {}, "ranges": {}}
        for who, (total, count) in sums.items():
            seen = count > 0
            mse = total[seen] / count[seen]
            result["per_frame_psnr"][who] = [round(10 * math.log10(1.0 / max(float(m), 1e-10)), 3) for m in mse]
            for lo, hi in HORIZON_RANGES:
                if lo >= len(mse):
                    continue
                hi = min(hi, len(mse) - 1)
                # frames weighted by how many rollouts reach them
                part = float(total[lo : hi + 1].sum() / count[lo : hi + 1].sum())
                result["ranges"].setdefault(f"h{lo}-{hi}", {})[who] = {"psnr": round(10 * math.log10(1.0 / max(part, 1e-10)), 3)}
        print(f"{len(index)} rollouts written to {out_dir}", flush=True)
        print(json.dumps(result["ranges"], indent=1), flush=True)
        if metrics_path:
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=1)
        for key, by_name in result["ranges"].items():
            for who, values in by_name.items():
                self.log(f"{namespace}/{key}/{who}_psnr", values["psnr"], on_step=False, on_epoch=True)
        self.log(f"{namespace}/rollouts", float(len(index)), on_step=False, on_epoch=True)

    LatentWorldModel.val_horizon_frames = int(os.environ["IWS_EVAL_HORIZON"])
    LatentWorldModel.validation_step = step
    LatentWorldModel.on_validation_epoch_end = epoch_end
    return True


def tolerate_latest_run_race():
    symlink_to = pathlib.Path.symlink_to

    def symlink_unless_taken(self, target, *args, **kwargs):
        try:
            return symlink_to(self, target, *args, **kwargs)
        except FileExistsError:
            if self.name != "latest-run":
                raise
            print(f"{self} was just repointed by another run: left as is", flush=True)

    pathlib.Path.symlink_to = symlink_unless_taken


def add_normalizer_hook():
    source = os.environ.get("IWS_NORMALIZER_DATA")
    if not source:
        return
    import zarr
    from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset

    get_normalizer = RealAlohaDataset.get_normalizer

    def normalizer_from_source(self, *args, **kwargs):
        if getattr(self, "is_val", False):
            return get_normalizer(self, *args, **kwargs)
        if self.lowdim_keys:
            raise RuntimeError(f"IWS_NORMALIZER_DATA covers the action only, not {self.lowdim_keys}")
        (cache,) = glob.glob(os.path.join(source, "train", "cache*.zarr.zip"))
        with zarr.ZipStore(cache, mode="r") as store:
            actions = zarr.open(store, mode="r")["data/action"][:]
        proxy = copy.copy(self)
        proxy.replay_buffer = {"action": actions}  # all that get_normalizer reads without low-dim observations
        low, high = ([round(float(v), 3) for v in a] for a in (actions.min(0), actions.max(0)))
        print(f"Action normalizer from {cache}: min {low}, max {high}", flush=True)
        return get_normalizer(proxy, *args, **kwargs)

    RealAlohaDataset.get_normalizer = normalizer_from_source


def add_real_share_sampler():
    """``IWS_REAL_SHARE``: the training loader draws each sample from the real windows with this probability."""
    if not os.environ.get("IWS_REAL_SHARE"):
        return
    share = float(os.environ["IWS_REAL_SHARE"])
    import numpy as np
    import torch
    from interactive_world_sim.experiments.exp_base import BaseLightningExperiment

    def build_training_loader(self):
        dataset = self._build_dataset("training")
        sampler = dataset.sampler
        starts = sampler.indices[:: sampler.skip_idx][: len(dataset), 0]  # first buffer step of every training window
        real = np.asarray(dataset.replay_buffer["domain"])[starts] == 0
        n_real, n_sim = int(real.sum()), int((~real).sum())
        if not n_real or not n_sim:
            raise RuntimeError(f"IWS_REAL_SHARE needs both domains in the training set: {n_real} real, {n_sim} sim windows")
        weights = np.where(real, share / n_real, (1 - share) / n_sim)
        print(f"Training samples: real with probability {share} ({n_real} real, {n_sim} sim windows; "
              f"{100 * n_real / len(real):.1f} % real without it)", flush=True)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
            sampler=torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), len(dataset), replacement=True),
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=1,
        )  # IWS's loader, with the sampler in place of shuffle

    BaseLightningExperiment._build_training_loader = build_training_loader


def main():
    iws = os.path.abspath(sys.argv[1])
    sys.argv = [os.path.join(iws, "main.py"), *sys.argv[2:]]
    sys.path.insert(0, iws)  # as when main.py is run directly

    import interactive_world_sim.algorithms.models.attention as attention

    attention.sdpa_kernel = sdpa_kernel_with_fallbacks
    tolerate_latest_run_race()
    add_normalizer_hook()
    add_real_share_sampler()
    add_eval_hooks()
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

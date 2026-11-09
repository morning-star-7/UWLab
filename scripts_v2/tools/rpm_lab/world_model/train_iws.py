# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Interactive World Simulator (``third_party/interactive_world_sim``) on an RPM Lab dataset.

Runs IWS's own ``main.py`` with the RPM Lab settings: the dataset's camera(s) and resolution (from its
``manifest.json``), Lightning's 7-dim action, and the three training stages of the IWS README:

1. autoencoder (encoder + consistency decoder): reconstruction should be near perfect before stage 2;
2. latent dynamics, from a stage-1 checkpoint (``--load_ae``);
3. decoder fine-tuning on noisy latents, from a stage-2 checkpoint.

Runs go to ``<runs>/<name>/`` (Hydra outputs, checkpoints, offline wandb logs; sync later with ``wandb sync`` if
wanted). Use the IWS venv::

    P=/data/cui00191/rpm_lab_envs/iws/bin/python
    $P scripts_v2/tools/rpm_lab/world_model/train_iws.py --stage 1 --name cf_r40_s500_ae --data $W/data/cook_fish_r40_s500
    $P scripts_v2/tools/rpm_lab/world_model/train_iws.py --stage 2 --name cf_r40_s500_dyn --data $W/data/cook_fish_r40_s500 \\
        --load_ae $W/runs/cf_r40_s500_ae/checkpoints/<ckpt>
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
IWS = os.path.join(REPO, "third_party", "interactive_world_sim")
VIEWS_OF_TWO_CAMERA_SET = {
    "both": ["camera_0_color", "camera_1_color"],
    "scene": ["camera_0_color"],
    "wrist": ["camera_1_color"],
}
ACTION_DIM = 7

STAGES = {
    # stage: (horizon, val_horizon, batch size, extra algorithm overrides): the IWS README's T-pushing recipe, with
    # stage 3's batch halved (16 runs out of memory on a 48 GB A6000)
    1: (1, 1, 1, []),
    2: (10, 200, 4, ["algorithm.noise_scheduler.loss_weighting=uniform", "algorithm.sampling_strategy=terminal_only"]),
    3: (1, 200, 8, ["algorithm.noise_scheduler.loss_weighting=uniform", "algorithm.sampling_strategy=terminal_only"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--name", required=True, help="Run name (directory under --runs).")
    parser.add_argument("--data", required=True, help="Dataset from build_iws_dataset.py.")
    parser.add_argument("--runs", default="/data/cui00191/rpm_lab_wm/runs")
    parser.add_argument(
        "--views",
        choices=list(VIEWS_OF_TWO_CAMERA_SET),
        default=None,
        help="Only for a two-camera dataset: which camera(s) to model. 'both' stacks them in one model, whose "
        "validation then skips FVD (I3D takes 3 channels); IWS's released models are one per camera.",
    )
    parser.add_argument("--load_ae", default=None, help="Checkpoint of the previous stage (stages 2 and 3).")
    parser.add_argument("--max_steps", type=int, default=1_000_005)
    parser.add_argument("--batch_size", type=int, default=None, help="Default per stage: 1 / 4 / 8.")
    parser.add_argument("--val_every", type=int, default=None, help="Validation interval (default 6000 / 30000).")
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true", help="Print the command only.")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides, passed through to IWS.")
    args = parser.parse_args()

    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)
    resolution, cameras = manifest["resolution"], manifest["cameras"]
    if len(cameras) == 1:
        if args.views:
            parser.error(f"--views is for two-camera datasets; this one has {list(cameras.values())}")
        obs_keys = list(cameras)
    else:
        obs_keys = VIEWS_OF_TWO_CAMERA_SET[args.views or "both"]
    if args.stage > 1 and not args.load_ae:
        parser.error(f"stage {args.stage} needs --load_ae (the previous stage's checkpoint)")
    horizon, val_horizon, batch_size, extra = STAGES[args.stage]
    run_dir = os.path.join(args.runs, args.name)
    shape_meta = [f"dataset.shape_meta.obs.{key}.shape=[3,{resolution},{resolution}]" for key in cameras]
    if "camera_0_color" not in cameras:
        shape_meta.append("~dataset.shape_meta.obs.camera_0_color")  # IWS's default lists both cameras
    command = [
        sys.executable,
        os.path.join(IWS, "main.py"),
        f"+name={args.name}",
        f"hydra.run.dir={run_dir}",
        "algorithm=latent_world_model",
        "experiment=exp_latent_dyn",
        "dataset=real_aloha_dataset",
        f"dataset.dataset_dir={os.path.abspath(args.data)}",
        f"dataset.resolution={resolution}",
        f"dataset.horizon={horizon}",
        f"dataset.val_horizon={val_horizon}",
        f"dataset.obs_keys=[{','.join(obs_keys)}]",
        f"dataset.shape_meta.action.shape=[{ACTION_DIM}]",
        *shape_meta,
        f"experiment.training.batch_size={args.batch_size or batch_size}",
        f"experiment.training.max_steps={args.max_steps}",
        "experiment.training.log_every_n_steps=100",
        f"experiment.training.checkpointing.every_n_train_steps={args.ckpt_every}",
        f"experiment.training.data.num_workers={args.workers}",
        f"experiment.validation.data.num_workers={args.workers}",
        "experiment.validation.limit_batch=1.0",
        f"experiment.validation.batch_size={10 if args.stage == 1 else 2}",
        f"experiment.validation.val_every_n_step={args.val_every or (6000 if args.stage == 1 else 30000)}",
        "algorithm.latent_dim=512",
        f"algorithm.action_dim={ACTION_DIM}",
        f"algorithm.training_stage={args.stage}",
        "wandb.entity=rpm_lab",
        "wandb.mode=offline",
        *extra,
        *(["algorithm.metrics=[]"] if len(obs_keys) > 1 else []),
        # quoted: Lightning checkpoint names contain "=", which Hydra's override grammar rejects unquoted
        *([f"algorithm.load_ae='{os.path.abspath(args.load_ae)}'"] if args.load_ae else []),
        *args.overrides,
    ]
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ, WANDB_DIR=run_dir, PYTHONUNBUFFERED="1")
    sys.exit(subprocess.call(command, cwd=IWS, env=env))


if __name__ == "__main__":
    main()

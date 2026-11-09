# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an Interactive World Simulator (IWS) training set from RPM Lab LeRobot datasets, real and sim mixed.

Inputs are LeRobot v3 datasets with the robot PC's ``multisensor_20hz`` features: real recordings, and sim episodes
from ``render_mimic_dataset.py`` + ``export_lerobot_dataset.py``. The output is what IWS's ``real_aloha_dataset``
loads, ``<out>/{train,val}/cache[_res_<R>].zarr.zip``:

* the camera images (``--cameras``), center-cropped to a square and resized to ``--resolution`` like IWS's own
  converter (JPEG 2000, level 50). One camera (default: scene_1) is stored as ``camera_1_color``, the key IWS names its
  cache after; ``both`` stores scene_1 as ``camera_0_color`` and the wrist D405 as ``camera_1_color``;
* ``data/action``: the dataset's 7-dim action (six joint commands, gripper command), at the dataset's 20 Hz;
* ``data/domain``: 0 real, 1 sim, per step (IWS ignores it; kept for domain-conditioned co-training);
* ``meta/episode_ends``;
* ``manifest.json``: which episode of which dataset went where.

With the cache in place IWS never runs its ALOHA conversion, so ``dataset.action_mode`` does not matter. Frames are
streamed into the zip one chunk at a time (no whole-dataset array in memory). Validation holds out real episodes
(``--val_real``): the co-training question is how well the model predicts the real robot.

Runs in the IWS venv (``/data/cui00191/rpm_lab_envs/iws``)::

    python scripts_v2/tools/rpm_lab/world_model/build_iws_dataset.py --out $W/data/cook_fish_r40_s500 \\
        --real $REAL_LEROBOT --sim $SIM_LEROBOT --sim_episodes 500 --val_real 10
"""

import argparse
import concurrent.futures
import glob
import json
import numpy as np
import os
from dataclasses import dataclass

import av
import cv2
import pandas as pd
import zarr
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs

register_codecs()

SCENE, WRIST = "observation.images.world.scene_1", "observation.images.lightning.wrist_1"
CAMERA_SETS = {
    "scene": {"camera_1_color": SCENE},
    "wrist": {"camera_1_color": WRIST},
    "both": {"camera_0_color": SCENE, "camera_1_color": WRIST},
}
"""IWS observation key -> LeRobot feature. IWS names its cache after camera_1_color, so a single camera goes there."""

REAL, SIM = 0, 1


@dataclass
class Episode:
    root: str  # LeRobot dataset root
    index: int  # episode_index in that dataset
    domain: int  # REAL or SIM
    length: int
    meta: dict  # its row of meta/episodes


def list_episodes(root: str, domain: int, cameras: dict) -> list[Episode]:
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    if info["fps"] != 20:
        raise ValueError(f"{root}: {info['fps']} fps, expected the real datasets' 20")
    missing = [f for f in cameras.values() if f not in info["features"]]
    if missing:
        raise ValueError(f"{root} has no {missing}")
    meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "meta/episodes/*/*.parquet")))])
    return [Episode(root, int(r["episode_index"]), domain, int(r["length"]), r.to_dict()) for _, r in meta.iterrows()]


def pick(episodes: list[Episode], n: int | None, rng: np.random.Generator) -> list[Episode]:
    if n is None or n >= len(episodes):
        return list(episodes)
    return [episodes[i] for i in sorted(rng.choice(len(episodes), n, replace=False))]


def actions_of(ep: Episode) -> np.ndarray:
    path = os.path.join(
        ep.root, f"data/chunk-{int(ep.meta['data/chunk_index']):03d}/file-{int(ep.meta['data/file_index']):03d}.parquet"
    )
    data = pd.read_parquet(path, columns=["episode_index", "frame_index", "action"])
    data = data[data["episode_index"] == ep.index].sort_values("frame_index")
    actions = np.stack(data["action"].to_numpy()).astype(np.float32)
    if len(actions) != ep.length:
        raise ValueError(f"{ep.root} episode {ep.index}: {len(actions)} actions, {ep.length} frames")
    return actions


def frames_of(ep: Episode, feature: str, resolution: int):
    """The episode's frames of one camera, square-cropped and resized, in order."""
    key = f"videos/{feature}"
    path = os.path.join(
        ep.root,
        f"videos/{feature}/chunk-{int(ep.meta[key + '/chunk_index']):03d}/file-{int(ep.meta[key + '/file_index']):03d}.mp4",
    )
    start, end = float(ep.meta[key + "/from_timestamp"]), float(ep.meta[key + "/to_timestamp"])
    half_frame = 0.5 / 20
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - half_frame:
                continue
            if t >= end - half_frame:
                break
            img = center_crop(frame.to_ndarray(format="rgb24"), (resolution, resolution))
            yield cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)


def write_split(path: str, episodes: list[Episode], cameras: dict, resolution: int, workers: int):
    """One IWS cache: images streamed into the zip chunk by chunk, low-dim arrays written at the end."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(path)
    lengths = np.array([ep.length for ep in episodes])
    n = int(lengths.sum())
    with zarr.ZipStore(path, mode="w") as store:
        root = zarr.group(store)
        data, meta = root.require_group("data"), root.require_group("meta")
        images = {
            key: data.require_dataset(
                key,
                shape=(n, resolution, resolution, 3),
                chunks=(1, resolution, resolution, 3),
                compressor=Jpeg2k(level=50),
                dtype=np.uint8,
            )
            for key in cameras
        }
        actions, domains = [], []
        offset = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for k, ep in enumerate(episodes):
                actions.append(actions_of(ep))
                domains.append(np.full(ep.length, ep.domain, dtype=np.int8))
                for key, feature in cameras.items():
                    frames = list(frames_of(ep, feature, resolution))
                    if len(frames) != ep.length:
                        raise ValueError(f"{ep.root} episode {ep.index}: {len(frames)} {key} frames, {ep.length} steps")

                    def put(i, img, arr=images[key], base=offset):
                        arr[base + i] = img

                    list(pool.map(put, range(len(frames)), frames))
                offset += ep.length
                print(f"  {os.path.basename(os.path.dirname(path))}: {k + 1}/{len(episodes)} episodes", flush=True)
        action = np.concatenate(actions)
        data.array("action", action, chunks=action.shape, compressor=None, dtype=np.float32)
        domain = np.concatenate(domains)
        data.array("domain", domain, chunks=domain.shape, compressor=None, dtype=np.int8)
        meta.array("episode_ends", np.cumsum(lengths), compressor=None, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output IWS dataset directory (gets train/ and val/).")
    parser.add_argument("--real", nargs="*", default=[], help="Real LeRobot dataset roots.")
    parser.add_argument("--sim", nargs="*", default=[], help="Sim LeRobot dataset roots.")
    parser.add_argument("--real_episodes", type=int, default=None, help="Real training episodes (default: all left).")
    parser.add_argument("--sim_episodes", type=int, default=None, help="Sim training episodes (default: all left).")
    parser.add_argument("--val_real", type=int, default=0, help="Real episodes held out for validation.")
    parser.add_argument("--val_sim", type=int, default=0, help="Sim episodes held out for validation.")
    parser.add_argument("--cameras", choices=list(CAMERA_SETS), default="scene")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32, help="Threads encoding JPEG 2000.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cameras = CAMERA_SETS[args.cameras]
    real = [ep for root in args.real for ep in list_episodes(root, REAL, cameras)]
    sim = [ep for root in args.sim for ep in list_episodes(root, SIM, cameras)]
    val_real = pick(real, args.val_real, rng) if args.val_real else []
    val_sim = pick(sim, args.val_sim, rng) if args.val_sim else []
    held = {(ep.root, ep.index) for ep in val_real + val_sim}
    train_real = pick([ep for ep in real if (ep.root, ep.index) not in held], args.real_episodes, rng)
    train_sim = pick([ep for ep in sim if (ep.root, ep.index) not in held], args.sim_episodes, rng)
    train, val = train_real + train_sim, val_real + val_sim
    if not train or not val:
        raise ValueError(f"Need training and validation episodes (train {len(train)}, val {len(val)})")
    print(
        f"train: {len(train_real)} real + {len(train_sim)} sim episodes, {sum(e.length for e in train)} steps; "
        f"val: {len(val_real)} real + {len(val_sim)} sim"
    )

    name = "cache.zarr.zip" if args.resolution == 128 else f"cache_res_{args.resolution}.zarr.zip"
    for split, episodes in (("train", train), ("val", val)):
        write_split(os.path.join(args.out, split, name), episodes, cameras, args.resolution, args.workers)
    manifest = {
        "resolution": args.resolution,
        "cameras": cameras,
        "action": "lightning_cmd_joint_1..6 (rad), lightning_cmd_gripper (0 open .. 1 closed)",
        "domains": {"0": "real", "1": "sim"},
        "seed": args.seed,
        "splits": {
            split: [{"root": e.root, "episode": e.index, "domain": e.domain, "length": e.length} for e in episodes]
            for split, episodes in (("train", train), ("val", val))
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

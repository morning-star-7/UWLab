# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write rendered sim episodes (``render_mimic_dataset.py`` output) as a LeRobot v3 dataset with the real schema.

The features match what the robot PC's converter (``convert_episode_bag_to_lerobot.py``, ``multisensor_20hz``)
publishes for a Lightning-only session with the wrist D405 and scene_1: ``observation.state`` (19),
``action`` (7), ``observation.images.world.scene_1`` and ``observation.images.lightning.wrist_1`` (h264 video), 20 fps,
``task`` = the language instruction. Needs lerobot 0.5.2 (the robot PC's version; Python >= 3.12), not Isaac Sim:

    /data/cui00191/rpm_lab_envs/lerobot/bin/python scripts_v2/tools/rpm_lab/export_lerobot_dataset.py \\
        --frames render_dir --root out_dataset_dir --repo_id rpm_lab/sim_lightning_pick_place
"""

import argparse
import av
import glob
import json
import numpy as np
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_FIELDS = {"scene_1": "observation.images.world.scene_1", "wrist_1": "observation.images.lightning.wrist_1"}


def read_video(path: str):
    """RGB frames (H, W, 3 uint8) of a video, one at a time."""
    with av.open(path) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True, help="render_mimic_dataset.py output (episode npz + mp4 files).")
    parser.add_argument("--root", required=True, help="Output dataset directory (must not exist).")
    parser.add_argument("--repo_id", default="rpm_lab/sim_lightning")
    parser.add_argument("--vcodec", default="h264", help="Video codec; the real datasets use h264.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz in {args.frames}")
    meta = json.loads(str(np.load(files[0])["meta"]))
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(meta["state_names"]),), "names": meta["state_names"]},
        "action": {"dtype": "float32", "shape": (len(meta["action_names"]),), "names": meta["action_names"]},
    }
    for key, field in IMAGE_FIELDS.items():
        shape = next(read_video(os.path.join(args.frames, meta["videos"][key]))).shape
        features[field] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(meta["fps"]),
        features=features,
        root=args.root,
        use_videos=True,
        vcodec=args.vcodec,
    )
    for n, path in enumerate(files):
        episode = np.load(path)
        info = json.loads(str(episode["meta"]))
        if episode["action"].shape[1] != len(meta["action_names"]):
            raise ValueError(
                f"{path}: action has {episode['action'].shape[1]} values, expected {len(meta['action_names'])}"
            )
        steps = episode["state"].shape[0]
        streams = {key: read_video(os.path.join(args.frames, info["videos"][key])) for key in IMAGE_FIELDS}
        for t in range(steps):
            frame = {
                "observation.state": episode["state"][t],
                "action": episode["action"][t],
                "task": info["instruction"] or info["task"],
            }
            for key, field in IMAGE_FIELDS.items():
                frame[field] = next(streams[key])
            dataset.add_frame(frame)
        for key, stream in streams.items():
            if next(stream, None) is not None:
                raise ValueError(f"{path}: {key} video has more frames than the {steps} states")
        dataset.save_episode()
        print(f"[{n + 1}/{len(files)}] {os.path.basename(path)}: {episode['state'].shape[0]} frames")
    dataset.finalize()

    reloaded = LeRobotDataset(args.repo_id, root=args.root)
    sample = reloaded[0]
    print(f"Wrote {args.root}: {reloaded.num_episodes} episodes, {len(reloaded)} frames")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key:40s} {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()

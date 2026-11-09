# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert raw real Lightning episodes (robot PC ``raw_episodes/episode-*``) into the co-training episode format.

Writes exactly what ``render_mimic_dataset.py`` writes for sim episodes, so ``export_lerobot_dataset.py`` turns real
and sim alike into ``multisensor_20hz`` LeRobot datasets: per episode ``episode_XXXX.npz`` (state, action, meta) and
one near-lossless H.264 video per camera (``scene_1`` L515, ``wrist_1`` D405, 640 x 480 as captured).

Resampled at 20 Hz over the span where teleop commands and both cameras are live (from each camera's second frame:
the first one of an episode can be stale). At each tick:

* images: the newest frame captured at or before the tick (host capture clock);
* action: the newest teleop command (six servoJ joint targets, gripper command 0 open .. 1 closed);
* state: measured joints, flange pose in the controller Base frame (rotation vector), gripper opening and TCP
  force/torque, interpolated to the tick.

Needs ``mcap mcap-ros2-support`` (e.g. the IWS venv) and ffmpeg; no Isaac Sim::

    python scripts_v2/tools/rpm_lab/convert_raw_episodes.py --task cook_fish \\
        --episodes /data/cui00191/raw_episodes/episode-* --out /data/cui00191/WM-cotrain/real_frames/cook_fish
"""

import argparse
import importlib.util
import json
import numpy as np
import os
import subprocess
from multiprocessing import Pool

from scipy.spatial.transform import Rotation, Slerp

from cotrain_schema import ACTION_NAMES, FPS, STATE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "extract_spark_recording", os.path.join(HERE, "extract_spark_recording.py")
)
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)

ARM = "lightning"
CAMERA_KEYS = {"scene_1": "scene_1", "lightning_wrist": "wrist_1"}  # extractor camera name -> episode video key


def write_video(path: str, frames: np.ndarray):
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    proc.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def latest(times: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """Index of the newest sample at or before each tick."""
    return np.clip(np.searchsorted(times, ticks, side="right") - 1, 0, len(times) - 1)


def convert(job: tuple[int, str, str]) -> str:
    index, directory, out = job
    data, meta = extract.read_raw_episode(directory, image_period=0.0, depth=False)
    if ARM not in meta["arms"]:
        raise ValueError(f"{directory}: no {ARM} data")
    with open(os.path.join(directory, "episode_manifest.json")) as f:
        manifest = json.load(f)["episode"]
    cmd_t, q_t = data[f"{ARM}.cmd_t"], data[f"{ARM}.t"]
    image_t = {name: data[f"image.{name}.t"] for name in CAMERA_KEYS}
    start = max(cmd_t[0], q_t[0], *(t[1] for t in image_t.values()))
    end = min(cmd_t[-1], q_t[-1], *(t[-1] for t in image_t.values()))
    ticks = start + np.arange(int((end - start) * FPS) + 1) / FPS

    # action: newest teleop command
    cmd_q = data[f"{ARM}.cmd_q"][latest(cmd_t, ticks)]
    cmd_g = data[f"{ARM}.cmd_gripper"][latest(data[f"{ARM}.cmd_gripper_t"], ticks)]
    action = np.concatenate([cmd_q, cmd_g[:, None]], axis=1)

    # state: measured, interpolated to the tick (some bags log a sample twice with the same time: keep the first)
    keep = np.concatenate([[True], np.diff(q_t) > 0])
    q_t = q_t[keep]
    q = np.stack([np.interp(ticks, q_t, data[f"{ARM}.q"][keep, j]) for j in range(6)], axis=1)
    pos = np.stack([np.interp(ticks, q_t, data[f"{ARM}.flange_pos"][keep, j]) for j in range(3)], axis=1)
    quat = data[f"{ARM}.flange_quat"][keep]  # w x y z
    rotvec = Slerp(q_t, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(np.clip(ticks, q_t[0], q_t[-1])).as_rotvec()
    gripper = np.interp(ticks, q_t, data[f"{ARM}.gripper"][keep])
    wrench_t, wrench = data[f"{ARM}.wrench_t"], data[f"{ARM}.wrench"]
    ft = (
        np.stack([np.interp(ticks, wrench_t, wrench[:, j]) for j in range(6)], axis=1)
        if len(wrench_t)
        else np.zeros((len(ticks), 6))
    )
    state = np.concatenate([q, pos, rotvec, gripper[:, None], ft], axis=1)

    videos = {}
    for name, key in CAMERA_KEYS.items():
        frames = data[f"image.{name}.rgb"][latest(image_t[name], ticks)]
        videos[key] = f"episode_{index:04d}_{key}.mp4"
        write_video(os.path.join(out, videos[key]), frames)
    info = {
        "source": os.path.abspath(directory),
        "demo": manifest.get("episode_id"),
        "task": manifest.get("task_name"),
        "instruction": manifest.get("language_instruction", ""),
        "domain": "real",
        "fps": FPS,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "videos": videos,
        "frame_lag_s": {
            key: float(np.mean(ticks - image_t[name][latest(image_t[name], ticks)]))
            for name, key in CAMERA_KEYS.items()
        },
    }
    np.savez(
        os.path.join(out, f"episode_{index:04d}.npz"),
        state=state.astype(np.float32),
        action=action.astype(np.float32),
        meta=np.array(json.dumps(info)),
    )
    return f"[{index}] {info['demo']}: {len(ticks)} steps ({len(ticks) / FPS:.1f} s)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", required=True, help="Raw episode directories.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default=None, help="Keep only episodes whose manifest task_name matches.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    episodes = []
    for directory in sorted(args.episodes):
        manifest = os.path.join(directory, "episode_manifest.json")
        if not os.path.isfile(manifest):
            continue
        with open(manifest) as f:
            task = json.load(f)["episode"].get("task_name")
        if args.task is None or task == args.task:
            episodes.append(directory)
    if not episodes:
        raise FileNotFoundError("No matching raw episodes")
    os.makedirs(args.out, exist_ok=True)
    print(f"Converting {len(episodes)} episodes -> {args.out}", flush=True)
    jobs = [(i, d, args.out) for i, d in enumerate(episodes)]
    with Pool(args.workers) as pool:
        for line in pool.imap(convert, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()

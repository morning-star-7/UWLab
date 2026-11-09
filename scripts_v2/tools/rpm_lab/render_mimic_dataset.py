# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a recorded/generated Isaac Lab demo dataset (HDF5 with per-step states) like the real Lightning recordings.

For every step the sim state is restored from the dataset and both cameras are rendered: the scene_1 L515 at its
calibrated pose and Lightning's wrist D405. The pinhole renders are then warped into each real camera's model
(principal point, plus lens distortion for the L515), since Omniverse cameras ignore the principal point, and scaled
by each camera's color gain (``CAMERA_COLOR_GAIN``: the real cameras' own exposure and white balance). The
low-dimensional data follows the real ``multisensor_20hz`` schema:

* ``observation.state`` (19): 6 joint positions; flange x, y, z and rotation vector in the controller Base frame;
  gripper position (0 open .. 1 closed); 6 F/T values (zero: no F/T sensor in the sim);
* ``action`` (7): 6 joint commands and the gripper command, as stored in the dataset.

Writes, per episode, ``episode_XXXX.npz`` (state, action, meta) and one near-lossless H.264 video per camera
(``episode_XXXX_scene_1.mp4``, ``episode_XXXX_wrist_1.mp4``; about 1/100 the size of raw frames), streamed while
rendering, for ``export_lerobot_dataset.py``.

    python scripts_v2/tools/rpm_lab/render_mimic_dataset.py --headless --dataset generated.hdf5 --out frames_dir

``--num_envs`` episodes are rendered at once, one per env, through tiled cameras.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--episodes", default=None, help="Comma-separated demo indices (default: all).")
parser.add_argument("--task", default=None, help="Gym id of the RPM Lab task (default: the dataset's env_name).")
parser.add_argument("--instruction", default=None, help="Language instruction (default: the task's INSTRUCTION).")
parser.add_argument("--warmup_renders", type=int, default=2, help="RTX renders per frame (more = cleaner).")
parser.add_argument("--num_envs", type=int, default=8, help="Episodes rendered side by side (tiled cameras).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import h5py
import importlib
import json
import numpy as np
import os
import subprocess
import torch
from scipy.spatial.transform import Rotation

import cv2
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import TiledCamera

from uwlab_tasks.manager_based.manipulation.rpm_lab import registry
from uwlab_tasks.manager_based.manipulation.rpm_lab import scene_cfg as lab
from uwlab_tasks.manager_based.manipulation.rpm_lab.pick_place_env_cfg import ARM_JOINTS, FINGER_CLOSED, FINGER_JOINT
from uwlab_tasks.manager_based.manipulation.rpm_lab.sensors import reattach_mounted_cameras

STATE_NAMES = (
    [f"lightning_joint_pos_{i}" for i in range(1, 7)]
    + [f"lightning_eef_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")]
    + ["lightning_gripper_position"]
    + [f"lightning_ft_{k}" for k in ("fx", "fy", "fz", "tx", "ty", "tz")]
)
ACTION_NAMES = [f"lightning_cmd_joint_{i}" for i in range(1, 7)] + ["lightning_cmd_gripper"]
CAMERA_SIZES = {"scene_1": (lab.L515_WIDTH, lab.L515_HEIGHT), "wrist_1": (lab.D405_WIDTH, lab.D405_HEIGHT)}


def sim_matrix(k: tuple, width: int, height: int) -> np.ndarray:
    """The pinhole Omniverse renders for intrinsics ``k``: mean focal length, principal point at the center."""
    f = 0.5 * (k[0] + k[4])
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def real_model_maps(k: tuple, distortion: tuple | None, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps that turn a sim render into the real camera's image (intrinsics, and distortion if given)."""
    k_real = np.array(k, dtype=np.float64).reshape(3, 3)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    dist = np.zeros(5) if distortion is None else np.array(distortion, dtype=np.float64)
    normalized = cv2.undistortPoints(pixels, k_real, dist).reshape(-1, 2)  # ideal rays of the real pixels
    k_sim = sim_matrix(k, width, height)
    map_x = (k_sim[0, 0] * normalized[:, 0] + k_sim[0, 2]).reshape(height, width).astype(np.float32)
    map_y = (k_sim[1, 1] * normalized[:, 1] + k_sim[1, 2]).reshape(height, width).astype(np.float32)
    return map_x, map_y


def as_real(rgb: np.ndarray, maps: tuple[np.ndarray, np.ndarray], gain: tuple[float, float, float]) -> np.ndarray:
    """A sim frame as the real camera would show it: warped to its lens model, times its color gain."""
    warped = cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return np.clip(warped * np.asarray(gain, dtype=np.float32), 0, 255).astype(np.uint8)


class VideoWriter:
    """Streams RGB frames into an H.264 file through ffmpeg (crf 10: visually lossless; the export re-encodes)."""

    def __init__(self, path: str, width: int, height: int, fps: int = 20):
        self.path = path
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def write(self, frame: np.ndarray):
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def load_states(group) -> dict:
    """An episode's recorded sim states (``states`` group of the HDF5), read into memory."""
    return {kind: {name: {k: v[:] for k, v in group[kind][name].items()} for name in group[kind]} for kind in group}


def batch_state(episodes: list[dict], t: int, device) -> dict:
    """Every env's state at step ``t``, as ``scene.reset_to`` takes it; an episode that has ended holds its last step."""
    first = episodes[0]
    return {
        kind: {
            name: {
                k: torch.tensor(
                    np.stack([e[kind][name][k][min(t, len(e[kind][name][k]) - 1)] for e in episodes]), device=device
                )
                for k in first[kind][name]
            }
            for name in first[kind]
        }
        for kind in first
    }


def task_config(dataset: str) -> tuple[str, object, str]:
    """(gym id, env cfg, instruction) of the dataset's task."""
    task = args.task
    if task is None:
        with h5py.File(dataset, "r") as f:
            task = json.loads(f["data"].attrs["env_args"]).get("env_name")
    if task not in registry.TASKS:
        raise ValueError(f"Unknown task {task!r} (pass --task, one of {list(registry.TASKS)})")
    module_name, class_name = registry.TASKS[task]["env_cfg_entry_point"].split(":")
    module = importlib.import_module(module_name)
    instruction = args.instruction or getattr(module, "INSTRUCTION", "")
    return task, getattr(module, class_name)(), instruction


def main():
    task, env_cfg, instruction = task_config(args.dataset)
    print(f"Task {task}: {instruction!r}")
    scene_cfg = env_cfg.scene
    scene_cfg.num_envs = args.num_envs
    scene_cfg.env_spacing = 8.0  # keep the neighbors out of every camera's view
    scene_cfg.scene_camera = lab.l515_scene_camera_cfg(data_types=("rgb",), update_period=0.0)
    scene_cfg.wrist_camera = lab.d405_wrist_camera_cfg("Lightning", data_types=("rgb",), update_period=0.0)
    for camera in (scene_cfg.scene_camera, scene_cfg.wrist_camera):
        camera.class_type = TiledCamera  # one tiled render for all envs
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=env_cfg.sim.dt, device=args.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    reattach_mounted_cameras(scene)  # the wrist camera follows its link (see rpm_lab/sensors.py)
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    robot = scene["lightning"]
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    finger_id = robot.find_joints(FINGER_JOINT)[0][0]
    w3 = robot.find_bodies("wrist_3_link")[0][0]
    rz180 = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], device=sim.device))

    scene_maps = real_model_maps(lab.L515_COLOR_INTRINSICS, lab.L515_COLOR_DISTORTION, lab.L515_WIDTH, lab.L515_HEIGHT)
    wrist_maps = real_model_maps(lab.D405_COLOR_INTRINSICS, None, lab.D405_WIDTH, lab.D405_HEIGHT)

    def pose(pos, quat) -> torch.Tensor:
        t = torch.eye(4, device=sim.device)
        t[:3, :3] = math_utils.matrix_from_quat(quat)
        t[:3, 3] = pos
        return t

    def robot_state(i: int) -> np.ndarray:
        """Env ``i``'s observation.state, as the robot PC records it."""
        base = pose(robot.data.root_pos_w[i], robot.data.root_quat_w[i]) @ rz180
        flange = torch.linalg.inv(base) @ pose(robot.data.body_pos_w[i, w3], robot.data.body_quat_w[i, w3])
        flange = flange.cpu().numpy()
        rotvec = Rotation.from_matrix(flange[:3, :3]).as_rotvec()
        q = robot.data.joint_pos[i, arm_ids].cpu().numpy()
        gripper = float(robot.data.joint_pos[i, finger_id]) / FINGER_CLOSED
        return np.concatenate([q, flange[:3, 3], rotvec, [np.clip(gripper, 0, 1)], np.zeros(6)])

    os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if args.episodes:
            demos = [f"demo_{int(e)}" for e in args.episodes.split(",")]
        for start in range(0, len(demos), args.num_envs):
            names = demos[start : start + args.num_envs]
            episodes = [load_states(f["data"][name]["states"]) for name in names]
            actions = [f["data"][name]["actions"][:] for name in names]
            lengths = [a.shape[0] for a in actions]
            episodes += [episodes[-1]] * (args.num_envs - len(names))  # fill idle envs
            states = [[] for _ in names]
            videos = [
                {
                    key: VideoWriter(os.path.join(args.out, f"episode_{start + i:04d}_{key}.mp4"), width, height)
                    for key, (width, height) in CAMERA_SIZES.items()
                }
                for i in range(len(names))
            ]
            for t in range(max(lengths)):
                scene.reset_to(batch_state(episodes, t, sim.device), is_relative=True)
                sim.forward()
                for _ in range(args.warmup_renders if t else args.warmup_renders + 8):
                    sim.render()
                scene.update(sim.get_physics_dt())
                scene_all = scene["scene_camera"].data.output["rgb"][..., :3].cpu().numpy()
                wrist_all = scene["wrist_camera"].data.output["rgb"][..., :3].cpu().numpy()
                for i in range(len(names)):
                    if t < lengths[i]:
                        videos[i]["scene_1"].write(as_real(scene_all[i], scene_maps, lab.CAMERA_COLOR_GAIN["scene_1"]))
                        videos[i]["wrist_1"].write(
                            as_real(wrist_all[i], wrist_maps, lab.CAMERA_COLOR_GAIN["lightning_wrist"])
                        )
                        states[i].append(robot_state(i))
            for i, name in enumerate(names):
                n = start + i
                for writer in videos[i].values():
                    writer.close()
                path = os.path.join(args.out, f"episode_{n:04d}.npz")
                np.savez(
                    path,
                    state=np.asarray(states[i], dtype=np.float32),
                    action=actions[i].astype(np.float32),
                    meta=np.array(
                        json.dumps(
                            {
                                "source": os.path.abspath(args.dataset),
                                "demo": name,
                                "task": task,
                                "instruction": instruction,
                                "fps": 20,
                                "state_names": STATE_NAMES,
                                "action_names": ACTION_NAMES,
                                "videos": {k: os.path.basename(w.path) for k, w in videos[i].items()},
                            }
                        )
                    ),
                )
                print(f"[{n + 1}/{len(demos)}] {name}: {lengths[i]} frames -> {path}")
    sim.clear_all_callbacks()
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

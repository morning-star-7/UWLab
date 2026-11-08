# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record scripted pick-and-place demos on the Lightning Mimic task as source data for Isaac Lab Mimic.

The script plans grasp-point waypoints from the object poses (approach, descend, close, lift, carry, place, release,
retreat), interpolates them at the 20 Hz control rate, and turns each target into an action through the task's own
``target_eef_pose_to_action``, the same conversion Mimic uses when it generates data. Only successful episodes are
written, in the HDF5 layout of Isaac Lab's ``record_demos.py``.

Example (then annotate and generate with ``mimic.py``):
    python scripts_v2/tools/rpm_lab/scripted_pick_place_demos.py --headless --num_demos 10 \\
        --dataset_file /data/cui00191/rpm_lab_mimic/source.hdf5
"""

import argparse
import importlib.util
import os
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def register_rpm_lab_tasks():
    """Register the RPM Lab gym ids without importing Isaac Lab (loads registry.py by path)."""
    path = os.path.join(REPO, "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/registry.py")
    spec = importlib.util.spec_from_file_location("rpm_lab_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="UWLab-RPMLab-Lightning-PickPlace-Mimic-v0")
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--max_attempts", type=int, default=30)
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--num_success_steps", type=int, default=5, help="Consecutive success steps that end a demo.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trace", action="store_true", help="Print tracking and grasp state every 10 steps.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
register_rpm_lab_tasks()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

EEF = "lightning"
CONTROL_DT = 0.05  # 20 Hz
OPEN, CLOSE = 1.0, -1.0


def waypoints(env) -> list[tuple[np.ndarray, np.ndarray, float, float, int]]:
    """(position, rotation matrix, gripper, speed m/s, hold steps) for each phase, env frame."""
    from uwlab_tasks.manager_based.manipulation.rpm_lab.pick_place_env_cfg import CUBE_SIZE, PLATE_SIZE

    origin = env.scene.env_origins[0]
    tcp = env.get_robot_eef_pose(EEF, [0])[0].cpu().numpy()
    cube_p = (env.scene["cube"].data.root_pos_w[0] - origin).cpu().numpy()
    cube_q = env.scene["cube"].data.root_quat_w[0].cpu().numpy()
    plate_p = (env.scene["plate"].data.root_pos_w[0] - origin).cpu().numpy()

    # Approach axis (grasp-point z) straight down, then turn about the vertical onto the nearest cube face.
    r0 = tcp[:3, :3]
    align, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [r0[:, 2]])
    down = align.as_matrix() @ r0
    cube_x = Rotation.from_quat(cube_q, scalar_first=True).as_matrix()[:, 0]
    delta = math.atan2(cube_x[1], cube_x[0]) - math.atan2(down[1, 0], down[0, 0])
    delta = (delta + math.pi / 4) % (math.pi / 2) - math.pi / 4
    grasp = Rotation.from_euler("z", delta).as_matrix() @ down

    plate_top = plate_p[2] + 0.5 * PLATE_SIZE[2]
    place = np.array([plate_p[0], plate_p[1], plate_top + 0.5 * CUBE_SIZE + 0.006])
    return [
        (cube_p + [0, 0, 0.10], grasp, OPEN, 0.25, 2),  # above the cube
        (cube_p + [0, 0, 0.003], grasp, OPEN, 0.10, 2),  # grasp height
        (cube_p + [0, 0, 0.003], grasp, CLOSE, 0.10, 12),  # close
        (cube_p + [0, 0, 0.12], grasp, CLOSE, 0.15, 2),  # lift
        (place + [0, 0, 0.12], grasp, CLOSE, 0.25, 2),  # above the plate
        (place, grasp, CLOSE, 0.10, 4),  # place height
        (place, grasp, OPEN, 0.10, 10),  # release
        (place + [0, 0, 0.10], grasp, OPEN, 0.15, 10),  # retreat
    ]


def interpolate(env) -> list[tuple[np.ndarray, float]]:
    """Per-step (4x4 target pose, gripper) from the current grasp-point pose through the waypoints."""
    current = env.get_robot_eef_pose(EEF, [0])[0].cpu().numpy()
    pos, rot = current[:3, 3], current[:3, :3]
    steps = []
    for target_pos, target_rot, gripper, speed, hold in waypoints(env):
        n = max(5, math.ceil(np.linalg.norm(target_pos - pos) / (speed * CONTROL_DT)))
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([rot, target_rot]))
        for i in range(1, n + 1 + hold):
            s = min(1.0, i / n)
            pose = np.eye(4)
            pose[:3, :3] = slerp(s).as_matrix()
            pose[:3, 3] = pos + s * (target_pos - pos)
            steps.append((pose, gripper))
        pos, rot = target_pos, target_rot
    return steps


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None  # checked here, as in record_demos.py
    env_cfg.terminations.time_out = None
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.dataset_file))
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    os.makedirs(env_cfg.recorders.dataset_export_dir_path, exist_ok=True)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    exported, attempts = 0, 0
    env.reset()
    while exported < args.num_demos and attempts < args.max_attempts:
        attempts += 1
        plan = interpolate(env)
        plan += [plan[-1]] * 20  # time for the success check after the retreat
        streak, ok, length = 0, False, 0
        for pose, gripper in plan:
            action = env.target_eef_pose_to_action(
                {EEF: torch.tensor(pose, dtype=torch.float32, device=env.device)},
                {EEF: torch.tensor([gripper], dtype=torch.float32, device=env.device)},
                env_id=0,
            )
            _, _, terminated, truncated, _ = env.step(action)
            length += 1
            if args.trace and length % 10 == 0:
                tcp = env.get_robot_eef_pose(EEF, [0])[0].cpu().numpy()
                obs = env.obs_buf
                print(
                    f"  step {length:3d} target {np.round(pose[:3, 3], 3)} tcp {np.round(tcp[:3, 3], 3)} "
                    f"err {np.linalg.norm(tcp[:3, 3] - pose[:3, 3]) * 1e3:5.1f} mm, "
                    f"rot err {math.degrees(Rotation.from_matrix(tcp[:3, :3].T @ pose[:3, :3]).magnitude()):4.1f} deg"
                    f" | grip cmd {gripper:+.0f} finger {float(obs['policy']['gripper'][0, 0]):.2f} rad"
                    f" | cube {np.round(obs['policy']['cube_pos'][0].cpu().numpy(), 3)}"
                    f" grasp {float(obs['subtask_terms']['grasp'][0]):.0f}"
                )
            if bool(terminated[0] or truncated[0]):
                break
            streak = streak + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
            if streak >= args.num_success_steps:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], device=env.device))
                env.recorder_manager.export_episodes([0])
                ok = True
                break
        exported += int(ok)
        print(f"attempt {attempts}: {'success' if ok else 'FAILED'} after {length} steps ({exported}/{args.num_demos})")
        env.recorder_manager.reset()
        env.reset()
    print(f"Exported {exported} demos in {attempts} attempts to {args.dataset_file}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()  # close() ends the process with os._exit, which would drop the traceback
        raise
    finally:
        simulation_app.close()

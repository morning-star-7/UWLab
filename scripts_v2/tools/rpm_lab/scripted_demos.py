# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record scripted demos on a Lightning Mimic task (pick-and-place or cook_fish) as source data for Isaac Lab Mimic.

A task's script is a list of phases. Each phase plans grasp-point waypoints from the state at its start (where the
objects are now, and where a held object sits in the gripper), the waypoints are interpolated at the 20 Hz control
rate (minimum-jerk timing: each move speeds up and slows down), and every target goes through the task's own
``target_eef_pose_to_action``, the conversion Mimic uses when it generates data. The gripper points down and turns about
the vertical so that its closing axis (grasp-point x) crosses the part it closes on, picking whichever such yaw is
nearest the current one; gripper commands change over a few steps in 0.1 steps, like the real teleop trigger. Only
successful episodes are written, in the HDF5 layout of Isaac Lab's ``record_demos.py``.

cook_fish follows the operators of the real 2026-10-28 episodes (phase durations, heights, gripper tilt and openings,
``STYLE``): each demo leans the gripper 8-15 deg, opens it to 0.2-0.3 on the lid and 0-0.1 on the fish, and moves at
its own pace.

Example (then annotate and generate with ``mimic.py``):
    python scripts_v2/tools/rpm_lab/scripted_demos.py --headless --task UWLab-RPMLab-Lightning-CookFish-Mimic-v0 \\
        --num_demos 10 --dataset_file /data/cui00191/rpm_lab_mimic/cook_fish/source.hdf5
"""

import argparse
import importlib.util
import os
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PICK_PLACE = "UWLab-RPMLab-Lightning-PickPlace-Mimic-v0"
COOK_FISH = "UWLab-RPMLab-Lightning-CookFish-Mimic-v0"


def register_rpm_lab_tasks():
    """Register the RPM Lab gym ids without importing Isaac Lab (loads registry.py by path)."""
    path = os.path.join(REPO, "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/registry.py")
    spec = importlib.util.spec_from_file_location("rpm_lab_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default=PICK_PLACE, choices=[PICK_PLACE, COOK_FISH])
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
OPEN, CLOSE = 0.0, 1.0  # gripper command, as in the real datasets
Waypoint = tuple  # (position (3,), rotation (3, 3), gripper, mean speed m/s, hold steps), env frame
MIN_SEGMENT_STEPS = 3
TURN_SPEED = 1.0  # rad/s, for moves that mostly turn
GRIPPER_RAMP_STEPS = {"close": 6, "open": 4}
"""Steps a gripper command takes to change: the real trigger goes 0.3 -> 1.0 in about 6 steps (0.1 per sample)."""

STYLE: dict = {}
"""The current cook_fish demo's operator style (``sample_style``); empty for pick-and-place (straight down, pace 1)."""


def sample_style(rng: np.random.Generator):
    """An operator style drawn from the real cook_fish episodes (10th-90th percentiles of 40 parsed episodes)."""
    STYLE.update(
        tilt=math.radians(rng.uniform(8.0, 15.0)),  # gripper axis from straight down (real median 12 deg)
        azimuth=math.radians(rng.uniform(-70.0, -27.0)),  # direction the fingertips lean to, env frame (median -50)
        pace=rng.uniform(0.85, 1.2),  # speed factor: real phase durations spread about +-20 %
        wait=int(rng.integers(10, 26)),  # steps before the first move (real: 2.7 s to the lid, 1.4 s of it moving)
        open_lid=float(rng.choice([0.2, 0.3])),  # gripper command opened on the lid (real median 0.3)
        open_fish=float(rng.choice([0.0, 0.1])),  # ... on the fish (real median 0.1)
    )


def tilted(rot: np.ndarray) -> np.ndarray:
    """``rot`` leaned by the demo's tilt so that its approach axis (z) points towards the demo's azimuth."""
    if not STYLE:
        return rot
    a = STYLE["azimuth"]
    lean = Rotation.from_rotvec(STYLE["tilt"] * np.array([math.sin(a), -math.cos(a), 0.0])).as_matrix()
    return lean @ rot


# ---------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------------


def pose_of(env, name: str) -> np.ndarray:
    """4x4 pose of a scene object, env frame."""
    obj = env.scene[name]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(obj.data.root_quat_w[0].cpu().numpy(), scalar_first=True).as_matrix()
    pose[:3, 3] = (obj.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy()
    return pose


def tcp_pose(env) -> np.ndarray:
    return env.get_robot_eef_pose(EEF, [0])[0].cpu().numpy()


def down_crossing(current: np.ndarray, part_axes: list[np.ndarray]) -> np.ndarray:
    """Grasp-point rotation pointing straight down whose closing axis (x) is horizontal and at a right angle to one of
    ``part_axes`` (the directions a finger pair may close across), choosing the nearest yaw to ``current``."""
    best, best_cos = None, -2.0
    for axis in part_axes:
        x = np.cross([0.0, 0.0, 1.0], axis)  # horizontal, at a right angle to the part axis
        if np.linalg.norm(x[:2]) < 1e-6:
            continue
        x /= np.linalg.norm(x)
        for sign in (1.0, -1.0):
            c = float(np.dot(sign * x, current[:3, 0]))
            if c > best_cos:
                best, best_cos = sign * x, c
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([best, np.cross(z, best), z], axis=1)


def carry_target(env, obj: str, obj_target: np.ndarray) -> np.ndarray:
    """Grasp-point pose that puts the held object ``obj`` at ``obj_target``, keeping its current pose in the hand."""
    in_hand = np.linalg.inv(tcp_pose(env)) @ pose_of(env, obj)
    return obj_target @ np.linalg.inv(in_hand)


def at(pose: np.ndarray, dz: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return pose[:3, 3] + [0.0, 0.0, dz], pose[:3, :3]


# ---------------------------------------------------------------------------------------------------------------------
# Pick-and-place
# ---------------------------------------------------------------------------------------------------------------------


def pick_place(env, last: np.ndarray) -> list[Waypoint]:
    from uwlab_tasks.manager_based.manipulation.rpm_lab.pick_place_env_cfg import CUBE_SIZE, PLATE_SIZE

    cube, plate = pose_of(env, "cube"), pose_of(env, "plate")
    grasp = down_crossing(last, [cube[:3, 0], cube[:3, 1]])
    cube_p = cube[:3, 3]
    place = np.array([plate[0, 3], plate[1, 3], plate[2, 3] + 0.5 * PLATE_SIZE[2] + 0.5 * CUBE_SIZE + 0.006])
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


# ---------------------------------------------------------------------------------------------------------------------
# cook_fish: lid off, fish in, lid on
# ---------------------------------------------------------------------------------------------------------------------

LID_GRASP_ABOVE_HANDLE = 0.006  # grasp point above the handle's axis: the fingertips stay clear of the lid
FISH_GRASP_ABOVE_CENTER = 0.004


def _cook_fish():
    from uwlab_tasks.manager_based.manipulation.rpm_lab import cook_fish_env_cfg as task
    from uwlab_tasks.manager_based.manipulation.rpm_lab import scene_cfg as lab

    return task, lab


def grasp_lid(env, last: np.ndarray) -> list[Waypoint]:
    task, _ = _cook_fish()
    lid = pose_of(env, "lid")
    handle = lid[:3, 3] + lid[:3, :3] @ [0.0, 0.0, task.LID["handle_height"] + LID_GRASP_ABOVE_HANDLE]
    rot = tilted(down_crossing(last, [lid[:3, 0]]))
    opened = STYLE.get("open_lid", OPEN)
    return [
        (handle + [0, 0, 0.06], rot, opened, 0.30, 0),
        (handle, rot, opened, 0.18, 1),
        (handle, rot, CLOSE, 0.12, 6),
        (handle + [0, 0, 0.09], rot, CLOSE, 0.25, 0),  # lid bottom 9 cm up (real carry top: 16 cm grasp point)
    ]


LID_ASIDE_REACH = 0.70
"""Farthest lid set-down spot from Lightning's base (m), short of ``MAX_REACH``: Mimic re-places the spot with the pot
(turned up to ~0.8 rad and moved), and the lid has to be grasped there again. The real ones lie up to 0.71 m out."""


def lid_aside_spot(env) -> np.ndarray:
    """Where to set the lid down: where the operator did in the real layout this scene was drawn from (the nearest in
    ``REAL_LAYOUTS``), kept relative to the pot and pushed out to clear its rim or handles, if that is
    ``LID_ASIDE_CLEARANCE`` from the fish and within ``LID_ASIDE_REACH`` of the base. Otherwise beside the pot inside
    the object area and within reach, as far from the fish as possible (the spot nearest the base if none is in
    reach)."""
    task, lab = _cook_fish()
    pot, fish = pose_of(env, "pot"), pose_of(env, "fish")[:3, 3]
    base = task.lightning_base_xy(env)[0].cpu().numpy()
    pot_x = math.atan2(pot[1, 0], pot[0, 0])  # the handles stick out along the pot's x axis
    z = lab.WORK_SURFACE_Z + task.LID["lip_depth"]

    def clear_of_pot(angle: float) -> float:
        """Least distance from the pot center for a lid set down in this direction."""
        by_handle = abs(math.cos(angle - pot_x)) > math.cos(math.radians(50))
        return (task.POT["handle_reach"] if by_handle else task.POT["radius"]) + task.LID["radius"] + 0.02

    layouts = task.REAL_LAYOUTS.numpy()
    gap = np.linalg.norm(layouts[:, 0] - pot[:2, 3], axis=1) + np.linalg.norm(layouts[:, 1] - fish[:2], axis=1)
    offset = layouts[np.argmin(gap), 2] - layouts[np.argmin(gap), 0]
    angle = math.atan2(offset[1], offset[0])
    spot = pot[:2, 3] + max(np.linalg.norm(offset), clear_of_pot(angle)) * np.array([math.cos(angle), math.sin(angle)])
    if np.linalg.norm(spot - fish[:2]) > task.LID_ASIDE_CLEARANCE and np.linalg.norm(spot - base) < LID_ASIDE_REACH:
        return np.array([spot[0], spot[1], z])

    margin = task.LID["radius"] - 0.03
    candidates = []
    for angle in np.radians(np.arange(0, 360, 15)):
        spot = pot[:2, 3] + clear_of_pot(angle) * np.array([math.cos(angle), math.sin(angle)])
        inside = (
            task.OBJECT_X_RANGE[0] - margin < spot[0] < task.OBJECT_X_RANGE[1] + margin
            and task.OBJECT_Y_RANGE[0] - margin < spot[1] < task.OBJECT_Y_RANGE[1] + margin
        )
        if inside:
            candidates.append((spot, np.linalg.norm(spot - fish[:2]), np.linalg.norm(spot - base)))
    in_reach = [c for c in candidates if c[2] < LID_ASIDE_REACH]
    best = max(in_reach, key=lambda c: c[1]) if in_reach else min(candidates, key=lambda c: c[2])
    return np.array([best[0][0], best[0][1], z])


def lid_down_beside(env, last: np.ndarray) -> list[Waypoint]:
    lid = pose_of(env, "lid")
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", Rotation.from_matrix(lid[:3, :3]).as_euler("ZYX")[0]).as_matrix()
    target[:3, 3] = lid_aside_spot(env) + [0.0, 0.0, 0.02]  # let go 2 cm up (the operators dropped it from higher)
    pose = carry_target(env, "lid", target)
    pos, rot = at(pose)
    opened = STYLE.get("open_lid", OPEN)
    return [
        (pos + [0, 0, 0.08], rot, CLOSE, 0.35, 0),
        (pos, rot, CLOSE, 0.20, 1),
        (pos, rot, opened, 0.12, 3),
        (pos + [0, 0, 0.055], rot, opened, 0.25, 0),  # to 11.6 cm, past RETREAT_HEIGHT (real: 11 cm on to the fish)
    ]


def grasp_fish(env, last: np.ndarray) -> list[Waypoint]:
    task, _ = _cook_fish()
    fish = pose_of(env, "fish")
    center = fish[:3, 3] + fish[:3, :3] @ [0.0, 0.0, 0.5 * task.FISH["size"][2] + FISH_GRASP_ABOVE_CENTER]
    rot = tilted(down_crossing(last, [fish[:3, 0], fish[:3, 1]]))
    opened = STYLE.get("open_fish", OPEN)
    return [
        (center + [0, 0, 0.09], rot, opened, 0.35, 0),
        (center, rot, opened, 0.18, 1),
        (center, rot, CLOSE, 0.12, 6),
        (center + [0, 0, 0.10], rot, CLOSE, 0.25, 0),
    ]


def fish_into_pot(env, last: np.ndarray) -> list[Waypoint]:
    task, _ = _cook_fish()
    pot, fish = pose_of(env, "pot"), pose_of(env, "fish")
    target = fish.copy()
    target[:3, 3] = pot[:3, 3] + [0.0, 0.0, task.POT["rim_height"] + 0.025]  # drop it from 25 mm above the rim
    pos, rot = at(carry_target(env, "fish", target))
    opened = STYLE.get("open_lid", OPEN)
    return [
        (pos + [0, 0, 0.06], rot, CLOSE, 0.35, 0),
        (pos, rot, CLOSE, 0.15, 1),
        (pos, rot, opened, 0.12, 3),
        (pos + [0, 0, 0.04], rot, opened, 0.25, 0),
    ]


def lid_onto_pot(env, last: np.ndarray) -> list[Waypoint]:
    task, _ = _cook_fish()
    pot, lid = pose_of(env, "pot"), pose_of(env, "lid")
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", Rotation.from_matrix(lid[:3, :3]).as_euler("ZYX")[0]).as_matrix()
    target[:3, 3] = pot[:3, 3] + [0.0, 0.0, task.POT["rim_height"] + 0.006]
    pos, rot = at(carry_target(env, "lid", target))
    opened = STYLE.get("open_lid", OPEN)
    return [
        (pos + [0, 0, 0.06], rot, CLOSE, 0.35, 0),
        (pos + [0, 0, 0.015], rot, CLOSE, 0.12, 0),
        (pos, rot, CLOSE, 0.04, 2),
        (pos, rot, opened, 0.05, 3),
        (pos + [0, 0, 0.16], rot, opened, 0.20, 4),  # away, up to about 25 cm (the operators' last move)
    ]


PHASES = {
    PICK_PLACE: [pick_place],
    COOK_FISH: [grasp_lid, lid_down_beside, grasp_fish, fish_into_pot, grasp_lid, lid_onto_pot],
}

# ---------------------------------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------------------------------


def interpolate(
    start: np.ndarray, gripper0: float, waypoints: list[Waypoint]
) -> tuple[list[tuple[np.ndarray, float]], np.ndarray, float]:
    """Per-step (4x4 target pose, gripper) from ``start`` (and gripper command ``gripper0``) through the waypoints,
    the last target and the last gripper command. Each move takes its distance over the waypoint's mean speed (times
    the demo's pace) with minimum-jerk timing; a gripper change ramps in 0.1 steps over ``GRIPPER_RAMP_STEPS``."""
    pos, rot, grip = start[:3, 3], start[:3, :3], gripper0
    pace = STYLE.get("pace", 1.0)
    steps = []
    for target_pos, target_rot, gripper, speed, hold in waypoints:
        turn = Rotation.from_matrix(rot.T @ target_rot).magnitude()
        duration = max(np.linalg.norm(target_pos - pos) / (speed * pace), turn / TURN_SPEED)
        n = max(MIN_SEGMENT_STEPS, math.ceil(duration / CONTROL_DT))
        ramp = GRIPPER_RAMP_STEPS["close" if gripper > grip else "open"] if gripper != grip else 1
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([rot, target_rot]))
        for i in range(1, max(n + hold, ramp) + 1):
            t = min(1.0, i / n)
            s = t**3 * (10 - 15 * t + 6 * t**2)  # minimum jerk
            pose = np.eye(4)
            pose[:3, :3] = slerp(s).as_matrix()
            pose[:3, 3] = pos + s * (target_pos - pos)
            g = gripper if i >= ramp else round((grip + (gripper - grip) * i / ramp) * 10) / 10
            steps.append((pose, g))
        pos, rot, grip = target_pos, target_rot, gripper
    last = np.eye(4)
    last[:3, :3], last[:3, 3] = rot, pos
    return steps, last, grip


def trace(env, length: int, pose: np.ndarray, gripper: float):
    tcp = tcp_pose(env)
    obs = env.obs_buf
    signals = " ".join(f"{k}={float(v[0]):.0f}" for k, v in obs["subtask_terms"].items())
    print(
        f"  step {length:3d} target {np.round(pose[:3, 3], 3)} tcp {np.round(tcp[:3, 3], 3)} "
        f"err {np.linalg.norm(tcp[:3, 3] - pose[:3, 3]) * 1e3:5.1f} mm, "
        f"rot err {math.degrees(Rotation.from_matrix(tcp[:3, :3].T @ pose[:3, :3]).magnitude()):4.1f} deg"
        f" | grip cmd {gripper:.0f} finger {float(obs['policy']['gripper'][0, 0]):.2f} rad | {signals}"
    )


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.env_name = args.task  # stored in the dataset (env_args)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None  # checked here, as in record_demos.py
    env_cfg.terminations.time_out = None
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.dataset_file))
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    os.makedirs(env_cfg.recorders.dataset_export_dir_path, exist_ok=True)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    def run(steps) -> str:
        """Execute steps; "ended" (termination) or "" (still going). ``streak`` counts consecutive success steps."""
        nonlocal streak, length
        for pose, gripper in steps:
            action = env.target_eef_pose_to_action(
                {EEF: torch.tensor(pose, dtype=torch.float32, device=env.device)},
                {EEF: torch.tensor([gripper], dtype=torch.float32, device=env.device)},
                env_id=0,
            )
            _, _, terminated, truncated, _ = env.step(action)
            length += 1
            if args.trace and length % 10 == 0:
                trace(env, length, pose, gripper)
            if bool(terminated[0] or truncated[0]):
                return "ended"
            streak = streak + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
        return ""

    exported, attempts = 0, 0
    rng = np.random.default_rng(args.seed)
    env.reset()
    while exported < args.num_demos and attempts < args.max_attempts:
        attempts += 1
        streak, length, outcome = 0, 0, ""
        if args.task == COOK_FISH:
            sample_style(rng)
        last, grip = tcp_pose(env), OPEN
        outcome = run([(last, grip)] * STYLE.get("wait", 0)) if STYLE else ""
        for phase in PHASES[args.task]:
            if outcome:
                break
            steps, last, grip = interpolate(last, grip, phase(env, last))
            outcome = run(steps)
        if not outcome and streak < args.num_success_steps:
            outcome = run([(last, grip)] * args.num_success_steps)  # time for the success check after the retreat
        ok = not outcome and streak >= args.num_success_steps
        if ok:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], device=env.device))
            env.recorder_manager.export_episodes([0])
        exported += int(ok)
        style = " ".join(f"{k} {math.degrees(v) if k in ('tilt', 'azimuth') else v:.2f}" for k, v in STYLE.items())
        outcome = "success" if ok else "FAILED"
        print(f"attempt {attempts}: {outcome} after {length} steps ({exported}/{args.num_demos}) {style}")
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

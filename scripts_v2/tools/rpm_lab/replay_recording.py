# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the RPM Lab twin against a real Thunder/Lightning recording (``extract_spark_recording.py`` output).

``--mode kinematic`` (many envs, no rendering): puts recorded frames on the sim arms (joints and gripper) and reports

* the sim flange vs the flange pose the real controller reported (calibration and frame conventions),
* the grasp point and the lowest point of the hand's collision geometry above the tabletop (mounting; a hand that
  dips into the table where the real one did not means the sim geometry is off).

``--mode replay`` (one env): runs the recording through physics and optionally renders a video.

* ``--arm teleport`` writes the measured joints every step; the gripper follows the measured opening.
* ``--arm servo`` drives the arms like the real controller: a stiff joint PD tracks the teleop commands passed
  through a servoJ model (delay + first-order lag) fitted to this recording, and the gripper follows the commanded
  opening. It reports how closely the sim reproduces the measured motion.
* ``--video out.mp4`` renders the sim (overview and wrist D405s) next to the recorded camera frames.

The gripper reading (0 open .. 1 closed, from the gripper's own calibrated range) is linear in stroke; it maps onto
finger_joint through the model's fingertip-pad gap, tabulated at start-up (``Arm.calibrate_aperture``); the other
gripper joints follow the model's mimic couplings.

Examples:
    python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode kinematic --recording ep1106.npz
    python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode replay --arm servo --recording ep1106.npz
    python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode replay --recording ep1028.npz \\
        --video replay.mp4 --stills ./stills
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--recording", required=True, help=".npz from extract_spark_recording.py")
parser.add_argument("--mode", choices=("kinematic", "replay"), default="kinematic")
parser.add_argument("--arm", choices=("teleport", "servo"), default="teleport", help="Replay arm control.")
parser.add_argument("--envs", type=int, default=128, help="Kinematic mode: frames evaluated in parallel.")
parser.add_argument("--max_frames", type=int, default=20000, help="Kinematic mode: frames sampled per recording.")
parser.add_argument("--start", type=float, default=0.0, help="Replay start (s).")
parser.add_argument("--duration", type=float, default=None, help="Replay length (s); default: to the end.")
parser.add_argument("--video", type=str, default=None, help="Replay: write an mp4 here.")
parser.add_argument("--video_fps", type=float, default=10.0)
parser.add_argument("--stills", type=str, default=None, help="Replay: directory for real-vs-sim wrist stills.")
parser.add_argument("--num_stills", type=int, default=6)
parser.add_argument("--out", type=str, default=None, help="Write the summary (and replay time series) here (.npz).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video or args.stills:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import json
import math
import numpy as np
import os
import torch
from scipy.signal import lfilter

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from pxr import Usd, UsdGeom, UsdPhysics

from uwlab_tasks.manager_based.manipulation.rpm_lab import scene_cfg as lab
from uwlab_tasks.manager_based.manipulation.rpm_lab.sensors import reattach_mounted_cameras

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
FINGER_JOINT = "finger_joint"
EE_BODY = "wrist_3_link"
GRIPPER_BODY = "robotiq_base_link"
HAND_BODIES = ("wrist_1_link", "wrist_2_link", "wrist_3_link")  # plus every body under OfficialHand
RZ180 = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))  # controller Base = base_link turned 180 deg about z
SERVO_STIFFNESS = (4000.0, 4000.0, 2500.0, 600.0, 600.0, 400.0)  # N m/rad, replay --arm servo
SERVO_DAMPING = (250.0, 250.0, 150.0, 25.0, 25.0, 15.0)


# ---------------------------------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------------------------------


class Recording:
    def __init__(self, path: str):
        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        self.arms = list(self.meta["arms"])
        self.data = {k: d[k] for k in d.files if not k.startswith("image.") and k != "meta"}
        self.images = {}
        for key in d.files:
            if key.startswith("image.") and key.endswith(".t"):
                name = key[len("image.") : -len(".t")]
                self.images[name] = (d[key], d[f"image.{name}.rgb"])
        self.t_end = max(self.data[f"{a}.t"][-1] for a in self.arms)

    def q(self, arm: str, t: np.ndarray) -> np.ndarray:
        rt, rq = self.data[f"{arm}.t"], self.data[f"{arm}.q"]
        return np.stack([np.interp(t, rt, rq[:, j]) for j in range(6)], axis=-1)

    def gripper(self, arm: str, t: np.ndarray, commanded: bool = False) -> np.ndarray:
        if commanded and len(self.data[f"{arm}.cmd_gripper_t"]):
            ct, cg = self.data[f"{arm}.cmd_gripper_t"], self.data[f"{arm}.cmd_gripper"]
            idx = np.clip(np.searchsorted(ct, t, side="right") - 1, 0, len(ct) - 1)
            measured = np.interp(t, self.data[f"{arm}.t"], self.data[f"{arm}.gripper"])
            return np.where(self.active(t) & (t >= ct[0]), cg[idx], measured)
        return np.interp(t, self.data[f"{arm}.t"], self.data[f"{arm}.gripper"])

    def active(self, t: np.ndarray) -> np.ndarray:
        at, av = self.data["teleop_active.t"], self.data["teleop_active.value"]
        if len(at) == 0:
            return np.ones_like(t, dtype=bool)
        idx = np.clip(np.searchsorted(at, t, side="right") - 1, 0, len(at) - 1)
        return av[idx] > 0.5

    def image(self, name: str, t: float, tol: float = 0.2):
        if name not in self.images:
            return None
        times, frames = self.images[name]
        i = int(np.argmin(np.abs(times - t)))
        return frames[i] if abs(times[i] - t) < tol else None


def fit_servo_model(rec: Recording, arm: str) -> tuple[float, float, float]:
    """Fit measured joints ~ first-order lag (tau) of the zero-order-held teleop command delayed by d, while active.

    Returns (delay_s, tau_s, rms_rad).
    """
    ct, cq = rec.data[f"{arm}.cmd_t"], rec.data[f"{arm}.cmd_q"]
    if len(ct) < 10:
        raise RuntimeError(f"{arm}: no teleop commands in the recording")
    dt = 0.002
    grid = np.arange(ct[0], ct[-1], dt)
    idx = np.clip(np.searchsorted(ct, grid, side="right") - 1, 0, len(ct) - 1)
    held = cq[idx]  # zero-order hold, (G, 6)
    mask = rec.active(grid)
    mask[: int(0.5 / dt)] = False  # skip the filter's start-up
    idx = np.flatnonzero(mask)
    measured = rec.q(arm, grid[idx])
    best = (0.0, 0.0, float("inf"))
    for tau in np.arange(0.0, 0.2001, 0.005):
        a = math.exp(-dt / tau) if tau > 0 else 0.0
        filt = lfilter([1 - a], [1, -a], held - held[0], axis=0) + held[0]
        for delay in np.arange(0.0, 0.3001, 0.004):
            pred = filt[np.clip(idx - int(round(delay / dt)), 0, None)]
            rms = float(np.sqrt(np.mean((pred - measured) ** 2)))
            if rms < best[2]:
                best = (float(delay), float(tau), rms)
    return best


def servo_targets(rec: Recording, arm: str, t: np.ndarray, delay: float, tau: float) -> np.ndarray:
    """Joint targets from the fitted servoJ model; the measured joints where teleop is inactive."""
    ct, cq = rec.data[f"{arm}.cmd_t"], rec.data[f"{arm}.cmd_q"]
    dt = 0.002
    grid = np.arange(min(t[0], ct[0]) - 1.0, t[-1] + dt, dt)
    idx = np.clip(np.searchsorted(ct, grid, side="right") - 1, 0, len(ct) - 1)
    held = np.where((grid >= ct[0])[:, None], cq[idx], rec.q(arm, grid))
    a = math.exp(-dt / tau) if tau > 0 else 0.0
    filt = lfilter([1 - a], [1, -a], held - held[0], axis=0) + held[0]
    model = np.stack([np.interp(t - delay, grid, filt[:, j]) for j in range(6)], axis=-1)
    return np.where(rec.active(t)[:, None], model, rec.q(arm, t))


# ---------------------------------------------------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------------------------------------------------


def robot_prim_path(robot) -> str:
    return robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def gripper_couplings(robot) -> dict[str, float]:
    """Follower gripper joint -> multiplier on finger_joint, from the model's PhysX mimic joints.

    PhysX mimic constraint: q + gearing * q_reference + offset = 0 (offsets are zero in these models).
    """
    stage = omni.usd.get_context().get_stage()
    couplings = {FINGER_JOINT: 1.0}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path(robot))):
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith("physxMimicJoint:") and name.endswith(":gearing"):
                axis = name.split(":")[1]
                ref = prim.GetRelationship(f"physxMimicJoint:{axis}:referenceJoint").GetTargets()
                offset = prim.GetAttribute(f"physxMimicJoint:{axis}:offset").Get() or 0.0
                if [r.name for r in ref] != [FINGER_JOINT] or abs(offset) > 1e-9:
                    raise RuntimeError(f"Unexpected mimic joint {prim.GetPath()}")
                couplings[prim.GetName()] = -float(attr.Get())
    return couplings


class Arm:
    """Index bookkeeping and gripper mapping for one robot in the scene."""

    def __init__(self, name: str, robot):
        self.name, self.robot = name, robot
        self.arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        couplings = gripper_couplings(robot)
        self.grip_ids = robot.find_joints(list(couplings), preserve_order=True)[0]
        self.grip_mult = torch.tensor(list(couplings.values()), device=robot.device)
        finger = robot.find_joints(FINGER_JOINT)[0][0]
        self.finger_closed = float(robot.data.joint_pos_limits[0, finger, 1])
        self.finger_id = finger
        self.ee = robot.find_bodies(EE_BODY)[0][0]
        self.gripper_body = robot.find_bodies(GRIPPER_BODY)[0][0]
        meta = os.path.join(os.path.dirname(robot.cfg.spawn.usd_path), "metadata.yaml")
        import yaml

        with open(meta) as f:
            off = yaml.safe_load(f)["gripper_offset"]
        self.tcp_pos = torch.tensor(off["pos"], device=robot.device)
        self.tcp_quat = torch.tensor(off["quat"], device=robot.device)
        self.hand_points, self.hand_point_body = hand_collision_points(robot)

        self.aperture = None  # (finger_joint angles, pad gaps), set by calibrate_aperture

    def calibrate_aperture(self, sim, steps: int = 48):
        """Tabulate the gap between the two fingertip pads (collision geometry) over finger_joint's range.

        The Robotiq position reading is linear in stroke (85 mm open .. 0 closed), while the fingertips move on a
        linkage, so the reading maps onto finger_joint through this table: gap(angle) = gap(0) * (1 - reading).
        """
        robot = self.robot
        tips = [robot.body_names.index(n) for n in ("left_fingertip", "right_fingertip")]
        angles = np.linspace(0.0, self.finger_closed, steps)
        saved = robot.data.joint_pos.clone()
        gaps = []
        for angle in angles:
            jp = saved.clone()
            jp[:, self.grip_ids] = float(angle) * self.grip_mult[None]
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            sim.forward()
            pos, quat = link_poses(robot)
            pads = []
            for k in tips:
                m = self.hand_point_body == k
                pads.append(math_utils.quat_apply(quat[0, k].expand(int(m.sum()), 4), self.hand_points[m]) + pos[0, k])
            gaps.append(float(torch.cdist(pads[0], pads[1]).min()))
        robot.write_joint_state_to_sim(saved, torch.zeros_like(saved))
        sim.forward()
        gaps = np.minimum.accumulate(np.asarray(gaps))  # closing never reopens the gap
        self.aperture = (angles, gaps)
        mid = float(np.interp(0.5 * gaps[0], gaps[::-1], angles[::-1]))
        print(
            f"{self.name}: fingertip gap {gaps[0] * 1e3:.1f} mm open -> {gaps[-1] * 1e3:.1f} mm at "
            f"{math.degrees(angles[-1]):.0f} deg; reading 0.5 -> finger_joint {math.degrees(mid):.1f} deg "
            f"(linear map: {math.degrees(0.5 * self.finger_closed):.1f})"
        )

    def reading(self, finger_angle: float) -> float:
        """Inverse of the mapping: finger_joint angle -> the gripper reading it corresponds to."""
        if self.aperture is None:
            return finger_angle / self.finger_closed
        angles, gaps = self.aperture
        return float(1.0 - np.interp(finger_angle, angles, gaps) / gaps[0])

    def gripper_joint_pos(self, opening: torch.Tensor) -> torch.Tensor:
        """(N,) gripper reading (0 open, 1 closed) -> (N, gripper joints) positions."""
        g = opening.clamp(0, 1)
        if self.aperture is None:
            angle = g * self.finger_closed
        else:
            angles, gaps = self.aperture
            target = gaps[0] * (1.0 - g.cpu().numpy())
            angle = torch.tensor(np.interp(target, gaps[::-1], angles[::-1]), dtype=g.dtype, device=g.device)
        return angle[:, None] * self.grip_mult[None]


def hand_collision_points(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Collision-geometry points of the hand bodies in their body frames (mesh vertices, or box corners otherwise)."""
    stage = omni.usd.get_context().get_stage()
    root = robot_prim_path(robot)
    xcache, bcache = UsdGeom.XformCache(), UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "proxy", "guide"])
    points, owners = [], []
    proxies = Usd.TraverseInstanceProxies()
    bodies = {
        p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(root), proxies) if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for k, name in enumerate(robot.body_names):
        body = bodies.get(name)
        if body is None or not (name in HAND_BODIES or "/OfficialHand/" in body.GetPath().pathString):
            continue
        to_body = np.linalg.inv(np.array(xcache.GetLocalToWorldTransform(body)).T)
        it = iter(Usd.PrimRange(body, proxies))
        for prim in it:
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Gprim)):
                continue
            m = to_body @ np.array(xcache.GetLocalToWorldTransform(prim)).T
            if prim.IsA(UsdGeom.Mesh):
                local = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            else:
                box = bcache.ComputeLocalBound(prim).GetRange()
                lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
                local = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts = local @ m[:3, :3].T + m[:3, 3]
            points.append(pts)
            owners.append(np.full(len(pts), k))
    if not points:
        raise RuntimeError(f"No hand collision geometry under {root}")
    return (
        torch.tensor(np.concatenate(points), dtype=torch.float32, device=robot.device),
        torch.tensor(np.concatenate(owners), dtype=torch.long, device=robot.device),
    )


def link_poses(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """(envs, links, 3) positions and (envs, links, 4) w-x-y-z quaternions straight from PhysX."""
    tf = robot.root_physx_view.get_link_transforms().clone()
    return tf[..., :3], tf[..., [6, 3, 4, 5]]


def hand_lowest(
    arm: Arm, pos: torch.Tensor, quat: torch.Tensor, origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest hand collision point over the tabletop footprint: height above the tabletop (envs,), +inf if none, and
    its env-local position (envs, 3)."""
    body_pos, body_quat = pos[:, arm.hand_point_body], quat[:, arm.hand_point_body]  # (E, P, ...)
    world = math_utils.quat_apply(body_quat, arm.hand_points[None].expand_as(body_pos)) + body_pos
    local = world - origins[:, None]
    inside = (
        (local[..., 0] > lab.TABLE_X_RANGE[0])
        & (local[..., 0] < lab.TABLE_X_RANGE[1])
        & (local[..., 1] > lab.TABLE_Y_RANGE[0])
        & (local[..., 1] < lab.TABLE_Y_RANGE[1])
    )
    height = torch.where(inside, local[..., 2] - lab.WORK_SURFACE_Z, torch.full_like(local[..., 2], float("inf")))
    low, idx = height.min(dim=1)
    return low, local[torch.arange(len(idx), device=idx.device), idx]


def flange_in_base(arm: Arm, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """(envs, 4, 4) sim flange (wrist_3_link) in the controller Base frame."""
    robot = arm.robot
    base = torch.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    base[:, :3, :3] = math_utils.matrix_from_quat(robot.data.root_quat_w)
    base[:, :3, 3] = robot.data.root_pos_w
    ee = torch.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    ee[:, :3, :3] = math_utils.matrix_from_quat(quat[:, arm.ee])
    ee[:, :3, 3] = pos[:, arm.ee]
    return RZ180.to(pos.device) @ torch.linalg.inv(base) @ ee


def tcp_world(arm: Arm, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    n = pos.shape[0]
    return math_utils.combine_frame_transforms(
        pos[:, arm.gripper_body], quat[:, arm.gripper_body], arm.tcp_pos.expand(n, 3), arm.tcp_quat.expand(n, 4)
    )[0]


def rotation_error_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = a.transpose(-1, -2) @ b
    sin = 0.5 * torch.stack([m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1).norm(
        dim=-1
    )
    cos = 0.5 * (m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)
    return torch.rad2deg(torch.atan2(sin, cos))


def place_bases(scene):
    for name in ("thunder", "lightning"):
        robot = scene[name]
        root = robot.data.default_root_state.clone()
        root[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_position_target(robot.data.default_joint_pos)


def stats(x: np.ndarray, scale: float = 1e3) -> str:
    x = x[np.isfinite(x)] * scale
    if len(x) == 0:
        return "n/a"
    return f"min {x.min():.1f} / p1 {np.percentile(x, 1):.1f} / median {np.median(x):.1f} / max {x.max():.1f}"


# ---------------------------------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------------------------------


def kinematic_check(sim, scene, rec: Recording) -> dict:
    arms = {name: Arm(name, scene[name]) for name in rec.arms}
    for arm in arms.values():
        arm.calibrate_aperture(sim)
    n_env = scene.num_envs
    t_all = np.concatenate([rec.data[f"{a}.t"] for a in rec.arms])
    times = np.unique(t_all)
    if len(times) > args.max_frames:
        times = times[np.linspace(0, len(times) - 1, args.max_frames).astype(int)]
    out = {a: {k: [] for k in ("t", "pos_err", "rot_err", "tcp_h", "hand_h", "hand_xyz", "grip")} for a in rec.arms}
    for b in range(0, len(times), n_env):
        t = times[b : b + n_env]
        t_pad = np.concatenate([t, np.repeat(t[-1:], n_env - len(t))])
        for name, arm in arms.items():
            robot = arm.robot
            q = torch.tensor(rec.q(name, t_pad), dtype=torch.float32, device=sim.device)
            g = torch.tensor(rec.gripper(name, t_pad), dtype=torch.float32, device=sim.device)
            jp = robot.data.default_joint_pos.clone()
            jp[:, arm.arm_ids] = q
            jp[:, arm.grip_ids] = arm.gripper_joint_pos(g)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        sim.forward()
        for name, arm in arms.items():
            pos, quat = link_poses(arm.robot)
            flange = flange_in_base(arm, pos, quat)
            rt = rec.data[f"{name}.t"]
            k = np.clip(np.searchsorted(rt, t_pad), 0, len(rt) - 1)
            rec_pos = torch.tensor(rec.data[f"{name}.flange_pos"][k], dtype=torch.float32, device=sim.device)
            rec_rot = math_utils.matrix_from_quat(
                torch.tensor(rec.data[f"{name}.flange_quat"][k], dtype=torch.float32, device=sim.device)
            )
            exact = torch.tensor(np.abs(rt[k] - t_pad) < 1e-6, device=sim.device)  # sample times of this arm only
            sel = slice(0, len(t))
            o = out[name]
            o["t"].append(t)
            o["pos_err"].append(
                torch.where(exact, (flange[:, :3, 3] - rec_pos).norm(dim=-1), torch.nan)[sel].cpu().numpy()
            )
            o["rot_err"].append(
                torch.where(exact, rotation_error_deg(flange[:, :3, :3], rec_rot), torch.nan)[sel].cpu().numpy()
            )
            o["tcp_h"].append(
                (tcp_world(arm, pos, quat)[:, 2] - scene.env_origins[:, 2] - lab.WORK_SURFACE_Z)[sel].cpu().numpy()
            )
            low, low_xyz = hand_lowest(arm, pos, quat, scene.env_origins)
            o["hand_h"].append(low[sel].cpu().numpy())
            o["hand_xyz"].append(low_xyz[sel].cpu().numpy())
            o["grip"].append(rec.gripper(name, t))
    summary = {}
    print(f"\n=== Kinematic check: {len(times)} recorded frames, {rec.meta.get('source')} ===")
    for name in rec.arms:
        o = {k: np.concatenate(v) for k, v in out[name].items()}
        below = o["hand_h"] < 0
        worst = int(np.nanargmin(np.where(np.isfinite(o["hand_h"]), o["hand_h"], np.inf)))
        print(f"{name}:")
        # The robot PC reads joints and flange pose in two separate RTDE calls, which can straddle a 2 ms control
        # cycle; the residual grows with joint speed. Frames with the arm nearly still isolate the sim's own error.
        speed = np.abs(rec.q(name, o["t"] + 0.01) - rec.q(name, o["t"] - 0.01)).max(axis=1) / 0.02
        still = speed < 0.02
        pe, re = o["pos_err"] * 1e3, o["rot_err"]
        print(
            f"  flange vs controller: median {np.nanmedian(pe):.4f} / p99 {np.nanpercentile(pe, 99):.4f} / max "
            f"{np.nanmax(pe):.4f} mm, max {np.nanmax(re):.4f} deg; arm nearly still (< 0.02 rad/s, "
            f"{still.mean() * 100:.0f}% of frames): max {np.nanmax(np.where(still, pe, np.nan)):.4f} mm, "
            f"{np.nanmax(np.where(still, re, np.nan)):.5f} deg"
        )
        o["joint_speed"] = speed
        print(f"  grasp point above the work surface (cloth top, mm): {stats(o['tcp_h'])}")
        print(f"  lowest hand collision point above the work surface (cloth top, mm): {stats(o['hand_h'])}")
        print(
            f"  hand below the work surface in {below.mean() * 100:.2f}% of frames (> 5 mm deep: "
            f"{(o['hand_h'] < -0.005).mean() * 100:.2f}%); deepest {o['hand_h'][worst] * 1e3:.1f} mm at "
            f"t = {o['t'][worst]:.2f} s, gripper {o['grip'][worst]:.2f}"
        )
        summary[name] = o
    return summary


def replay(sim, scene, rec: Recording) -> dict:
    arms = {name: Arm(name, scene[name]) for name in rec.arms}
    for arm in arms.values():
        arm.calibrate_aperture(sim)
    dt = sim.get_physics_dt()
    t0 = max(args.start, max(rec.data[f"{a}.t"][0] for a in rec.arms))
    t1 = rec.t_end if args.duration is None else min(rec.t_end, t0 + args.duration)
    times = np.arange(t0, t1, dt)
    targets, grip = {}, {}
    for name, arm in arms.items():
        if args.arm == "servo":
            delay, tau, rms = fit_servo_model(rec, name)
            print(
                f"{name}: servoJ model fitted to the recording: delay {delay * 1e3:.0f} ms, lag {tau * 1e3:.0f} ms "
                f"(measured vs model {math.degrees(rms):.3f} deg RMS while teleop is active)"
            )
            targets[name] = servo_targets(rec, name, times, delay, tau)
        else:
            targets[name] = rec.q(name, times)
        grip[name] = rec.gripper(name, times, commanded=args.arm == "servo")
    # Start from the measured state.
    for name, arm in arms.items():
        robot = arm.robot
        jp = robot.data.default_joint_pos.clone()
        jp[:, arm.arm_ids] = torch.tensor(rec.q(name, times[:1]), dtype=torch.float32, device=sim.device)
        jp[:, arm.grip_ids] = arm.gripper_joint_pos(
            torch.tensor(grip[name][:1], device=sim.device, dtype=torch.float32)
        )
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
    scene.write_data_to_sim()
    sim.forward()

    video = None
    render_every = None
    if args.video or args.stills:
        cam = scene["overview"]
        names = rec.arms
        mid = np.mean([np.asarray(getattr(lab, f"{n.upper()}_POS")) for n in names], axis=0)
        eye = torch.tensor([[float(mid[0]), -1.05, 1.60]], device=sim.device) + scene.env_origins[:1]
        tgt = torch.tensor([[float(mid[0]), 0.0, lab.TABLE_TOP_Z]], device=sim.device) + scene.env_origins[:1]
        cam.set_world_poses_from_view(eye, tgt)
        for _ in range(20):  # RTX warm-up
            sim.render()
    if args.video:
        import imageio.v2 as imageio

        video = imageio.get_writer(args.video, fps=args.video_fps, codec="libx264", quality=7, macro_block_size=8)
        render_every = max(1, int(round(1.0 / (args.video_fps * dt))))

    log = {f"{n}.{k}": [] for n in arms for k in ("q", "grip", "hand_h", "tcp_h")}
    log["t"] = []
    for i, t in enumerate(times):
        for name, arm in arms.items():
            robot = arm.robot
            q = torch.tensor(targets[name][i : i + 1], dtype=torch.float32, device=sim.device)
            if args.arm == "teleport":
                qd = (targets[name][min(i + 1, len(times) - 1)] - targets[name][i]) / dt
                robot.write_joint_state_to_sim(
                    q, torch.tensor(qd[None], dtype=torch.float32, device=sim.device), joint_ids=arm.arm_ids
                )
            robot.set_joint_position_target(q, joint_ids=arm.arm_ids)
            g = torch.tensor(grip[name][i : i + 1], dtype=torch.float32, device=sim.device)
            robot.set_joint_position_target(arm.gripper_joint_pos(g)[:, :1], joint_ids=[arm.grip_ids[0]])
        render = video is not None and i % render_every == 0
        scene.write_data_to_sim()
        sim.step(render=render)
        scene.update(dt)
        log["t"].append(t)
        for name, arm in arms.items():
            robot = arm.robot
            pos, quat = link_poses(robot)
            log[f"{name}.q"].append(robot.data.joint_pos[0, arm.arm_ids].cpu().numpy())
            log[f"{name}.grip"].append(arm.reading(float(robot.data.joint_pos[0, arm.finger_id])))
            log[f"{name}.hand_h"].append(float(hand_lowest(arm, pos, quat, scene.env_origins)[0][0]))
            log[f"{name}.tcp_h"].append(
                float(tcp_world(arm, pos, quat)[0, 2] - scene.env_origins[0, 2] - lab.WORK_SURFACE_Z)
            )
        if render:
            video.append_data(compose_frame(scene, rec, t))
        if i % int(round(10.0 / dt)) == 0:
            print(f"  replay t = {t:.1f} / {t1:.1f} s")
    if video is not None:
        video.close()
        print(f"Wrote {args.video}")
    log = {k: np.asarray(v) for k, v in log.items()}

    print(f"\n=== Replay ({args.arm}), {t0:.2f}-{t1:.2f} s, {rec.meta.get('source')} ===")
    for name in arms:
        measured = rec.q(name, log["t"])
        err = np.degrees(np.abs(log[f"{name}.q"] - measured))
        active = rec.active(log["t"])
        g_err = np.abs(log[f"{name}.grip"] - rec.gripper(name, log["t"]))
        print(f"{name}:")
        print(
            f"  sim joints vs measured (deg): RMS {np.sqrt(np.mean(err**2, axis=0)).round(3).tolist()}, "
            f"max {err.max():.3f}" + (f" (teleop active: max {err[active].max():.3f})" if active.any() else "")
        )
        print(
            f"  gripper opening vs measured: mean |error| {g_err.mean():.3f}, p95 {np.percentile(g_err, 95):.3f} "
            "(0..1 scale; the real fingers stop on objects, the sim's close freely)"
        )
        print(f"  grasp point above the work surface (cloth top, mm): {stats(log[f'{name}.tcp_h'])}")
        print(f"  lowest hand collision point above the work surface (cloth top, mm): {stats(log[f'{name}.hand_h'])}")
    if args.stills:
        save_stills(sim, scene, rec, arms)
    return log


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sim_image(scene, name: str) -> np.ndarray:
    return scene[name].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)


def undistort_l515(img: np.ndarray) -> np.ndarray:
    """Remove the scene_1 L515's lens distortion so the frame matches the pinhole sim camera."""
    import cv2

    k = np.array(lab.L515_COLOR_INTRINSICS, dtype=np.float64).reshape(3, 3)
    return cv2.undistort(np.ascontiguousarray(img), k, np.array(lab.L515_COLOR_DISTORTION))


def compose_frame(scene, rec: Recording, t: float) -> np.ndarray:
    blank = np.zeros((lab.D405_HEIGHT, lab.D405_WIDTH, 3), dtype=np.uint8)
    panels = []
    # A panel for every camera in the recording, blank while it has no frame, so the frame size stays fixed.
    for arm in rec.arms:
        if f"{arm}_wrist" in rec.images:
            real = rec.image(f"{arm}_wrist", t)
            panels.append(label(blank.copy() if real is None else real.copy(), f"real {arm} wrist D405  t={t:.1f}s"))
        panels.append(label(sim_image(scene, f"{arm}_wrist_camera"), f"sim {arm} wrist D405"))
    if "scene_1" in rec.images:
        real_scene = rec.image("scene_1", t)
        panels.append(
            label(blank.copy() if real_scene is None else undistort_l515(real_scene), "real scene_1 L515 (undistorted)")
        )
        panels.append(label(sim_image(scene, "scene_camera"), f"sim scene_1 L515 (calibrated pose)  t={t:.1f}s"))
    else:
        panels.append(label(sim_image(scene, "overview"), f"sim  t={t:.1f}s"))
    if len(panels) % 2:
        panels.append(blank)
    rows = [np.concatenate(panels[i : i + 2], axis=1) for i in range(0, len(panels), 2)]
    return np.concatenate(rows, axis=0)


def save_stills(sim, scene, rec: Recording, arms: dict):
    """Real camera frames next to the sim camera at the same recorded joints and gripper (kinematic): each arm's wrist
    D405 and the scene_1 L515 (real frame undistorted)."""
    from PIL import Image

    os.makedirs(args.stills, exist_ok=True)
    streams = {f"{name}_wrist": f"{name}_wrist_camera" for name in arms}
    streams["scene_1"] = "scene_camera"
    for key, sim_camera in streams.items():
        if key not in rec.images:
            continue
        times = rec.images[key][0]
        for t in times[np.linspace(0, len(times) - 1, args.num_stills).astype(int)]:
            for other, a in arms.items():
                robot = a.robot
                jp = robot.data.joint_pos.clone()
                jp[:, a.arm_ids] = torch.tensor(rec.q(other, np.array([t])), dtype=torch.float32, device=sim.device)
                g = torch.tensor(rec.gripper(other, np.array([t])), dtype=torch.float32, device=sim.device)
                jp[:, a.grip_ids] = a.gripper_joint_pos(g)
                robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
                robot.set_joint_position_target(jp)
            scene.write_data_to_sim()
            sim.forward()
            for _ in range(12):
                sim.render()
            scene.update(sim.get_physics_dt())
            real = rec.image(key, t)
            if key == "scene_1":
                real = undistort_l515(real)
            simulated = sim_image(scene, sim_camera)
            blend = (0.5 * real.astype(np.float32) + 0.5 * simulated.astype(np.float32)).astype(np.uint8)
            row = np.concatenate(
                [label(real.copy(), f"real {key} t={t:.2f}s"), label(simulated, "sim"), label(blend, "50/50 overlay")],
                1,
            )
            path = os.path.join(args.stills, f"{key}_{t:06.2f}.png")
            Image.fromarray(row).save(path)
            print(f"Saved {path}")


# ---------------------------------------------------------------------------------------------------------------------


def main():
    rec = Recording(args.recording)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    cameras = args.mode == "replay" and (args.video or args.stills)
    cfg_cls = lab.RpmLabDualArmCameraSceneCfg if cameras else lab.RpmLabDualArmSceneCfg
    scene_cfg = cfg_cls(num_envs=args.envs if args.mode == "kinematic" else 1, env_spacing=4.0)
    if cameras:
        scene_cfg.thunder_wrist_camera.update_period = 0.0
        scene_cfg.lightning_wrist_camera.update_period = 0.0
        scene_cfg.scene_camera.update_period = 0.0
        scene_cfg.overview = CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverviewCamera",
            update_period=0.0,
            width=lab.D405_WIDTH,
            height=lab.D405_HEIGHT,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 20.0)),
        )
    if args.mode == "replay" and args.arm == "servo":
        for name in ("thunder", "lightning"):
            arm_actuator = getattr(scene_cfg, name).actuators["arm"]
            arm_actuator.stiffness = dict(zip(ARM_JOINTS, SERVO_STIFFNESS))
            arm_actuator.damping = dict(zip(ARM_JOINTS, SERVO_DAMPING))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    reattach_mounted_cameras(scene)  # wrist cameras follow their links (rpm_lab/sensors.py)
    place_bases(scene)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())
    try:
        result = kinematic_check(sim, scene, rec) if args.mode == "kinematic" else replay(sim, scene, rec)
        if args.out:
            flat = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
                else:
                    flat[k] = v
            np.savez_compressed(args.out, **flat)
            print(f"Wrote {args.out}")
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()  # close() ends the process with os._exit, which would drop the traceback
        raise
    finally:
        # Also on errors: an exception that skips close() leaves the Kit app running and the process hangs.
        simulation_app.close()

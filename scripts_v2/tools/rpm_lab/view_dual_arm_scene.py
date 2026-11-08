# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the RPM Lab workcell with Thunder and Lightning, check it against the real arms, optionally render it.

Checks: both mounts (Lightning's base is the mirror image of Thunder's, each base sits on its mounting plate), each
arm's kinematics against that arm's UR factory calibration, the gripper pose at the home postures, and that both arms
hold still.

Examples:
    # GUI
    python scripts_v2/tools/rpm_lab/view_dual_arm_scene.py
    # Headless check and three rendered views
    python scripts_v2/tools/rpm_lab/view_dual_arm_scene.py --headless --save_images ./rpm_lab_views
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# simulation_app.close() ends the process with os._exit, which drops block-buffered output when stdout is a file.
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hold_steps", type=int, default=240, help="Physics steps used to check that both arms stay put.")
parser.add_argument(
    "--save_images", type=str, default=None, help="Directory for the rendered views and wrist-camera images."
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.save_images:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import math
import os
import torch
import yaml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg
from pxr import Usd, UsdGeom

from uwlab_tasks.manager_based.manipulation.rpm_lab import scene_cfg as lab

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
CALIBRATION_ENTRIES = ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur5e_calibration")
EE_BODY = "wrist_3_link"  # its frame is the controller's tool0 (flange)
GRIPPER_BODY = "robotiq_base_link"
X = lab.MIRROR_PLANE_X


def mirror_points(p: torch.Tensor) -> torch.Tensor:
    """Reflect env-local points across the plane x = MIRROR_PLANE_X."""
    out = p.clone()
    out[..., 0] = 2.0 * X - out[..., 0]
    return out


def body_pos(robot, scene, name: str) -> torch.Tensor:
    return robot.data.body_pos_w[:, robot.find_bodies(name)[0][0]] - scene.env_origins


def body_quat(robot, name: str) -> torch.Tensor:
    return robot.data.body_quat_w[:, robot.find_bodies(name)[0][0]]


def metadata(robot) -> dict:
    with open(os.path.join(os.path.dirname(robot.cfg.spawn.usd_path), "metadata.yaml")) as f:
        return yaml.safe_load(f)


def tcp(robot, scene) -> tuple[torch.Tensor, torch.Tensor]:
    """Grasp point: ``gripper_offset`` from the asset's metadata.yaml applied to robotiq_base_link (as in OmniReset)."""
    offset = metadata(robot)["gripper_offset"]
    pos, quat = body_pos(robot, scene, GRIPPER_BODY), body_quat(robot, GRIPPER_BODY)
    offset_pos = torch.tensor(offset["pos"], device=pos.device).expand_as(pos)
    offset_quat = torch.tensor(offset["quat"], device=pos.device).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, offset_pos, offset_quat)


def reset_to_defaults(scene):
    """Write each arm's ``init_state`` (base_link pose and joints) into the simulation, as task reset events do.

    InteractiveScene only stores ``init_state`` as default buffers. Without this, the arms start at the USD's zero
    joints, and base_link keeps the asset's authored 180 deg about z under the robot prim (see scene_cfg).
    """
    for name in ("thunder", "lightning"):
        robot = scene[name]
        root = robot.data.default_root_state.clone()
        root[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_position_target(robot.data.default_joint_pos)


def step(sim, scene, render: bool = False):
    scene.write_data_to_sim()
    sim.step(render=render)
    scene.update(sim.get_physics_dt())


def mounting_plates(env_index: int = 0) -> list[tuple[list[float], list[float]]]:
    """World AABBs of the two side mounting plates (27 mm along x, 180 mm square) in the lab CAD."""
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    root = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Table")
    plates = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Cube":
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = list(box.GetMin()), list(box.GetMax())
        size = [b - a for a, b in zip(mn, mx)]
        if 0.02 < size[0] < 0.035 and 0.17 < size[1] < 0.19 and 0.17 < size[2] < 0.19:
            plates.append((mn, mx))
    return plates


def factory_flange_pose(robot) -> torch.Tensor:
    """tool0 in the controller Base frame, from the arm's UR factory calibration, as the real controller computes it.

    The calibration file is chosen by the asset's ``metadata.yaml`` and must carry the same hash.
    """
    cal = metadata(robot)["kinematics_calibration"]
    with open(os.path.join(CALIBRATION_DIR, f"{cal['robot']}_kinematics.yaml")) as f:
        k = yaml.safe_load(f)["kinematics"]
    if k["hash"] != cal["hash"]:
        raise RuntimeError(f"{robot.cfg.prim_path}: asset calibration {cal['hash']}, file {k['hash']}")
    q = robot.data.joint_pos[0, robot.find_joints(ARM_JOINTS, preserve_order=True)[0]].tolist()
    T = torch.eye(4, dtype=torch.float64)
    for entry, qi in zip(CALIBRATION_ENTRIES, q):
        e = {a: float(v) for a, v in k[entry].items()}
        origin = torch.eye(4, dtype=torch.float64)
        origin[:3, :3] = _rot(2, e["yaw"]) @ _rot(1, e["pitch"]) @ _rot(0, e["roll"])  # URDF rpy
        origin[:3, 3] = torch.tensor([e["x"], e["y"], e["z"]], dtype=torch.float64)
        joint = torch.eye(4, dtype=torch.float64)
        joint[:3, :3] = _rot(2, qi)
        T = T @ origin @ joint
    return T


def rotation_angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angle between two rotation matrices. atan2 keeps small angles exact, unlike acos of the trace."""
    m = a.double().T @ b.double()
    sin = 0.5 * torch.stack([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]).norm()
    return math.degrees(math.atan2(sin.item(), 0.5 * (m.trace().item() - 1.0)))


def _rot(axis: int, angle: float) -> torch.Tensor:
    """Rotation about x (0), y (1) or z (2)."""
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    r = torch.eye(3, dtype=torch.float64)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def sim_flange_pose(robot) -> torch.Tensor:
    """wrist_3_link in the controller Base frame, which is base_link turned 180 deg about z (ur_description)."""
    to_base = torch.eye(4, dtype=torch.float64)
    to_base[:3, :3] = math_utils.matrix_from_quat(robot.data.root_quat_w[0]).double().cpu()
    to_base[:3, 3] = robot.data.root_pos_w[0].double().cpu()
    ee = torch.eye(4, dtype=torch.float64)
    k = robot.find_bodies(EE_BODY)[0][0]
    ee[:3, :3] = math_utils.matrix_from_quat(robot.data.body_quat_w[0, k]).double().cpu()
    ee[:3, 3] = robot.data.body_pos_w[0, k].double().cpu()
    rz180 = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64))
    return rz180 @ torch.linalg.inv(to_base) @ ee


def report(sim, scene) -> dict[str, float]:
    thunder, lightning = scene["thunder"], scene["lightning"]
    out = {}
    print("\n=== Mount poses (env-local, base_link) ===")
    for name, robot, cfg in (("Thunder", thunder, lab.THUNDER_CFG), ("Lightning", lightning, lab.LIGHTNING_CFG)):
        pos = robot.data.root_pos_w[0] - scene.env_origins[0]
        quat = robot.data.root_quat_w[0]
        rot = math_utils.matrix_from_quat(quat)
        cfg_err = (pos - torch.tensor(cfg.init_state.pos, device=pos.device)).norm().item()
        print(f"{name:9s} pos={[round(v, 6) for v in pos.tolist()]} quat={[round(v, 4) for v in quat.tolist()]}")
        print(
            f"{'':9s} base x->{rot[:, 0].round().tolist()} base y->{rot[:, 1].round().tolist()} "
            f"base z->{rot[:, 2].round().tolist()}   |pos - cfg| = {cfg_err * 1e3:.3f} mm   "
            f"cable opening (base -y) -> {(-rot[:, 1]).round().tolist()}"
        )
    t_pos = thunder.data.root_pos_w[0] - scene.env_origins[0]
    l_pos = lightning.data.root_pos_w[0] - scene.env_origins[0]
    mirrored = math_utils.matrix_from_quat(thunder.data.root_quat_w[0]).clone()
    mirrored[0, :] *= -1.0  # reflect world x
    mirrored[:, 0] *= -1.0  # negate base x: right-handed again (scene_cfg.mirror_mount_pose)
    out["base_mirror_mm"] = (mirror_points(t_pos) - l_pos).norm().item() * 1e3
    out["base_mirror_deg"] = rotation_angle_deg(mirrored, math_utils.matrix_from_quat(lightning.data.root_quat_w[0]))
    print(
        f"Lightning base vs mirror image of Thunder's: {out['base_mirror_mm']:.3f} mm, {out['base_mirror_deg']:.4f} deg"
    )

    print("\n=== Mounting plates (lab CAD) ===")
    plates = mounting_plates()
    print(f"Found {len(plates)} plates")
    origin = scene.env_origins[0].tolist()
    for name, robot in (("Thunder", thunder), ("Lightning", lightning)):
        base = (robot.data.root_pos_w[0]).tolist()
        best = min(plates, key=lambda b: abs(0.5 * (b[0][0] + b[1][0]) - base[0]))
        side = 1.0 if 0.5 * (best[0][0] + best[1][0]) - origin[0] > X else -1.0
        outer = best[1][0] if side > 0 else best[0][0]  # the face away from the column
        depth = side * (outer - base[0]) * 1e3  # > 0: base_link origin lies behind the face
        dy = (base[1] - 0.5 * (best[0][1] + best[1][1])) * 1e3
        dz = (base[2] - 0.5 * (best[0][2] + best[1][2])) * 1e3
        print(
            f"{name:9s} plate x[{best[0][0] - origin[0]:.5f}, {best[1][0] - origin[0]:.5f}] outer face "
            f"x={outer - origin[0]:.5f}; base_link is {depth:+.2f} mm inside the face, offset from plate center "
            f"y {dy:+.2f} mm, z {dz:+.2f} mm"
        )

    print("\n=== Kinematics vs each arm's UR factory calibration (flange in controller Base) ===")
    for name, robot in (("Thunder", thunder), ("Lightning", lightning)):
        sim_T, cal_T = sim_flange_pose(robot), factory_flange_pose(robot)
        pos_err = (sim_T[:3, 3] - cal_T[:3, 3]).norm().item() * 1e3
        rot_err = rotation_angle_deg(sim_T[:3, :3], cal_T[:3, :3])
        cal = metadata(robot)["kinematics_calibration"]
        out[f"{name.lower()}_calibration_mm"] = pos_err
        print(f"{name:9s} {cal['robot']} {cal['hash']}: {pos_err:.4f} mm, {rot_err:.5f} deg")

    print("\n=== Arm joint limits (real UR5e: +-360 deg, elbow +-180 deg, 180 deg/s) ===")
    for name, robot in (("Thunder", thunder), ("Lightning", lightning)):
        ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos_lim = torch.rad2deg(robot.data.joint_pos_limits[0, ids]).round().int().tolist()
        vel_lim = torch.rad2deg(robot.data.joint_vel_limits[0, ids]).round().int().tolist()
        print(f"{name:9s} position {pos_lim} deg, speed {vel_lim} deg/s")

    print("\n=== Gripper at the home postures ===")
    for name, robot in (("Thunder", thunder), ("Lightning", lightning)):
        pos, _ = tcp(robot, scene)
        approach = math_utils.matrix_from_quat(body_quat(robot, GRIPPER_BODY))[0, :, 0]  # robotiq_base_link +x
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -approach[2].item()))))
        offset = metadata(robot)["gripper_offset"]["pos"][0]
        print(
            f"{name:9s} TCP {[round(v, 4) for v in pos[0].tolist()]}, {(pos[0, 2].item() - lab.TABLE_TOP_Z) * 1e3:.1f} mm "
            f"above the tabletop, approach axis {tilt:.1f} deg from straight down (gripper_offset {offset:.4f} m)"
        )
    return out


def hold_check(sim, scene, steps: int):
    print(f"\n=== Hold check: {steps} physics steps ({steps * sim.get_physics_dt():.1f} s) ===")
    start = {n: scene[n].data.joint_pos.clone() for n in ("thunder", "lightning")}
    for _ in range(steps):
        step(sim, scene)
    for name in ("thunder", "lightning"):
        robot = scene[name]
        ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        drift = (robot.data.joint_pos[:, ids] - start[name][:, ids]).abs().max().item()
        print(f"{name:9s} max arm joint drift {math.degrees(drift):.4f} deg")


def save_views(sim, scene, directory: str):
    os.makedirs(directory, exist_ok=True)
    from PIL import Image

    camera = scene["camera"]
    origin = scene.env_origins[0]
    views = {
        "front": ((X, -2.6, 2.0), (X, 0.15, 1.15)),
        "oblique": ((X + 2.3, -2.1, 2.3), (X, 0.1, 1.1)),
        "top": ((X, -0.7, 4.2), (X, -0.05, 0.84)),
    }
    for name, (eye, target) in views.items():
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device) + origin, torch.tensor([target], device=sim.device) + origin
        )
        for _ in range(30):
            step(sim, scene, render=True)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        path = os.path.join(directory, f"rpm_lab_{name}.png")
        Image.fromarray(rgb).save(path)
        print(f"Saved {path}")
    for arm in ("thunder", "lightning"):  # the wrist D405s at the home postures
        rgb = scene[f"{arm}_wrist_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
        path = os.path.join(directory, f"rpm_lab_{arm}_wrist.png")
        Image.fromarray(rgb).save(path)
        print(f"Saved {path}")


def main():
    for path in (lab.LAB_USD, lab.THUNDER_USD, lab.LIGHTNING_USD):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim.set_camera_view((X + 2.3, -2.1, 2.3), (X, 0.1, 1.1))
    scene_cfg_class = lab.RpmLabDualArmCameraSceneCfg if args.save_images else lab.RpmLabDualArmSceneCfg
    scene_cfg = scene_cfg_class(num_envs=1, env_spacing=4.0)
    if args.save_images:
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ViewCamera",
            update_period=0.0,
            height=800,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 50.0)),
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    reset_to_defaults(scene)
    step(sim, scene)

    try:
        report(sim, scene)
        hold_check(sim, scene, args.hold_steps)
        if args.save_images:
            save_views(sim, scene, args.save_images)
        if not args.headless:
            print("\nViewer open; close the window to exit.")
            while simulation_app.is_running():
                step(sim, scene, render=True)
    finally:
        # Without this, the timeline-stop callback keeps rendering inside simulation_app.close() and never returns.
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

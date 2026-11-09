# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit an arm's base position in the twin from the calibrated scene_1 camera and real recordings.

The scene_1 L515 is calibrated in Thunder's base frame, so its images measure where the other arm really is relative to
Thunder. For frames of a recording (``extract_spark_recording.py --images``), the arm is put at the recorded joints and
gripper and rendered alone through the sim scene camera; its exact silhouette comes from the depth image. The score is
the mean distance from the silhouette outline to the nearest edge in the real frame (Canny, truncated at 15 px), which
needs no segmentation of the real arm and is insensitive to shading, shadows and similar colors. A coordinate search
over the base offset (x, y, z, world) minimizes it.

Real frames are undistorted straight into the sim camera's model (Omniverse ignores the principal point and averages
fx and fy), so the two images compare pixel for pixel.

    python scripts_v2/tools/rpm_lab/fit_arm_base_from_scene_camera.py --headless --arm lightning \\
        --recording ep1.npz --recording ep2.npz --out fit_dir
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--recording", action="append", required=True, help="Recording .npz with scene_1 frames.")
parser.add_argument("--arm", default="lightning", choices=("lightning", "thunder"))
parser.add_argument("--frames", type=int, default=12, help="Frames used per recording.")
parser.add_argument("--steps", default="0.008,0.004,0.002", help="Search step sizes (m), one pass each.")
parser.add_argument("--out", required=True, help="Directory for overlays and the result.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import json
import numpy as np
import os
import torch

import cv2
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.scene import InteractiveScene
from PIL import Image
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
W, H = lab.L515_WIDTH, lab.L515_HEIGHT
EDGE_CLIP_PX = 15.0


def sim_camera_matrix() -> np.ndarray:
    """The pinhole Omniverse actually renders: mean focal length, principal point at the image center."""
    k = lab.L515_COLOR_INTRINSICS
    f = 0.5 * (k[0] + k[4])
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)


def to_sim_model(img: np.ndarray) -> np.ndarray:
    k_real = np.array(lab.L515_COLOR_INTRINSICS, dtype=np.float64).reshape(3, 3)
    return cv2.undistort(
        np.ascontiguousarray(img), k_real, np.array(lab.L515_COLOR_DISTORTION), None, sim_camera_matrix()
    )


def gripper_couplings(stage, root: str) -> dict[str, float]:
    """Follower gripper joint -> multiplier on finger_joint (PhysX mimic: q + gearing * q_ref = 0)."""
    couplings = {"finger_joint": 1.0}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("physxMimicJoint:") and attr.GetName().endswith(":gearing"):
                couplings[prim.GetName()] = -float(attr.Get())
    return couplings


def load_frames(path: str, arm: str, count: int) -> list[dict]:
    """Real frames (sim camera model) with the distance to their nearest edge, and the recorded joints and gripper."""
    d = np.load(path)
    times, rgb = d["image.scene_1.t"], d["image.scene_1.rgb"]
    t_q, q, g = d[f"{arm}.t"], d[f"{arm}.q"], d[f"{arm}.gripper"]
    inside = np.flatnonzero((times > t_q[0]) & (times < t_q[-1]))
    pick = inside[np.linspace(0, len(inside) - 1, min(count, len(inside))).astype(int)]
    frames = []
    for i in pick:
        image = to_sim_model(rgb[i])
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, 20, 60)
        frames.append(
            {
                "name": f"{os.path.basename(path)} t={times[i]:.2f}s",
                "image": image,
                "edge_distance": np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), EDGE_CLIP_PX),
                "joints": np.array([np.interp(times[i], t_q, q[:, j]) for j in range(6)]),
                "gripper": float(np.interp(times[i], t_q, g)),
            }
        )
    return frames


def outline(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    cfg = lab.RpmLabDualArmSceneCfg(num_envs=1, env_spacing=4.0)
    cfg.scene_camera = lab.l515_scene_camera_cfg(data_types=("distance_to_image_plane",), update_period=0.0)
    scene = InteractiveScene(cfg)
    sim.reset()
    stage = omni.usd.get_context().get_stage()
    other = "Thunder" if args.arm == "lightning" else "Lightning"
    for path in ("/World/envs/env_0/Table", "/World/GroundPlane", f"/World/envs/env_0/{other}"):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    robot = scene[args.arm]
    root_default = robot.data.default_root_state.clone()
    root_default[:, :3] += scene.env_origins
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    couplings = gripper_couplings(stage, f"/World/envs/env_0/{args.arm.capitalize()}")
    grip_ids = robot.find_joints(list(couplings), preserve_order=True)[0]
    grip_mult = torch.tensor(list(couplings.values()), device=sim.device)
    finger_closed = float(robot.data.joint_pos_limits[0, grip_ids[0], 1])
    camera = scene["scene_camera"]

    def silhouette(offset, frame) -> np.ndarray:
        root = root_default.clone()
        root[:, :3] += torch.tensor(offset, dtype=torch.float32, device=sim.device)
        robot.write_root_pose_to_sim(root[:, :7])
        jp = robot.data.default_joint_pos.clone()
        jp[0, arm_ids] = torch.tensor(frame["joints"], dtype=torch.float32, device=sim.device)
        jp[0, grip_ids] = frame["gripper"] * finger_closed * grip_mult
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        sim.forward()
        for _ in range(3):
            sim.render()
        scene.update(sim.get_physics_dt())
        depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        return np.isfinite(depth) & (depth > 0.01) & (depth < 5.0)  # only the arm is visible

    def score(offset) -> float:
        """Mean distance (px) from the sim outline to the nearest real edge, over frames with the arm in view."""
        dists = []
        for frame in frames:
            edge = outline(silhouette(offset, frame))
            if edge.sum() > 50:
                dists.append(float(frame["edge_distance"][edge].mean()))
        return float(np.mean(dists))

    frames = [f for path in args.recording for f in load_frames(path, args.arm, args.frames)]
    print(f"{len(frames)} frames from {len(args.recording)} recording(s)")
    best = np.zeros(3)
    best_score = score(best)
    history = [{"offset": best.tolist(), "score": best_score}]
    print(f"nominal base: mean outline-to-edge distance {best_score:.3f} px")
    for step in (float(s) for s in args.steps.split(",")):
        for axis in range(3):
            improved = True
            while improved:
                improved = False
                for sign in (-1, 1):
                    cand = best.copy()
                    cand[axis] += sign * step
                    s = score(cand)
                    history.append({"offset": cand.tolist(), "score": s})
                    if s < best_score - 1e-3:
                        best, best_score, improved = cand, s, True
                        print(f"  {step * 1e3:.0f} mm steps, {'xyz'[axis]}: {np.round(best * 1e3, 1)} mm, {s:.3f} px")
                        break
    print(f"best base offset (world, mm): {np.round(best * 1e3, 1).tolist()}, mean distance {best_score:.3f} px")

    # Score along each axis around the optimum, to judge how well it is determined.
    for axis in range(3):
        curve = []
        for delta in (-0.01, -0.005, 0.0, 0.005, 0.01):
            cand = best.copy()
            cand[axis] += delta
            curve.append(f"{delta * 1e3:+.0f}:{score(cand):.3f}")
        print(f"  around the optimum, {'xyz'[axis]} (mm:px): {' '.join(curve)}")

    os.makedirs(args.out, exist_ok=True)
    for i, frame in enumerate(frames[:: max(1, len(frames) // 8)]):
        panels = []
        for offset, name in ((np.zeros(3), "nominal"), (best, "fitted")):
            overlay = frame["image"].copy()
            overlay[outline(silhouette(offset, frame))] = (255, 0, 0)  # sim outline on the real frame
            cv2.putText(overlay, f"{name}  {frame['name']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            panels.append(overlay)
        Image.fromarray(np.concatenate(panels, 1)).save(os.path.join(args.out, f"overlay_{i:02d}.png"))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(
            {
                "arm": args.arm,
                "offset_m": best.tolist(),
                "mean_edge_distance_px": best_score,
                "nominal_edge_distance_px": history[0]["score"],
                "recordings": args.recording,
                "frames": len(frames),
                "history": history,
            },
            f,
            indent=1,
        )
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

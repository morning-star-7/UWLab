# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit the colors of a twin arm's paint (the UR5e materials) to the real arm in scene_1 frames.

The arm is put at the recorded joints of real frames and rendered in the full scene (lighting, table) through the
scene_1 camera, warped into the real camera model like the dataset renders (``render_mimic_dataset.py``). Which material
covers each pixel comes from one extra render with each material painted a pure, distinct color (the meshes mix
materials over face subsets, so no segmentation output tells them apart). For every material, the median color of the
pixels it covers in the real frames is compared with the sim render, inside masks eroded by ``ERODE_PX`` (the arm
lines up to about 4.5 px; edges, the gripper cable and shadows stay out of the median), and its diffuse color is scaled
by real/sim in linear RGB (damped). A few passes converge.

    python scripts_v2/tools/rpm_lab/fit_arm_paint_from_scene_camera.py --headless \\
        --episodes /data/cui00191/WM-cotrain/real_frames/cook_fish --exclude 13,15,24,29,39 --out fit_dir

Writes ``paint.json`` (material name -> diffuse color) for ``build_calibrated_ur5e_usd.py --paint`` and real | before |
after panels.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episodes", required=True, help="Directory of real episode_XXXX.npz + _scene_1.mp4.")
parser.add_argument("--exclude", default="", help="Comma-separated episode indices to leave out.")
parser.add_argument("--num_episodes", type=int, default=12)
parser.add_argument("--frames", type=int, default=6, help="Frames per episode (spread over its middle 90 %%).")
parser.add_argument("--passes", type=int, default=6)
parser.add_argument(
    "--materials",
    default="URBlue,LinkGrey,wrist_3_link/LinkGrey,JointGrey,Black",
    help="Materials to fit: a material name (all its copies) or <link>/<name> (that link's copy only).",
)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import glob
import json
import numpy as np
import os
import torch

import cv2
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.scene import InteractiveScene
from pxr import Gf, Usd, UsdGeom, UsdShade

from uwlab_tasks.manager_based.manipulation.rpm_lab import scene_cfg as lab
from uwlab_tasks.manager_based.manipulation.rpm_lab.cook_fish_env_cfg import LightningCookFishMimicEnvCfg
from uwlab_tasks.manager_based.manipulation.rpm_lab.pick_place_env_cfg import ARM_JOINTS, FINGER_CLOSED

ROBOT = "/World/envs/env_0/Lightning"
ERODE_PX = 4
DAMPING = 0.7
"""Exponent on each pass's real/sim ratio: a full step overshoots (the render is not quite linear in the albedo)."""
LABEL_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1)]
"""Pure colors the materials are painted for the label render (one each, in ``--materials`` order)."""


def real_model_maps(k, distortion, width, height):
    """cv2.remap maps that turn a sim render into the real camera's image (as in render_mimic_dataset.py)."""
    k_real = np.array(k, dtype=np.float64).reshape(3, 3)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel()], axis=-1).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, k_real, np.array(distortion, dtype=np.float64)).reshape(-1, 2)
    f = 0.5 * (k[0] + k[4])
    map_x = (f * normalized[:, 0] + width / 2).reshape(height, width).astype(np.float32)
    map_y = (f * normalized[:, 1] + height / 2).reshape(height, width).astype(np.float32)
    return map_x, map_y


def diffuse_input(prim):
    """A shader's base color: OmniPBR's ``diffuse_color_constant`` or UsdPreviewSurface's ``diffuseColor``."""
    if prim.IsA(UsdShade.Shader):
        for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
            if prim.GetAttribute(name) and prim.GetAttribute(name).Get() is not None:
                return prim.GetAttribute(name)
    return None


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def gripper_couplings(stage, root: str) -> dict[str, float]:
    """Follower gripper joint -> multiplier on finger_joint (PhysX mimic: q + gearing * q_ref = 0)."""
    couplings = {"finger_joint": 1.0}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("physxMimicJoint:") and attr.GetName().endswith(":gearing"):
                couplings[prim.GetName()] = -float(attr.Get())
    return couplings


def load_frames() -> list[dict]:
    exclude = {int(e) for e in args.exclude.split(",") if e}
    paths = [
        p
        for p in sorted(glob.glob(os.path.join(args.episodes, "episode_*.npz")))
        if int(os.path.basename(p)[8:12]) not in exclude
    ]
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.num_episodes, len(paths))).astype(int)]
    frames = []
    for path in paths:
        state = np.load(path)["state"]
        cap = cv2.VideoCapture(path.replace(".npz", "_scene_1.mp4"))
        video = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            video.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        n = min(len(video), len(state))
        for i in np.linspace(0.05 * n, 0.95 * n, args.frames).astype(int):
            frames.append(
                {
                    "name": f"{os.path.basename(path)[:-4]}:{i}",
                    "real": video[i],
                    "joints": state[i, :6],
                    "gripper": float(np.clip(state[i, 12], 0, 1)),
                }
            )
    return frames


def main():
    env_cfg = LightningCookFishMimicEnvCfg()
    cfg = env_cfg.scene
    cfg.num_envs = 1
    cfg.env_spacing = 8.0
    cfg.scene_camera = lab.l515_scene_camera_cfg(data_types=("rgb",), update_period=0.0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=env_cfg.sim.dt, device=args.device))
    scene = InteractiveScene(cfg)
    sim.reset()
    stage = omni.usd.get_context().get_stage()
    for name in ("Pot", "Lid", "Fish"):  # the real objects lie elsewhere; keep them out of the arm's pixels
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_0/{name}")).MakeInvisible()

    robot = scene["lightning"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    couplings = gripper_couplings(stage, ROBOT)
    grip_ids = robot.find_joints(list(couplings), preserve_order=True)[0]
    grip_mult = torch.tensor(list(couplings.values()), device=sim.device)
    camera = scene["scene_camera"]
    maps = real_model_maps(lab.L515_COLOR_INTRINSICS, lab.L515_COLOR_DISTORTION, lab.L515_WIDTH, lab.L515_HEIGHT)
    gain = np.asarray(lab.CAMERA_COLOR_GAIN["scene_1"], dtype=np.float32)

    names = args.materials.split(",")
    shaders = {name: [] for name in names}  # the copies of each paint material (each link has its own)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if not prim.IsA(UsdShade.Material):
            continue
        link = str(prim.GetPath())[len(ROBOT) + 1 :].split("/")[0]
        key = f"{link}/{prim.GetName()}"
        key = key if key in shaders else prim.GetName()
        if key in shaders:
            for child in Usd.PrimRange(prim):
                attr = diffuse_input(child)
                if attr:
                    shaders[key].append(attr)
    paint = {}
    for name, attrs in shaders.items():
        if not attrs:
            raise RuntimeError(f"No material {name!r} under {ROBOT}")
        paint[name] = np.array(attrs[0].Get(), dtype=np.float64)
        print(f"{name}: {len(attrs)} shaders, diffuse {np.round(paint[name], 4).tolist()}")
    original = {k: v.copy() for k, v in paint.items()}

    def set_paint(colors: dict):
        for name, color in colors.items():
            for attr in shaders[name]:
                attr.Set(Gf.Vec3f(*[float(c) for c in color]))

    def render(frame) -> np.ndarray:
        robot.write_root_pose_to_sim(root[:, :7])  # base_link is authored Rz(180) off until the root pose is written
        jp = robot.data.default_joint_pos.clone()
        jp[0, arm_ids] = torch.tensor(frame["joints"], dtype=torch.float32, device=sim.device)
        jp[0, grip_ids] = frame["gripper"] * FINGER_CLOSED * grip_mult
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        sim.forward()
        for _ in range(6):
            sim.render()
        scene.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        warped = cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        return np.clip(warped * gain, 0, 255).astype(np.uint8)

    frames = load_frames()
    print(f"{len(frames)} real frames")

    # Label render: each material in a pure color; a pixel belongs to the one whose color it shows clearly, where the
    # repaint changed it (the hand has a few saturated parts of its own).
    for frame in frames:
        frame["before"] = render(frame)
    set_paint({name: LABEL_COLORS[i] for i, name in enumerate(names)})
    kernel = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
    for frame in frames:
        img = render(frame).astype(np.float32)
        changed = np.abs(img - frame["before"]).sum(axis=-1) > 60
        lit = changed & (img.max(axis=-1) - img.min(axis=-1) > 40)  # saturated: a label color, not the scene
        unit = img / np.maximum(np.linalg.norm(img, axis=-1, keepdims=True), 1.0)
        refs = np.array(LABEL_COLORS[: len(names)], dtype=np.float32)
        refs /= np.linalg.norm(refs, axis=-1, keepdims=True)
        label = np.argmax(unit @ refs.T, axis=-1)
        frame["masks"] = {
            name: cv2.erode(((label == i) & lit).astype(np.uint8), kernel) > 0 for i, name in enumerate(names)
        }
    counts = {name: sum(int(f["masks"][name].sum()) for f in frames) for name in names}
    print(f"pixels per material (eroded): {counts}")

    history = []
    for p in range(args.passes + 1):
        set_paint(paint)
        sim_px = {name: [] for name in names}
        real_px = {name: [] for name in names}
        for frame in frames:
            img = render(frame)
            frame["after"] = img
            for name in names:
                sim_px[name].append(img[frame["masks"][name]])
                real_px[name].append(frame["real"][frame["masks"][name]])
        report = {}
        for name in names:
            s, r = np.concatenate(sim_px[name]), np.concatenate(real_px[name])
            if len(s) < 200:
                print(f"  {name}: only {len(s)} pixels, left as is")
                continue
            s_med, r_med = np.median(s, axis=0), np.median(r, axis=0)
            report[name] = {"real": r_med.tolist(), "sim": s_med.tolist(), "diffuse": paint[name].tolist()}
            if p < args.passes:
                ratio = srgb_to_linear(r_med) / np.maximum(srgb_to_linear(s_med), 1e-4)
                paint[name] = np.clip(paint[name] * ratio**DAMPING, 0.003, 1.0)
        history.append(report)
        medians = [
            f"{n} real {np.round(v['real']).astype(int).tolist()} sim {np.round(v['sim']).astype(int).tolist()}"
            for n, v in report.items()
        ]
        print(f"pass {p}: " + "; ".join(medians))

    os.makedirs(args.out, exist_ok=True)
    for i, frame in enumerate(frames[:: max(1, len(frames) // 10)]):
        panel = np.concatenate([frame["real"], frame["before"], frame["after"]], axis=1)
        cv2.putText(
            panel, f"{frame['name']}  real | before | after", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )
        cv2.imwrite(os.path.join(args.out, f"paint_{i:02d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    with open(os.path.join(args.out, "paint.json"), "w") as f:
        json.dump(
            {
                "materials": {name: [round(float(c), 4) for c in paint[name]] for name in names},
                "original": {name: [round(float(c), 4) for c in original[name]] for name in names},
                "fitted_on": sorted({f["name"].split(":")[0] for f in frames}),
                "frames": len(frames),
                "eroded_pixels": counts,
                "passes": history,
            },
            f,
            indent=1,
        )
    print(f"paint: { {n: np.round(c, 4).tolist() for n, c in paint.items()} }")
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

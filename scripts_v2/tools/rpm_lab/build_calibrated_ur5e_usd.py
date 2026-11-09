# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Give a calibrated UR5e robot USD a given arm's factory kinematic calibration and the real UR5e joint limits.

Each UR5e controller stores its own factory calibration (``rpm_lab/calibration/<arm>_kinematics.yaml``, copied from
``Hardware/robot_calibration/ur5e`` in RPM-lab-UMN/spark-data-collection; each matches its controller's reported flange
pose to 0.002 mm). The source asset's arm joints carry one arm's calibration in ``physics:localPos0/localRot0``. This
tool writes ``robot.usda``, a small layer that sublayers the source USD unchanged and overrides:

* the six arm joint frames, with the target calibration;
* the authored link poses, rebuilt along the chain at the same authored joint angles, so the stage stays consistent;
* the hand subtree (gripper, fingers, camera mount), moved rigidly with wrist_3_link;
* with ``--real_joint_limits``, the arm joint limits: the controller's +-360 deg (elbow +-180 deg, ur_description
  ``joint_limits.yaml``) instead of the source's +-180 deg wrists. The real arms use wrist angles beyond +-180 deg;
* any light in the source asset (the builds carry a 1000-unit DomeLight), deactivated: the scene owns the lighting,
  and an asset's light would be added again for every robot and every env.

It writes a matching ``metadata.yaml``. Before writing, it checks that the source calibration reproduces the source
USD's joint frames and link poses exactly. Run it with any Python that has ``pxr`` (no Isaac Sim app needed):

    A=/data/cui00191/rpm_lab_assets   # sources/: verbatim copies of kanth042's reviewed builds
    C=source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/calibration
    python scripts_v2/tools/rpm_lab/build_calibrated_ur5e_usd.py --real_joint_limits \\
        --source_usd $A/sources/124_official_robotiq_d405_20260926_visualfix/robot.usd \\
        --source_calibration $C/thunder_kinematics.yaml --calibration $C/lightning_kinematics.yaml \\
        --robot lightning --out_dir $A/lightning_124_official_robotiq_d405/robot
    python scripts_v2/tools/rpm_lab/build_calibrated_ur5e_usd.py --real_joint_limits \\
        --source_usd $A/sources/189_official_umi_fingertips_20261003_visualfix/robot.usd \\
        --source_calibration $C/thunder_kinematics.yaml --calibration $C/thunder_kinematics.yaml \\
        --robot thunder --out_dir $A/thunder_189_official_umi_fingertips/robot
"""

import argparse
import hashlib
import math
import numpy as np
import os
import yaml

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ARM = [  # (USD joint, calibration entry, parent link, child link)
    ("shoulder_pan_joint", "shoulder", "base_link", "shoulder_link"),
    ("shoulder_lift_joint", "upper_arm", "shoulder_link", "upper_arm_link"),
    ("elbow_joint", "forearm", "upper_arm_link", "forearm_link"),
    ("wrist_1_joint", "wrist_1", "forearm_link", "wrist_1_link"),
    ("wrist_2_joint", "wrist_2", "wrist_1_link", "wrist_2_link"),
    ("wrist_3_joint", "wrist_3", "wrist_2_link", "wrist_3_link"),
]
UR5E_JOINT_LIMITS_DEG = {  # symmetric; ur_description config/ur5e/joint_limits.yaml (the controller's ranges)
    "shoulder_pan_joint": 360.0,
    "shoulder_lift_joint": 360.0,
    "elbow_joint": 180.0,
    "wrist_1_joint": 360.0,
    "wrist_2_joint": 360.0,
    "wrist_3_joint": 360.0,
}
POS_TOL = 2e-6  # m; joint frames are float32
ROT_TOL = 5e-6  # rad


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose(t=(0.0, 0.0, 0.0), r=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if r is None else r
    T[:3, 3] = t
    return T


def quat_to_matrix(q) -> np.ndarray:
    """Gf.Quatf/Quatd -> 3x3 rotation."""
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


def rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1) / 2)))


def check_close(label: str, a: np.ndarray, b: np.ndarray, pos_tol=POS_TOL, rot_tol=ROT_TOL):
    dp, dr = np.linalg.norm(a[:3, 3] - b[:3, 3]), rot_angle(a[:3, :3], b[:3, :3])
    if dp > pos_tol or dr > rot_tol:
        raise RuntimeError(f"{label}: differs by {dp * 1e3:.4f} mm / {math.degrees(dr):.5f} deg")


def load_calibration(path: str) -> dict:
    with open(path) as f:
        k = yaml.safe_load(f)["kinematics"]
    return {j: [float(k[j][a]) for a in ("x", "y", "z", "roll", "pitch", "yaw")] for _, j, _, _ in ARM} | {
        "hash": k["hash"]
    }


def joint_frames(joint: str, c: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(localPose0, localPose1) of an arm joint, in the source asset's convention (checked against the source).

    The URDF joint origin is translation (x, y, z) then rpy = Rz(yaw) Ry(pitch) Rx(roll). The shoulder joint also
    absorbs the URDF's fixed base_link -> base_link_inertia turn of 180 deg about z, and the importer put its yaw in
    localRot1.
    """
    x, y, z, roll, pitch, yaw = c
    if joint == "shoulder_pan_joint":
        if abs(roll) > 1e-12 or abs(pitch) > 1e-12:
            raise RuntimeError("A shoulder entry with roll or pitch does not fit the source asset's convention")
        return pose((x, y, z), rot_z(math.pi)), pose(r=rot_z(-yaw))
    return pose((x, y, z), rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)), pose()


def authored_link_pose(prim) -> np.ndarray:
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    if set(ops) != {"xformOp:translate", "xformOp:orient", "xformOp:scale"}:
        raise RuntimeError(f"{prim.GetPath()}: unexpected xform ops {list(ops)}")
    if not Gf.IsClose(ops["xformOp:scale"].Get(), Gf.Vec3d(1, 1, 1), 1e-9):
        raise RuntimeError(f"{prim.GetPath()}: scaled link")
    return pose(tuple(ops["xformOp:translate"].Get()), quat_to_matrix(ops["xformOp:orient"].Get()))


def local_pose(prim, i: int) -> np.ndarray:
    pos = prim.GetAttribute(f"physics:localPos{i}").Get()
    return pose(tuple(pos), quat_to_matrix(prim.GetAttribute(f"physics:localRot{i}").Get()))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--source_calibration", required=True, help="Calibration the source USD was built with.")
    parser.add_argument("--calibration", required=True, help="Calibration to apply.")
    parser.add_argument("--robot", required=True, help="Arm name recorded in metadata.yaml, e.g. lightning.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--real_joint_limits", action="store_true", help="Set the arm joint limits to the UR5e's.")
    args = parser.parse_args()

    source_usd = os.path.abspath(args.source_usd)
    with open(os.path.join(os.path.dirname(source_usd), "metadata.yaml")) as f:
        metadata = yaml.safe_load(f)
    src_cal, new_cal = load_calibration(args.source_calibration), load_calibration(args.calibration)
    if metadata["kinematics_calibration"]["hash"] != src_cal["hash"]:
        raise RuntimeError(
            f"Source metadata says {metadata['kinematics_calibration']['hash']}, --source_calibration is "
            f"{src_cal['hash']}"
        )

    stage = Usd.Stage.Open(source_usd)
    root = stage.GetDefaultPrim()
    prims = {p.GetName(): p for p in root.GetChildren()}
    links = {name: authored_link_pose(prims[name]) for name in ["base_link"] + [a[3] for a in ARM]}

    # 1) The source calibration must reproduce the source joint frames and the authored link poses.
    angles = {}
    for joint, entry, parent, child in ARM:
        j = UsdPhysics.Joint(prims[joint])
        if [t.name for t in j.GetBody0Rel().GetTargets()] != [parent] or [
            t.name for t in j.GetBody1Rel().GetTargets()
        ] != [child]:
            raise RuntimeError(f"{joint} does not connect {parent} -> {child}")
        if prims[joint].GetAttribute("physics:axis").Get() != "Z":
            raise RuntimeError(f"{joint}: axis is not Z")
        l0, l1 = local_pose(prims[joint], 0), local_pose(prims[joint], 1)
        e0, e1 = joint_frames(joint, src_cal[entry])
        check_close(f"{joint} localPose0 vs source calibration", l0, e0)
        check_close(f"{joint} localPose1 vs source calibration", l1, e1)
        # Joint angle at which the links are authored: frame1 relative to frame0 must be a pure turn about z.
        rel = np.linalg.inv(links[parent] @ l0) @ links[child] @ l1
        angles[joint] = math.atan2(rel[1, 0], rel[0, 0])
        check_close(f"{joint} authored link poses", rel, pose(r=rot_z(angles[joint])), 1e-6, 1e-6)

    # 2) Rebuild the chain with the target calibration at the same authored joint angles.
    new_links = {"base_link": links["base_link"]}
    new_frames = {}
    for joint, entry, parent, child in ARM:
        l0, l1 = joint_frames(joint, new_cal[entry])
        new_frames[joint] = (l0, l1)
        new_links[child] = new_links[parent] @ l0 @ pose(r=rot_z(angles[joint])) @ np.linalg.inv(l1)
    delta = new_links["wrist_3_link"] @ np.linalg.inv(links["wrist_3_link"])

    # Every other rigid body hangs off wrist_3_link inside one top-level Xform (the hand); it moves rigidly with it.
    arm_links = set(links)
    hand_roots = set()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() not in arm_links:
            top = prim
            while top.GetParent() != root:
                top = top.GetParent()
            if top.GetName() in arm_links:
                continue  # nested under an arm link: moves with it
            hand_roots.add(top.GetName())
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() not in prims:
            j = UsdPhysics.Joint(prim)
            for rel_targets in (j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()):
                names = [t.name for t in rel_targets]
                if names and names[0] in arm_links - {"wrist_3_link"}:
                    raise RuntimeError(f"{prim.GetPath()} attaches to {names[0]}, not wrist_3_link")
    if not hand_roots:
        raise RuntimeError("No hand subtree found under the robot")

    # 3) Write the override layer.
    os.makedirs(args.out_dir, exist_ok=True)
    out_usd = os.path.join(args.out_dir, "robot.usda")
    if os.path.exists(out_usd):
        os.remove(out_usd)
    layer = Sdf.Layer.CreateNew(out_usd)
    layer.subLayerPaths.append(source_usd)
    layer.defaultPrim = root.GetName()
    layer.customLayerData = {
        "rpm_lab:source_usd": source_usd,
        "rpm_lab:source_usd_sha256": sha256(source_usd),
        "rpm_lab:source_calibration": src_cal["hash"],
        "rpm_lab:calibration": new_cal["hash"],
        "rpm_lab:robot": args.robot,
    }
    out = Usd.Stage.Open(layer)
    out.SetEditTarget(layer)
    r = out.GetDefaultPrim()
    for joint, (l0, l1) in new_frames.items():
        prim = r.GetChild(joint)
        for i, frame in ((0, l0), (1, l1)):
            w, x, y, z = matrix_to_quat(frame[:3, :3])
            prim.GetAttribute(f"physics:localPos{i}").Set(Gf.Vec3f(*frame[:3, 3].tolist()))
            prim.GetAttribute(f"physics:localRot{i}").Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        if args.real_joint_limits:
            joint_api = UsdPhysics.RevoluteJoint(prim)
            joint_api.GetLowerLimitAttr().Set(-UR5E_JOINT_LIMITS_DEG[joint])
            joint_api.GetUpperLimitAttr().Set(UR5E_JOINT_LIMITS_DEG[joint])
    for name, T in new_links.items():
        prim = r.GetChild(name)
        w, x, y, z = matrix_to_quat(T[:3, :3])
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*T[:3, 3].tolist()))
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    for name in sorted(hand_roots):
        xf = UsdGeom.Xformable(r.GetChild(name))
        ops = xf.GetOrderedXformOps()
        if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
            raise RuntimeError(f"{name}: expected one transform op, found {[op.GetOpName() for op in ops]}")
        old = np.array(ops[0].Get()).T  # Gf matrices are row-vector
        ops[0].Set(Gf.Matrix4d((delta @ old).T.tolist()))
    for prim in out.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            prim.SetActive(False)
            print(f"Deactivated the asset's light {prim.GetPath()}")
    layer.Save()

    # 4) Re-open and check the result against the target calibration.
    check = Usd.Stage.Open(out_usd)
    c = {p.GetName(): p for p in check.GetDefaultPrim().GetChildren()}
    for joint, entry, parent, child in ARM:
        e0, e1 = joint_frames(joint, new_cal[entry])
        check_close(f"written {joint} localPose0", local_pose(c[joint], 0), e0)
        check_close(f"written {joint} localPose1", local_pose(c[joint], 1), e1)
        rel = np.linalg.inv(authored_link_pose(c[parent]) @ e0) @ authored_link_pose(c[child]) @ e1
        check_close(f"written {child} pose", rel, pose(r=rot_z(angles[joint])), 1e-6, 1e-6)
        if args.real_joint_limits:
            limits = UsdPhysics.RevoluteJoint(c[joint])
            if (limits.GetLowerLimitAttr().Get(), limits.GetUpperLimitAttr().Get()) != (
                -UR5E_JOINT_LIMITS_DEG[joint],
                UR5E_JOINT_LIMITS_DEG[joint],
            ):
                raise RuntimeError(f"written {joint} limits")

    if args.real_joint_limits:
        metadata["arm_joint_limits_deg"] = {j: [-v, v] for j, v in UR5E_JOINT_LIMITS_DEG.items()}
    metadata["kinematics_calibration"] = {
        "robot": args.robot,
        "hash": new_cal["hash"],
        "source_repository": "https://github.com/RPM-lab-UMN/spark-data-collection",
        "source_path": f"Hardware/robot_calibration/ur5e/{os.path.basename(args.calibration)}",
        "applied_by": "scripts_v2/tools/rpm_lab/build_calibrated_ur5e_usd.py",
        "base_asset": source_usd,
        "base_asset_calibration": src_cal["hash"],
    }
    metadata["calibrated_joints"] = {
        "xyz": [new_cal[e][:3] for _, e, _, _ in ARM],
        "rpy": [new_cal[e][3:] for _, e, _, _ in ARM],
    }
    with open(os.path.join(args.out_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    moved = np.linalg.norm(delta[:3, 3]) * 1e3, math.degrees(rot_angle(np.eye(3), delta[:3, :3]))
    print(f"Wrote {out_usd} ({src_cal['hash']} -> {new_cal['hash']})")
    print(f"Authored joint angles kept (deg): {[round(math.degrees(a), 4) for a in angles.values()]}")
    print(f"wrist_3_link moved {moved[0]:.4f} mm / {moved[1]:.5f} deg at the authored pose; hand roots: {hand_roots}")
    if args.real_joint_limits:
        print(f"Arm joint limits (deg, +-): {UR5E_JOINT_LIMITS_DEG}")


if __name__ == "__main__":
    main()

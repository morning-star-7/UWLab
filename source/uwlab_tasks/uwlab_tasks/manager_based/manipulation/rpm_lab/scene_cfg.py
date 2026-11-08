# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RPM Lab workcell: the Vention table with two columns, Thunder on one column and Lightning on the other.

World frame: meters, z up, ground at z = 0. Every number below comes from the lab CAD or from Thunder's
measured mounting (branch ``thunder-umi-sim2real``: ``umi_reset_cfg.py`` and
``scripts_v2/tools/thunder_sim2real/MOUNTING_CORRECTION.md``).

* The lab CAD (``lab_vention.usd``) is a millimeter, Y-up Vention export. It contains the table, both columns and
  a 27 mm side mounting plate on each column. It is placed with ``TABLE_POS`` / ``TABLE_ROT`` and scale 1e-3.
* The lab is mirror-symmetric about the vertical plane ``x = MIRROR_PLANE_X``. The column centers, the two
  mounting-plate faces and the tabletop center all lie on that plane to within 1 micrometer.
* Thunder's ``base_link`` sits on the +x plate (the measured pose). Lightning sits on the -x plate at the mirror
  image of that pose (see :func:`mirror_mount_pose`). Both real arms have the base cable opening pointing down, unlike
  the CAD (horizontal).
* Each arm model carries that arm's own UR factory kinematic calibration
  (``scripts_v2/tools/rpm_lab/ur5e_calibration``) and the real UR5e joint position and speed limits.

The real recordings on the lab's robot PC (Zeus, ``spark-data-collection``) confirm the mounting. The controllers
report flange poses in their base frames. Placed with these mounts, the recorded pick-and-place gripper paths stay
0.06-0.35 m above the tabletop for both arms (11/2026 dual-arm session and 561 Lightning episodes from 05-09/2026). With
the CAD orientation they would float 1.0-1.4 m above it.

Asset paths default to the shared copies under ``/data``. Set ``RPM_LAB_USD``, ``RPM_THUNDER_USD`` and
``RPM_LIGHTNING_USD`` to use other copies.

.. warning::
    ``init_state.rot`` is the world orientation of ``base_link`` *after the root pose is written*, as every
    Thunder task does on reset (``reset_scene_to_default`` / ``reset_robot_pose``). The robot USD authors
    ``base_link`` at 180 deg about z under its root prim, so a scene that only spawns the robot, without writing
    ``default_root_state`` to the simulation, has both arms turned 180 deg about their mount normal.
"""

from __future__ import annotations

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

# ---------------------------------------------------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------------------------------------------------

LAB_USD = os.environ.get("RPM_LAB_USD", "/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd")
"""Lab CAD: table, two columns and their mounting plates (millimeter units)."""

THUNDER_USD = os.environ.get(
    "RPM_THUNDER_USD", "/data/cui00191/rpm_lab_assets/thunder_189_official_umi_fingertips/robot/robot.usda"
)
"""Thunder: UR5e with Thunder's factory kinematic calibration, official Robotiq 2F-85 model with the rigid UMI
fingertips, D405 mount.

A small override layer on the reviewed build behind the R214 policy that runs on the real Thunder
(``189_official_umi_fingertips_20261003/assets_visualfix/robot/robot.usd``, sha256 d6372619...). The layer changes only
the arm joint limits to the real UR5e's (+-360 deg, elbow +-180 deg). The source build limits the wrists to +-180 deg,
and the real Thunder goes beyond that (wrist_3 at -5.1 rad in the 11/2026 recordings). Rebuild it with
``scripts_v2/tools/rpm_lab/build_calibrated_ur5e_usd.py``.
"""

LIGHTNING_USD = os.environ.get(
    "RPM_LIGHTNING_USD", "/data/cui00191/rpm_lab_assets/lightning_124_official_robotiq_d405/robot/robot.usda"
)
"""Lightning: the official Robotiq 2F-85 model with the stock finger pads, D405 mount, and Lightning's own calibration.

Lightning keeps its original fingers, and its D405 wrist camera mount is the same part as Thunder's. The file is a small
override layer on the stock-pad build ``124_official_robotiq_d405_20260926/assets_visualfix/robot/robot.usd`` (sha256
31b9852c...; R209: the same D405/adapter visual-placement fix as Thunder's R189 ``assets_visualfix``, physics unchanged).
Relative to wrist_3_link, its whole hand (D405 mount, camera, optical frame, flange adapter, gripper) matches Thunder's
exactly except the fingertips. It carries Thunder's calibration. The layer replaces the arm joint
frames with Lightning's factory calibration (``calib_16320005684383956165``; 0.40 mm / 0.05 deg at the flange) and sets
the real UR5e joint limits as for Thunder (the real Lightning's wrist_1 reaches -3.44 rad). Rebuild it with
``scripts_v2/tools/rpm_lab/build_calibrated_ur5e_usd.py`` (its docstring has the command).
"""

# ---------------------------------------------------------------------------------------------------------------------
# Lab geometry (world frame)
# ---------------------------------------------------------------------------------------------------------------------

TABLE_POS = (1.793445, 0.340075, -0.030851)
TABLE_ROT = (0.5, 0.5, 0.5, 0.5)  # (w, x, y, z): CAD +Y (vertical) -> world +z, CAD +X -> world +y, CAD +Z -> world +x
TABLE_SCALE = (0.001, 0.001, 0.001)
TABLE_TOP_Z = 0.84235
"""Height of the tabletop surface. The tabletop spans x in [-0.83484, 0.65016] and y in [-0.43230, 0.28770]."""

MIRROR_PLANE_X = -0.09234
"""The lab's symmetry plane, x = const.

Measured from the lab CAD: column centers 0.065160 and -0.249840, mounting-plate faces 0.182660 and -0.367340,
tabletop center -0.0923396.
"""

THUNDER_POS = (0.177660, 0.377695, 1.466000)
"""Thunder's measured ``base_link`` position: centered on the +x plate (y 0.2877..0.4677, z 1.376..1.556)."""

THUNDER_ROT = (0.5, 0.5, 0.5, 0.5)
"""Thunder's ``base_link`` orientation (w, x, y, z), corrected to the physical cable opening (pointing down).

Columns of the rotation: base x -> world +y, base y -> world +z (up), base z (mount normal) -> world +x.
The cable opening points along base -y. The Vention CAD shows the opening horizontal, (0.7071068, 0, 0.7071068, 0),
which is 90 deg off about the mount normal (MOUNTING_CORRECTION.md). Neither real arm is mounted that way.
"""

# ---------------------------------------------------------------------------------------------------------------------
# Mirror construction
# ---------------------------------------------------------------------------------------------------------------------


def _quat_to_matrix(q: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _matrix_to_quat(m: list[list[float]]) -> tuple[float, float, float, float]:
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    sign = -1.0 if q[0] < 0.0 else 1.0
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(sign * v / norm for v in q)  # type: ignore[return-value]


def mirror_mount_pose(
    pos: tuple[float, float, float], quat: tuple[float, float, float, float], plane_x: float
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Mount pose of the twin arm on the opposite column, mirrored across the plane ``x = plane_x``.

    A reflection of a rigid body is left-handed, and no real robot can be mounted that way. The twin's base frame is the
    reflected frame with its x axis negated. The reflected base z (the mount normal, now facing -x) and base y are
    kept, so the cable opening (base -y on the UR5e) is mirrored as well.
    """
    r = _quat_to_matrix(quat)
    mirrored = [[-r[0][j] for j in range(3)], list(r[1]), list(r[2])]  # reflect world x
    proper = [[-row[0], row[1], row[2]] for row in mirrored]  # negate base x -> right-handed again
    return (2.0 * plane_x - pos[0], pos[1], pos[2]), _matrix_to_quat(proper)


LIGHTNING_POS, LIGHTNING_ROT = mirror_mount_pose(THUNDER_POS, THUNDER_ROT, MIRROR_PLANE_X)
"""Lightning's ``base_link`` pose: (-0.36234, 0.377695, 1.466) and (0.5, 0.5, -0.5, -0.5).

The mirror image of Thunder's corrected mount. Columns of the rotation: base x -> world -y, base y -> world +z (up),
base z (mount normal) -> world -x. The cable opening (base -y) points down, as on the real Lightning.
"""

# ---------------------------------------------------------------------------------------------------------------------
# Home postures
# ---------------------------------------------------------------------------------------------------------------------

THUNDER_HOME_JOINT_POS = {
    "shoulder_pan_joint": 0.778399,
    "shoulder_lift_joint": -2.568071,
    "elbow_joint": -1.872467,
    "wrist_1_joint": -1.927923,
    "wrist_2_joint": -2.296463,
    "wrist_3_joint": 2.327399,
}
"""Thunder's hardware start pose for the R214 state policy (R217 Reaching bank row 5692).

S-W-E- arm shape, gripper 5 deg from straight down, hand 0.30 m above the table over the trained cube region
(``scripts_v2/tools/thunder_sim2real/workstation/run_state_policy.py`` on ``thunder-umi-sim2real``). The real
controller and this asset use the same joint angles.
"""

LIGHTNING_HOME_JOINT_POS = {
    "shoulder_pan_joint": -3.369978189468384,
    "shoulder_lift_joint": -2.8929593563079834,
    "elbow_joint": -1.1633856296539307,
    "wrist_1_joint": -1.9551736116409302,
    "wrist_2_joint": -0.06568319350481033,
    "wrist_3_joint": 1.2720727920532227,
}
"""Lightning's real start pose: the joint angles its controller reported at the start of a recorded episode.

This is the medoid of the 561 recorded Lightning episode starts (05-11/2026), from dataset ``0519_cup``. It is within
0.09 rad of their per-joint median, and the gripper was open. The grasp point is 94 mm above the tabletop, with the
gripper 9 deg from straight down.
wrist_2 is near 0, close to the wrist singularity, as on the real arm; an IK or OSC task should start elsewhere.
"""

OFFICIAL_ROBOTIQ_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
}
"""Open gripper. The official Robotiq model replaces the stock asset's inner-knuckle hinges with finger couplings."""

UR5E_MAX_JOINT_SPEED = math.pi
"""UR5e datasheet: 180 deg/s on every joint. Upstream's UR5e config caps shoulder and elbow at 90 deg/s, but the real
arms exceed that (elbow at 1.66 rad/s in the 11/2026 Thunder recording)."""

# ---------------------------------------------------------------------------------------------------------------------
# Asset configurations
# ---------------------------------------------------------------------------------------------------------------------


def ur5e_official_robotiq_cfg(
    prim_name: str,
    usd_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    arm_joint_pos: dict[str, float],
) -> ArticulationCfg:
    """UR5e + official Robotiq 2F-85 on a fixed mount, with the actuator settings of Thunder's training configs.

    The arm keeps upstream's zero-gain implicit drive (a task's OSC action supplies torques) and its effort limits,
    with the real joint speed limit. The robot bodies keep gravity disabled (the UR controller compensates gravity).
    The gripper uses the vendor-authored drive.
    """
    robot = copy.deepcopy(IMPLICIT_UR5E_ROBOTIQ_2F85)
    robot.prim_path = f"{{ENV_REGEX_NS}}/{prim_name}"
    robot.spawn.usd_path = usd_path
    robot.spawn.mass_props = None
    robot.spawn.articulation_props.solver_position_iteration_count = 64
    robot.init_state.pos = pos
    robot.init_state.rot = rot
    robot.init_state.joint_pos = {**arm_joint_pos, **OFFICIAL_ROBOTIQ_JOINT_POS}
    robot.actuators["arm"].velocity_limit_sim = UR5E_MAX_JOINT_SPEED
    gripper = robot.actuators["gripper"]
    gripper.stiffness = None
    gripper.damping = None
    gripper.effort_limit_sim = None
    gripper.velocity_limit_sim = None
    gripper.armature = None
    return robot


THUNDER_CFG = ur5e_official_robotiq_cfg("Thunder", THUNDER_USD, THUNDER_POS, THUNDER_ROT, THUNDER_HOME_JOINT_POS)
LIGHTNING_CFG = ur5e_official_robotiq_cfg(
    "Lightning", LIGHTNING_USD, LIGHTNING_POS, LIGHTNING_ROT, LIGHTNING_HOME_JOINT_POS
)

LAB_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=LAB_USD,
        scale=TABLE_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS, rot=TABLE_ROT),
)
"""The whole workcell as one kinematic body, named ``table`` in the scene as in Thunder's task configs."""


@configclass
class RpmLabDualArmSceneCfg(InteractiveSceneCfg):
    """The lab workcell with Thunder and Lightning. Use ``env_spacing`` >= 4 m; the workcell is about 1.5 m x 0.9 m."""

    table: RigidObjectCfg = LAB_CFG
    thunder: ArticulationCfg = THUNDER_CFG
    lightning: ArticulationCfg = LIGHTNING_CFG

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.95, 0.95, 0.95)),
    )

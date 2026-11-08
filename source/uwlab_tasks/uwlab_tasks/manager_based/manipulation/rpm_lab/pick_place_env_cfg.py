# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightning single-arm pick-and-place on the RPM Lab twin, set up for scripted demos and Isaac Lab Mimic.

A 45 mm cube goes onto a 100 mm plate; Thunder stays parked at its home. The arm takes absolute grasp-point (TCP)
poses in Lightning's base_link frame (differential IK) at 20 Hz, the rate of the real datasets; the gripper takes a
binary open (>= 0) / close (< 0) command on finger_joint. The IK uses an exact geometric Jacobian (``actions.py``);
Isaac Lab's own IK term gets the offset grasp point's Jacobian wrong on these arms. Mimic splits a demo into "grasp the cube" (relative to the
cube) and "place on the plate" (relative to the plate).
"""

from __future__ import annotations

import copy
import os
import torch
import yaml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import scene_cfg as lab
from .actions import GeometricJacobianDiffIKActionCfg

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ARM_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
GRIPPER_BODY = "robotiq_base_link"
FINGER_JOINT = "finger_joint"
FINGER_CLOSED_TARGET = 0.7  # rad; the fingers meet at 0.82
ARM_STIFFNESS = (4000.0, 4000.0, 2500.0, 600.0, 600.0, 400.0)  # joint PD that tracks the IK targets
ARM_DAMPING = (250.0, 250.0, 150.0, 25.0, 25.0, 15.0)

CUBE_SIZE = 0.045
PLATE_SIZE = (0.10, 0.10, 0.01)
CUBE_POS = (-0.30, -0.16, lab.TABLE_TOP_Z + 0.5 * CUBE_SIZE + 0.001)
PLATE_POS = (-0.13, -0.30, lab.TABLE_TOP_Z + 0.5 * PLATE_SIZE[2])
"""Nominal object positions (world). Lightning's real teleop workspace spans x -0.47..0.09, y -0.41..0.06; these keep
both objects clear of the parked Thunder hand (x 0.07, y -0.06) and of each other under the reset ranges below."""


def _gripper_offset(usd_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with open(os.path.join(os.path.dirname(usd_path), "metadata.yaml")) as f:
        offset = yaml.safe_load(f)["gripper_offset"]
    return tuple(offset["pos"]), tuple(offset["quat"])


TCP_POS, TCP_QUAT = _gripper_offset(lab.LIGHTNING_USD)
"""Grasp point on robotiq_base_link (the asset's gripper_offset): its z axis is the approach direction."""


# ---------------------------------------------------------------------------------------------------------------------
# MDP terms
# ---------------------------------------------------------------------------------------------------------------------


def tcp_pose(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("lightning")) -> torch.Tensor:
    """(N, 7) grasp-point pose in the env frame: position and w-x-y-z quaternion."""
    robot: Articulation = env.scene[robot_cfg.name]
    k = robot.find_bodies(GRIPPER_BODY)[0][0]
    n = env.num_envs
    pos, quat = math_utils.combine_frame_transforms(
        robot.data.body_pos_w[:, k],
        robot.data.body_quat_w[:, k],
        torch.tensor(TCP_POS, device=env.device).expand(n, 3),
        torch.tensor(TCP_QUAT, device=env.device).expand(n, 4),
    )
    return torch.cat([pos - env.scene.env_origins, quat], dim=-1)


def tcp_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return tcp_pose(env)[:, :3]


def tcp_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return tcp_pose(env)[:, 3:]


def object_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def object_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_quat_w


def finger_angle(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("lightning")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.joint_pos[:, robot.find_joints(FINGER_JOINT)[0]]


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    max_distance: float = 0.03,
    min_closure: float = 0.2,
) -> torch.Tensor:
    """1 while the cube's center is within ``max_distance`` of the grasp point and the fingers are closing on it."""
    near = (object_pos(env, object_cfg) - tcp_pos(env)).norm(dim=-1) < max_distance
    closed = finger_angle(env)[:, 0] > min_closure
    return (near & closed).float()


def object_placed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    xy_tolerance: float = 0.03,
    z_tolerance: float = 0.01,
    max_closure: float = 0.15,
) -> torch.Tensor:
    """The cube rests on the plate (centered within ``xy_tolerance``) and the gripper has let go."""
    cube, plate = object_pos(env, object_cfg), object_pos(env, plate_cfg)
    rest_z = plate[:, 2] + 0.5 * PLATE_SIZE[2] + 0.5 * CUBE_SIZE
    on_plate = ((cube[:, :2] - plate[:, :2]).norm(dim=-1) < xy_tolerance) & ((cube[:, 2] - rest_z).abs() < z_tolerance)
    return on_plate & (finger_angle(env)[:, 0] < max_closure)


def object_fell(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    return object_pos(env, object_cfg)[:, 2] < lab.TABLE_TOP_Z - 0.05


# ---------------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class LightningPickPlaceSceneCfg(lab.RpmLabDualArmSceneCfg):
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.15, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CuboidCfg(
            size=PLATE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_POS),
    )


@configclass
class ActionsCfg:
    arm = GeometricJacobianDiffIKActionCfg(
        asset_name="lightning",
        joint_names=ARM_JOINTS,
        joint_child_bodies=ARM_LINKS,
        body_name=GRIPPER_BODY,
        body_offset=GeometricJacobianDiffIKActionCfg.OffsetCfg(pos=TCP_POS, rot=TCP_QUAT),
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
    )
    gripper = BinaryJointPositionActionCfg(
        asset_name="lightning",
        joint_names=[FINGER_JOINT],
        open_command_expr={FINGER_JOINT: 0.0},
        close_command_expr={FINGER_JOINT: FINGER_CLOSED_TARGET},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("lightning")})
        eef_pos = ObsTerm(func=tcp_pos)
        eef_quat = ObsTerm(func=tcp_quat)
        gripper = ObsTerm(func=finger_angle)
        cube_pos = ObsTerm(func=object_pos, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_quat = ObsTerm(func=object_quat, params={"asset_cfg": SceneEntityCfg("cube")})
        plate_pos = ObsTerm(func=object_pos, params={"asset_cfg": SceneEntityCfg("plate")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(func=object_grasped)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    # Writes every articulation's default root pose too (the robot USDs author base_link turned 180 deg).
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    place_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.5, 0.5)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    place_plate = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("plate"),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_fell = DoneTerm(func=object_fell)
    success = DoneTerm(func=object_placed)


@configclass
class LightningPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: LightningPickPlaceSceneCfg = LightningPickPlaceSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None

    def __post_init__(self):
        self.decimation = 6  # 120 Hz physics, 20 Hz control as in the real datasets
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        # Lightning's arm tracks the IK joint targets with a stiff PD (the scene default is torque control).
        self.scene.lightning = copy.deepcopy(lab.LIGHTNING_CFG)
        arm = self.scene.lightning.actuators["arm"]
        arm.stiffness = dict(zip(ARM_JOINTS, ARM_STIFFNESS))
        arm.damping = dict(zip(ARM_JOINTS, ARM_DAMPING))


@configclass
class LightningPickPlaceMimicEnvCfg(LightningPickPlaceEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "lightning_pick_place"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1
        common = dict(
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.subtask_configs["lightning"] = [
            SubTaskConfig(object_ref="cube", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10), **common),
            SubTaskConfig(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0), **common),
        ]

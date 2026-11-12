# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightning single-arm cook_fish on the RPM Lab twin: open the lid, put the fish in the pot, close the lid.

Set up like the 2026-10-28 real cook_fish recordings: the same objects (``build_cook_fish_assets.py``: sizes from those
recordings), Lightning starting where the operator started, Thunder parked at its teleop home outside the scene_1
view, and the real action (six joint commands plus a gripper command at 20 Hz, executed through the servoJ model). The
reset lays the pot and the fish out like one of the real episodes (a little moved), the lid on the pot with its handle
turned the way the operators grasped it. Mimic splits a demo into six subtasks: grasp the lid, set it down beside the
pot, grasp the fish, drop it into the pot, grasp the lid again, put it back on the pot.
"""

from __future__ import annotations

import copy
import json
import math
import os
import torch
import yaml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import scene_cfg as lab
from .pick_place_env_cfg import (
    ARM_DAMPING,
    ARM_JOINTS,
    ARM_STIFFNESS,
    FINGER_CLOSED,
    ActionsCfg,
    finger_angle,
    object_pos,
    object_quat,
    tcp_pos,
    tcp_quat,
)
from .sensors import reattach_mounted_cameras_event

INSTRUCTION = "open lid, place fish inside and close lid"
"""The real episodes' language instruction (``task`` in the LeRobot datasets)."""

OBJECTS_DIR = os.path.join(lab.ASSET_ROOT, "objects")


def _object(name: str) -> tuple[str, dict]:
    with open(os.path.join(OBJECTS_DIR, name, "metadata.yaml")) as f:
        return os.path.join(OBJECTS_DIR, name, f"{name}.usda"), yaml.safe_load(f)


POT_USD, POT = _object("pot")
LID_USD, LID = _object("lid")
FISH_USD, FISH = _object("fish")
"""USDs and dimensions (``metadata.yaml``) from ``build_cook_fish_assets.py``. Origins are at the bottom center; the lid
handle runs along the lid's x axis, the fish's length along its x axis."""

COOK_FISH_START_JOINT_POS = {
    "shoulder_pan_joint": -3.163,
    "shoulder_lift_joint": -2.918,
    "elbow_joint": -1.172,
    "wrist_1_joint": -2.939,
    "wrist_2_joint": -0.090,
    "wrist_3_joint": 2.258,
}
"""Lightning's start in real episode 20261028-185333, the one closest to the per-joint median of the 10/28 cook_fish
starts: grasp point about 0.23 m above the table, gripper pointing down."""

REAL_LAYOUTS_FILE = os.path.join(lab.CALIBRATION_DIR, "cook_fish_layouts_20261028.json")
"""Where the operators found the pot and the fish, and set the lid aside, in 45 of the 2026-10-28 real episodes (the
world model's validation episodes left out), read off the grasps by ``extract_cook_fish_layouts.py``."""
LAYOUT_JITTER = 0.02
"""Each object of a real layout is moved by up to this much (m) along x and y at reset, about the estimate's own
spread."""
FISH_POT_CLEARANCE = 0.12
"""Least fish-to-pot distance (center to center, m) after the jitter: clear of the pot's handles (0.09 m out). The real
ones lie 0.13-0.25 m apart."""
LID_YAW_RANGE = (1.50, 2.25)
"""Handle direction (world yaw, rad). The operators closed the gripper with its closing axis (grasp-point x) at -4..39
deg yaw, at a right angle to the handle; scene_1 shows the handle running along world y."""
FISH_YAW_RANGE = (1.27, 2.37)  # the fish's length across the operators' closing axis
OBJECT_X_RANGE = (-0.36, 0.06)
OBJECT_Y_RANGE = (-0.36, -0.04)
"""Where objects may lie (world): inside Lightning's real teleop workspace, clear of the table's front edge."""
MAX_REACH = 0.74
"""Farthest object (pot, fish) from Lightning's base, horizontally (m); real layouts farther out are not used. The real
fish grasps lie 0.49-0.76 m out (2 of 45 beyond 0.74). The scripted demos lean the gripper like the operators (about 12
deg, away from the base), which reaches about 3 cm past the 0.73 m of a straight-down grasp with the elbow straight."""
LID_ASIDE_CLEARANCE = LID["radius"] + 0.5 * math.hypot(*FISH["size"][:2]) + 0.01
"""Least distance (m) from the fish to a lid set down beside the pot (lid radius + fish half-diagonal + 1 cm)."""
POT_POS = (-0.07, -0.13, lab.WORK_SURFACE_Z)
FISH_POS = (-0.22, -0.16, lab.WORK_SURFACE_Z)


def _real_layouts(path: str) -> torch.Tensor:
    """(L, 3, 2) env-frame xy of the pot, the fish and the set-aside lid in each real layout within ``MAX_REACH``.

    The file holds grasp points in Lightning's controller Base frame (base_link turned 180 deg about z); they are moved
    into the env frame with the twin's base pose, the one fitted to scene_1 views of the real arm.
    """
    with open(path) as f:
        layouts = json.load(f)["layouts"]
    points = torch.tensor([[lay["pot"], lay["fish"], lay["lid_aside"]] for lay in layouts], dtype=torch.float64)
    base_rot = math_utils.matrix_from_quat(torch.tensor(lab.LIGHTNING_ROT, dtype=torch.float64))
    rot = base_rot @ torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    xy = (points @ rot.T + torch.tensor(lab.LIGHTNING_POS, dtype=torch.float64))[..., :2]
    reach = (xy[:, :2] - torch.tensor(lab.LIGHTNING_POS[:2], dtype=torch.float64)).norm(dim=-1)
    return xy[(reach < MAX_REACH).all(dim=1)].float()


REAL_LAYOUTS = _real_layouts(REAL_LAYOUTS_FILE)
"""The real layouts the reset draws from (``_real_layouts``)."""

RELEASE_SLACK = 0.05
"""A gripper commanded open counts as released once its finger_joint is within this much (rad) of the commanded
opening: nothing holds the fingers. The commanded opening varies like the operators' (0.2-0.3 after letting go, from the
0.0-1.0 trigger), and a gripper opened to 0.3 is closed further (0.25 rad) than one holding the fish (0.28 rad when told
1.0)."""
LIFT_CLEARANCE = 0.02
"""A held object counts as lifted once its bottom (its origin) is this far above the pot rim, clear of everything on the
table. The grasp signals wait for it: Mimic starts the next subtask's segment where a signal first rises, so the move
between two segments (to the next segment's start, transformed with another object) happens in free space instead of
dragging the object over the table."""
RETREAT_HEIGHT = 0.11
"""Grasp-point height above the table at which the open gripper is clear after letting go; the release signals wait
for it, for the same reason. The operators move on to the next object at about this height (11-13 cm)."""


# ---------------------------------------------------------------------------------------------------------------------
# MDP terms
# ---------------------------------------------------------------------------------------------------------------------


def handle_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 3) lid handle center (the bar's axis at mid-length), env frame."""
    lid: RigidObject = env.scene["lid"]
    offset = torch.tensor([0.0, 0.0, LID["handle_height"]], device=env.device).expand(env.num_envs, 3)
    return lid.data.root_pos_w - env.scene.env_origins + math_utils.quat_apply(lid.data.root_quat_w, offset)


def lightning_base_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 2) Lightning's base position on the table plane, env frame."""
    return (env.scene["lightning"].data.root_pos_w - env.scene.env_origins)[:, :2]


def _lifted(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    z = object_pos(env, SceneEntityCfg(name))[:, 2]
    return z > lab.WORK_SURFACE_Z + POT["rim_height"] + LIFT_CLEARANCE


def _released(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The gripper is commanded open (below 0.5) and its fingers have opened that far (``RELEASE_SLACK``)."""
    command = env.action_manager.get_term("gripper").raw_actions[:, 0]
    return (command < 0.5) & (finger_angle(env)[:, 0] < command * FINGER_CLOSED + RELEASE_SLACK)


def _retreated(env: ManagerBasedRLEnv) -> torch.Tensor:
    return tcp_pos(env)[:, 2] > lab.WORK_SURFACE_Z + RETREAT_HEIGHT


def lid_grasped(env: ManagerBasedRLEnv, max_distance: float = 0.025, min_closure: float = 0.4) -> torch.Tensor:
    """1 while the fingers hold the handle (the bar is thin: nearly shut) and the lid is lifted (``LIFT_CLEARANCE``)."""
    near = (handle_pos(env) - tcp_pos(env)).norm(dim=-1) < max_distance
    return (near & (finger_angle(env)[:, 0] > min_closure) & _lifted(env, "lid")).float()


def fish_grasped(env: ManagerBasedRLEnv, max_distance: float = 0.03, min_closure: float = 0.15) -> torch.Tensor:
    """1 while the fingers hold the fish and it is lifted (``LIFT_CLEARANCE``)."""
    fish = object_pos(env, SceneEntityCfg("fish")) + torch.tensor([0.0, 0.0, 0.5 * FISH["size"][2]], device=env.device)
    near = (fish - tcp_pos(env)).norm(dim=-1) < max_distance
    return (near & (finger_angle(env)[:, 0] > min_closure) & _lifted(env, "fish")).float()


def _fish_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    pot, fish = object_pos(env, SceneEntityCfg("pot")), object_pos(env, SceneEntityCfg("fish"))
    centered = (fish[:, :2] - pot[:, :2]).norm(dim=-1) < POT["inner_radius"] - 0.02
    return centered & (fish[:, 2] < pot[:, 2] + POT["rim_height"] - FISH["size"][2] + 0.01)


def _lid_on_pot(env: ManagerBasedRLEnv, xy_tolerance: float = 0.012, z_tolerance: float = 0.006) -> torch.Tensor:
    pot, lid = object_pos(env, SceneEntityCfg("pot")), object_pos(env, SceneEntityCfg("lid"))
    centered = (lid[:, :2] - pot[:, :2]).norm(dim=-1) < xy_tolerance
    seated = (lid[:, 2] - pot[:, 2] - POT["rim_height"]).abs() < z_tolerance
    lid_up = math_utils.quat_apply(
        object_quat(env, SceneEntityCfg("lid")),
        torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(env.num_envs, 3),
    )
    return centered & seated & (lid_up[:, 2] > math.cos(0.15))


def lid_off(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 once the lid lies on the table beside the pot and the gripper has let go and moved up (``RETREAT_HEIGHT``)."""
    pot, lid = object_pos(env, SceneEntityCfg("pot")), object_pos(env, SceneEntityCfg("lid"))
    beside = (lid[:, :2] - pot[:, :2]).norm(dim=-1) > POT["radius"] + LID["radius"] - 0.01
    on_table = lid[:, 2] < lab.WORK_SURFACE_Z + LID["lip_depth"] + 0.01
    return (beside & on_table & _released(env) & _retreated(env)).float()


def fish_in_pot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 once the fish lies in the pot and the gripper has let go and moved up (``RETREAT_HEIGHT``)."""
    return (_fish_inside(env) & _released(env) & _retreated(env)).float()


def lid_regrasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The second lid grasp (fish already in the pot): its own signal, since Mimic splits on a signal's first rise."""
    return lid_grasped(env) * _fish_inside(env).float()


def cook_fish_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _lid_on_pot(env) & _fish_inside(env) & _released(env)


def object_fell(env: ManagerBasedRLEnv) -> torch.Tensor:
    low = lab.WORK_SURFACE_Z - 0.05
    return torch.stack([object_pos(env, SceneEntityCfg(n))[:, 2] < low for n in ("pot", "lid", "fish")]).any(dim=0)


def reset_cook_fish_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor, candidates: int = 64):
    """A real layout (``REAL_LAYOUTS``) with the pot and the fish each moved by up to ``LAYOUT_JITTER``, the lid seated
    on the pot: the first of several random draws that keeps the fish ``FISH_POT_CLEARANCE`` from the pot and both
    within ``MAX_REACH``."""
    n, device = len(env_ids), env.device
    origins = env.scene.env_origins[env_ids]

    def uniform(low, high, *shape):
        return torch.empty(*(shape or (n,)), device=device).uniform_(low, high)

    def yaw_quat(yaw):
        zeros = torch.zeros_like(yaw)
        return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    layouts = REAL_LAYOUTS.to(device)
    xy = layouts[torch.randint(len(layouts), (n, candidates), device=device), :2]  # (n, candidates, pot/fish, 2)
    xy = xy + uniform(-LAYOUT_JITTER, LAYOUT_JITTER, n, candidates, 2, 2)
    base = lightning_base_xy(env)[env_ids, None, None]
    valid = ((xy[:, :, 1] - xy[:, :, 0]).norm(dim=-1) > FISH_POT_CLEARANCE) & (
        (xy - base).norm(dim=-1) < MAX_REACH
    ).all(dim=-1)
    xy = xy[torch.arange(n, device=device), valid.int().argmax(dim=1)]
    z = torch.full((n, 1), lab.WORK_SURFACE_Z + 0.0005, device=device)
    pot, fish = torch.cat([xy[:, 0], z], dim=-1), torch.cat([xy[:, 1], z], dim=-1)
    lid = pot + torch.tensor([0.0, 0.0, POT["rim_height"] + 0.0005], device=device)

    poses = {
        "pot": (pot, yaw_quat(uniform(-0.4, 0.4))),
        "lid": (lid, yaw_quat(uniform(*LID_YAW_RANGE))),
        "fish": (fish, yaw_quat(uniform(*FISH_YAW_RANGE))),
    }
    for name, (pos, quat) in poses.items():
        obj: RigidObject = env.scene[name]
        obj.write_root_pose_to_sim(torch.cat([pos + origins, quat], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


# ---------------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------------


def _rigid_object(name: str, usd_path: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class LightningCookFishSceneCfg(lab.RpmLabDualArmSceneCfg):
    pot = _rigid_object("Pot", POT_USD, POT_POS)
    lid = _rigid_object("Lid", LID_USD, (POT_POS[0], POT_POS[1], POT_POS[2] + POT["rim_height"]))
    fish = _rigid_object("Fish", FISH_USD, FISH_POS)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("lightning")})
        eef_pos = ObsTerm(func=tcp_pos)
        eef_quat = ObsTerm(func=tcp_quat)
        gripper = ObsTerm(func=finger_angle)
        pot_pos = ObsTerm(func=object_pos, params={"asset_cfg": SceneEntityCfg("pot")})
        lid_pos = ObsTerm(func=object_pos, params={"asset_cfg": SceneEntityCfg("lid")})
        lid_quat = ObsTerm(func=object_quat, params={"asset_cfg": SceneEntityCfg("lid")})
        fish_pos = ObsTerm(func=object_pos, params={"asset_cfg": SceneEntityCfg("fish")})
        fish_quat = ObsTerm(func=object_quat, params={"asset_cfg": SceneEntityCfg("fish")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        lid_grasped = ObsTerm(func=lid_grasped)
        lid_off = ObsTerm(func=lid_off)
        fish_grasped = ObsTerm(func=fish_grasped)
        fish_in_pot = ObsTerm(func=fish_in_pot)
        lid_regrasped = ObsTerm(func=lid_regrasped)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reattach_cameras = EventTerm(func=reattach_mounted_cameras_event, mode="reset")
    layout = EventTerm(func=reset_cook_fish_layout, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_fell = DoneTerm(func=object_fell)
    success = DoneTerm(func=cook_fish_success)


@configclass
class LightningCookFishEnvCfg(ManagerBasedRLEnvCfg):
    scene: LightningCookFishSceneCfg = LightningCookFishSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None

    def __post_init__(self):
        self.decimation = 6  # 120 Hz physics, 20 Hz control as in the real datasets
        self.episode_length_s = 45.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        # Lightning tracks the joint commands with a stiff PD (the scene default is torque control).
        self.scene.lightning = copy.deepcopy(lab.LIGHTNING_CFG)
        self.scene.lightning.init_state.joint_pos = {
            **COOK_FISH_START_JOINT_POS,
            **lab.OFFICIAL_ROBOTIQ_JOINT_POS,
        }
        # Thunder holds its teleop home, outside the scene_1 view as in the real Lightning-only recordings.
        self.scene.thunder = copy.deepcopy(lab.THUNDER_CFG)
        self.scene.thunder.init_state.joint_pos = {
            **lab.THUNDER_TELEOP_HOME_JOINT_POS,
            **lab.OFFICIAL_ROBOTIQ_JOINT_POS,
        }
        for robot in (self.scene.lightning, self.scene.thunder):
            arm = robot.actuators["arm"]
            arm.stiffness = dict(zip(ARM_JOINTS, ARM_STIFFNESS))
            arm.damping = dict(zip(ARM_JOINTS, ARM_DAMPING))


@configclass
class LightningCookFishMimicEnvCfg(LightningCookFishEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "lightning_cook_fish"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True  # failed attempts go to <output>_failed.hdf5, for diagnosis
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        # Mimic only selects again after the first subtask with this set too (data_generator.py: without it, the first
        # selection holds for the whole episode); with one arm it changes nothing else
        self.datagen_config.generation_select_src_per_arm = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1
        # source segments Lightning can follow here (lightning_mimic_env.NearestNeighborReachableStrategy): targets
        # within reach of the base (MAX_REACH, plus 1 cm for the noise on the targets)
        reachable = {"nn_k": 3, "max_reach": MAX_REACH + 0.01}
        common = dict(
            selection_strategy="nearest_neighbor_reachable",
            selection_strategy_kwargs=reachable,
            action_noise=0.003,
            # 0.4 s from one segment to the next, in free space (see LIFT_CLEARANCE) and short since each subtask picks
            # a nearby source segment; with 5 steps and up to 30 cm (one source per episode) the joint commands jumped
            # up to 0.22 rad per step (4.4 rad/s, beyond the UR5e's 3.14) and a held fish slipped
            num_interpolation_steps=8,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        grasp = dict(subtask_term_offset_range=(5, 10), **common)
        place = dict(subtask_term_offset_range=(0, 5), **common)
        self.subtask_configs["lightning"] = [
            SubTaskConfig(object_ref="lid", subtask_term_signal="lid_grasped", **grasp),
            SubTaskConfig(
                object_ref="pot",
                subtask_term_signal="lid_off",
                **{
                    **place,
                    # the set-down spot moves with the pot: keep it off the fish (v3 set the lid on the fish in 10 of
                    # 69 attempts) and in reach of the regrasp (the real ones lie up to 0.71 m out)
                    "selection_strategy_kwargs": {
                        **reachable,
                        "release_reach": MAX_REACH - 0.02,
                        "release_clearance": {"fish": LID_ASIDE_CLEARANCE},
                    },
                },
            ),
            SubTaskConfig(object_ref="fish", subtask_term_signal="fish_grasped", **grasp),
            SubTaskConfig(object_ref="pot", subtask_term_signal="fish_in_pot", **place),
            SubTaskConfig(object_ref="lid", subtask_term_signal="lid_regrasped", **grasp),
            SubTaskConfig(object_ref="pot", subtask_term_signal=None, subtask_term_offset_range=(0, 0), **common),
        ]

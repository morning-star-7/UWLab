# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic interface for the Lightning single-arm tasks (pick-and-place, cook_fish).

Mimic reasons about grasp-point (TCP) poses in one frame shared with the objects: the env frame (``get_object_poses``
uses env-relative object poses). The action, as in the real datasets, is a joint command plus a gripper command, so:

* ``target_eef_pose_to_action`` solves IK on Lightning's factory calibration, seeded with the previous command (the
  way a joint-space teleop device moves), and
* ``action_to_target_eef_pose`` is forward kinematics of the commanded joints.

Both go through Lightning's controller Base frame (base_link turned 180 deg about z). Object poses are reduced to
position and yaw (``get_object_poses``). The subtask signals are the task's ``subtask_terms`` observation group,
whatever terms it defines. ``nearest_neighbor_reachable`` is a source selection that leaves out segments Lightning
could not follow in the current scene.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

try:  # isaaclab_mimic's datagen package imports IPython, which only mimic.py (Mimic's own scripts) puts on the path
    from isaaclab_mimic.datagen.selection_strategy import NearestNeighborObjectStrategy
except ImportError:  # e.g. scripted_demos.py: no source selection there
    NearestNeighborObjectStrategy = None

from . import scene_cfg as lab
from .kinematics import URKinematics
from .pick_place_env_cfg import GRIPPER_BODY, TCP_POS, TCP_QUAT, tcp_pose

EEF = "lightning"
ELBOW_BEND = 0.2
"""Least elbow bend (rad) the IK may command. The elbow is negative in every Lightning start pose and scripted demo; a
target out of reach would otherwise straighten it and fold it positive, where the arm stays and misses everything after.
At 0.2 rad the reach is 4 mm short of a straight arm."""

_selection_context: dict = {}
"""The scene Mimic is selecting a source segment for (object poses and Lightning's base, env frame), which Mimic does
not pass to a selection strategy: it calls ``get_object_poses`` for that one env just before selecting."""


class LightningMimicEnv(ManagerBasedRLMimicEnv):
    _kinematics: URKinematics | None = None

    def _setup_kinematics(self):
        """Calibrated kinematics, the grasp point in tool0 and the controller Base pose (env frame), once."""
        if self._kinematics is not None:
            return
        robot = self.scene[EEF]
        self._kinematics = URKinematics(lab.UR_CALIBRATION[EEF], device=self.device)
        # Grasp point in tool0 (= wrist_3_link): fixed, read off the sim once.
        w3 = robot.find_bodies("wrist_3_link")[0][0]
        rb = robot.find_bodies(GRIPPER_BODY)[0][0]
        tcp_p, tcp_q = PoseUtils.combine_frame_transforms(
            robot.data.body_pos_w[0:1, rb],
            robot.data.body_quat_w[0:1, rb],
            torch.tensor([TCP_POS], device=self.device),
            torch.tensor([TCP_QUAT], device=self.device),
        )
        tool_p, tool_q = PoseUtils.subtract_frame_transforms(
            robot.data.body_pos_w[0:1, w3], robot.data.body_quat_w[0:1, w3], tcp_p, tcp_q
        )
        self._tool = PoseUtils.make_pose(tool_p, PoseUtils.matrix_from_quat(tool_q))[0]
        base = PoseUtils.make_pose(
            (robot.data.root_pos_w[0] - self.scene.env_origins[0])[None],
            PoseUtils.matrix_from_quat(robot.data.root_quat_w[0:1]),
        )[0]
        rz180 = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], device=self.device))
        self._env_from_base = base @ rz180
        self._base_from_env = torch.linalg.inv(self._env_from_base)
        arm = self.action_manager.get_term("arm")._joint_ids
        limits = robot.data.soft_joint_pos_limits[0, arm]
        self._ik_lower, self._ik_upper = limits[:, 0].clone(), limits[:, 1].clone()
        self._ik_upper[2] = min(self._ik_upper[2].item(), -ELBOW_BEND)

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Object poses reduced to their table-plane part (position and yaw).

        Mimic moves each source segment with its object's pose. The objects here lie on the table, so a grasp should
        follow their position and heading only: a tilt (a resting object leaning a little, or settling differently
        from the source demo) would otherwise tilt the whole approach and miss.
        """
        poses = super().get_object_poses(env_ids)
        for name, pose in poses.items():
            yaw = torch.atan2(pose[..., 1, 0], pose[..., 0, 0])
            flat = torch.zeros_like(pose[..., :3, :3])
            flat[..., 0, 0], flat[..., 0, 1] = torch.cos(yaw), -torch.sin(yaw)
            flat[..., 1, 0], flat[..., 1, 1] = torch.sin(yaw), torch.cos(yaw)
            flat[..., 2, 2] = 1.0
            pose = pose.clone()
            pose[..., :3, :3] = flat
            poses[name] = pose
        if env_ids is not None and len(env_ids) == 1:
            robot = self.scene[EEF]
            _selection_context["objects"] = {name: pose[0] for name, pose in poses.items()}
            _selection_context["base_xy"] = (robot.data.root_pos_w - self.scene.env_origins)[env_ids[0], :2]
        return poses

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        pose = tcp_pose(self)[env_ids]
        return PoseUtils.make_pose(pose[:, :3], PoseUtils.matrix_from_quat(pose[:, 3:]))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        self._setup_kinematics()
        (target,) = target_eef_pose_dict.values()  # (4, 4), env frame
        target = target.clone()
        if action_noise_dict is not None:
            noise = float(action_noise_dict[EEF])
            target[:3, 3] += noise * torch.randn(3, device=self.device)
            rot_noise = PoseUtils.quat_from_angle_axis(
                torch.tensor([noise], device=self.device), PoseUtils.normalize(torch.randn(1, 3, device=self.device))
            )
            target[:3, :3] = PoseUtils.matrix_from_quat(rot_noise)[0] @ target[:3, :3]
        seed = self.action_manager.get_term("arm").last_command(env_id)
        q, _, _ = self._kinematics.ik(
            (self._base_from_env @ target)[None],
            seed[None],
            self._tool[None],
            lower=self._ik_lower,
            upper=self._ik_upper,
        )
        (gripper,) = gripper_action_dict.values()
        return torch.cat([q[0], gripper.reshape(-1).to(q.dtype)]).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        self._setup_kinematics()
        tcp = self._env_from_base @ self._kinematics.fk(action[:, :6]) @ self._tool
        return {EEF: tcp.clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF: actions[..., -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        return {name: signal[env_ids] for name, signal in self.obs_buf["subtask_terms"].items()}


if NearestNeighborObjectStrategy is not None:

    class NearestNeighborReachableStrategy(NearestNeighborObjectStrategy):
        """``nearest_neighbor_object`` among the source segments Lightning can still follow once Mimic has moved them
        with the current object pose, or the segment that misses least if none can.

        A segment is followed when every target lies within ``max_reach`` of the base (horizontally) and, where the
        gripper opens, the grasp point is within ``release_reach`` of the base and at least ``release_clearance[name]``
        from each named object. A segment carries the move from where the last one ended, which for a grasp can be 30 cm
        from the object: turned with the object's yaw it may swing out of reach. And a set-down spot kept relative to
        the pot may land on the fish.
        """

        NAME = "nearest_neighbor_reachable"

        def select_source_demo(
            self,
            eef_pose,
            object_pose,
            src_subtask_datagen_infos,
            pos_weight=1.0,
            rot_weight=1.0,
            nn_k=3,
            max_reach=0.72,
            release_reach=None,
            release_clearance=None,
        ):
            base, objects = _selection_context["base_xy"], _selection_context["objects"]
            misses = []
            for info in src_subtask_datagen_infos:
                (src_object,) = info.object_poses.values()
                xy = (object_pose @ torch.linalg.inv(src_object[0]) @ info.target_eef_pose)[:, :2, 3]
                miss = ((xy - base).norm(dim=-1).max() - max_reach).clamp(min=0)
                closed = info.gripper_action[:, 0] > 0.5
                opens = (closed[:-1] & ~closed[1:]).nonzero()
                if len(opens):
                    spot = xy[opens[0, 0] + 1]
                    if release_reach is not None:
                        miss = miss + ((spot - base).norm() - release_reach).clamp(min=0)
                    for name, clearance in (release_clearance or {}).items():
                        miss = miss + (clearance - (spot - objects[name][:2, 3]).norm()).clamp(min=0)
                misses.append(miss)
            misses = torch.stack(misses)
            feasible = (misses == 0).nonzero()[:, 0]
            if len(feasible) == 0:
                return int(misses.argmin())
            pick = super().select_source_demo(
                eef_pose, object_pose, [src_subtask_datagen_infos[i] for i in feasible], pos_weight, rot_weight, nn_k
            )
            return int(feasible[pick])

# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab Mimic interface for the Lightning pick-and-place task.

Mimic reasons in one frame for the end effector and the objects: the env frame (``get_object_poses`` uses
env-relative object poses). The arm action is the grasp-point pose in Lightning's base_link frame, which is rotated
and offset from the env frame, so the two conversions below go through the base pose.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from .pick_place_env_cfg import tcp_pose

EEF = "lightning"


class LightningPickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    def _base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Lightning's base_link pose in the env frame (identical in every env)."""
        robot = self.scene[EEF]
        return robot.data.root_pos_w[0] - self.scene.env_origins[0], robot.data.root_quat_w[0]

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
        (target,) = target_eef_pose_dict.values()  # (4, 4), env frame
        pos, rot = PoseUtils.unmake_pose(target)
        base_pos, base_quat = self._base_pose()
        pos_b, quat_b = PoseUtils.subtract_frame_transforms(
            base_pos[None], base_quat[None], pos[None], PoseUtils.quat_from_matrix(rot)[None]
        )
        pose_action = torch.cat([pos_b[0], quat_b[0]])
        if action_noise_dict is not None:
            pose_action = pose_action + action_noise_dict[EEF] * torch.randn_like(pose_action)
            pose_action[3:] = PoseUtils.normalize(pose_action[3:])
        (gripper,) = gripper_action_dict.values()
        return torch.cat([pose_action, gripper.reshape(-1)]).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        base_pos, base_quat = self._base_pose()
        n = action.shape[0]
        pos, quat = PoseUtils.combine_frame_transforms(
            base_pos.expand(n, 3), base_quat.expand(n, 4), action[:, :3], action[:, 3:7]
        )
        return {EEF: PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat)).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF: actions[..., -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        return {"grasp": self.obs_buf["subtask_terms"]["grasp"][env_ids]}

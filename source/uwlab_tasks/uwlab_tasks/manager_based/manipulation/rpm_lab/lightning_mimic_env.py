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

Both go through Lightning's controller Base frame (base_link turned 180 deg about z). The subtask signals are the
task's ``subtask_terms`` observation group, whatever terms it defines.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from . import scene_cfg as lab
from .kinematics import URKinematics
from .pick_place_env_cfg import GRIPPER_BODY, TCP_POS, TCP_QUAT, tcp_pose

EEF = "lightning"


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
        q, _, _ = self._kinematics.ik((self._base_from_env @ target)[None], seed[None], self._tool[None])
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

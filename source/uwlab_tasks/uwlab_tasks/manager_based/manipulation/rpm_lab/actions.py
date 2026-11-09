# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential IK on a frame offset from a body, with an exact Jacobian.

Isaac Lab's ``DifferentialInverseKinematicsAction`` (Isaac Lab 2.x, ``_compute_frame_jacobian``) gets the Jacobian
of a ``body_offset`` frame wrong whenever the offset is rotated or the body is not aligned with the base:

* it rotates the angular block by the offset rotation, although a rigidly attached frame has the link's angular
  velocity (for the RPM Lab grasp point, whose offset rotation cycles x -> y -> z, this permutes the angular rows);
* it applies the offset translation, given in the body frame, to a base-frame Jacobian.

PhysX's linear Jacobian also refers to the link's center of mass rather than its origin. With the wrong Jacobian the
IK diverges on Lightning. This term keeps Isaac Lab's controller and frame pose, and builds the geometric Jacobian of
the controlled frame from the link frames: each revolute joint turns about its child link's z axis through the child
link's origin (true for the UR5e assets: joint axis Z, identity ``localRot1``). Checked against finite differences of
the controlled frame (max 6e-4 per column).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class GeometricJacobianDiffIKAction(DifferentialInverseKinematicsAction):
    cfg: GeometricJacobianDiffIKActionCfg

    def __init__(self, cfg: GeometricJacobianDiffIKActionCfg, env):
        super().__init__(cfg, env)
        if len(cfg.joint_child_bodies) != self._num_joints:
            raise ValueError(f"{len(cfg.joint_child_bodies)} child bodies for {self._num_joints} joints")
        self._child_ids = [self._asset.find_bodies(name, preserve_order=True)[0][0] for name in cfg.joint_child_bodies]

    def _compute_frame_jacobian(self) -> torch.Tensor:
        data = self._asset.data
        axes = math_utils.matrix_from_quat(data.body_quat_w[:, self._child_ids])[..., :, 2]  # (N, J, 3), world
        origins = data.body_pos_w[:, self._child_ids]
        point = data.body_pos_w[:, self._body_idx]
        if self.cfg.body_offset is not None:
            point = point + math_utils.quat_apply(data.body_quat_w[:, self._body_idx], self._offset_pos)
        linear = torch.cross(axes, point[:, None] - origins, dim=-1)
        to_base = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w))
        return torch.cat([to_base @ linear.transpose(1, 2), to_base @ axes.transpose(1, 2)], dim=1)


@configclass
class GeometricJacobianDiffIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = GeometricJacobianDiffIKAction

    joint_child_bodies: list[str] = MISSING
    """Child link of each controlled joint, in the order of ``joint_names``."""


class ServoJointPositionAction(JointPositionAction):
    """Absolute joint-position commands executed like the UR controller's servoJ.

    The real arms receive teleop joint commands through ``servoJ`` and reach them after a delay and a first-order lag
    (``scene_cfg.SERVOJ_MODEL``, fitted on recordings). This term applies the same model at the physics rate: the
    command goes through a delay line and a first-order filter, and the result becomes the joint position target of a
    stiff PD. The action is the command itself, as in the real datasets.
    """

    cfg: ServoJointPositionActionCfg

    def __init__(self, cfg: ServoJointPositionActionCfg, env):
        super().__init__(cfg, env)
        dt = env.physics_dt
        self._delay_steps = max(0, int(round(cfg.delay_s / dt)))
        self._alpha = 1.0 if cfg.lag_s <= 0 else 1.0 - math.exp(-dt / cfg.lag_s)
        n, j = self.num_envs, self.action_dim
        self._line = torch.zeros(n, self._delay_steps + 1, j, device=self.device)  # oldest first
        self._filtered = torch.zeros(n, j, device=self.device)
        self._needs_init = torch.ones(n, dtype=torch.bool, device=self.device)

    def _initialize(self, env_ids: torch.Tensor):
        """Start the delay line and filter at the measured joints (after any reset or state write)."""
        q = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._line[env_ids] = q.unsqueeze(1)
        self._filtered[env_ids] = q
        self._needs_init[env_ids] = False

    def last_command(self, env_id: int) -> torch.Tensor:
        """The most recent command for one env (the measured joints right after a reset): an IK seed."""
        if self._needs_init[env_id]:
            return self._asset.data.joint_pos[env_id, self._joint_ids].clone()
        return self._line[env_id, -1].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._needs_init.any():
            self._initialize(self._needs_init.nonzero().squeeze(-1))

    def apply_actions(self):
        self._line = torch.cat([self._line[:, 1:], self.processed_actions.unsqueeze(1)], dim=1)
        self._filtered = self._filtered + self._alpha * (self._line[:, 0] - self._filtered)
        self._asset.set_joint_position_target(self._filtered, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._needs_init[slice(None) if env_ids is None else env_ids] = True


@configclass
class ServoJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ServoJointPositionAction
    use_default_offset: bool = False

    delay_s: float = 0.0
    """Delay from command to the start of the motion (s)."""
    lag_s: float = 0.0
    """First-order time constant of the tracking (s)."""

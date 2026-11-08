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

import torch
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
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

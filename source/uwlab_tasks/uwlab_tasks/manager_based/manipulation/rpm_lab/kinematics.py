# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e kinematics from an arm's factory calibration, independent of the simulation state.

The same chain the real controller uses (URDF-style joint origins x, y, z, roll, pitch, yaw, each joint turning about
its frame's z): forward kinematics of tool0 (= the sim's wrist_3_link) in the controller Base frame, the geometric
Jacobian of a point on the tool, and damped-least-squares IK. Batched over the first dimension, in torch.
"""

from __future__ import annotations

import torch
import yaml

import isaaclab.utils.math as math_utils

ENTRIES = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")


def _rot(axis: int, angle: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation about x (0), y (1) or z (2)."""
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


class URKinematics:
    def __init__(self, calibration_file: str, device: str | torch.device = "cpu"):
        with open(calibration_file) as f:
            k = yaml.safe_load(f)["kinematics"]
        self.hash = k["hash"]
        origins = torch.zeros(6, 4, 4, dtype=torch.float64)
        for i, name in enumerate(ENTRIES):
            e = {
                a: torch.tensor(float(k[name][a]), dtype=torch.float64) for a in ("x", "y", "z", "roll", "pitch", "yaw")
            }
            origins[i, :3, :3] = _rot(2, e["yaw"]) @ _rot(1, e["pitch"]) @ _rot(0, e["roll"])
            origins[i, :3, 3] = torch.stack([e["x"], e["y"], e["z"]])
            origins[i, 3, 3] = 1.0
        self.origins = origins.to(device=device, dtype=torch.float32)

    def chain(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint frames (N, 6, 4, 4; joint axis = z) and tool0 (N, 4, 4), in the controller Base frame."""
        n = q.shape[0]
        t = torch.eye(4, device=q.device, dtype=q.dtype).expand(n, 4, 4)
        frames = []
        for i in range(6):
            t = t @ self.origins[i].to(q.dtype)
            frames.append(t)
            rz = torch.eye(4, device=q.device, dtype=q.dtype).repeat(n, 1, 1)
            rz[:, :3, :3] = _rot(2, q[:, i])
            t = t @ rz
        return torch.stack(frames, dim=1), t

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        return self.chain(q)[1]

    def jacobian(self, q: torch.Tensor, tool_point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Geometric Jacobian (N, 6, 6) of ``tool_point`` (N, 3, in tool0) in the Base frame, and tool0 (N, 4, 4)."""
        frames, tool0 = self.chain(q)
        point = (tool0[:, :3, :3] @ tool_point.unsqueeze(-1)).squeeze(-1) + tool0[:, :3, 3]
        axes, origins = frames[..., :3, 2], frames[..., :3, 3]
        linear = torch.cross(axes, point.unsqueeze(1) - origins, dim=-1)
        return torch.cat([linear.transpose(1, 2), axes.transpose(1, 2)], dim=1), tool0

    def ik(
        self,
        target: torch.Tensor,
        q_seed: torch.Tensor,
        tool: torch.Tensor,
        iterations: int = 12,
        damping: float = 0.02,
        max_step: float = 0.3,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joints (N, 6) that put ``tool`` (N, 4, 4, in tool0) at ``target`` (N, 4, 4, Base frame), from ``q_seed``.

        Damped least squares, which stays on the seed's branch through the wrist singularity. ``lower``/``upper`` (6,)
        bound the joints after every step: an elbow bound keeps an out-of-reach target from straightening the elbow
        and folding it the other way, a branch it does not come back from. Returns the joints and the remaining
        position (m) and rotation (rad) errors.
        """
        q = q_seed.clone()
        eye = torch.eye(6, device=q.device, dtype=q.dtype)
        for _ in range(iterations):
            jac, tool0 = self.jacobian(q, tool[:, :3, 3])
            current = tool0 @ tool
            pos_err = target[:, :3, 3] - current[:, :3, 3]
            rot_err = math_utils.axis_angle_from_quat(
                math_utils.quat_from_matrix(target[:, :3, :3] @ current[:, :3, :3].transpose(1, 2))
            )
            err = torch.cat([pos_err, rot_err], dim=-1).unsqueeze(-1)
            dq = jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + damping**2 * eye, err)
            q = q + dq.squeeze(-1).clamp(-max_step, max_step)
            if lower is not None:
                q = torch.max(q, lower)
            if upper is not None:
                q = torch.min(q, upper)
        current = self.fk(q) @ tool
        pos_err = (target[:, :3, 3] - current[:, :3, 3]).norm(dim=-1)
        rot_err = math_utils.axis_angle_from_quat(
            math_utils.quat_from_matrix(target[:, :3, :3] @ current[:, :3, :3].transpose(1, 2))
        ).norm(dim=-1)
        return q, pos_err, rot_err

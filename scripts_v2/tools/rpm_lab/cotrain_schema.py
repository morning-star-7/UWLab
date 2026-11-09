# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot PC's ``multisensor_20hz`` episode schema (Lightning only), shared by the sim render and the real
conversion so both produce identical episodes for ``export_lerobot_dataset.py``."""

FPS = 20

STATE_NAMES = (
    [f"lightning_joint_pos_{i}" for i in range(1, 7)]
    + [f"lightning_eef_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")]
    + ["lightning_gripper_position"]
    + [f"lightning_ft_{k}" for k in ("fx", "fy", "fz", "tx", "ty", "tz")]
)
ACTION_NAMES = [f"lightning_cmd_joint_{i}" for i in range(1, 7)] + ["lightning_cmd_gripper"]
CAMERAS = ("scene_1", "wrist_1")
"""Episode video keys: the scene_1 L515 and Lightning's wrist D405 (``wrist_1``)."""

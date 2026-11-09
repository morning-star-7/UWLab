# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration of the RPM Lab tasks.

Imports nothing from Isaac Lab, so tools that must register tasks before the simulation app starts (Isaac Lab's Mimic
scripts, through ``scripts_v2/tools/rpm_lab/mimic.py``) can load this file by path.
"""

import gymnasium as gym

_PKG = "uwlab_tasks.manager_based.manipulation.rpm_lab"

TASKS = {
    "UWLab-RPMLab-Lightning-PickPlace-Mimic-v0": {
        "entry_point": f"{_PKG}.lightning_mimic_env:LightningMimicEnv",
        "env_cfg_entry_point": f"{_PKG}.pick_place_env_cfg:LightningPickPlaceMimicEnvCfg",
    },
    "UWLab-RPMLab-Lightning-CookFish-Mimic-v0": {
        "entry_point": f"{_PKG}.lightning_mimic_env:LightningMimicEnv",
        "env_cfg_entry_point": f"{_PKG}.cook_fish_env_cfg:LightningCookFishMimicEnvCfg",
    },
}


def register():
    for task_id, spec in TASKS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point=spec["entry_point"],
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": spec["env_cfg_entry_point"]},
        )

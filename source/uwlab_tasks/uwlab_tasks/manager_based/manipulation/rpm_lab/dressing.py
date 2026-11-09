# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-only set dressing for the twin: things the real cameras see that the robots never touch (a loose cable)."""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Usd


@clone
def spawn_cable(
    prim_path: str,
    cfg: CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """A cable lying along ``cfg.points`` (parent frame): one capsule per segment, no collision."""
    stage = get_current_stage()
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    points = np.asarray(cfg.points, dtype=np.float64)
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        direction = b - a
        length = float(np.linalg.norm(direction))
        axis = np.cross([0.0, 0.0, 1.0], direction / length)  # rotate the capsule's z axis onto the segment
        angle = math.acos(float(np.clip(direction[2] / length, -1.0, 1.0)))
        if np.linalg.norm(axis) > 1e-9:
            axis = axis / np.linalg.norm(axis) * math.sin(angle / 2)
        quat = (math.cos(angle / 2), *axis)
        create_prim(
            f"{prim_path}/segment_{i:02d}",
            prim_type="Capsule",
            translation=tuple(0.5 * (a + b)),
            orientation=quat,
            attributes={"radius": cfg.radius, "height": length, "axis": "Z"},
            stage=stage,
        )
    material_path = f"{prim_path}/Looks/material"
    cfg.visual_material.func(material_path, cfg.visual_material)
    bind_visual_material(prim_path, material_path, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class CableCfg(sim_utils.SpawnerCfg):
    """A visual-only cable through ``points`` (meters, in the parent prim's frame)."""

    func: Callable = spawn_cable
    points: tuple[tuple[float, float, float], ...] = MISSING
    radius: float = 0.0035
    visual_material: sim_utils.PreviewSurfaceCfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.015, 0.015, 0.015), roughness=0.5
    )

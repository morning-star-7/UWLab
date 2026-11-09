# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep cameras that are mounted under robot links (the wrist D405s) attached to their links in Fabric.

Isaac Lab's ``Camera`` reads its pose through an ``XformPrimView`` in Fabric mode. On its first pose read (the
camera's ``reset``, e.g. the env's first reset) that view syncs Fabric from USD once: it writes each camera's USD world
pose into Fabric as a world matrix. USD still holds the spawn configuration, because PhysX writes link poses only to
Fabric. If the robot has moved by then (an env's reset puts it at its home), the Fabric hierarchy turns that stale world
pose into a wrong local offset from the link, and the camera keeps rendering from near its spawn pose instead of
following the link. Writing the camera's authored local transform back into Fabric re-attaches it; later pose reads
do not sync again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import InteractiveScene


def reattach_mounted_cameras(scene: InteractiveScene):
    """Re-attach each camera of ``scene`` to its parent prim in Fabric, once per camera (later calls are no-ops)."""
    import omni.usd
    import usdrt
    from isaaclab.sensors import Camera
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    done = 0
    for camera in scene.sensors.values():
        if not isinstance(camera, Camera) or getattr(camera, "_fabric_reattached", False):
            continue
        camera.reset()  # its first pose read does the one-time USD -> Fabric sync; get that over with first
        for path in sim_utils.find_matching_prim_paths(camera.cfg.prim_path):
            local = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
            rt_stage.GetPrimAtPath(path).GetAttribute("omni:fabric:localMatrix").Set(
                usdrt.Gf.Matrix4d(*[local[i][j] for i in range(4) for j in range(4)])
            )
        camera._fabric_reattached = True
        done += 1
    if done:
        hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
        )
        hierarchy.update_world_xforms()


def reattach_mounted_cameras_event(env: ManagerBasedEnv, env_ids: Sequence[int] | None):
    """Event term (mode "reset") form of :func:`reattach_mounted_cameras`: the env's first reset does the sync (in
    ``scene.reset``, before reset events), this re-attaches right after; later resets cost nothing."""
    reattach_mounted_cameras(env.scene)

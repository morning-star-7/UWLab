# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the cook_fish objects (saucepan, lid, plush fish) as rigid-body USDs.

Sizes come from the 2026-10-28 cook_fish recordings: diameters from scene_1 frames back-projected through the calibrated
camera (pot and lid about 130 mm across, the fish about 63 x 63 mm); heights from Lightning's grasp-point height when
the gripper closed on the lid handle, about 82 mm above the table with the lid on the pot and 39 mm with the lid on the
table (so the rim is about 45 mm up and the handle's top about 35 mm above the lid's underside). The real plush fish
squashes under the lid; the rigid one is modeled flat (34 mm) so the lid closes over it.

Collision: the pot is hollow (a bottom plus 24 wall segments); the lid is a disc, a centering lip that drops inside the
rim, and a raised bar handle the gripper closes on; the fish is a flat-bottomed elliptic puck (a stable rest, unlike a hull of the ellipsoid). Visuals are smooth trimesh
meshes. Each object's origin is the center of its bottom face. Runs with any Python that has ``pxr`` and ``trimesh``
(e.g. the Isaac Sim env's):

    python scripts_v2/tools/rpm_lab/build_cook_fish_assets.py --out /data/cui00191/rpm_lab_assets/objects
"""

import argparse
import math
import numpy as np
import os
import trimesh
import yaml

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

POT_RADIUS = 0.065  # outer
POT_WALL = 0.003
POT_HEIGHT = 0.045
LID_RADIUS = 0.067
LID_THICKNESS = 0.006
LID_HANDLE_HEIGHT = 0.032  # bar axis above the lid's bottom face (an arched strap on the real lid)
LID_HANDLE_LENGTH = 0.08
POT_BOTTOM = 0.003
POT_HANDLE_RISE = 0.025  # how far the wire loop handles stick out from the wall (along the pot's x axis)
LID_LIP_DEPTH = 0.003  # centering lip under the lid, inside the pot's rim
LID_LIP_CLEARANCE = 0.010  # radial play of the lip inside the rim: a lid set down up to 1 cm off center still sits flat
FISH_SIZE = (0.063, 0.060, 0.034)  # x (length), y, z: the plush fish lying flat, squashed to fit under the lid
STEEL_COLOR, STEEL_METALLIC, STEEL_ROUGHNESS = (0.80, 0.81, 0.83), 0.6, 0.3
"""Stainless steel under the twin's lighting (RTX reflections off): closest to the real pot and lid in scene_1 and the
wrist camera among the variants tried; fully metallic renders black, matte renders chalk white."""
FISH_COLOR = (0.88, 0.84, 0.62)  # the plush fish is white with yellow patches


def _material(stage, path: str, color, metallic: float, roughness: float, friction: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction * 0.9)
    physics.CreateRestitutionAttr(0.0)
    return material


def _mesh(stage, path: str, mesh: trimesh.Trimesh, material: UsdShade.Material):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh.vertices])
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr([int(i) for i in mesh.faces.flatten()])
    prim.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)


def _collider(stage, path: str, kind: str, material: UsdShade.Material, translate, rotate_z_deg=0.0, **dims):
    """A collision primitive (invisible): kind cube (size xyz), cylinder or capsule (radius, height, axis)."""
    if kind == "cube":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
    elif kind == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr(dims["radius"])
        prim.CreateHeightAttr(dims["height"])
        prim.CreateAxisAttr(dims.get("axis", "Z"))
    else:
        prim = UsdGeom.Capsule.Define(stage, path)
        prim.CreateRadiusAttr(dims["radius"])
        prim.CreateHeightAttr(dims["height"])
        prim.CreateAxisAttr(dims.get("axis", "Z"))
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddRotateZOp().Set(rotate_z_deg)
    if kind == "cube":
        xform.AddScaleOp().Set(Gf.Vec3f(*dims["size"]))
    prim.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material, materialPurpose="physics")


def _new_stage(path: str, name: str, mass: float) -> tuple[Usd.Stage, Usd.Prim]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    return stage, root


def _handle_loop(radius: float, span: float, rise: float, wire: float) -> trimesh.Trimesh:
    """A wire loop (half torus) of the given span and rise, in the x-z plane, ends at x = +-span / 2, z = 0."""
    torus = trimesh.creation.torus(major_radius=span / 2, minor_radius=wire, major_sections=48, minor_sections=10)
    torus.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))  # into the x-z plane
    keep = torus.triangles_center[:, 2] >= 0.0
    loop = trimesh.Trimesh(torus.vertices, torus.faces[keep])
    loop.apply_scale([1.0, 1.0, rise / (span / 2)])
    return loop


def build_pot(path: str):
    stage, root = _new_stage(path, "Pot", mass=0.35)
    steel = _material(stage, "/Pot/Looks/Steel", STEEL_COLOR, STEEL_METALLIC, STEEL_ROUGHNESS, friction=0.6)
    wall = trimesh.creation.annulus(r_min=POT_RADIUS - POT_WALL, r_max=POT_RADIUS, height=POT_HEIGHT, sections=96)
    wall.apply_translation([0, 0, POT_HEIGHT / 2])
    bottom = trimesh.creation.cylinder(radius=POT_RADIUS - 0.0005, height=POT_BOTTOM, sections=96)
    bottom.apply_translation([0, 0, POT_BOTTOM / 2])
    handles = []
    for side in (-1, 1):
        loop = _handle_loop(POT_RADIUS, span=0.05, rise=POT_HANDLE_RISE, wire=0.0025)
        loop.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2 * side, [0, 0, 1]))  # spans along y
        loop.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2 * side, [0, 1, 0]))  # sticks out
        loop.apply_translation([side * (POT_RADIUS - 0.002), 0, POT_HEIGHT - 0.012])
        handles.append(loop)
    _mesh(stage, "/Pot/visuals/body", trimesh.util.concatenate([wall, bottom] + handles), steel)
    _collider(
        stage, "/Pot/collisions/bottom", "cylinder", steel, (0, 0, POT_BOTTOM / 2), radius=POT_RADIUS, height=POT_BOTTOM
    )
    segments = 24
    width = 2 * math.pi * POT_RADIUS / segments * 1.15
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        r = POT_RADIUS - POT_WALL / 2
        _collider(
            stage,
            f"/Pot/collisions/wall_{i:02d}",
            "cube",
            steel,
            (r * math.cos(angle), r * math.sin(angle), POT_HEIGHT / 2),
            rotate_z_deg=math.degrees(angle),
            size=(POT_WALL + 0.001, width, POT_HEIGHT),
        )
    for side in (-1, 1):
        _collider(
            stage,
            f"/Pot/collisions/handle_{'l' if side < 0 else 'r'}",
            "capsule",
            steel,
            (side * (POT_RADIUS - 0.003 + POT_HANDLE_RISE), 0, POT_HEIGHT - 0.012),
            radius=0.003,
            height=0.05,
            axis="Y",
        )
    stage.GetRootLayer().Save()


def build_lid(path: str):
    stage, root = _new_stage(path, "Lid", mass=0.15)
    steel = _material(stage, "/Lid/Looks/Steel", STEEL_COLOR, STEEL_METALLIC, STEEL_ROUGHNESS, friction=0.6)
    disc = trimesh.creation.cylinder(radius=LID_RADIUS, height=LID_THICKNESS, sections=96)
    disc.apply_translation([0, 0, LID_THICKNESS / 2])
    lip = trimesh.creation.cylinder(radius=POT_RADIUS - POT_WALL - LID_LIP_CLEARANCE, height=LID_LIP_DEPTH, sections=96)
    lip.apply_translation([0, 0, -LID_LIP_DEPTH / 2])
    bar = trimesh.creation.capsule(radius=0.004, height=LID_HANDLE_LENGTH, count=[16, 16])
    bar.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))  # along x
    bar.apply_translation([0, 0, LID_HANDLE_HEIGHT])
    legs = []
    for side in (-1, 1):
        leg = trimesh.creation.cylinder(radius=0.003, height=LID_HANDLE_HEIGHT - LID_THICKNESS, sections=16)
        leg.apply_translation([side * LID_HANDLE_LENGTH / 2, 0, (LID_HANDLE_HEIGHT + LID_THICKNESS) / 2])
        legs.append(leg)
    _mesh(stage, "/Lid/visuals/body", trimesh.util.concatenate([disc, lip, bar] + legs), steel)
    _collider(
        stage,
        "/Lid/collisions/disc",
        "cylinder",
        steel,
        (0, 0, LID_THICKNESS / 2),
        radius=LID_RADIUS,
        height=LID_THICKNESS,
    )
    _collider(
        stage,
        "/Lid/collisions/lip",
        "cylinder",
        steel,
        (0, 0, -LID_LIP_DEPTH / 2),
        radius=POT_RADIUS - POT_WALL - LID_LIP_CLEARANCE,
        height=LID_LIP_DEPTH,
    )
    _collider(
        stage,
        "/Lid/collisions/bar",
        "capsule",
        steel,
        (0, 0, LID_HANDLE_HEIGHT),
        radius=0.004,
        height=LID_HANDLE_LENGTH,
        axis="X",
    )
    for side in (-1, 1):
        _collider(
            stage,
            f"/Lid/collisions/leg_{'l' if side < 0 else 'r'}",
            "cylinder",
            steel,
            (side * LID_HANDLE_LENGTH / 2, 0, (LID_HANDLE_HEIGHT + LID_THICKNESS) / 2),
            radius=0.003,
            height=LID_HANDLE_HEIGHT - LID_THICKNESS,
        )
    stage.GetRootLayer().Save()


def build_fish(path: str):
    stage, root = _new_stage(path, "Fish", mass=0.03)
    plush = _material(stage, "/Fish/Looks/Plush", FISH_COLOR, metallic=0.0, roughness=0.95, friction=1.2)
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([FISH_SIZE[0] / 2 * 0.85, FISH_SIZE[1] / 2, FISH_SIZE[2] / 2])
    body.apply_translation([0, 0, FISH_SIZE[2] / 2])
    tail = trimesh.creation.cone(radius=0.014, height=0.016, sections=24)
    tail.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0]))
    tail.apply_translation([-FISH_SIZE[0] / 2 * 0.85 + 0.002, 0, FISH_SIZE[2] / 2])
    _mesh(stage, "/Fish/visuals/body", trimesh.util.concatenate([body, tail]), plush)
    # Collision: a flat-bottomed elliptic puck with rounded edges (the tail is visual only). 60 vertices: PhysX
    # simplifies convex hulls to at most 64, and the simplified hull of the visual ellipsoid rested on a facet 20 deg
    # off level, so the fish lay tilted, rolled over easily and Mimic carried the tilt into its grasps.
    a, b, h = FISH_SIZE[0] / 2 * 0.85, FISH_SIZE[1] / 2, FISH_SIZE[2]
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    points = [
        [scale * a * math.cos(t), scale * b * math.sin(t), z]
        for z, scale in ((0.0, 0.7), (0.15 * h, 0.95), (0.5 * h, 1.0), (0.85 * h, 0.95), (h, 0.7))
        for t in angles
    ]
    puck = trimesh.convex.convex_hull(np.array(points))
    hull = UsdGeom.Mesh.Define(stage, "/Fish/collisions/body")
    hull.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in puck.vertices])
    hull.CreateFaceVertexCountsAttr([3] * len(puck.faces))
    hull.CreateFaceVertexIndicesAttr([int(i) for i in puck.faces.flatten()])
    hull.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr("convexHull")
    UsdShade.MaterialBindingAPI.Apply(hull.GetPrim()).Bind(plush, materialPurpose="physics")
    UsdPhysics.MassAPI(root).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, h / 2))  # geometric center, not computed
    stage.GetRootLayer().Save()


METADATA = {
    "pot": {
        "radius": POT_RADIUS,
        "inner_radius": POT_RADIUS - POT_WALL,
        "rim_height": POT_HEIGHT,
        "bottom_thickness": POT_BOTTOM,
        "handle_reach": POT_RADIUS + POT_HANDLE_RISE,  # handles' outer extent from the center, along x
    },
    "lid": {
        "radius": LID_RADIUS,
        "thickness": LID_THICKNESS,
        "lip_depth": LID_LIP_DEPTH,
        "handle_height": LID_HANDLE_HEIGHT,
        "handle_length": LID_HANDLE_LENGTH,
    },
    "fish": {"size": list(FISH_SIZE)},
}
"""Dimensions (m) written next to each USD as ``metadata.yaml``; the cook_fish task reads them from there. Every
object's origin is the center of its bottom face; the lid handle runs along the lid's x axis."""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="/data/cui00191/rpm_lab_assets/objects")
    args = parser.parse_args()
    for name, build in (("pot", build_pot), ("lid", build_lid), ("fish", build_fish)):
        path = os.path.join(args.out, name, f"{name}.usda")
        if os.path.exists(path):
            os.remove(path)
        build(path)
        with open(os.path.join(args.out, name, "metadata.yaml"), "w") as f:
            yaml.safe_dump(METADATA[name], f, sort_keys=False)
        print(f"Wrote {path} and its metadata.yaml")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()

# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the object layouts off real cook_fish episodes: where the operators found the pot and the fish, and where they
set the lid aside.

Nothing in a recording measures the objects, but the gripper does at each grasp: the grasp point (``GRASP_DEPTH``
along the flange z) is over the lid handle (= pot center) when the gripper first closes on the lid, over the fish when
it closes on it, and over the set-aside handle when it closes on the lid again. An episode counts when its gripper
command (closed above 0.65: some operators let go to only 0.6) runs close-open-close-open-close with the fish let go
where the lid was picked up and the lid picked up again where it was set down; a fumbled grasp breaks that pattern, and
an episode without it is skipped. Run on the twin's own renders, the estimate is within 0.3 cm of the true object
positions.

Points are written in Lightning's controller Base frame (``ur_base``), as recorded; the task turns them into the
env frame with the twin's base pose. Leave out the world model's validation episodes (``--exclude``) so that the sim
data never shows their layouts.

    python scripts_v2/tools/rpm_lab/extract_cook_fish_layouts.py \\
        --episodes /data/cui00191/WM-cotrain/real_frames/cook_fish --exclude 13,15,24,29,39 \\
        --out \\
    source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/calibration/cook_fish_layouts_20261028.json
"""

import argparse
import glob
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation

GRASP_DEPTH = 0.150
"""Grasp point along the flange z (m): the twin's TCP in tool0 at runtime (no rotation)."""
SAME_SPOT = 0.05
"""Horizontal distance (m) under which a release and a grasp count as the same spot (lid pick-up vs fish drop-off)."""
SAME_LID_SPOT = 0.06
"""The same for the lid's set-down vs pick-up; it is dropped from a few cm up and may slide."""
MOVED = 0.08
"""Least horizontal move (m) between the lid pick-up and its set-down or the fish."""


def grasp_points(state: np.ndarray) -> np.ndarray:
    """(T, 3) grasp point in the controller Base frame, from the recorded flange pose (state[6:12])."""
    rot = Rotation.from_rotvec(state[:, 9:12]).as_matrix()
    return state[:, 6:9] + rot[:, :, 2] * GRASP_DEPTH


def horizontal(p: np.ndarray) -> np.ndarray:
    """The tabletop components: Lightning's Base y points down (its mount is on a column), x and z are horizontal."""
    return p[..., [0, 2]]


def layout(state: np.ndarray, action: np.ndarray) -> dict | None:
    points = grasp_points(state)
    closed = action[:, 6] > 0.65
    events = [(int(i), bool(closed[i])) for i in np.flatnonzero(closed[1:] != closed[:-1]) + 1]
    for k in range(len(events) - 4):
        (lid, c1), (aside, c2), (fish, c3), (drop, c4), (regrasp, c5) = events[k : k + 5]
        if not (c1 and not c2 and c3 and not c4 and c5):
            continue

        def apart(a, b):
            return float(np.linalg.norm(horizontal(points[a]) - horizontal(points[b])))

        if (
            apart(lid, drop) < SAME_SPOT
            and apart(aside, regrasp) < SAME_LID_SPOT
            and apart(lid, aside) > MOVED
            and apart(lid, fish) > MOVED
        ):
            return {
                "pot": points[lid].tolist(),
                "fish": points[fish].tolist(),
                "lid_aside": points[regrasp].tolist(),
                "steps": [lid, fish, regrasp],
            }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="Directory of real episode_XXXX.npz (co-training format).")
    parser.add_argument("--exclude", default="", help="Comma-separated episode indices to leave out.")
    parser.add_argument("--out", required=True, help="Output .json.")
    args = parser.parse_args()
    exclude = {int(e) for e in args.exclude.split(",") if e}

    layouts, skipped = [], []
    for path in sorted(glob.glob(os.path.join(args.episodes, "episode_*.npz"))):
        index = int(os.path.basename(path)[8:12])
        if index in exclude:
            continue
        data = np.load(path)
        found = layout(data["state"], data["action"])
        meta = json.loads(str(data["meta"]))
        if found is None:
            skipped.append(meta["demo"])
            continue
        layouts.append({"episode": meta["demo"], **found})
    with open(args.out, "w") as f:
        json.dump(
            {
                "frame": "Lightning controller Base (ur_base), meters",
                "points": f"grasp point ({GRASP_DEPTH} m along the flange z) at the grasp that locates each object",
                "source": os.path.abspath(args.episodes),
                "excluded_episodes": sorted(exclude),
                "skipped": skipped,
                "layouts": layouts,
            },
            f,
            indent=1,
        )
    print(f"{len(layouts)} layouts, {len(skipped)} episodes without a clean grasp sequence: {skipped}")


if __name__ == "__main__":
    main()

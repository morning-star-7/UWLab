# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Isaac Lab Mimic's own scripts on the RPM Lab tasks.

Isaac Lab's scripts only know the tasks in ``isaaclab_tasks``. This wrapper registers the RPM Lab gym ids (without
importing Isaac Lab, so before the scripts start the app), puts ``isaaclab_mimic`` and its missing dependencies on the
path (neither is installed in the shared environment; see ``EXTRA_DEPS``), and runs the script unchanged.

    python scripts_v2/tools/rpm_lab/mimic.py annotate --auto --headless \\
        --input_file source.hdf5 --output_file annotated.hdf5
    python scripts_v2/tools/rpm_lab/mimic.py generate --headless --num_envs 4 --generation_num_trials 20 \\
        --input_file annotated.hdf5 --output_file generated.hdf5
    python scripts_v2/tools/rpm_lab/mimic.py replay --dataset_file generated.hdf5 --select_episodes 0

``--task`` defaults to the Lightning pick-and-place Mimic task for ``annotate`` and ``replay``.
"""

import importlib.util
import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXTRA_DEPS = os.environ.get("RPM_LAB_PYDEPS", "/data/cui00191/rpm_lab_pydeps")
"""IPython, ipywidgets and tomli for isaaclab_mimic, installed with ``pip install --target`` because the shared
environment lacks them. Appended to the end of the path, so the environment's own packages take precedence."""
DEFAULT_TASK = "UWLab-RPMLab-Lightning-PickPlace-Mimic-v0"
SCRIPTS = {
    "annotate": "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py",
    "generate": "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
    "replay": "scripts/tools/replay_demos.py",
}


def main():
    # Kit's close() ends the process with os._exit, which drops block-buffered output (e.g. Mimic's success rate).
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        sys.exit(f"usage: mimic.py {{{','.join(SCRIPTS)}}} [script arguments]\n\n{__doc__}")
    command, script_args = sys.argv[1], sys.argv[2:]

    # Isaac Lab's repository root, from the installed isaaclab package (source/isaaclab/isaaclab).
    package_dir = importlib.util.find_spec("isaaclab").submodule_search_locations[0]
    isaaclab_root = os.path.abspath(os.path.join(package_dir, "..", "..", ".."))
    sys.path.insert(0, os.path.join(isaaclab_root, "source", "isaaclab_mimic"))
    if os.path.isdir(EXTRA_DEPS):
        sys.path.append(EXTRA_DEPS)

    registry_path = os.path.join(REPO, "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/registry.py")
    spec = importlib.util.spec_from_file_location("rpm_lab_registry", registry_path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    registry.register()

    if command in ("annotate", "replay") and "--task" not in script_args:
        script_args = ["--task", DEFAULT_TASK] + script_args
    script = os.path.join(isaaclab_root, SCRIPTS[command])
    sys.argv = [script] + script_args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

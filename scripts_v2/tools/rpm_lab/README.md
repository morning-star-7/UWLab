# RPM Lab digital twin tools

Scene: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/scene_cfg.py` (Thunder and Lightning on the
Vention workcell; `RpmLabDualArmCameraSceneCfg` adds both wrist D405s). Assets live in `/data/cui00191/rpm_lab_assets`,
real recordings copied from the lab robot PC (Zeus) in `/data/cui00191/rpm_lab_recordings`.

| Tool | Runs in | What it does |
|---|---|---|
| `build_calibrated_ur5e_usd.py` | any Python with `pxr` | Writes a robot override layer: an arm's UR factory calibration (`ur5e_calibration/`) and the real UR5e joint limits. |
| `view_dual_arm_scene.py` | Isaac Sim | Checks mounts, mounting plates, calibration, limits and home poses; `--save_images` renders views and wrist cameras. |
| `extract_spark_recording.py` | Python with `mcap mcap-ros2-support` (raw bags) or `pandas pyarrow` (LeRobot) | Converts a Zeus recording into one `.npz`. |
| `replay_recording.py` | Isaac Sim | Checks the twin against a recording: `--mode kinematic` (flange vs controller, hand height over the table), `--mode replay --arm teleport\|servo` (physics replay, servoJ model fitted to the recording, video, real-vs-sim wrist stills). |
| `scripted_pick_place_demos.py` | Isaac Sim | Records scripted source demos for Mimic (Isaac Lab demo HDF5). |
| `mimic.py` | Isaac Sim | Runs Isaac Lab Mimic's `annotate` / `generate` / `replay` scripts on the RPM Lab tasks. |

Isaac Sim on this server (no personal env): `PATH=/data/kanth042/envs/uwlab-isaac51/bin:$PATH OMNI_KIT_ACCEPT_EULA=YES
PYTHONPATH=$PWD/source/uwlab:$PWD/source/uwlab_assets:$PWD/source/uwlab_tasks:$PWD/source/uwlab_rl python ... --headless`.

```bash
R=/data/cui00191/rpm_lab_recordings
python scripts_v2/tools/rpm_lab/extract_spark_recording.py --images 0.1 --depth \
    --episode $R/episode-20261028-185259 --out $R/extracted/ep1028_185259.npz
python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode kinematic --recording $R/extracted/ep1106.npz
python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode replay --arm servo --recording $R/extracted/ep1106.npz
python scripts_v2/tools/rpm_lab/replay_recording.py --headless --mode replay --recording $R/extracted/ep1028_185259.npz \
    --video $R/replays/ep1028.mp4 --stills $R/replays/ep1028_stills
```

## Scripted demos + MimicGen (Lightning pick-and-place)

Task `UWLab-RPMLab-Lightning-PickPlace-Mimic-v0` (`rpm_lab/pick_place_env_cfg.py`, `pick_place_mimic_env.py`): a
45 mm cube onto a 100 mm plate, 20 Hz, absolute grasp-point pose in Lightning's base frame + binary gripper. Mimic
subtasks: grasp (relative to the cube), place (relative to the plate). `isaaclab_mimic` and its missing deps (IPython,
ipywidgets, tomli in `/data/cui00191/rpm_lab_pydeps`) are put on the path by `mimic.py`; nothing is installed into the
shared environment.

```bash
M=/data/cui00191/rpm_lab_mimic
python scripts_v2/tools/rpm_lab/scripted_pick_place_demos.py --headless --num_demos 10 --dataset_file $M/source.hdf5
python scripts_v2/tools/rpm_lab/mimic.py annotate --auto --headless --input_file $M/source.hdf5 --output_file $M/annotated.hdf5
python scripts_v2/tools/rpm_lab/mimic.py generate --headless --num_envs 8 --generation_num_trials 100 \
    --input_file $M/annotated.hdf5 --output_file $M/generated_100.hdf5
```

First run (11/2026): 10/10 scripted demos, 10/10 annotated, 100/101 generated (99%) in 6.6 min with 8 envs on a
shared A6000. The datasets keep full sim states, so they can be re-rendered once the scene camera and appearance are
fixed. Isaac Lab's IK term gets the Jacobian of an offset frame wrong on these arms (`rpm_lab/actions.py` explains and
replaces it).

Rendering a Mimic dataset from its states (`scene.reset_to` per step): in this task env a camera parented under a robot
link does not follow the link (it renders from its spawn pose, even after physics steps or `set_world_poses`), unlike
in the bare scene. Workaround that works: a free camera under the env prim, posed every frame from the link pose times
the fixed link-to-camera transform (`D405_OPTICAL_FRAME` + `D405_IMAGER_OFFSET`). Preview of three generated demos:
`/data/cui00191/rpm_lab_mimic/generated_100_preview.mp4`.

## Checked against the real arms (11/2026)

- Mounts: recorded gripper paths sit on the tabletop only with both cable openings down (Lightning = mirror of Thunder).
- Kinematics: sim flange vs controller-reported flange, median 0.002-0.004 mm (still arm: < 0.05 mm; larger values
  only during fast motion, where the robot PC reads joints and pose in separate RTDE calls).
- Wrist D405: finger silhouettes in the real wrist images fix the camera at the left imager (`D405_IMAGER_OFFSET`).
- Motion: driving the sim with the teleop commands through the fitted servoJ model reproduces the measured joints.

## Known gaps

- Lightning vs the table: during real table-level grasps Lightning's hand reaches 12 mm (10/2026) to ~30 mm (09/2026)
  below the sim tabletop, so its height or tilt relative to the table is off. Needs a touch measurement (fingertip on
  marked table points, record `getActualTCPPose`). The real table also has a cloth, which the sim lacks.
- Wrist depth: back-projected real D405 depth disagrees with the sim geometry in a range-dependent way (apparent 14 deg
  table tilt) while the color images agree; unresolved, so depth is not used for calibration yet.
- Scene cameras (L515 `scene_1`): intrinsics known, pose never calibrated.
- Thunder's D405 intrinsics were never recorded (Lightning's are used); joint friction/armature are reference values.

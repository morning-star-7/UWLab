# RPM Lab digital twin tools

Scene: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/rpm_lab/scene_cfg.py` (Thunder and Lightning on the
Vention workcell; `RpmLabDualArmCameraSceneCfg` adds both wrist D405s and the calibrated scene_1 L515). Assets live in `/data/cui00191/rpm_lab_assets`,
real recordings copied from the lab robot PC (Zeus) in `/data/cui00191/rpm_lab_recordings`.

| Tool | Runs in | What it does |
|---|---|---|
| `build_calibrated_ur5e_usd.py` | any Python with `pxr` | Writes a robot override layer: an arm's UR factory calibration (`rpm_lab/calibration/`) and the real UR5e joint limits. |
| `view_dual_arm_scene.py` | Isaac Sim | Checks mounts, mounting plates, calibration, limits and home poses; `--save_images` renders views and wrist cameras. |
| `extract_spark_recording.py` | Python with `mcap mcap-ros2-support` (raw bags) or `pandas pyarrow` (LeRobot) | Converts a Zeus recording into one `.npz`. |
| `fit_arm_base_from_scene_camera.py` | Isaac Sim | Fits an arm's base offset from scene_1 frames of real recordings (silhouette outline vs image edges). |
| `replay_recording.py` | Isaac Sim | Checks the twin against a recording: `--mode kinematic` (flange vs controller, hand height over the table), `--mode replay --arm teleport\|servo` (physics replay, servoJ model fitted to the recording, video, real-vs-sim wrist stills). |
| `build_cook_fish_assets.py` | any Python with `pxr` and `trimesh` | Builds the cook_fish pot, lid and fish USDs (+ `metadata.yaml` dimensions) from the real recordings' measurements. |
| `scripted_demos.py` | Isaac Sim | Records scripted source demos for Mimic (Isaac Lab demo HDF5) on the pick-and-place or cook_fish task. |
| `mimic.py` | Isaac Sim | Runs Isaac Lab Mimic's `annotate` / `generate` / `replay` scripts on the RPM Lab tasks. |
| `render_mimic_dataset.py` | Isaac Sim | Renders a demo HDF5 through scene_1 and the wrist D405 in the real cameras' models; writes per-episode `.npz` with the real state/action schema. |
| `export_lerobot_dataset.py` | lerobot 0.5.2 venv | Writes those episodes as a LeRobot v3 dataset with the real feature names (h264, 20 fps). |
| `world_model/build_iws_dataset.py` | IWS venv | Mixes real and sim LeRobot datasets into an Interactive World Simulator training set. |
| `world_model/train_iws.py` | IWS venv | Trains the Interactive World Simulator (three stages) on such a set. |

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

## Scripted demos + MimicGen (Lightning tasks)

Both tasks use the real datasets' action: six absolute joint commands plus a gripper command in [0 open, 1 closed] at
20 Hz, executed through the servoJ model fitted on real recordings (`rpm_lab/actions.py`). Mimic reasons in grasp-point
poses; `rpm_lab/lightning_mimic_env.py` turns them into joint commands with IK on Lightning's factory calibration
(`rpm_lab/kinematics.py`), seeded with the previous command, and back with FK. `isaaclab_mimic` and its missing deps
(IPython, ipywidgets, tomli in `/data/cui00191/rpm_lab_pydeps`) are put on the path by `mimic.py`; nothing is installed
into the shared environment.

- `UWLab-RPMLab-Lightning-PickPlace-Mimic-v0` (`pick_place_env_cfg.py`): a 45 mm cube onto a 100 mm plate; subtasks
  grasp (cube) and place (plate). 11/2026: 10/10 scripted, 10/10 annotated, 50/50 generated.
- `UWLab-RPMLab-Lightning-CookFish-Mimic-v0` (`cook_fish_env_cfg.py`): "open lid, place fish inside and close lid",
  set up like the 10/28 real episodes (below); subtasks lid grasped, lid set down beside the pot, fish grasped, fish in
  the pot, lid grasped again, lid on the pot.

```bash
M=/data/cui00191/rpm_lab_mimic/cook_fish T=UWLab-RPMLab-Lightning-CookFish-Mimic-v0
python scripts_v2/tools/rpm_lab/scripted_demos.py --headless --task $T --num_demos 10 --dataset_file $M/source.hdf5
python scripts_v2/tools/rpm_lab/mimic.py annotate --task $T --auto --headless --input_file $M/source.hdf5 --output_file $M/annotated.hdf5
python scripts_v2/tools/rpm_lab/mimic.py generate --headless --num_envs 8 --generation_num_trials 50 \
    --input_file $M/annotated.hdf5 --output_file $M/generated_50.hdf5
```

The datasets keep full sim states, so they can be re-rendered whenever the cameras or the appearance change.

### cook_fish, matched to the 10/28 recordings

- Objects (`build_cook_fish_assets.py`): pot and lid about 130 mm across (scene_1 back-projection); heights from where
  the operator's grasp point was when the gripper closed on the lid handle, about 82 mm above the table with the lid on
  the pot and 39 mm with it on the table, so the rim is 45 mm up and the handle's axis 32 mm above the lid's underside.
  The plush fish squashes under the real lid; the rigid one is 34 mm flat so the lid closes over it. The lid's centering
  lip has 1 cm of radial play so a lid set down slightly off center still sits flat.
- Layout: pot where the real pots were (x -0.17..0.03, y -0.17..-0.10), the lid on it with its handle along world y
  (the operators closed the gripper with its closing axis within 40 deg of world x, at a right angle to the handle), the
  fish 0.13-0.20 m from the pot in any direction that stays in Lightning's real workspace.
- Lightning starts at real episode 185333's start; Thunder is parked at its teleop home, outside the scene_1 view.
- The scripted policy re-plans each phase from the current state and carries held objects by their actual pose in the
  hand; it drops the fish from 25 mm above the rim (the operators let go 25-60 mm above it).

## Co-training data (LeRobot)

```bash
python scripts_v2/tools/rpm_lab/render_mimic_dataset.py --headless --num_envs 8 --dataset $M/generated_50.hdf5 --out $M/render
/data/cui00191/rpm_lab_envs/lerobot/bin/python scripts_v2/tools/rpm_lab/export_lerobot_dataset.py \
    --frames $M/render --root $M/lerobot --repo_id rpm_lab/sim_lightning_cook_fish
```

The render restores each step's sim state, renders scene_1 and the wrist D405, warps the pinhole renders into the real
cameras' models (`cv2.remap`: principal point, L515 distortion) and applies each camera's color gain. It renders
`--num_envs` episodes at once (one per env, tiled cameras; 4 cook_fish episodes, 1736 frames, in about 90 s plus 60 s
startup on a shared A6000) and streams each camera into a near-lossless H.264 file (crf 10, 43-44 dB PSNR vs the raw
render, about 5 MB per camera per episode instead of 400 MB raw) next to a small `.npz` of state, action and meta. The
LeRobot export re-encodes like the robot PC (37-38 dB vs the raw render; 4 episodes, 7 MB). The export matches
the robot PC's `multisensor_20hz` LeRobot v3 datasets feature for feature (names, dtypes, h264/yuv420p, 20 fps;
checked against `published/cup_random`), and `observation.state`'s flange pose equals calibrated FK of the joints, as on
the robot PC. The lerobot venv (`uv`, Python 3.12, lerobot 0.5.2 from git 5de7aa5a, CPU torch) is separate from Isaac Sim.

Wrist cameras and Fabric: Isaac Lab's camera reads its pose through a Fabric `XformPrimView`, which on the first read
(the camera's `reset`) copies the USD world pose into Fabric once. USD still has the spawn configuration (PhysX writes
link poses only to Fabric), so if the arm has moved by then, as after a task env's first reset, the camera ends up at a
fixed wrong offset from its link and stops following it. `rpm_lab/sensors.py` re-attaches mounted cameras (writes their
authored local transform back into Fabric); the tasks run it as a reset event and the tools call it after
`sim.reset()`. Preview of three generated demos: `/data/cui00191/rpm_lab_mimic/generated_100_preview.mp4`.

## World model (Interactive World Simulator)

`third_party/interactive_world_sim` (git submodule, MIT; Wang et al., RSS 2026: latent consistency-model world model,
trained from scratch, fast enough for closed-loop policy evaluation) is trained unchanged on RPM Lab data through two
tools in `world_model/`:

- `build_iws_dataset.py` mixes real and sim LeRobot datasets into IWS's training cache (`{train,val}/cache*.zarr.zip`):
  the camera images (`--cameras`, default `scene`: scene_1 only, for fast iteration; `wrist`; `both`), square
  center-cropped and resized (`--resolution`), the 7-dim joint-command action, a per-step real/sim `domain` array,
  and `manifest.json`. Validation holds out real episodes (`--val_real`). With the cache present, IWS's
  ALOHA-specific conversion never runs. IWS names its cache after the `camera_1_color` key, so a single camera is
  stored under that key (the manifest records which one).
- `train_iws.py` runs IWS's `main.py` for its three stages (1 autoencoder, 2 latent dynamics, 3 decoder fine-tuning,
  each from the previous checkpoint) on the dataset's camera(s) with Lightning's 7-dim action; runs, checkpoints and
  offline wandb logs go to `/data/cui00191/WM-cotrain/runs/<name>`. IWS's released models are one per camera; on a
  `both` dataset `--views both` stacks the views in one model, whose validation skips FVD (I3D takes 3 channels).

```bash
# environment (once): plain uv venv, no conda
export UV_CACHE_DIR=/data/cui00191/rpm_lab_envs/uv_cache UV_PYTHON_INSTALL_DIR=/data/cui00191/rpm_lab_envs/uv_python
D=scripts_v2/tools/rpm_lab/world_model
uv venv --python 3.11 /data/cui00191/rpm_lab_envs/iws
uv pip install --python /data/cui00191/rpm_lab_envs/iws/bin/python -r $D/requirements.txt \
    --build-constraint $D/build_constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match
uv pip install --python /data/cui00191/rpm_lab_envs/iws/bin/python --no-deps -e third_party/interactive_world_sim

P=/data/cui00191/rpm_lab_envs/iws/bin/python W=/data/cui00191/WM-cotrain
$P $D/build_iws_dataset.py --out $W/data/cook_fish_r40_s500 --real $REAL --sim $SIM --sim_episodes 500 --val_real 10
$P $D/train_iws.py --stage 1 --name cf_ae --data $W/data/cook_fish_r40_s500
$P $D/train_iws.py --stage 2 --name cf_dyn --data $W/data/cook_fish_r40_s500 --load_ae $W/runs/cf_ae/checkpoints/<ckpt>
$P $D/train_iws.py --stage 3 --name cf_dec --data $W/data/cook_fish_r40_s500 --load_ae $W/runs/cf_dyn/checkpoints/<ckpt>
```

Checked end to end on 4 sim cook_fish episodes (11/2026): all three stages train, checkpoint and validate (stages 2
and 3 roll out 200 frames and compute FVD), for one camera and for both stacked. Batch size vs memory and speed, one
camera at 128 px on an otherwise idle A6000 (48 GB), 60 steps each:

| stage | batch | peak memory | it/s | samples/s |
|---|---|---|---|---|
| 1 autoencoder | 1 (IWS README) | 5.2 GB | 3.26 | 3.3 |
| | 4 | 17.5 GB | 2.28 | 9.1 |
| | 8 | 34.0 GB | 1.28 | 10.2 |
| | 16 | out of memory | | |
| 2 dynamics (10-frame clips) | 4 (IWS README) | 7.2 GB | 2.78 | 11.1 |
| | 8 | 13.4 GB | 1.52 | 12.2 |
| | 16 | 26.0 GB | 0.82 | 13.1 |
| 3 decoder | 8 | 34.0 GB | 1.31 | 10.5 |
| | 16 (IWS README, on an H200) | out of memory | | |

The GPU saturates early: past batch 4 the samples per second barely grow. `train_iws.py` defaults to 1 / 4 / 8.

## Training on MSI (Agate)

`msi/` runs the world-model training on MSI's `msigpu` partition (A100 / H100, not preemptible, 24 h per job), after
the framework MSI users in the lab already use (thin Apptainer container, venv on project storage, one pinned commit
per job, one git worktree per run). Isaac Sim rendering stays on the workstation (MSI's A100/H100 have no RT cores).

| File | Where | What it does |
|---|---|---|
| `msi/msi.env` | both | Site settings: ssh alias, Slurm account (PI group), MSI project directory, local ledger. |
| `msi/rpm_lab_iws.def` | MSI | Apptainer definition: CUDA 12.6 devel, Python 3.11 (deadsnakes), uv. No Python packages. |
| `msi/setup_msi.sh` | MSI login node | One time: clone the repo + IWS submodule, build the sandbox, create `venv-iws` through it, import check. |
| `msi/msi_iws.slurm` | MSI job | One training stage: checks the commit pin, makes the run's worktree under `flock` (IWS submodule cloned from the main checkout, no network), checks the dataset, runs `train_iws.py --resume` in the sandbox. Stops before the time limit (Lightning counts training time across resumed sessions; `train_iws.py` adds the time recorded in the checkpoint) (10% early, 3 to 30 min) and keeps the newest two checkpoints (saved every 2000 steps); `--requeue` and chained jobs continue the run. |
| `msi/msi_submit.py` | workstation | Submits a stage at the pushed tip of the branch (launcher text read at that commit; msigpu only; sandbox, venv, dataset and queue checked over ssh); `--chain N` for stages longer than 24 h; appends to the job ledger. |
| `msi/msi_sync.sh` | workstation | `push-data <dataset dir>`, `pull-run <run>` (checkpoints, config, logs; sha256 checked), `status`. |

One-time setup:

1. `~/.ssh/config` on the workstation has `Host msi` with a ControlMaster (`ControlPath ~/.ssh/sockets/%r@%h:%p`,
   `ControlPersist 4h`). Open the connection by hand (password + Duo): `ssh -fN msi`; the tools only reuse it
   (`ssh -O check msi`).
2. Fill `MSI_ACCOUNT` (PI group) and `MSI_PROJECT_DIR` (`/projects/standard/<group>/shared/<user>/WM-cotrain`) in
   `msi/msi.env`.
3. `ssh msi "bash -s -- <MSI_PROJECT_DIR>" < scripts_v2/tools/rpm_lab/msi/setup_msi.sh`

Each experiment (push the code first: jobs train the branch tip on origin):

```bash
M=scripts_v2/tools/rpm_lab/msi
$M/msi_sync.sh push-data /data/cui00191/WM-cotrain/data/cook_fish_r40_s500
python $M/msi_submit.py --run cf_ae --stage 1 --dataset cook_fish_r40_s500
python $M/msi_submit.py --run cf_dyn --stage 2 --dataset cook_fish_r40_s500 --load_ae_run cf_ae \
    --dependency <stage-1 job> --chain 2
$M/msi_sync.sh status
$M/msi_sync.sh pull-run cf_dyn
```

Checked on the workstation (11/2026; no MSI access yet): the launcher end to end with `apptainer exec` replaced by
a plain shell (refuses a job without the submit tool's mark; worktrees and the submodule from the local copy; stage 1
trains; stage 2 starts from stage 1's newest checkpoint; a second job of a run resumes it), and `train_iws.py`
resume and `--max_time`. Resuming needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (set by `train_iws.py`): torch >= 2.6
loads weights only by default and IWS's Lightning checkpoints also hold OmegaConf objects.

## Checked against the real arms (11/2026)

- Mounts: recorded gripper paths sit on the tabletop only with both cable openings down (Lightning = mirror of Thunder).
- Kinematics: sim flange vs controller-reported flange, median 0.002-0.004 mm (still arm: < 0.05 mm; larger values
  only during fast motion, where the robot PC reads joints and pose in separate RTDE calls).
- Wrist D405: finger silhouettes in the real wrist images fix the camera at the left imager (`D405_IMAGER_OFFSET`).
- Motion: driving the sim with the teleop commands through the fitted servoJ model reproduces the measured joints.
- Scene camera `scene_1` (L515): pose from the lab's 2026-10-08 ChArUco calibration in Thunder's Base frame, saved
  verbatim in `rpm_lab/calibration/scene_1_l515_thunder_20261008.json` (override with `RPM_SCENE_1_CALIBRATION`). Checked on the 10/28 Lightning episode: Lightning's recorded flange/grasp point
  projects onto the real gripper in every frame, the sim render overlays the real arm (stills in
  `replays/ep1028_185259_scene_stills`), and the tabletop plane in the L515 depth puts the camera 0.738 m above the
  cloth at 22.6 deg (calibration: 0.746 m above the bare top, 23.6 deg). So the camera had not moved and Lightning's
  placement relative to Thunder holds to the pixel level. Real frames have lens distortion: `cv2.undistort` them with
  `L515_COLOR_DISTORTION` before comparing with or mixing into the pinhole sim renders.
- Lightning's base vs Thunder: fitted through the calibrated scene_1 camera (`fit_arm_base_from_scene_camera.py`, 40
  frames of five 10/28 cook_fish episodes): +2, -10, +6 mm from the mirror image (`LIGHTNING_POS_CORRECTION`).
- Table: the real tabletop carries a cloth; a 3 mm cloth collider (`CLOTH_CFG`, `WORK_SURFACE_Z`) puts the work
  surface where Lightning's table-level grasps touch it (-6..+7 mm across the five episodes, after the correction).

- Appearance (scene_1, 10/28 cook_fish frames): `DOME_LIGHT_INTENSITY` and `CLOTH_COLOR` render the cloth at the
  real median RGB (sim 108/112/109 vs real 107/113/108; before, the cloth and the UR's blue caps came out near white).
  The scene's dome is the only light: the robot builds' own dome lights are deactivated in their override layers
  (they added up per env); the wrist D405 gets `CAMERA_COLOR_GAIN` (1.17, 1.14, 1.12) for its own exposure
  (cloth 125/126/120 vs real 126/127/121). The cloth continues past the table's far edge (visual strip) and a loose
  black cable lies near the column (`CABLE_PATH`, median of five episodes). Image timing: scene_1 frames line up with
  the joint stream within 0.1 s, except each episode's first frame, which is stale.

## Known gaps

- Wrist depth: back-projected real D405 depth disagrees with the sim geometry in a range-dependent way (apparent 14 deg
  table tilt) while the color images agree; unresolved, so depth is not used for calibration yet.
- Omniverse cameras ignore the principal point (cx, cy forced to the image center; fx, fy averaged): raw sim frames are
  shifted against real ones by up to 7.6 px (L515) / 4.5 px (D405); `render_mimic_dataset.py` warps them.
- Not modeled: the dark band along the bottom of scene_1 (the real cloth seems to end near y = 0.17, the CAD tabletop
  runs to 0.29), the USB cable looping from Lightning's wrist camera, the lid's arched strap handle (a straight bar
  here), soft shadows from the ceiling lights, and steel reflections (RTX reflections are off).
- Thunder's D405 intrinsics were never recorded (Lightning's are used); joint friction/armature are reference values.

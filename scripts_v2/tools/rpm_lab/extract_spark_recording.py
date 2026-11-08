# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a real Thunder/Lightning recording from the lab robot PC into one ``.npz`` for the sim tools.

Sources (RPM-lab-UMN/spark-data-collection on the robot PC, Zeus):

* a raw episode directory (``raw_episodes/episode-*``, ROS 2 MCAP bag, robot state at ~128 Hz), or
* a published LeRobot v3 dataset (``published/*``, Lightning only, 20 Hz). Its episodes are concatenated;
  ``episode`` marks them.

Output arrays, per arm present (``thunder``, ``lightning``), times in seconds from the recording start:

* ``<arm>.t``, ``<arm>.q`` (N, 6) measured joints, ``<arm>.flange_pos`` (N, 3) and ``<arm>.flange_quat`` (N, 4, w x y z):
  the flange (``getActualTCPPose`` with the TCP at the flange) in the controller Base frame, ``<arm>.gripper`` (N,)
  measured opening, 0 open .. 1 closed;
* ``<arm>.cmd_t``, ``<arm>.cmd_q``, ``<arm>.cmd_gripper``: the teleop commands sent with servoJ (only while teleop
  is active);
* ``teleop_active.t``, ``teleop_active.value``; ``meta`` (JSON: source, arms, task, start time);
* with ``--images``: ``image.<camera>.t`` and ``image.<camera>.rgb`` (M, H, W, 3), e.g. ``lightning_wrist``; with
  ``--depth`` also ``depth.<camera>.t`` and ``depth.<camera>.raw`` (M, H, W, uint16; meters = raw * meta
  ``depth_scale_m``; the D405 depth stream has its own intrinsics, see ``scene_cfg.D405_DEPTH_INTRINSICS``).

Needs ``pip install mcap mcap-ros2-support`` (raw bags) or ``pandas pyarrow`` (LeRobot); no Isaac Sim.

Examples:
    python scripts_v2/tools/rpm_lab/extract_spark_recording.py \\
        --episode /data/cui00191/rpm_lab_recordings/episode-20261106-120011 --out ep1106.npz
    python scripts_v2/tools/rpm_lab/extract_spark_recording.py --images 2.0 \\
        --episode /data/cui00191/rpm_lab_recordings/episode-20261028-185259 --out ep1028.npz
"""

import argparse
import glob
import json
import numpy as np
import os
from collections import defaultdict
from scipy.spatial.transform import Rotation

ARMS = ("thunder", "lightning")
CAMERAS = {  # image topic prefix -> name in the output
    "/spark/cameras/lightning/wrist_1": "lightning_wrist",
    "/spark/cameras/thunder/wrist_1": "thunder_wrist",
    "/spark/cameras/world/scene_1": "scene_1",
    "/spark/cameras/world/scene_2": "scene_2",
    "/spark/cameras/world/scene_3": "scene_3",
}


def decode_image(msg) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        img = data.reshape(msg.height, msg.step)[:, : msg.width * 4].reshape(msg.height, msg.width, 4)[..., :3]
        return img[..., ::-1].copy() if msg.encoding == "bgra8" else img.copy()
    raise ValueError(f"Unsupported image encoding {msg.encoding}")


def decode_depth(msg) -> np.ndarray:
    if msg.encoding not in ("16UC1", "mono16"):
        raise ValueError(f"Unsupported depth encoding {msg.encoding}")
    data = np.frombuffer(msg.data, dtype=">u2" if msg.is_bigendian else "<u2")
    return data.reshape(msg.height, msg.step // 2)[:, : msg.width].astype(np.uint16)


def read_raw_episode(directory: str, image_period: float | None, depth: bool) -> dict:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bags = sorted(glob.glob(os.path.join(directory, "bag", "*.mcap")))
    if not bags:
        raise FileNotFoundError(f"No MCAP bag under {directory}/bag")
    rows = defaultdict(list)
    images = defaultdict(list)
    depths = defaultdict(list)
    last_image = defaultdict(lambda: -1e18)
    for bag in bags:
        with open(bag, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            wanted = None
            if image_period is None:
                summary = reader.get_summary()
                wanted = [c.topic for c in summary.channels.values() if "/cameras/" not in c.topic]
            for schema, channel, message, msg in reader.iter_decoded_messages(topics=wanted):
                topic, t = channel.topic, message.log_time
                if schema.name == "sensor_msgs/msg/Image":
                    prefix = topic.rsplit("/", 2)[0]
                    stream = "depth" if topic.endswith("/depth/image_rect_raw") else "color"
                    if prefix not in CAMERAS or (stream == "color") != topic.endswith("/color/image_raw"):
                        continue
                    if stream == "depth" and not depth:
                        continue
                    if (t - last_image[(prefix, stream)]) * 1e-9 < image_period:
                        continue
                    last_image[(prefix, stream)] = t
                    if stream == "color":
                        images[CAMERAS[prefix]].append((t, decode_image(msg)))
                    else:
                        depths[CAMERAS[prefix]].append((t, decode_depth(msg)))
                elif schema.name == "sensor_msgs/msg/JointState":
                    rows[topic].append([t, *msg.position])
                elif schema.name == "geometry_msgs/msg/PoseStamped":
                    p, q = msg.pose.position, msg.pose.orientation
                    rows[topic].append([t, p.x, p.y, p.z, q.w, q.x, q.y, q.z])
                elif schema.name == "std_msgs/msg/Bool":
                    rows[topic].append([t, float(msg.data)])
    rows = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    starts = [v[0, 0] for v in rows.values()] + [v[0][0] for v in images.values()]
    t0 = min(starts)
    out = {}
    arms = [a for a in ARMS if f"/spark/{a}/robot/joint_state" in rows]
    for arm in arms:
        js, ee, gr = (rows[f"/spark/{arm}/robot/{k}"] for k in ("joint_state", "eef_pose", "gripper_state"))
        n = min(len(js), len(ee), len(gr))  # published from the same tick
        out[f"{arm}.t"] = (js[:n, 0] - t0) * 1e-9
        out[f"{arm}.q"] = js[:n, 1:7]
        out[f"{arm}.flange_pos"] = ee[:n, 1:4]
        out[f"{arm}.flange_quat"] = ee[:n, 4:8]
        out[f"{arm}.gripper"] = gr[:n, 1]
        cmd = rows.get(f"/spark/{arm}/teleop/cmd_joint_state", np.zeros((0, 7)))
        cmd_g = rows.get(f"/spark/{arm}/teleop/cmd_gripper_state", np.zeros((0, 2)))
        out[f"{arm}.cmd_t"] = (cmd[:, 0] - t0) * 1e-9
        out[f"{arm}.cmd_q"] = cmd[:, 1:7]
        out[f"{arm}.cmd_gripper_t"] = (cmd_g[:, 0] - t0) * 1e-9
        out[f"{arm}.cmd_gripper"] = cmd_g[:, 1]
    active = rows.get("/spark/session/teleop_active", np.zeros((0, 2)))
    out["teleop_active.t"] = (active[:, 0] - t0) * 1e-9
    out["teleop_active.value"] = active[:, 1]
    for name, frames in images.items():
        out[f"image.{name}.t"] = (np.array([t for t, _ in frames], dtype=np.float64) - t0) * 1e-9
        out[f"image.{name}.rgb"] = np.stack([img for _, img in frames])
    for name, frames in depths.items():
        out[f"depth.{name}.t"] = (np.array([t for t, _ in frames], dtype=np.float64) - t0) * 1e-9
        out[f"depth.{name}.raw"] = np.stack([img for _, img in frames])
    meta = {"source": os.path.abspath(directory), "arms": arms, "start_time_ns": int(t0)}
    manifest = os.path.join(directory, "episode_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest) as f:
            m = json.load(f)
        episode = m.get("episode", {})
        meta.update(task=episode.get("task_name"), episode_id=episode.get("episode_id"))
        meta["depth_scale_m"] = {
            CAMERAS[d["sensor_key"]]: d["depth_scale_meters_per_unit"]
            for d in m.get("sensors", {}).get("devices", [])
            if d.get("sensor_key") in CAMERAS and "depth_scale_meters_per_unit" in d
        }
    return out, meta


def read_lerobot(directory: str, episodes: list[int] | None) -> dict:
    import pandas as pd

    with open(os.path.join(directory, "meta", "info.json")) as f:
        info = json.load(f)
    names = info["features"]["observation.state"]["names"]
    arms = [a for a in ARMS if f"{a}_joint_pos_1" in names]
    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(directory, "data", "*", "*.parquet")))]
    )
    if episodes is not None:
        frames = frames[frames["episode_index"].isin(episodes)]
    frames = frames.sort_values(["episode_index", "frame_index"])
    state = np.stack(frames["observation.state"].to_numpy())
    action = np.stack(frames["action"].to_numpy())
    action_names = info["features"]["action"]["names"]
    # Concatenate episodes on one clock, 1 s apart, so that time stays increasing.
    t = frames["timestamp"].to_numpy(dtype=np.float64)
    episode = frames["episode_index"].to_numpy()
    offset, clock = 0.0, np.empty_like(t)
    for e in np.unique(episode):
        m = episode == e
        clock[m] = t[m] - t[m].min() + offset
        offset = clock[m].max() + 1.0
    out = {"episode": episode}
    for arm in arms:
        col = {n: i for i, n in enumerate(names)}
        q = state[:, [col[f"{arm}_joint_pos_{i}"] for i in range(1, 7)]]
        pos = state[:, [col[f"{arm}_eef_{k}"] for k in ("x", "y", "z")]]
        rotvec = state[:, [col[f"{arm}_eef_{k}"] for k in ("rx", "ry", "rz")]]
        out[f"{arm}.t"] = clock
        out[f"{arm}.q"] = q
        out[f"{arm}.flange_pos"] = pos
        out[f"{arm}.flange_quat"] = Rotation.from_rotvec(rotvec).as_quat(scalar_first=True)
        out[f"{arm}.gripper"] = state[:, col[f"{arm}_gripper_position"]]
        acol = {n: i for i, n in enumerate(action_names)}
        out[f"{arm}.cmd_t"] = clock
        out[f"{arm}.cmd_q"] = action[:, [acol[f"{arm}_cmd_joint_{i}"] for i in range(1, 7)]]
        out[f"{arm}.cmd_gripper_t"] = clock
        out[f"{arm}.cmd_gripper"] = action[:, acol[f"{arm}_cmd_gripper"]]
    meta = {"source": os.path.abspath(directory), "arms": arms, "episodes": sorted(int(e) for e in np.unique(episode))}
    return out, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", help="Raw episode directory (contains bag/*.mcap).")
    source.add_argument("--lerobot", help="Published LeRobot dataset directory.")
    parser.add_argument("--episodes", type=str, default=None, help="LeRobot episode indices, e.g. 0,3,7.")
    parser.add_argument("--images", type=float, default=None, help="Keep one color frame per camera every N s.")
    parser.add_argument("--depth", action="store_true", help="With --images, also keep the depth frames.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.episode:
        out, meta = read_raw_episode(args.episode, args.images, args.depth)
    else:
        episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else None
        out, meta = read_lerobot(args.lerobot, episodes)
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out}: {json.dumps(meta)}")
    for key in sorted(out):
        if key != "meta":
            print(f"  {key:32s} {out[key].shape}")


if __name__ == "__main__":
    main()

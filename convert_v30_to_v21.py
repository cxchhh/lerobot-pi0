#!/usr/bin/env python
"""Convert a LeRobot v3.0 dataset (Humanoid-WAM nvenc export) to the v2.1
layout expected by this repo (CODEBASE_VERSION = "v2.1").

Differences handled:
  * v3.0 concatenated data/chunk-XXX/file-YYY.parquet  -> per-episode
    data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet
  * v3.0 concatenated videos/{key}/chunk-XXX/file-YYY.mp4 -> per-episode
    videos/chunk-XXX/{key}/episode_XXXXXX.mp4, resized (aspect kept)
  * meta/episodes/*.parquet + tasks.parquet -> episodes.jsonl, tasks.jsonl,
    episodes_stats.jsonl, v2.1 info.json
  * action padded 39 -> 42 dims with zeros (pelvis SE(2) cols 39:42 are
    filled at train time by chunk_transforms.reanchor_chunk_keep_state_local;
    their stats are copied from a reference v2.1 dataset so normalization
    matches previous runs)

Usage:
  python convert_v30_to_v21.py \
    --src  /path/to/v30_root   (dir containing data/ meta/ videos/) \
    --dst  /mnt/kpfs/chenxuchuan/sandbox/G1-VLA/bfm-rubbish-v0 \
    [--width 224] [--pad-action-to 42] \
    [--pelvis-stats-src /mnt/kpfs/chenxuchuan/sandbox/G1-VLA/bfm-v2.4/meta]
"""

import argparse
import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd

CHUNK_SIZE = 1000
PELVIS_NAMES = ["pelvis_dx_from_chunk0", "pelvis_dy_from_chunk0", "pelvis_dyaw_from_chunk0"]

# Aggregated action[39:42] stats from bfm-v2.4 (675 eps, weighted by count).
# Used when --pelvis-stats-src is unavailable (e.g. running off-cluster) so
# normalization of the padded pelvis cols matches previous training runs.
PELVIS_STATS_DEFAULT = {
    "min": np.array([-0.1281977891921997, -0.5192219018936157, -0.4456104338169098]),
    "max": np.array([1.1347023248672485, 0.7886660099029541, 1.9242887496948242]),
    "mean": np.array([0.10443641819335796, 0.012399410431731936, 0.20798273313928534]),
    "std": np.array([0.18953125590318454, 0.12088733504448364, 0.3373943924059068]),
}

# First available encoder is used; codec name goes into info.json.
ENCODER_PREFS = [("libsvtav1", "av1", {"g": "2", "crf": "30", "preset": "8"}),
                 ("libaom-av1", "av1", {"g": "2", "crf": "30", "cpu-used": "8", "row-mt": "1"}),
                 ("libx264", "h264", {"g": "2", "crf": "23"})]


def pick_encoder():
    for name, codec, opts in ENCODER_PREFS:
        try:
            av.CodecContext.create(name, "w")
            return name, codec, opts
        except Exception:
            continue
    raise RuntimeError("no usable video encoder in this pyav build")


def load_episodes_meta(src: Path) -> pd.DataFrame:
    files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def aggregate_pelvis_stats(meta_dir: Path, cols=slice(39, 42)) -> dict:
    """Aggregate action stats for the padded pelvis cols from a reference
    v2.1 dataset's episodes_stats.jsonl (weighted by count)."""
    mins, maxs, means, m2s, counts = [], [], [], [], []
    with open(meta_dir / "episodes_stats.jsonl") as f:
        for line in f:
            st = json.loads(line)["stats"]["action"]
            n = np.asarray(st["count"]).reshape(-1)[0]
            mean = np.asarray(st["mean"])[cols]
            std = np.asarray(st["std"])[cols]
            mins.append(np.asarray(st["min"])[cols])
            maxs.append(np.asarray(st["max"])[cols])
            means.append(mean * n)
            m2s.append((std**2 + mean**2) * n)
            counts.append(n)
    total = sum(counts)
    mean = np.sum(means, axis=0) / total
    var = np.sum(m2s, axis=0) / total - mean**2
    return {
        "min": np.min(mins, axis=0),
        "max": np.max(maxs, axis=0),
        "mean": mean,
        "std": np.sqrt(np.clip(var, 0, None)),
    }


def stats_from_row(row: pd.Series, feature_keys: list[str], length: int,
                   pad_action_to: int, pelvis_stats: dict | None,
                   image_keys: list[str]) -> dict:
    out = {}
    for key in feature_keys:
        st = {}
        for field in ["min", "max", "mean", "std"]:
            v = np.asarray(row[f"stats/{key}/{field}"], dtype=np.float64)
            if key in image_keys:
                v = v.reshape(3, 1, 1)
            else:
                v = v.reshape(-1)
            if key == "action" and pad_action_to and v.shape[0] < pad_action_to:
                pad = np.zeros(pad_action_to - v.shape[0])
                if pelvis_stats is not None:
                    pad = pelvis_stats[field][: pad_action_to - v.shape[0]]
                v = np.concatenate([v, pad])
            st[field] = v.tolist()
        st["count"] = [int(length)]
        out[key] = st
    return out


def split_video(src_file: Path, jobs: list[dict], width: int, fps: int,
                encoder: str, enc_opts: dict):
    """Decode src_file once; route frame ranges to per-episode outputs.

    jobs: [{start, end, out_path}] with start/end frame indices in src_file,
    sorted by start."""
    with av.open(str(src_file)) as inp:
        istream = inp.streams.video[0]
        istream.thread_type = "AUTO"
        h = int(round(istream.height * width / istream.width / 2) * 2)

        job_i, frame_i = 0, 0
        out, ostream = None, None

        def open_out(path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            o = av.open(str(path), "w")
            s = o.add_stream(encoder, rate=fps)
            s.width, s.height = width, h
            s.pix_fmt = "yuv420p"
            s.codec_context.time_base = Fraction(1, fps)
            s.options = dict(enc_opts)
            return o, s

        def close_out():
            nonlocal out, ostream
            if out is not None:
                for pkt in ostream.encode():
                    out.mux(pkt)
                out.close()
                out, ostream = None, None

        for frame in inp.decode(istream):
            while job_i < len(jobs) and frame_i >= jobs[job_i]["end"]:
                close_out()
                jobs[job_i]["written"] = jobs[job_i].get("written", 0)
                job_i += 1
            if job_i >= len(jobs):
                break
            job = jobs[job_i]
            if frame_i >= job["start"]:
                if out is None:
                    out, ostream = open_out(job["out_path"])
                small = frame.reformat(width=width, height=h, format="yuv420p")
                small.pts = job.get("written", 0)
                small.time_base = Fraction(1, fps)
                for pkt in ostream.encode(small):
                    out.mux(pkt)
                job["written"] = job.get("written", 0) + 1
            frame_i += 1
        close_out()
    return h


def count_frames(path: Path) -> int:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.frames:
            return s.frames
        return sum(1 for _ in c.decode(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--pad-action-to", type=int, default=42)
    ap.add_argument("--pelvis-stats-src", type=Path,
                    default=Path("/mnt/kpfs/chenxuchuan/sandbox/G1-VLA/bfm-v2.4/meta"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to replace")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    info30 = json.load(open(src / "meta" / "info.json"))
    fps = info30["fps"]
    eps = load_episodes_meta(src)
    n_eps = len(eps)
    image_keys = [k for k, v in info30["features"].items() if v["dtype"] == "video"]
    feature_keys = list(info30["features"].keys())

    # ---- tasks ----
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_names = list(tasks_df.index)  # index = task string, col = task_index
    task_to_idx = {t: int(tasks_df.loc[t, "task_index"]) for t in task_names}
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for t, i in sorted(task_to_idx.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    pelvis_stats = None
    if args.pad_action_to:
        if args.pelvis_stats_src and args.pelvis_stats_src.exists():
            pelvis_stats = aggregate_pelvis_stats(args.pelvis_stats_src)
            print(f"pelvis col stats from {args.pelvis_stats_src}: "
                  f"mean={pelvis_stats['mean'].round(4)} std={pelvis_stats['std'].round(4)}")
        else:
            pelvis_stats = PELVIS_STATS_DEFAULT
            print("pelvis col stats: built-in defaults (bfm-v2.4 aggregate)")

    # ---- data parquets ----
    action_dim = None
    global_index = 0
    ep_lengths = {}
    for (ck, fk), grp in eps.groupby([eps["data/chunk_index"], eps["data/file_index"]]):
        df = pd.read_parquet(src / "data" / f"chunk-{ck:03d}" / f"file-{fk:03d}.parquet")
        for _, ep_row in grp.sort_values("episode_index").iterrows():
            ei = int(ep_row["episode_index"])
            sub = df[df["episode_index"] == ei].copy().reset_index(drop=True)
            n = len(sub)
            assert n == int(ep_row["length"]), f"ep {ei}: rows {n} != length {ep_row['length']}"
            ep_lengths[ei] = n

            action = np.stack(sub["action"].to_numpy()).astype(np.float32)
            action_dim = action.shape[1]
            if args.pad_action_to and action.shape[1] < args.pad_action_to:
                action = np.pad(action, ((0, 0), (0, args.pad_action_to - action.shape[1])))
            sub["action"] = list(action)
            sub["frame_index"] = np.arange(n, dtype=np.int64)
            sub["timestamp"] = (np.arange(n) / fps).astype(np.float32)
            sub["episode_index"] = np.int64(ei)
            sub["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
            global_index += n

            out = dst / "data" / f"chunk-{ei // CHUNK_SIZE:03d}" / f"episode_{ei:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(out, index=False)
    total_frames = global_index
    print(f"data: {n_eps} episodes, {total_frames} frames, action {action_dim}->{args.pad_action_to}")

    # ---- videos ----
    encoder, codec_name, enc_opts = pick_encoder()
    print(f"video encoder: {encoder}")
    out_h = None
    for key in image_keys:
        gcols = [eps[f"videos/{key}/chunk_index"], eps[f"videos/{key}/file_index"]]
        for (ck, fk), grp in eps.groupby(gcols):
            src_file = src / "videos" / key / f"chunk-{ck:03d}" / f"file-{fk:03d}.mp4"
            jobs = []
            for _, ep_row in grp.sort_values(f"videos/{key}/from_timestamp").iterrows():
                ei = int(ep_row["episode_index"])
                start = int(round(ep_row[f"videos/{key}/from_timestamp"] * fps))
                end = int(round(ep_row[f"videos/{key}/to_timestamp"] * fps))
                jobs.append({
                    "start": start, "end": end, "episode_index": ei,
                    "out_path": dst / "videos" / f"chunk-{ei // CHUNK_SIZE:03d}" / key
                                / f"episode_{ei:06d}.mp4",
                })
            out_h = split_video(src_file, jobs, args.width, fps, encoder, enc_opts)
            for j in jobs:
                exp = ep_lengths[j["episode_index"]]
                got = count_frames(j["out_path"])
                status = "OK" if got == exp else f"MISMATCH (expected {exp})"
                print(f"  {key} ep{j['episode_index']:06d}: {got} frames {status}")
                assert got == exp

    # ---- episodes.jsonl + episodes_stats.jsonl ----
    with open(dst / "meta" / "episodes.jsonl", "w") as fe, \
         open(dst / "meta" / "episodes_stats.jsonl", "w") as fs:
        for _, ep_row in eps.iterrows():
            ei = int(ep_row["episode_index"])
            tasks = list(ep_row["tasks"])
            fe.write(json.dumps({"episode_index": ei, "tasks": tasks,
                                 "length": int(ep_row["length"])}) + "\n")
            stats = stats_from_row(ep_row, feature_keys, ep_lengths[ei],
                                   args.pad_action_to, pelvis_stats, image_keys)
            fs.write(json.dumps({"episode_index": ei, "stats": stats}) + "\n")

    # ---- info.json ----
    features = {}
    for key, feat in info30["features"].items():
        feat = json.loads(json.dumps(feat))  # deep copy
        if key in image_keys:
            feat["shape"] = [3, out_h, args.width]
            feat["info"] = {
                "video.height": out_h, "video.width": args.width,
                "video.codec": codec_name, "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": fps,
                "video.channels": 3, "has_audio": False,
            }
        if key == "action" and args.pad_action_to and feat["shape"][0] < args.pad_action_to:
            feat["shape"] = [args.pad_action_to]
            if feat.get("names"):
                feat["names"] = feat["names"] + PELVIS_NAMES[: args.pad_action_to - action_dim]
        features[key] = feat

    info21 = {
        "codebase_version": "v2.1",
        "robot_type": info30.get("robot_type"),
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(task_to_idx),
        "total_videos": n_eps * len(image_keys),
        "total_chunks": (n_eps + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info21, f, indent=4)

    print(f"done -> {dst}")


if __name__ == "__main__":
    main()

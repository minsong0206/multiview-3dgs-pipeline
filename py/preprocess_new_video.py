#!/usr/bin/env python3
"""
New video preprocessing pipeline for KIST HUGSIM.

Steps:
  1. Parse GPS from all 6 DJI MP4 files → find GPS-lock stable start frame
  2. Estimate camera frame offsets → trim to common aligned window
  3. Extract frames at 12.5 fps (every 4th frame of 50fps) with 800x450 resize
  4. Generate gps.txt for COLMAP model_aligner

Usage:
  python preprocess_new_video.py \
      --video_dir /media/ms/WD_BLACK_4TB/KIST/raw_video \
      --out_dir   /home/ms/260308-KIST-Videos/KIST_NEW/recon \
      --chunk_start 0.0 \
      --chunk_end   30.0

  # chunk_start / chunk_end are in seconds relative to the aligned start.
  # Omit both to process the full aligned window.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Camera mapping ─────────────────────────────────────────────────────────────
# filename → HUGSIM camera name
CAM_MAP = {
    "0_front.MP4":       "CAM_FRONT",
    "1_right_front.MP4": "CAM_FRONT_RIGHT",
    "2_right_back.MP4":  "CAM_BACK_RIGHT",
    "3_back.MP4":        "CAM_BACK",
    "4_left_back.MP4":   "CAM_BACK_LEFT",
    "5_left_front.MP4":  "CAM_FRONT_LEFT",
}
CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
]

# ── GPS parsing ────────────────────────────────────────────────────────────────

def _extract_dji_meta_bin(mp4_path: str) -> tuple[bytes, list[int], list[float]]:
    """Extract DJI meta stream bytes + per-packet sizes + pts_times."""
    tmp = tempfile.mktemp(suffix=".bin")
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", mp4_path, "-map", "0:2", "-f", "rawvideo", tmp],
        check=True,
    )
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_packets", "-select_streams", "2", mp4_path],
        capture_output=True, text=True, check=True,
    )
    pkts = json.loads(r.stdout)["packets"]
    sizes = [int(p["size"]) for p in pkts]
    pts_times = [float(p["pts_time"]) for p in pkts]
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data, sizes, pts_times


def parse_gps(mp4_path: str) -> list[tuple[float, float, float]]:
    """
    Return list of (pts_time, lat, lon) for every frame in the DJI meta stream.
    Frames where GPS is invalid get (pts_time, None, None).
    """
    data, sizes, pts_times = _extract_dji_meta_bin(mp4_path)
    result = []
    offset = 0
    for sz, pts in zip(sizes, pts_times):
        pkt = data[offset: offset + sz]
        found = False
        for i in range(len(pkt) - 16):
            lat = struct.unpack_from("<d", pkt, i)[0]
            if 36.0 < lat < 39.0:
                for j in range(i + 8, min(i + 20, len(pkt) - 8)):
                    lon = struct.unpack_from("<d", pkt, j)[0]
                    if 126.0 < lon < 129.0:
                        result.append((pts, lat, lon))
                        found = True
                        break
                if found:
                    break
        if not found:
            result.append((pts, None, None))
        offset += sz
    return result


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in meters between two WGS84 coordinates."""
    import math
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _gps_valid(lat, lon) -> bool:
    return lat is not None and 36.0 < lat < 39.0 and 126.0 < lon < 129.0


def find_gps_lock_frame(
    gps: list,
    min_stable: int = 25,
    move_threshold_m: float = 10.0,
    jump_threshold_m: float = 50.0,
) -> int:
    """
    Return the first frame index where GPS is considered locked and reliable.

    Conditions:
      1. Valid range (36–39°N, 126–129°E).
      2. Has moved at least `move_threshold_m` from the initial frozen fix,
         indicating the GPS is actively updating rather than replaying a cached position.
      3. No abnormal jump > `jump_threshold_m` between consecutive frames.
      4. Followed by `min_stable` consecutive valid, non-jumping frames.
         If stability check fails, count resets and search continues.
    """
    # Find initial frozen GPS value (first valid reading, typically cached from last session)
    init_lat, init_lon = None, None
    for _, lat, lon in gps:
        if _gps_valid(lat, lon):
            init_lat, init_lon = lat, lon
            break
    if init_lat is None:
        raise RuntimeError("No valid GPS found in the entire stream.")

    # Find first frame where GPS has moved >= move_threshold_m from init
    first_moved = None
    for i, (_, lat, lon) in enumerate(gps):
        if not _gps_valid(lat, lon):
            continue
        if _haversine_m(init_lat, init_lon, lat, lon) >= move_threshold_m:
            first_moved = i
            break

    if first_moved is None:
        # Vehicle was stationary — no movement detected; return first valid frame
        for i, (_, lat, lon) in enumerate(gps):
            if _gps_valid(lat, lon):
                return i
        raise RuntimeError("GPS lock never found: no valid GPS frames.")

    # From first_moved, require min_stable consecutive frames that are:
    #   - valid range
    #   - no jump > jump_threshold_m from previous valid frame
    count = 0
    candidate_start = first_moved
    prev_lat, prev_lon = None, None

    for i in range(first_moved, len(gps)):
        _, lat, lon = gps[i]

        if not _gps_valid(lat, lon):
            # invalid frame → reset stability count
            count = 0
            candidate_start = i + 1
            prev_lat, prev_lon = None, None
            continue

        if prev_lat is not None:
            dist = _haversine_m(prev_lat, prev_lon, lat, lon)
            if dist > jump_threshold_m:
                # abnormal jump → reset
                count = 0
                candidate_start = i + 1
                prev_lat, prev_lon = lat, lon
                continue

        prev_lat, prev_lon = lat, lon
        count += 1
        if count >= min_stable:
            return candidate_start

    raise RuntimeError(
        f"GPS lock never stabilized: best stable run was {count} frames "
        f"(need {min_stable})."
    )


# ── Camera synchronization ─────────────────────────────────────────────────────

def get_nb_frames(mp4_path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", mp4_path],
        capture_output=True, text=True, check=True,
    )
    return int(json.loads(r.stdout)["streams"][0]["nb_frames"])


def estimate_sync_offsets(
    gps_all: dict[str, list],
    ref_cam: str = "CAM_FRONT",
) -> dict[str, int]:
    """
    All DJI cameras record GPS independently at 50 Hz.
    GPS cross-correlation gives lag=0 because the trajectories are identical.

    Strategy:
      - Find GPS-lock frame for each camera independently.
      - Compute the difference vs the reference camera's GPS-lock frame.
      - This gives the frame offset to trim at the start.

    Returns dict cam_name → frames_to_skip_at_start (non-negative).
    """
    lock_frames = {}
    for cam, gps in gps_all.items():
        lock_frames[cam] = find_gps_lock_frame(gps)
        print(f"  {cam}: GPS lock @ frame {lock_frames[cam]} "
              f"(t={lock_frames[cam]/50:.3f}s)")

    ref_lock = lock_frames[ref_cam]
    offsets = {}
    for cam in CAM_ORDER:
        # positive offset → this camera's GPS locks later than ref → trim more from front
        offsets[cam] = lock_frames[cam] - ref_lock
    return offsets, lock_frames


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(
    mp4_path: str,
    cam_name: str,
    out_dir: Path,
    skip_start: int,      # frames to skip at start of this camera
    n_frames: int,        # total aligned frames to extract
    sample_every: int,    # subsample (4 → 12.5 fps from 50 fps)
    width: int,
    height: int,
    fps: int = 50,
) -> int:
    """Extract frames using ffmpeg -ss seek + fps filter. Returns number of frames written."""
    cam_dir = out_dir / "images" / cam_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    # Convert frame indices to timestamps for accurate seeking.
    # -ss before -i triggers fast keyframe seek to the start of the window,
    # then -frames:v limits output to exactly the frames we need.
    start_sec = skip_start / fps
    end_sec   = (skip_start + n_frames) / fps
    duration  = end_sec - start_sec
    out_fps   = fps / sample_every   # 50/4 = 12.5

    out_pattern = str(cam_dir / "%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",   # seek before -i → fast
        "-i", mp4_path,
        "-t", f"{duration:.6f}",     # duration of window
        "-vf", f"fps={out_fps},scale={width}:{height}",
        "-q:v", "2",
        "-start_number", "0",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)

    written = len(list(cam_dir.glob("*.jpg")))
    return written


# ── GPS txt generation ─────────────────────────────────────────────────────────

def generate_gps_txt(
    gps_all: dict[str, list],
    skip_starts: dict[str, int],   # frame index in original video where aligned window begins
    n_aligned: int,                # total frames in aligned window (50 fps)
    sample_every: int,             # 4 for 12.5 fps
    out_path: Path,
) -> None:
    """
    Write gps.txt for COLMAP model_aligner.

    Uses only 1Hz GPS anchors (one per unique GPS coordinate change).
    For each unique GPS coordinate, picks the first extracted CAM_FRONT
    frame that falls within that GPS epoch. This avoids interpolation noise.
    """
    ref_cam = "CAM_FRONT"
    gps_ref = gps_all[ref_cam]
    ref_skip = skip_starts[ref_cam]
    n_sampled = n_aligned // sample_every

    lines = []
    prev_lat, prev_lon = None, None

    for s in range(n_sampled):
        orig_frame = ref_skip + s * sample_every
        if orig_frame >= len(gps_ref):
            break
        _, lat, lon = gps_ref[orig_frame]
        if lat is None:
            continue

        # Only emit when GPS coordinate actually changes (1Hz anchor)
        if lat == prev_lat and lon == prev_lon:
            continue

        prev_lat, prev_lon = lat, lon
        img_name = f"CAM_FRONT/{s:06d}.jpg"
        lines.append(f"{img_name} {lat:.8f} {lon:.8f} 0.000")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  gps.txt: {len(lines)} lines (1Hz anchors) → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KIST new video preprocessing")
    parser.add_argument("--video_dir", required=True, help="Directory with 0_front.MP4 ... 5_left_front.MP4")
    parser.add_argument("--out_dir",   required=True, help="Output directory (recon root)")
    parser.add_argument("--chunk_start", type=float, default=None,
                        help="Chunk start in seconds (relative to aligned window start)")
    parser.add_argument("--chunk_end",   type=float, default=None,
                        help="Chunk end in seconds (relative to aligned window start)")
    parser.add_argument("--sample_every", type=int, default=4,
                        help="Subsample factor: 4 → 12.5 fps from 50 fps")
    parser.add_argument("--width",  type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--min_stable_gps", type=int, default=25,
                        help="Minimum consecutive valid GPS frames to declare lock")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir   = Path(args.out_dir)
    FPS = 50

    # ── resolve MP4 paths ──────────────────────────────────────────────────────
    mp4s: dict[str, str] = {}
    for filename, cam_name in CAM_MAP.items():
        p = video_dir / filename
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        mp4s[cam_name] = str(p)

    # ── Step 1: parse GPS from all cameras ────────────────────────────────────
    print("\n[Step 1] Parsing GPS from all cameras...")
    gps_all: dict[str, list] = {}
    nb_frames: dict[str, int] = {}
    for cam_name in CAM_ORDER:
        print(f"  {cam_name} ...", end=" ", flush=True)
        gps_all[cam_name] = parse_gps(mp4s[cam_name])
        nb_frames[cam_name] = len(gps_all[cam_name])
        print(f"{nb_frames[cam_name]} frames")

    # ── Step 2: find GPS lock + synchronization offsets ───────────────────────
    print("\n[Step 2] Estimating GPS-lock and sync offsets...")
    offsets, lock_frames = estimate_sync_offsets(gps_all, ref_cam="CAM_FRONT")

    # aligned start: latest GPS-lock among all cameras
    max_lock = max(lock_frames.values())
    print(f"\n  Latest GPS lock: frame {max_lock} (t={max_lock/FPS:.3f}s)")
    print(f"  Using frame {max_lock} as common aligned start for all cameras.")

    # All cameras started recording at the same UTC second (verified via GPS metadata).
    # GPS-lock frame differences between cameras reflect GPS processing delay, not
    # actual recording time offsets. Use the same raw frame index for all cameras.
    skip_starts: dict[str, int] = {}
    for cam in CAM_ORDER:
        skip_starts[cam] = max_lock

    # aligned window length: min available frames after skip
    avail = {cam: nb_frames[cam] - skip_starts[cam] for cam in CAM_ORDER}
    n_aligned_full = min(avail.values())
    print(f"  Aligned window: {n_aligned_full} frames = {n_aligned_full/FPS:.2f}s")
    for cam in CAM_ORDER:
        print(f"    {cam}: skip {skip_starts[cam]} frames, avail {avail[cam]} frames")

    # ── Apply chunk_start / chunk_end ─────────────────────────────────────────
    chunk_start_frame = 0
    chunk_end_frame   = n_aligned_full

    if args.chunk_start is not None:
        chunk_start_frame = int(args.chunk_start * FPS)
    if args.chunk_end is not None:
        chunk_end_frame = int(args.chunk_end * FPS)

    chunk_end_frame = min(chunk_end_frame, n_aligned_full)
    n_chunk = chunk_end_frame - chunk_start_frame

    if n_chunk <= 0:
        sys.exit(f"ERROR: chunk window is empty ({chunk_start_frame} .. {chunk_end_frame})")

    print(f"\n  Chunk: frames {chunk_start_frame}..{chunk_end_frame-1} "
          f"({n_chunk} frames = {n_chunk/FPS:.2f}s)")

    # adjust skip_starts for chunk offset
    final_skips = {cam: skip_starts[cam] + chunk_start_frame for cam in CAM_ORDER}

    # ── Step 3: extract frames ────────────────────────────────────────────────
    print(f"\n[Step 3] Extracting frames (every {args.sample_every} → {FPS/args.sample_every:.1f} fps)...")
    n_sampled_expected = n_chunk // args.sample_every

    for cam_name in CAM_ORDER:
        print(f"  {cam_name} ...", end=" ", flush=True)
        written = extract_frames(
            mp4_path    = mp4s[cam_name],
            cam_name    = cam_name,
            out_dir     = out_dir,
            skip_start  = final_skips[cam_name],
            n_frames    = n_chunk,
            sample_every= args.sample_every,
            width       = args.width,
            height      = args.height,
        )
        status = "✓" if written == n_sampled_expected else f"WARNING: expected {n_sampled_expected}"
        print(f"{written} frames  {status}")

    # ── Step 4: generate gps.txt ──────────────────────────────────────────────
    print("\n[Step 4] Generating gps.txt...")
    gps_txt_path = out_dir / "gps.txt"

    # rebuild gps_all with chunk-adjusted skip
    generate_gps_txt(
        gps_all     = gps_all,
        skip_starts = final_skips,
        n_aligned   = n_chunk,
        sample_every= args.sample_every,
        out_path    = gps_txt_path,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DONE")
    print(f"  Output dir   : {out_dir}")
    print(f"  Images       : {out_dir}/images/<CAM>/000000.jpg ...")
    print(f"  GPS txt      : {gps_txt_path}")
    print(f"  Frames/cam   : {n_sampled_expected}")
    print(f"  FPS          : {FPS/args.sample_every:.1f}")
    print(f"  Resolution   : {args.width}x{args.height}")
    print("="*60)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    errors = []
    for cam in CAM_ORDER:
        cam_dir = out_dir / "images" / cam
        n = len(list(cam_dir.glob("*.jpg"))) if cam_dir.exists() else 0
        if n != n_sampled_expected:
            errors.append(f"  {cam}: {n} frames (expected {n_sampled_expected})")
    if not gps_txt_path.exists():
        errors.append("  gps.txt not created")

    if errors:
        print("\nWARNINGS:")
        for e in errors:
            print(e)
    else:
        print("\nAll outputs verified ✓")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Make a NuScenes-style 6-camera video from HUGSIM/StreetGaussian save_test PNGs.

Default layout:
  CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
  CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT
"""

import argparse
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

IMAGE_RE = re.compile(r"^(CAM_[A-Z_]+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a NuScenes-style 3x2 video from a save_test directory."
    )
    parser.add_argument(
        "input",
        help="Path to recon directory or directly to save_test directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path. Default: <recon>/save_test_nuscenes_6cam.webm",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Output FPS.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=DEFAULT_CAMERAS,
        help="Camera order. Six cameras are arranged row-major in a 3x2 grid.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize each camera tile to this width. Default: source width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize each camera tile to this height. Default: source height.",
    )
    parser.add_argument(
        "--sync-by",
        choices=["ordinal", "frame-id"],
        default="ordinal",
        help=(
            "How to align cameras. 'ordinal' pairs each camera's nth sorted image "
            "(default, works with interleaved save_test names). 'frame-id' requires "
            "matching numeric frame IDs across cameras."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Write only the first N synchronized frames.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use black tiles when a frame is missing instead of using only common frames.",
    )
    return parser.parse_args()


def resolve_paths(input_path, output_path):
    path = Path(input_path).expanduser().resolve()
    if path.name == "save_test":
        save_test_dir = path
        recon_dir = path.parent
    else:
        recon_dir = path
        save_test_dir = path / "save_test"

    if not save_test_dir.is_dir():
        raise FileNotFoundError(f"save_test directory not found: {save_test_dir}")

    if output_path is None:
        output = recon_dir / "save_test_nuscenes_6cam.webm"
    else:
        output = Path(output_path).expanduser().resolve()

    output.parent.mkdir(parents=True, exist_ok=True)
    return save_test_dir, output


def fourcc_for_output(output):
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if suffix == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def open_ffmpeg_webm_writer(output, fps, video_size):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for .webm output but was not found")

    width, height = video_size
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libvpx",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def write_frame(writer, frame, output):
    if output.suffix.lower() == ".webm":
        writer.stdin.write(frame.tobytes())
    else:
        writer.write(frame)


def close_writer(writer, output):
    if output.suffix.lower() == ".webm":
        writer.stdin.close()
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {output}")
    else:
        writer.release()


def index_images(save_test_dir):
    images = defaultdict(dict)
    for image_path in save_test_dir.iterdir():
        if not image_path.is_file():
            continue
        match = IMAGE_RE.match(image_path.name)
        if not match:
            continue
        cam, frame_text, _ = match.groups()
        images[cam.upper()][int(frame_text)] = image_path
    return images


def collect_frame_ids(images, cameras, allow_missing):
    frame_sets = [set(images.get(cam, {})) for cam in cameras]
    if not frame_sets:
        return []
    if allow_missing:
        frame_ids = sorted(set().union(*frame_sets))
    else:
        frame_ids = sorted(set.intersection(*frame_sets))
    return frame_ids


def collect_sequences(images, cameras, allow_missing):
    sequences = {}
    lengths = []
    for cam in cameras:
        paths = [path for _, path in sorted(images.get(cam, {}).items())]
        sequences[cam] = paths
        lengths.append(len(paths))

    if not lengths or (not allow_missing and any(length == 0 for length in lengths)):
        return sequences, 0

    if allow_missing:
        return sequences, max(lengths)
    return sequences, min(lengths)


def read_first_image(images, cameras):
    for cam in cameras:
        paths = images.get(cam, {})
        if paths:
            return cv2.imread(str(next(iter(paths.values()))), cv2.IMREAD_COLOR)
    return None


def make_tile(image_path, tile_size):
    if image_path is None:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    if (image.shape[1], image.shape[0]) != tile_size:
        image = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
    return image


def make_mosaic(images, cameras, frame_id, tile_size):
    tiles = [
        make_tile(images.get(cam, {}).get(frame_id), tile_size)
        for cam in cameras
    ]
    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:6])
    return np.vstack([top, bottom])


def make_mosaic_from_sequences(sequences, cameras, index, tile_size):
    tiles = []
    for cam in cameras:
        paths = sequences.get(cam, [])
        image_path = paths[index] if index < len(paths) else None
        tiles.append(make_tile(image_path, tile_size))
    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:6])
    return np.vstack([top, bottom])


def main():
    args = parse_args()
    cameras = [cam.upper() for cam in args.cameras]
    if len(cameras) != 6:
        raise ValueError("--cameras must contain exactly 6 camera names")

    save_test_dir, output = resolve_paths(args.input, args.output)
    images = index_images(save_test_dir)
    if args.sync_by == "frame-id":
        frame_ids = collect_frame_ids(images, cameras, args.allow_missing)
        if args.max_frames is not None:
            frame_ids = frame_ids[: args.max_frames]

        if not frame_ids:
            available = ", ".join(sorted(images)) or "(none)"
            raise RuntimeError(
                "No synchronized frames found. "
                f"Requested cameras: {cameras}. Available cameras: {available}"
            )
        frame_count = len(frame_ids)
        frame_range = f"{frame_ids[0]}..{frame_ids[-1]}"
        sequences = None
    else:
        sequences, frame_count = collect_sequences(images, cameras, args.allow_missing)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)

        if frame_count == 0:
            available = ", ".join(sorted(images)) or "(none)"
            raise RuntimeError(
                "No camera sequences found. "
                f"Requested cameras: {cameras}. Available cameras: {available}"
            )
        frame_ids = None
        frame_range = f"0..{frame_count - 1}"

    first = read_first_image(images, cameras)
    if first is None:
        raise RuntimeError(f"No readable images found in {save_test_dir}")

    tile_width = args.width or first.shape[1]
    tile_height = args.height or first.shape[0]
    tile_size = (tile_width, tile_height)
    video_size = (tile_width * 3, tile_height * 2)

    if output.suffix.lower() == ".webm":
        writer = open_ffmpeg_webm_writer(output, args.fps, video_size)
    else:
        writer = cv2.VideoWriter(
            str(output),
            fourcc_for_output(output),
            args.fps,
            video_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output}")

    try:
        if args.sync_by == "frame-id":
            for frame_id in frame_ids:
                frame = make_mosaic(images, cameras, frame_id, tile_size)
                write_frame(writer, frame, output)
        else:
            for index in range(frame_count):
                frame = make_mosaic_from_sequences(sequences, cameras, index, tile_size)
                write_frame(writer, frame, output)
    finally:
        close_writer(writer, output)

    print(f"save_test : {save_test_dir}")
    print(f"output    : {output}")
    print(f"frames    : {frame_count} ({frame_range})")
    print(f"sync_by   : {args.sync_by}")
    print(f"fps       : {args.fps}")
    print(f"size      : {video_size[0]}x{video_size[1]}")
    print(f"layout    : {cameras[:3]} / {cameras[3:]}")


if __name__ == "__main__":
    main()

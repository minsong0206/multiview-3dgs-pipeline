#!/usr/bin/env python3
"""
COLMAP automation script for KIST new video pipeline.

Runs inside Docker container (colmap_cudss), where:
  - /data → /home/ms/260308-KIST-Videos (host mount)
  - colmap is available in PATH

Steps:
  1. feature_extractor
  2. exhaustive_matcher
  3. mapper
  4. model_aligner  (GPS-based, using gps.txt)
  5. model_converter (→ sparse_ba.ply)
  6. Validation      (registered images, reprojection error)

Usage (inside Docker):
  python3 /data/py/run_colmap.py --recon_dir /data/KIST_NEW/test_chunk/recon

  # Resume from a specific step:
  python3 /data/py/run_colmap.py --recon_dir /data/KIST_NEW/test_chunk/recon \\
      --skip_feature_extraction --skip_matching --skip_mapper

Host equivalent path:
  /home/ms/260308-KIST-Videos/KIST_NEW/test_chunk/recon
"""

import argparse
import os
import subprocess
import sys
import struct
from pathlib import Path


CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# ── Unified log file ───────────────────────────────────────────────────────────

_log_fh = None

def _open_log(log_path: str):
    global _log_fh
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(log_path, 'a', buffering=1)  # line-buffered


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + '\n')
        _log_fh.flush()


def banner(title: str):
    log("")
    log("═" * 60)
    log(f"  {title}")
    log("═" * 60)


DRY_RUN = False

def run(cmd: list[str], desc: str):
    log(f"▶ {desc}")
    log("  " + " ".join(cmd))
    if DRY_RUN:
        log(f"  [DRY RUN] skipped")
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"✖ FAILED: {desc}  (exit {result.returncode})")
    log(f"✔ OK: {desc}")


def check_file(path: Path, desc: str):
    if DRY_RUN:
        log(f"  [DRY RUN] check_file: {path}")
        return
    if not path.exists() or path.stat().st_size == 0:
        sys.exit(f"✖ MISSING or EMPTY: {path}  [{desc}]")
    log(f"  ✔ {desc}: {path}")


def check_dir(path: Path, desc: str, min_files: int = 1):
    if DRY_RUN:
        log(f"  [DRY RUN] check_dir: {path}")
        return
    if not path.is_dir():
        sys.exit(f"✖ MISSING directory: {path}  [{desc}]")
    n = len(list(path.iterdir()))
    if n < min_files:
        sys.exit(f"✖ EMPTY directory ({n} files): {path}  [{desc}]")
    log(f"  ✔ {desc}: {path}  ({n} entries)")


# ── COLMAP binary reader (minimal) ─────────────────────────────────────────────

def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)


def read_colmap_stats(sparse_dir: Path) -> dict:
    """Convert sparse model to TXT and parse for stats."""
    import subprocess as _sp
    stats = {}

    # Convert to TXT for reliable parsing
    txt_dir = sparse_dir.parent / (sparse_dir.name + "_txt")
    txt_dir.mkdir(exist_ok=True)
    _sp.run([
        "colmap", "model_converter",
        "--input_path",  str(sparse_dir),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ], capture_output=True)

    # images.txt → registered image count and camera breakdown
    images_txt = txt_dir / "images.txt"
    if images_txt.exists():
        cam_counts = {}
        num_images = 0
        for line in images_txt.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10 and not parts[0].isdigit() is False:
                try:
                    int(parts[0])  # image_id
                except ValueError:
                    continue
                name = parts[9]
                cam_name = name.split("/")[0] if "/" in name else name
                cam_counts[cam_name] = cam_counts.get(cam_name, 0) + 1
                num_images += 1
        stats["registered_images"] = num_images
        stats["cam_counts"] = cam_counts
    else:
        stats["registered_images"] = 0
        stats["cam_counts"] = {}

    # points3D.bin → point count and mean reprojection error
    pts_bin = sparse_dir / "points3D.bin"
    if pts_bin.exists():
        with open(pts_bin, "rb") as f:
            num_pts = _read_next_bytes(f, 8, "<Q")[0]
            stats["num_points3d"] = num_pts
            errors = []
            for _ in range(num_pts):
                _read_next_bytes(f, 8, "<Q")    # point3d_id
                _read_next_bytes(f, 24, "<3d")  # xyz
                _read_next_bytes(f, 3, "3B")    # rgb
                err = _read_next_bytes(f, 8, "<d")[0]  # reprojection error
                errors.append(err)
                track_len = _read_next_bytes(f, 8, "<Q")[0]
                f.read(track_len * 8)  # track elements (image_id + point2d_idx)
            if errors:
                stats["mean_reproj_error"] = sum(errors) / len(errors)
                stats["max_reproj_error"] = max(errors)
    else:
        stats["num_points3d"] = 0

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COLMAP pipeline for KIST new video")
    parser.add_argument("--recon_dir", required=True,
                        help="Recon root dir (contains gps.txt, and images/ unless --image_dir is set)")
    parser.add_argument("--image_dir", default=None,
                        help="Override image directory (default: recon_dir/images/)")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL",
                        help="COLMAP camera model (default: SIMPLE_RADIAL)")
    parser.add_argument("--single_camera_per_folder", default="1",
                        help="1 = one camera model per subfolder (default: 1)")
    parser.add_argument("--gpu_index", default="1",
                        help="GPU index to use (default: 1 = RTX 4090)")
    parser.add_argument("--rig_config", default=None,
                        help="Path to rig_config.json (default: /data/rig_config.json)")
    parser.add_argument("--skip_feature_extraction", action="store_true")
    parser.add_argument("--skip_rig_configurator",   action="store_true")
    parser.add_argument("--matcher", default="exhaustive",
                        choices=["exhaustive", "sequential"],
                        help="Matcher type (default: exhaustive)")
    parser.add_argument("--skip_matching",           action="store_true")
    parser.add_argument("--skip_mapper",             action="store_true")
    parser.add_argument("--skip_aligner",            action="store_true")
    parser.add_argument("--skip_converter",          action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run

    # ── Open unified log ──────────────────────────────────────
    recon_dir  = Path(args.recon_dir)
    default_log = str(recon_dir.parent / "pipeline.log")
    log_path = os.environ.get("LOG_FILE", default_log)
    if not DRY_RUN:
        _open_log(log_path)
        log(f"\n{'█'*60}")
        log(f"  COLMAP pipeline  [run_colmap.py]")
        log(f"  recon_dir : {args.recon_dir}")
        log(f"  LOG       : {log_path}")
        log(f"  START     : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'█'*60}")

    images_dir = Path(args.image_dir) if args.image_dir else recon_dir / "images"
    db_path    = recon_dir / "database.db"
    sparse_dir = recon_dir / "colmap" / "sparse"
    sparse_0   = sparse_dir / "0"
    aligner_0  = sparse_dir / "0_aligner"
    gps_txt    = recon_dir / "gps.txt"
    sparse_ply = recon_dir / "sparse_ba.ply"
    rig_config = Path(args.rig_config) if args.rig_config else Path("/data/rig_config.json")

    # ── Pre-checks ────────────────────────────────────────────
    banner("Pre-flight checks")
    check_dir(images_dir, "images/")
    check_file(gps_txt, "gps.txt")

    if not DRY_RUN:
        n_images = sum(1 for _ in images_dir.rglob("*.jpg"))
        n_gps    = sum(1 for line in gps_txt.read_text().splitlines() if line.strip())
        log(f"  images : {n_images}")
        log(f"  gps.txt: {n_gps} lines  (expected = {n_images})")
        if n_gps != n_images:
            log(f"  WARNING: gps.txt line count ({n_gps}) != image count ({n_images})")

    if not DRY_RUN:
        sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Feature extraction ─────────────────────────────
    banner("Step 1: Feature extraction")
    if args.skip_feature_extraction:
        log("  SKIPPED")
        check_file(db_path, "database.db (from previous run)")
    else:
        run([
            "colmap", "feature_extractor",
            "--database_path",       str(db_path),
            "--image_path",          str(images_dir),
            "--ImageReader.camera_model",             args.camera_model,
            "--ImageReader.single_camera_per_folder", args.single_camera_per_folder,
            "--FeatureExtraction.use_gpu",            "1",
            "--FeatureExtraction.gpu_index",          args.gpu_index,
        ], "colmap feature_extractor")
        check_file(db_path, "database.db")

    # ── Step 2: Matching ──────────────────────────────────────
    banner(f"Step 2: Matching ({args.matcher})")
    if args.skip_matching:
        log("  SKIPPED")
    elif args.matcher == "exhaustive":
        run([
            "colmap", "exhaustive_matcher",
            "--database_path",               str(db_path),
            "--FeatureMatching.use_gpu",     "1",
            "--FeatureMatching.gpu_index",   args.gpu_index,
            "--FeatureMatching.guided_matching", "1",
        ], "colmap exhaustive_matcher")
    else:
        run([
            "colmap", "sequential_matcher",
            "--database_path",                        str(db_path),
            "--SequentialMatching.overlap",           "20",
            "--SequentialMatching.quadratic_overlap",  "0",
            "--SequentialMatching.expand_rig_images",  "1",
            "--FeatureMatching.use_gpu",              "1",
            "--FeatureMatching.gpu_index",            args.gpu_index,
            "--FeatureMatching.guided_matching",      "1",
        ], "colmap sequential_matcher")

    # ── Step 3: Mapper ────────────────────────────────────────
    banner("Step 3: Mapper")
    if args.skip_mapper:
        log("  SKIPPED")
        check_dir(sparse_0, "sparse/0/")
    else:
        # Remove auto-created rigs from feature_extractor (single_camera_per_folder)
        # so mapper runs without rig constraints
        if not DRY_RUN:
            import sqlite3
            _db = sqlite3.connect(str(db_path))
            _db.execute("DELETE FROM rigs")
            _db.execute("DELETE FROM rig_sensors")
            _db.execute("DELETE FROM frames")
            _db.execute("DELETE FROM frame_data")
            _db.commit()
            _db.close()
            log("  Cleared auto-created rigs from DB")
        sparse_0.mkdir(parents=True, exist_ok=True)
        run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path",    str(images_dir),
            "--output_path",   str(sparse_dir),
            "--Mapper.ba_refine_focal_length",    "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params",    "1",
            "--Mapper.ba_use_gpu",               "1",
            "--Mapper.ba_gpu_index",             args.gpu_index,
        ], "colmap mapper")
        check_dir(sparse_0, "sparse/0/")

        # Mapper validation
        log("")
        log("  Mapper stats:")
        stats = read_colmap_stats(sparse_0)
        log(f"    registered images : {stats.get('registered_images', '?')}")
        log(f"    points3D          : {stats.get('num_points3d', '?')}")
        if "mean_reproj_error" in stats:
            log(f"    mean reproj error : {stats['mean_reproj_error']:.4f} px")
            log(f"    max  reproj error : {stats['max_reproj_error']:.4f} px")
            if stats["mean_reproj_error"] > 2.0:
                log("  WARNING: mean reprojection error > 2.0 px — pose quality may be poor")

        # Camera registration check
        cam_counts = stats.get("cam_counts", {})
        log(f"    registered cameras:")
        for cam in sorted(cam_counts):
            log(f"      {cam}: {cam_counts[cam]} images")
        missing_cams = [c for c in CAMERAS if c not in cam_counts]
        if missing_cams:
            sys.exit(f"✖ ERROR: {len(missing_cams)} camera(s) not registered: {missing_cams}\n"
                     f"  Re-run with exhaustive matcher or check image quality.")

    # ── Step 3.5: rig_configurator ────────────────────────────
    banner("Step 3.5: rig_configurator")
    if args.skip_rig_configurator:
        log("  SKIPPED")
    else:
        check_file(rig_config, "rig_config.json")
        run([
            "colmap", "rig_configurator",
            "--database_path",   str(db_path),
            "--rig_config_path", str(rig_config),
            "--input_path",      str(sparse_0),
        ], "colmap rig_configurator")

    # ── Step 3.6: Mapper with rig constraints ─────────────────
    banner("Step 3.6: Mapper with rig constraints")
    sparse_rig_out = sparse_dir / "0_rig"      # mapper writes 0_rig/0/
    sparse_rig     = sparse_rig_out / "0"      # actual reconstruction dir
    if args.skip_rig_configurator:
        log("  SKIPPED (rig_configurator skipped)")
    else:
        sparse_rig_out.mkdir(parents=True, exist_ok=True)
        run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path",    str(images_dir),
            "--output_path",   str(sparse_rig_out),
            "--Mapper.ba_refine_focal_length",    "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params",    "1",
            "--Mapper.ba_use_gpu",                "1",
            "--Mapper.ba_gpu_index",              args.gpu_index,
        ], "colmap mapper (rig)")
        check_dir(sparse_rig, "sparse/0_rig/0/")

    # ── Step 4: model_aligner (GPS) ───────────────────────────
    banner("Step 4: model_aligner (GPS)")
    if args.skip_aligner:
        log("  SKIPPED")
        check_dir(aligner_0, "sparse/0_aligner/")
    else:
        aligner_0.mkdir(parents=True, exist_ok=True)
        run([
            "colmap", "model_aligner",
            "--input_path",      str(sparse_rig) if not args.skip_rig_configurator else str(sparse_0),
            "--output_path",     str(aligner_0),
            "--ref_images_path", str(gps_txt),
            "--ref_is_gps",      "1",
            "--alignment_type",  "enu",
            "--alignment_max_error", "3.0",
        ], "colmap model_aligner")
        check_dir(aligner_0, "sparse/0_aligner/")
        check_file(aligner_0 / "images.bin",  "0_aligner/images.bin")
        check_file(aligner_0 / "cameras.bin", "0_aligner/cameras.bin")

        # Aligner validation: compare stats vs pre-alignment
        log("")
        log("  Aligner stats:")
        stats_a = read_colmap_stats(aligner_0)
        log(f"    registered images : {stats_a.get('registered_images', '?')}")
        log(f"    points3D          : {stats_a.get('num_points3d', '?')}")

    # ── Step 5: model_converter → sparse_ba.ply ───────────────
    banner("Step 5: model_converter → sparse_ba.ply")
    if args.skip_converter:
        log("  SKIPPED")
    else:
        run([
            "colmap", "model_converter",
            "--input_path",  str(aligner_0),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ], "colmap model_converter → sparse_ba.ply")
        check_file(sparse_ply, "sparse_ba.ply")

    # ── Final summary ─────────────────────────────────────────
    banner("Pipeline complete")
    log(f"  recon_dir   : {recon_dir}")
    log(f"  sparse/0    : {sparse_0}")
    log(f"  0_aligner   : {aligner_0}")
    log(f"  sparse_ba   : {sparse_ply}")
    log(f"  gps.txt     : {gps_txt}")
    log("")
    log("Next step:")
    log("  python3 /data/py/make_meta_data_new.py --recon_dir <recon_dir>")


if __name__ == "__main__":
    main()

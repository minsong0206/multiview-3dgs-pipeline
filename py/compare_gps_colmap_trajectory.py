import argparse
import csv
import os
import site
import sys
from pathlib import Path

import numpy as np


HUGSIM_ROOT = "/workspace" if os.path.isdir("/workspace") else "/home/ms/HUGSIM_N/HUGSIM"
sys.path.insert(0, os.path.join(HUGSIM_ROOT, "data"))
from colmap.colmap_reader import qvec2rotmat, read_extrinsics_binary  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GPS trajectory with COLMAP-estimated trajectory and save PNG."
    )
    parser.add_argument(
        "--gps-csv",
        type=Path,
        default=Path("/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1.csv"),
        help="GPS CSV path",
    )
    parser.add_argument(
        "--images-bin",
        type=Path,
        default=Path(
            "/home/ms/260308-KIST-Videos/KIST_CURVE_ALL/"
            "kist_curve_all_exhaustive/sparse_with_rig/0/images.bin"
        ),
        help="COLMAP images.bin path",
    )
    parser.add_argument("--cam", type=str, default="CAM_FRONT", help="Camera folder name to use")
    parser.add_argument(
        "--gps-sampling",
        choices=["uniform", "first_n"],
        default="first_n",
        help="How to match GPS points to COLMAP frame count",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1_vs_kist_curve_all_exhaustive.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def import_matplotlib_agg():
    user_site = site.getusersitepackages()
    user_paths = [user_site] if isinstance(user_site, str) else list(user_site)
    user_paths = {p for p in user_paths if p}
    if user_paths:
        sys.path = [p for p in sys.path if p not in user_paths]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_gps_latlon(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"lat_deg", "lon_deg"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in GPS CSV: {sorted(missing)}")
        for r in reader:
            rows.append((float(r["lat_deg"]), float(r["lon_deg"])))
    if not rows:
        raise ValueError("GPS CSV has no rows.")
    return np.asarray(rows, dtype=np.float64)


def latlon_to_local_m(latlon):
    # Equirectangular projection around first point (good for local trajectory scale)
    lat0 = np.deg2rad(latlon[0, 0])
    lon0 = np.deg2rad(latlon[0, 1])
    lat = np.deg2rad(latlon[:, 0])
    lon = np.deg2rad(latlon[:, 1])
    r = 6371000.0
    east = (lon - lon0) * np.cos(lat0) * r
    north = (lat - lat0) * r
    return np.stack([east, north], axis=1)


def load_colmap_cam_centers(images_bin: Path, cam_name: str):
    ex = read_extrinsics_binary(str(images_bin))
    centers = []
    for img in ex.values():
        if not img.name.startswith(f"{cam_name}/"):
            continue
        frame_name = os.path.basename(img.name)
        stem, ext = os.path.splitext(frame_name)
        if ext.lower() != ".jpg" or not stem.isdigit():
            continue
        frame_idx = int(stem)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(img.qvec)
        w2c[:3, 3] = img.tvec
        c2w = np.linalg.inv(w2c)
        c = c2w[:3, 3]
        centers.append((frame_idx, c))
    if not centers:
        raise ValueError(f"No poses found for camera '{cam_name}' in {images_bin}")
    centers.sort(key=lambda x: x[0])
    frame_ids = np.asarray([x[0] for x in centers], dtype=np.int32)
    xyz = np.asarray([x[1] for x in centers], dtype=np.float64)
    return frame_ids, xyz


def project_3d_to_2d_pca(xyz):
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T  # 3x2
    proj = centered @ basis
    return proj


def sample_gps_for_match(gps_xy_m, n_target: int, mode: str):
    if len(gps_xy_m) < n_target:
        raise ValueError(f"GPS rows ({len(gps_xy_m)}) < target frames ({n_target})")
    if mode == "first_n":
        idx = np.arange(n_target, dtype=np.int64)
    else:
        idx = np.linspace(0, len(gps_xy_m) - 1, n_target)
        idx = np.round(idx).astype(np.int64)
    return gps_xy_m[idx], idx


def similarity_align_2d(src_xy, dst_xy):
    # Find s, R, t minimizing ||s * src * R + t - dst||^2
    src_mu = src_xy.mean(axis=0, keepdims=True)
    dst_mu = dst_xy.mean(axis=0, keepdims=True)
    src_c = src_xy - src_mu
    dst_c = dst_xy - dst_mu

    m = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    denom = np.sum(src_c * src_c)
    scale = np.sum((src_c @ r) * dst_c) / denom
    t = (dst_mu - scale * (src_mu @ r)).reshape(2)
    aligned = scale * (src_xy @ r) + t
    return aligned, scale, r, t


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def plot_comparison(out_path: Path, gps_xy, est_xy_raw, est_xy_aligned, info_text: str):
    plt = import_matplotlib_agg()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0, ax1 = axes

    ax0.plot(gps_xy[:, 0], gps_xy[:, 1], color="#2ca02c", linewidth=1.6, label="GPS")
    ax0.plot(est_xy_raw[:, 0], est_xy_raw[:, 1], color="#1f77b4", linewidth=1.6, label="COLMAP (raw PCA)")
    ax0.scatter(gps_xy[0, 0], gps_xy[0, 1], color="#2ca02c", s=28)
    ax0.scatter(est_xy_raw[0, 0], est_xy_raw[0, 1], color="#1f77b4", s=28)
    ax0.set_title("Raw Trajectory (No Alignment)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(alpha=0.3)
    ax0.axis("equal")
    ax0.legend(loc="best")

    ax1.plot(gps_xy[:, 0], gps_xy[:, 1], color="#2ca02c", linewidth=2.0, label="GPS")
    ax1.plot(est_xy_aligned[:, 0], est_xy_aligned[:, 1], color="#d62728", linewidth=1.8, label="COLMAP (aligned)")
    ax1.scatter(gps_xy[0, 0], gps_xy[0, 1], color="#2ca02c", s=36, label="GPS start")
    ax1.scatter(gps_xy[-1, 0], gps_xy[-1, 1], color="#1b7f35", s=36, label="GPS end")
    ax1.scatter(est_xy_aligned[0, 0], est_xy_aligned[0, 1], color="#d62728", s=36, label="COLMAP start")
    ax1.scatter(est_xy_aligned[-1, 0], est_xy_aligned[-1, 1], color="#8c1c1c", s=36, label="COLMAP end")
    ax1.set_title("Aligned Trajectory Comparison")
    ax1.set_xlabel("East (m, local)")
    ax1.set_ylabel("North (m, local)")
    ax1.grid(alpha=0.3)
    ax1.axis("equal")
    ax1.legend(loc="best", fontsize=9)

    fig.suptitle("GPS vs COLMAP Trajectory", fontsize=14)
    fig.text(0.5, 0.01, info_text, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    gps_latlon = load_gps_latlon(args.gps_csv)
    gps_xy_full = latlon_to_local_m(gps_latlon)

    frame_ids, est_xyz = load_colmap_cam_centers(args.images_bin, args.cam)
    est_xy_raw = project_3d_to_2d_pca(est_xyz)

    gps_xy_match, gps_idx = sample_gps_for_match(gps_xy_full, len(est_xy_raw), args.gps_sampling)
    est_xy_aligned, scale, _, _ = similarity_align_2d(est_xy_raw, gps_xy_match)
    err_rmse = rmse(est_xy_aligned, gps_xy_match)

    info = (
        f"cam={args.cam}, colmap_frames={len(est_xy_raw)}, gps_rows={len(gps_xy_full)}, "
        f"gps_sampling={args.gps_sampling}, gps_idx=[{gps_idx[0]}..{gps_idx[-1]}], "
        f"similarity_scale={scale:.6f}, aligned_rmse={err_rmse:.3f} m"
    )
    plot_comparison(args.out, gps_xy_match, est_xy_raw, est_xy_aligned, info)
    print(f"Saved: {args.out}")
    print(info)
    print(f"COLMAP frame range: {frame_ids[0]}..{frame_ids[-1]}")


if __name__ == "__main__":
    main()

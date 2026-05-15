#!/usr/bin/env python3
"""
COLMAP ENU 정렬 결과 + raw GPS anchors → QGIS GeoJSON (new video pipeline)

Outputs:
  <out_dir>/camera_poses.geojson  — all 6-camera poses + CAM_FRONT LineString trajectory
  <out_dir>/gps_anchors.geojson   — raw 1Hz GPS anchors from gps.txt

ENU origin is read from the first line of gps.txt (same origin used by model_aligner).

Usage:
  python3 export_qgis.py --recon_dir /data/KIST_NEW/test_chunk/recon
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))
from colmap.colmap_reader import read_extrinsics_binary, qvec2rotmat

# WGS84 constants
_A  = 6378137.0
_E2 = 0.00669437999014


def enu_to_geodetic(e, n, u, lat0_deg, lon0_deg, alt0):
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    N0   = _A / np.sqrt(1 - _E2 * np.sin(lat0) ** 2)

    x0 = (N0 + alt0) * np.cos(lat0) * np.cos(lon0)
    y0 = (N0 + alt0) * np.cos(lat0) * np.sin(lon0)
    z0 = (N0 * (1 - _E2) + alt0) * np.sin(lat0)

    dx = -np.sin(lon0)*e - np.sin(lat0)*np.cos(lon0)*n + np.cos(lat0)*np.cos(lon0)*u
    dy =  np.cos(lon0)*e - np.sin(lat0)*np.sin(lon0)*n + np.cos(lat0)*np.sin(lon0)*u
    dz =  np.cos(lat0)*n + np.sin(lat0)*u

    x, y, z = x0 + dx, y0 + dy, z0 + dz
    lon = np.degrees(np.arctan2(y, x))
    p   = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, p * (1 - _E2)))
    for _ in range(5):
        lr = np.radians(lat)
        N  = _A / np.sqrt(1 - _E2 * np.sin(lr)**2)
        lat = np.degrees(np.arctan2(z + _E2 * N * np.sin(lr), p))
    return lat, lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_dir', required=True,
                        help='Recon root containing images/ and colmap/sparse/0_aligner/')
    parser.add_argument('--colmap_path', default=None,
                        help='Override aligned sparse model path')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: recon_dir)')
    parser.add_argument('--out_prefix', default=None,
                        help='Filename prefix for output files, e.g. "chunk00" '
                             '→ camera_poses_chunk00.geojson, gps_chunk00.geojson')
    args = parser.parse_args()

    recon_dir   = Path(args.recon_dir)
    colmap_path = Path(args.colmap_path) if args.colmap_path \
                  else recon_dir / 'colmap' / 'sparse' / '0_aligner'
    out_dir     = Path(args.out_dir) if args.out_dir else recon_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix      = f'_{args.out_prefix}' if args.out_prefix else ''

    gps_txt = recon_dir / 'gps.txt'
    if not gps_txt.exists():
        sys.exit(f'ERROR: {gps_txt} not found')

    images_bin = colmap_path / 'images.bin'
    if not images_bin.exists():
        sys.exit(f'ERROR: {images_bin} not found')

    # ── Load GPS anchors ──────────────────────────────────────
    gps_anchors = []   # list of (frame_idx, lat, lon, alt)
    with open(gps_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name  = parts[0]
            lat   = float(parts[1])
            lon   = float(parts[2])
            alt   = float(parts[3]) if len(parts) >= 4 else 0.0
            idx   = int(name.split('/')[1].split('.')[0])
            gps_anchors.append((idx, lat, lon, alt))

    if not gps_anchors:
        sys.exit('ERROR: gps.txt is empty')

    gps_anchors.sort(key=lambda x: x[0])
    origin_idx, lat0, lon0, alt0 = gps_anchors[0]
    print(f'ENU origin: frame {origin_idx:06d}  lat={lat0:.8f} lon={lon0:.8f} alt={alt0:.3f}')

    # ── Load COLMAP poses ─────────────────────────────────────
    cam_extrinsics = read_extrinsics_binary(str(images_bin))
    print(f'Loaded {len(cam_extrinsics)} images from {images_bin}')

    # ── File 1: camera_poses.geojson ─────────────────────────
    point_features = []
    for iid, image in cam_extrinsics.items():
        name = image.name
        if '/' not in name:
            name = f'CAM_FRONT/{name}'
        cam_name = name.split('/')[0]
        frame_idx = int(name.split('/')[1].split('.')[0])

        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3]  = image.tvec
        c2w = np.linalg.inv(w2c)
        e, n, u = float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3])

        lat, lon = enu_to_geodetic(e, n, u, lat0, lon0, alt0)

        point_features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {
                'name':       name,
                'cam':        cam_name,
                'frame':      frame_idx,
                'timestamp_s': round(frame_idx / 12.5, 4),
                'enu_e':      round(e, 4),
                'enu_n':      round(n, 4),
                'enu_u':      round(u, 4),
            }
        })

    # CAM_FRONT LineString trajectory
    front_pts = sorted(
        [f for f in point_features if f['properties']['cam'] == 'CAM_FRONT'],
        key=lambda f: f['properties']['frame']
    )
    trajectory = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [f['geometry']['coordinates'] for f in front_pts]
        },
        'properties': {'label': 'CAM_FRONT trajectory (COLMAP ENU)'}
    }

    cam_geojson = {
        'type': 'FeatureCollection',
        'features': point_features + [trajectory]
    }
    cam_out = out_dir / f'camera_poses{prefix}.geojson'
    with open(cam_out, 'w') as f:
        json.dump(cam_geojson, f, indent=2)
    print(f'Saved: {cam_out}  ({len(point_features)} points + 1 LineString)')

    # ── File 2: gps_anchors.geojson ──────────────────────────
    gps_features = []
    gps_coords   = []
    for idx, lat, lon, alt in gps_anchors:
        gps_features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': {
                'frame':       idx,
                'timestamp_s': round(idx / 12.5, 4),
                'lat':         lat,
                'lon':         lon,
                'alt':         alt,
                'source':      'raw_gps_1hz',
            }
        })
        gps_coords.append([lon, lat])

    gps_line = {
        'type': 'Feature',
        'geometry': {'type': 'LineString', 'coordinates': gps_coords},
        'properties': {'label': 'GPS anchor trajectory (raw 1Hz)'}
    }
    gps_geojson = {
        'type': 'FeatureCollection',
        'features': gps_features + [gps_line]
    }
    gps_out = out_dir / f'gps{prefix}.geojson'
    with open(gps_out, 'w') as f:
        json.dump(gps_geojson, f, indent=2)
    print(f'Saved: {gps_out}  ({len(gps_features)} points + 1 LineString)')

    # ── Alignment check ───────────────────────────────────────
    colmap_by_frame = {f['properties']['frame']: f for f in front_pts}
    print('\n=== Alignment check (GPS anchor frames vs COLMAP CAM_FRONT) ===')
    print(f"{'frame':>6}  {'GPS lat':>12} {'GPS lon':>13}  {'CAM lat':>12} {'CAM lon':>13}  {'dist_m':>7}")
    for idx, gps_lat, gps_lon, _ in gps_anchors:
        if idx not in colmap_by_frame:
            print(f"{idx:6d}  {gps_lat:.8f} {gps_lon:.8f}  (not in COLMAP)")
            continue
        coords = colmap_by_frame[idx]['geometry']['coordinates']
        cam_lon, cam_lat = coords[0], coords[1]
        dlat = (cam_lat - gps_lat) * 111111
        dlon = (cam_lon - gps_lon) * 111111 * np.cos(np.radians(gps_lat))
        dist = np.sqrt(dlat**2 + dlon**2)
        print(f"{idx:6d}  {gps_lat:.8f} {gps_lon:.8f}  {cam_lat:.8f} {cam_lon:.8f}  {dist:7.3f} m")


if __name__ == '__main__':
    main()

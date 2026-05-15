"""
COLMAP ENU 정렬 결과 → GeoJSON 변환 (QGIS 확인용)
- model_aligner --alignment_type enu 결과의 카메라 위치를 GPS lon/lat 좌표로 변환
- ENU origin은 gps.txt 첫 줄에서 자동으로 읽음
- 출력: chunk_xx/colmap/sparse/cameras.geojson
"""

import os
import sys
import json
import numpy as np

# =========================
# 설정
# =========================
chunk = 'chunk_03'

base = f'/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks/{chunk}'
model_path = f'{base}/colmap/sparse/0_aligner'
out_path   = f'{base}/colmap/sparse/cameras.geojson'
gps_path   = f'{base}/colmap/sparse/0_txt/gps.txt'

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))

from colmap.colmap_reader import read_extrinsics_binary, qvec2rotmat


# =========================
# ENU origin 읽기
# =========================
def load_enu_origin_from_gps(gps_path):
    """
    gps.txt 형식:
    image_name lat lon alt

    예:
    CAM_FRONT/000000.jpg 37.59982222 127.04338611 95.026
    """
    if not os.path.exists(gps_path):
        raise FileNotFoundError(f'gps.txt not found: {gps_path}')

    with open(gps_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            image_name = parts[0]
            lat0 = float(parts[1])
            lon0 = float(parts[2])
            alt0 = float(parts[3])

            print('[ENU origin loaded from gps.txt]')
            print(f'  image : {image_name}')
            print(f'  LAT0  : {lat0}')
            print(f'  LON0  : {lon0}')
            print(f'  ALT0  : {alt0}')

            return lat0, lon0, alt0

    raise ValueError(f'No valid GPS line found in {gps_path}')


LAT0, LON0, ALT0 = load_enu_origin_from_gps(gps_path)


# =========================
# WGS84 constants
# =========================
A = 6378137.0
E2 = 0.00669437999014


def enu_to_geodetic(e, n, u):
    """
    ENU meter 좌표를 WGS84 geodetic 좌표(lat, lon)로 변환.
    ENU origin은 LAT0, LON0, ALT0 사용.
    """
    lat0 = np.radians(LAT0)
    lon0 = np.radians(LON0)

    N0 = A / np.sqrt(1 - E2 * np.sin(lat0) ** 2)

    x0 = (N0 + ALT0) * np.cos(lat0) * np.cos(lon0)
    y0 = (N0 + ALT0) * np.cos(lat0) * np.sin(lon0)
    z0 = (N0 * (1 - E2) + ALT0) * np.sin(lat0)

    dx = (
        -np.sin(lon0) * e
        - np.sin(lat0) * np.cos(lon0) * n
        + np.cos(lat0) * np.cos(lon0) * u
    )
    dy = (
        np.cos(lon0) * e
        - np.sin(lat0) * np.sin(lon0) * n
        + np.cos(lat0) * np.sin(lon0) * u
    )
    dz = np.cos(lat0) * n + np.sin(lat0) * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x ** 2 + y ** 2)

    lat = np.degrees(np.arctan2(z, p * (1 - E2)))

    for _ in range(5):
        lat_rad = np.radians(lat)
        N = A / np.sqrt(1 - E2 * np.sin(lat_rad) ** 2)
        lat = np.degrees(np.arctan2(z + E2 * N * np.sin(lat_rad), p))

    return lat, lon


# =========================
# COLMAP images.bin 읽기
# =========================
images_bin_path = os.path.join(model_path, 'images.bin')

if not os.path.exists(images_bin_path):
    raise FileNotFoundError(f'images.bin not found: {images_bin_path}')

cam_extrinsics = read_extrinsics_binary(images_bin_path)
print(f'Loaded {len(cam_extrinsics)} images from {images_bin_path}')


# =========================
# Camera center → GeoJSON
# =========================
features = []

for iid, image in cam_extrinsics.items():
    # COLMAP image pose:
    # qvec, tvec은 world-to-camera 변환
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3] = image.tvec

    # camera center는 c2w translation
    c2w = np.linalg.inv(w2c)

    # model_aligner --alignment_type enu 결과라고 가정
    e = c2w[0, 3]
    n = c2w[1, 3]
    u = c2w[2, 3]

    lat, lon = enu_to_geodetic(e, n, u)

    cam_name = image.name.split('/')[0] if '/' in image.name else 'UNKNOWN'

    features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            # GeoJSON/QGIS 좌표 순서: [lon, lat]
            'coordinates': [lon, lat]
        },
        'properties': {
            'name': image.name,
            'cam': cam_name,
            'id': iid,
            'enu_e': float(e),
            'enu_n': float(n),
            'enu_u': float(u)
        }
    })


# CAM_FRONT trajectory as LineString (sorted by frame number)
front_features = sorted(
    [f for f in features if f['properties']['name'].startswith('CAM_FRONT/')],
    key=lambda f: f['properties']['name']
)
trajectory = {
    'type': 'Feature',
    'geometry': {
        'type': 'LineString',
        'coordinates': [f['geometry']['coordinates'] for f in front_features]
    },
    'properties': {'label': f'{chunk} CAM_FRONT trajectory'}
}

geojson = {
    'type': 'FeatureCollection',
    'features': features + [trajectory]
}

with open(out_path, 'w') as f:
    json.dump(geojson, f, indent=2)

print(f'Saved: {out_path} ({len(features)} points + 1 LineString trajectory)')
print(f'QGIS에서 열기: Layer > Add Layer > Add Vector Layer > {out_path}')
"""
COLMAP ENU 정렬 결과 → GeoJSON 변환 (QGIS 확인용)
- 카메라 위치를 GPS (lat/lon) 좌표로 변환
- 출력: chunk_03/colmap/sparse/cameras.geojson
"""

import os
import sys
import json
import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))
from colmap.colmap_reader import read_extrinsics_binary, qvec2rotmat

# ENU origin (model_aligner 로그에서 확인)
# "Using the first GPS coordinate as the ENU origin"
LAT0 = 37.59982222
LON0 = 127.04338611
ALT0 = 95.026

# WGS84 constants
A = 6378137.0
E2 = 0.00669437999014

def enu_to_geodetic(e, n, u):
    lat0 = np.radians(LAT0)
    lon0 = np.radians(LON0)
    N0 = A / np.sqrt(1 - E2 * np.sin(lat0)**2)
    x0 = (N0 + ALT0) * np.cos(lat0) * np.cos(lon0)
    y0 = (N0 + ALT0) * np.cos(lat0) * np.sin(lon0)
    z0 = (N0 * (1 - E2) + ALT0) * np.sin(lat0)
    dx = -np.sin(lon0)*e - np.sin(lat0)*np.cos(lon0)*n + np.cos(lat0)*np.cos(lon0)*u
    dy =  np.cos(lon0)*e - np.sin(lat0)*np.sin(lon0)*n + np.cos(lat0)*np.sin(lon0)*u
    dz =  np.cos(lat0)*n + np.sin(lat0)*u
    x, y, z = x0+dx, y0+dy, z0+dz
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, p * (1 - E2)))
    for _ in range(5):
        N = A / np.sqrt(1 - E2 * np.sin(np.radians(lat))**2)
        lat = np.degrees(np.arctan2(z + E2 * N * np.sin(np.radians(lat)), p))
    return lat, lon

chunk = 'chunk_03'
base = f'/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks/{chunk}'
model_path = f'{base}/colmap/sparse/0_aligner'
out_path   = f'{base}/colmap/sparse/cameras.geojson'

cam_extrinsics = read_extrinsics_binary(os.path.join(model_path, 'images.bin'))
print(f'Loaded {len(cam_extrinsics)} images')

features = []
for iid, image in cam_extrinsics.items():
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3]  = image.tvec
    c2w = np.linalg.inv(w2c)
    e, n, u = c2w[0, 3], c2w[1, 3], c2w[2, 3]
    lat, lon = enu_to_geodetic(e, n, u)
    cam_name = image.name.split('/')[0] if '/' in image.name else 'UNKNOWN'
    features.append({
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
        'properties': {'name': image.name, 'cam': cam_name, 'id': iid}
    })

geojson = {'type': 'FeatureCollection', 'features': features}
with open(out_path, 'w') as f:
    json.dump(geojson, f)
print(f'Saved: {out_path}  ({len(features)} points)')
print(f'QGIS에서 열기: Layer > Add Layer > Add Vector Layer > {out_path}')

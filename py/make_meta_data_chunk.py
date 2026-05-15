"""
COLMAP BA 결과 → HUGSIM meta_data.json 생성 (chunk_00 ~ chunk_04 공용)
"""

import os
import sys
import json
import argparse
import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))
from colmap.colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

VIDEO_FPS = 12.5

# chunk별 frame 범위 (1-based global index, GPS 매핑용)
CHUNK_FRAMES = {
    'chunk_00': (1,    525),
    'chunk_01': (526,  987),
    'chunk_02': (988,  1450),
    'chunk_03': (1451, 1912),
    'chunk_04': (1913, 2788),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', required=True, help='e.g. chunk_00')
    parser.add_argument('--colmap_path', default=None,
                        help='aligned sparse model path (default: chunks/<chunk>/colmap/sparse/0_aligner)')
    parser.add_argument('--out_dir', default=None,
                        help='출력 디렉토리 (default: chunks/<chunk>/recon)')
    args = parser.parse_args()

    chunk = args.chunk
    base = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks'

    colmap_path = args.colmap_path or os.path.join(base, chunk, 'colmap/sparse/0_aligner')
    out_dir     = args.out_dir     or os.path.join(base, chunk, 'recon')

    if chunk not in CHUNK_FRAMES:
        print(f'ERROR: unknown chunk {chunk}. choices: {list(CHUNK_FRAMES.keys())}')
        sys.exit(1)

    frame_start, frame_end = CHUNK_FRAMES[chunk]
    n_frames = frame_end - frame_start + 1

    print(f'chunk      : {chunk}')
    print(f'frames     : {frame_start} ~ {frame_end}  ({n_frames} frames)')
    print(f'colmap_path: {colmap_path}')
    print(f'out_dir    : {out_dir}')

    cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_path, 'images.bin'))
    cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_path, 'cameras.bin'))
    print(f'Loaded {len(cam_extrinsics)} images, {len(cam_intrinsics)} cameras')

    name2pose  = {}
    name2camid = {}
    for iid, image in cam_extrinsics.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3]  = image.tvec
        c2w  = np.linalg.inv(w2c)
        name = image.name
        if '/' not in name:
            name = f'CAM_FRONT/{name}'
        name2pose[name]  = c2w
        name2camid[name] = image.camera_id

    camid2intr = {}
    for cid, cam in cam_intrinsics.items():
        if cam.model in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            fx, fy = f, f
        else:
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        camid2intr[cid] = K

    # origin: chunk 내 첫 번째 CAM_FRONT 프레임
    front_names = sorted([n for n in name2pose if n.startswith('CAM_FRONT/')])
    origin_c2w  = name2pose[front_names[0]]
    inv_pose    = np.linalg.inv(origin_c2w)

    meta_data = {
        'camera_model': 'SIMPLE_RADIAL',
        'verts':        {},
        'frames':       [],
        'inv_pose':     inv_pose.tolist(),
    }

    missing = []
    for frame_idx in range(frame_start, frame_end + 1):
        local_idx = frame_idx - frame_start                         # 0-based, matches images.bin names
        frame_str = f'{local_idx:06d}.jpg'
        timestamp = local_idx / VIDEO_FPS

        for cam in CAMERAS:
            img_name = f'{cam}/{frame_str}'
            rgb_path = f'./images/{cam}/{frame_str}'

            if img_name not in name2pose:
                missing.append(img_name)
                continue

            # model_aligner already applied rotation, translation, and scale.
            # No additional scale factor needed.
            c2w = inv_pose @ name2pose[img_name]
            cid  = name2camid[img_name]
            intr = camid2intr[cid]
            w    = cam_intrinsics[cid].width
            h    = cam_intrinsics[cid].height

            meta_data['frames'].append({
                'rgb_path':   rgb_path,
                'camtoworld': c2w.tolist(),
                'intrinsics': intr.tolist(),
                'width':      w,
                'height':     h,
                'timestamp':  round(timestamp, 6),
                'dynamics':   {},
            })

    print(f'Total frames written: {len(meta_data["frames"])}')
    if missing:
        print(f'WARNING: {len(missing)} images not found in COLMAP (first 5: {missing[:5]})')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'meta_data.json')
    with open(out_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()

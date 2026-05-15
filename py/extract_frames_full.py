"""
RAW_DATA 6카메라 MP4 → 12.5fps 프레임 추출
출력: KIST_ALL_FULL/kist_all_full/images/{CAM_*/000001.jpg ...}
해상도: 800x450, 공통 구간 223초, 총 2787장/카메라
"""

import os
import subprocess

RAW_DATA = '/home/ms/260308-KIST-Videos/RAW_DATA'
OUT_BASE = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/kist_all_full/images'
FPS = 12.5
DURATION = 223
W, H = 800, 450

CAM_MAP = {
    '0_front':       'CAM_FRONT',
    '1_right_front': 'CAM_FRONT_RIGHT',
    '2_right_back':  'CAM_BACK_RIGHT',
    '3_back':        'CAM_BACK',
    '4_left_back':   'CAM_BACK_LEFT',
    '5_left_front':  'CAM_FRONT_LEFT',
}

expected = int(DURATION * FPS)
print(f"Expected frames per camera: {expected}")

for folder, cam_name in CAM_MAP.items():
    mp4 = os.path.join(RAW_DATA, folder, '2_Entrance-L1.MP4')
    out_dir = os.path.join(OUT_BASE, cam_name)
    os.makedirs(out_dir, exist_ok=True)

    # existing = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    # if existing >= expected:
    #     print(f'  ⏭  {cam_name}: already done ({existing} files), skipping')
    #     continue

    print(f'  === {cam_name} ===')
    cmd = [
        'ffmpeg', '-y',
        '-i', mp4,
        '-t', str(DURATION),
        '-vf', f'fps={FPS},scale={W}:{H}',
        '-q:v', '2',
        '-start_number', '0',
        os.path.join(out_dir, '%06d.jpg')
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'  ✖ FAILED: {ret.stderr[-300:]}')
        continue
    count = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    print(f'  ✔ Done: {count} frames')

print('\nAll cameras done.')

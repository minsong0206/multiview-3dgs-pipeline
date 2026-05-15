"""
KIST_ALL_FULL GPS 기반 chunk 분할
- 기존 이미지를 복사 없이 symlink로 chunk별 디렉토리 구성
- 200m 간격 5개 chunk (overlap 없음)
- 출력: KIST_ALL_FULL/chunks/chunk_00 ~ chunk_04/images/{CAM_*}/
"""

import os

SRC_BASE   = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/kist_all_full/images'
CHUNK_BASE = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks'

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# GPS 기반 200m 간격 chunk 경계 (frame_idx, 1-based)
CHUNKS = [
    {'name': 'chunk_00', 'start':    1, 'end':  525},
    {'name': 'chunk_01', 'start':  526, 'end':  987},
    {'name': 'chunk_02', 'start':  988, 'end': 1450},
    {'name': 'chunk_03', 'start': 1451, 'end': 1912},
    {'name': 'chunk_04', 'start': 1913, 'end': 2788},
]

for chunk in CHUNKS:
    name  = chunk['name']
    start = chunk['start']
    end   = chunk['end']
    n_frames = end - start + 1

    print(f'\n=== {name} (frames {start:06d} ~ {end:06d}, {n_frames} frames) ===')

    for cam in CAMS:
        src_dir  = os.path.join(SRC_BASE, cam)
        dst_dir  = os.path.join(CHUNK_BASE, name, 'images', cam)
        os.makedirs(dst_dir, exist_ok=True)

        count = 0
        for idx in range(start, end + 1):
            fname    = f'{idx:06d}.jpg'
            src_file = os.path.join(src_dir, fname)
            dst_link = os.path.join(dst_dir, fname)

            if not os.path.exists(src_file):
                print(f'  ✖ missing: {src_file}')
                continue

            if os.path.islink(dst_link):
                os.remove(dst_link)

            # 상대경로 symlink → Docker/호스트 모두 동작
            rel_src = os.path.relpath(src_file, dst_dir)
            os.symlink(rel_src, dst_link)
            count += 1

        print(f'  ✔ {cam}: {count} symlinks')

print('\nAll chunks done.')
print(f'Output: {CHUNK_BASE}')

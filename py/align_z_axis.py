#!/usr/bin/env python3
"""
meta_data.json의 world Z축을 gravity(Up) 방향으로 정렬.

CAM_FRONT trajectory로 도로 평면을 RANSAC fit하고,
평면 법선이 Z=Up이 되도록 회전행렬 R_fix를 구해
모든 camtoworld와 inv_pose에 적용.

출력: meta_data_aligned.json (같은 디렉토리)

Usage:
  python3 align_z_axis.py --meta_data /path/to/meta_data.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_R_fix(points):
    """
    CAM_FRONT trajectory 포인트들로부터 world→Z-up 정렬 회전행렬 계산.

    SVD의 최소 분산 방향(= 도로 평면 법선)을 구하고,
    Rodrigues 공식으로 그 법선을 Z=[0,0,1]로 돌리는 R 반환.
    """
    pts = np.array(points)
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[2]   # 최소 분산 = 도로 법선

    # Up 방향 부호 통일 (분산이 두 번째로 작은 축 쪽으로)
    # 현재 world에서 Y축이 Up에 가장 가까우므로 Y 성분 양수로
    if normal[1] < 0:
        normal = -normal

    n = normal / np.linalg.norm(normal)
    z = np.array([0., 0., 1.])
    axis = np.cross(n, z)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(n, z))

    if sin_a < 1e-8:
        R = np.eye(3) if cos_a > 0 else np.diag([1., -1., -1.])
    else:
        axis /= sin_a
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    print(f'SVD 법선(up): [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]')
    print(f'법선→Z 회전각: {angle:.2f}°')
    print(f'R_fix det={np.linalg.det(R):.4f}  (1이면 정상 회전)')
    return R


def apply_rotation_to_pose(c2w_4x4, R_fix_4x4):
    """world 좌표계에 R_fix 적용: new_c2w = R_fix @ c2w"""
    return R_fix_4x4 @ c2w_4x4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data', required=True,
                        help='입력 meta_data.json 경로')
    parser.add_argument('--out', default=None,
                        help='출력 경로 (기본: 같은 폴더의 meta_data_aligned.json)')
    args = parser.parse_args()

    meta_path = Path(args.meta_data)
    out_path  = Path(args.out) if args.out else meta_path.parent / 'meta_data_aligned.json'

    with open(meta_path) as f:
        data = json.load(f)

    # ── CAM_FRONT translation 수집 ────────────────────────────
    front_pts = []
    for fr in data['frames']:
        if 'CAM_FRONT/' in fr['rgb_path']:
            c2w = np.array(fr['camtoworld'])
            front_pts.append(c2w[:3, 3])

    if len(front_pts) < 10:
        sys.exit(f'ERROR: CAM_FRONT 프레임이 너무 적음 ({len(front_pts)}개)')
    print(f'CAM_FRONT 포인트 수: {len(front_pts)}')

    # ── R_fix 계산 (SVD 기반) ─────────────────────────────────
    R_fix = compute_R_fix(front_pts)
    R_fix_4x4 = np.eye(4)
    R_fix_4x4[:3, :3] = R_fix.T   # R_fix.T @ p 가 변환식이므로

    print(f'R_fix.T (적용 행렬):')
    for row in R_fix_4x4[:3, :3]:
        print(f'  [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]')

    # ── 모든 camtoworld에 R_fix 적용 ─────────────────────────
    for fr in data['frames']:
        c2w = np.array(fr['camtoworld'])
        new_c2w = apply_rotation_to_pose(c2w, R_fix_4x4)
        fr['camtoworld'] = new_c2w.tolist()

    # ── inv_pose에 R_fix 적용 ────────────────────────────────
    # camtoworld_new = R_fix_4x4 @ camtoworld_old
    # inv_pose = inv(camtoworld[origin]) 이고 origin은 identity이므로
    # inv_pose_new = inv(R_fix_4x4) = R_fix_4x4.T
    inv = np.array(data['inv_pose'])
    new_inv = R_fix_4x4 @ inv
    data['inv_pose'] = new_inv.tolist()

    # ── 저장 ─────────────────────────────────────────────────
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'저장: {out_path}')

    # ── 결과 검증 ─────────────────────────────────────────────
    new_front_pts = []
    for fr in data['frames']:
        if 'CAM_FRONT/' in fr['rgb_path']:
            c2w = np.array(fr['camtoworld'])
            new_front_pts.append(c2w[:3, 3])

    zs = [p[2] for p in new_front_pts]
    print(f'\n=== 정렬 후 검증 ===')
    print(f'CAM_FRONT Z 범위: {min(zs):.4f} ~ {max(zs):.4f} m  (작을수록 수평)')
    print(f'CAM_FRONT Z 표준편차: {np.std(zs):.4f} m')

    # CAM_BACK 첫 프레임 확인
    for fr in data['frames']:
        if fr['rgb_path'] == './images/CAM_BACK/000000.jpg':
            R = np.array(fr['camtoworld'])[:3, :3]
            t = np.array(fr['camtoworld'])[:3, 3]
            print(f'\nCAM_BACK/000000 정렬 후:')
            print(f'  t = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]')
            print(f'  X-axis(right): [{R[0,0]:+.4f}, {R[1,0]:+.4f}, {R[2,0]:+.4f}]')
            yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
            pitch = np.degrees(np.arcsin(np.clip(-R[2,0], -1, 1)))
            roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
            print(f'  Euler: yaw={yaw:.2f}°  pitch={pitch:.2f}°  roll={roll:.2f}°')
            break


if __name__ == '__main__':
    main()

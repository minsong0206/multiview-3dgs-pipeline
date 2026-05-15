"""
Generate gps.txt for COLMAP model_aligner geo-registration.

Format (one line per registered image):
    image_name lat lon alt

GPS CSV (25Hz rows, 1Hz updates): RAW_DATA/6_GPS/2_Entrance-L1.csv
  - doc column = original 25fps frame index (1-based)
  - Images extracted at 12.5fps → image N → GPS doc = round((N-1)*2) + 1

Usage:
    python make_gps_txt.py --chunk chunk_03
    python make_gps_txt.py --chunk chunk_03 --cam CAM_FRONT   # one-camera mode
    python make_gps_txt.py --chunk all                        # all five chunks

Output:
    KIST_ALL_FULL/chunks/<chunk>/colmap/sparse/0_txt/gps.txt
"""

import argparse
import csv
import os
import sys

GPS_CSV   = '/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1.csv'
CHUNK_BASE = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks'
CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

CHUNK_FRAMES = {
    'chunk_00': (1,    525),
    'chunk_01': (526,  987),
    'chunk_02': (988,  1450),
    'chunk_03': (1451, 1912),
    'chunk_04': (1913, 2788),
}

FPS_EXTRACTED = 12.5
FPS_ORIGINAL  = 25.0


def load_gps(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((float(r['lat_deg']), float(r['lon_deg']), float(r['alt_m'])))
    return rows  # index 0 = doc 1


def gps_for_frame(frame_idx, gps_rows):
    """Map 1-based extracted-frame index to GPS (lat, lon, alt) via linear interpolation."""
    t = (frame_idx - 1) / FPS_EXTRACTED          # seconds from video start
    doc_float = t * FPS_ORIGINAL + 1              # fractional 25fps doc index (1-based)

    lo = max(0, int(doc_float) - 1)               # lower GPS row (0-based)
    hi = min(len(gps_rows) - 1, lo + 1)           # upper GPS row

    if lo == hi:
        return gps_rows[lo]

    frac = doc_float - (lo + 1)                   # interpolation weight toward hi
    lat = gps_rows[lo][0] + frac * (gps_rows[hi][0] - gps_rows[lo][0])
    lon = gps_rows[lo][1] + frac * (gps_rows[hi][1] - gps_rows[lo][1])
    alt = gps_rows[lo][2] + frac * (gps_rows[hi][2] - gps_rows[lo][2])
    return lat, lon, alt


def make_gps_txt(chunk, cams, gps_rows, dry_run=False):
    if chunk not in CHUNK_FRAMES:
        print(f'ERROR: unknown chunk {chunk}', file=sys.stderr)
        return

    start, end = CHUNK_FRAMES[chunk]
    out_dir = os.path.join(CHUNK_BASE, chunk, 'colmap', 'sparse', '0_txt')
    out_path = os.path.join(out_dir, 'gps.txt')

    lines = []
    for frame_idx in range(start, end + 1):
        lat, lon, alt = gps_for_frame(frame_idx, gps_rows)
        local_idx = frame_idx - start          # 0-based chunk-local index (matches images.bin)
        for cam in cams:
            img_name = f'{cam}/{local_idx:06d}.jpg'
            lines.append(f'{img_name} {lat:.8f} {lon:.8f} {alt:.3f}')

    print(f'[{chunk}] {len(lines)} lines → {out_path}')
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'  Written.')
    else:
        for l in lines[:6]:
            print('  ', l)
        print('  ...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', default='chunk_03',
                        help='chunk name (chunk_00..chunk_04) or "all"')
    parser.add_argument('--cam', default=None,
                        help='single camera name to include (default: all 6)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print first few lines without writing')
    args = parser.parse_args()

    cams = [args.cam] if args.cam else CAMS

    print(f'Loading GPS: {GPS_CSV}')
    gps_rows = load_gps(GPS_CSV)
    print(f'  {len(gps_rows)} rows loaded.')

    chunks = list(CHUNK_FRAMES.keys()) if args.chunk == 'all' else [args.chunk]
    for chunk in chunks:
        make_gps_txt(chunk, cams, gps_rows, dry_run=args.dry_run)


if __name__ == '__main__':
    main()

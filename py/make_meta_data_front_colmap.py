"""
COLMAP SfM 결과 → HUGSIM meta_data.json 생성 (CAM_FRONT only)
- sparse/0/cameras.bin  → intrinsics (SIMPLE_RADIAL)
- sparse/0/images.bin   → camtoworld (c2w)
- 출력: kist_curve_front_colmap/meta_data.json
"""

import os, json, struct
import numpy as np

SPARSE_DIR = '/home/ms/260308-KIST-Videos/kist_curve_front_colmap/sparse/0'
IMG_DIR    = '/home/ms/260308-KIST-Videos/KIST_CURVE/kist_curve_front/images/CAM_FRONT'
OUT_JSON   = '/home/ms/260308-KIST-Videos/kist_curve_front_colmap/meta_data.json'
W, H       = 800, 450
CAM_FPS    = 12.5


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])


def read_cameras_bin(path):
    """SIMPLE_RADIAL: f, cx, cy, k1"""
    cams = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid     = struct.unpack('<i', f.read(4))[0]
            model   = struct.unpack('<i', f.read(4))[0]
            width   = struct.unpack('<Q', f.read(8))[0]
            height  = struct.unpack('<Q', f.read(8))[0]
            # model 2 = SIMPLE_RADIAL: f, cx, cy, k1
            nparams = {0:3, 1:4, 2:4, 3:5, 4:8}.get(model, 4)
            params  = struct.unpack('<' + 'd'*nparams, f.read(8*nparams))
            cams[cid] = {'model': model, 'w': width, 'h': height, 'params': params}
            print(f'  cam {cid}: model={model}, params={[round(p,4) for p in params]}')
    return cams


def read_images_bin(path):
    """returns dict: image_id → {name, qvec, tvec, camera_id}"""
    images = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            iid  = struct.unpack('<i', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            cid  = struct.unpack('<i', f.read(4))[0]
            # read name (null-terminated)
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            name = name.decode()
            # skip 2D points
            n2d  = struct.unpack('<Q', f.read(8))[0]
            f.read(n2d * 24)  # x(8) + y(8) + point3D_id(8)
            images[iid] = {'name': name, 'qvec': qvec, 'tvec': tvec, 'cam_id': cid}
    return images


print('=== Reading COLMAP SfM results ===')
cams   = read_cameras_bin(os.path.join(SPARSE_DIR, 'cameras.bin'))
images = read_images_bin(os.path.join(SPARSE_DIR, 'images.bin'))
print(f'cameras: {len(cams)}, images: {len(images)}')

# intrinsics matrix 생성 (SIMPLE_RADIAL: f, cx, cy, k1)
cam = cams[1]
f, cx, cy, k1 = cam['params']
print(f'\nIntrinsics: fx=fy={f:.4f}, cx={cx:.4f}, cy={cy:.4f}, k1={k1:.6f}')

# 3x3 K matrix (HUGSIM 형식)
K = [
    [f,   0.0, cx],
    [0.0, f,   cy],
    [0.0, 0.0, 1.0],
]

# name → c2w 매핑
name2c2w = {}
for iid, info in images.items():
    R   = qvec2rotmat(info['qvec'])
    t   = np.array(info['tvec'])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3]  = t
    c2w = np.linalg.inv(w2c)
    name2c2w[info['name']] = c2w

# origin 정렬: 000001.jpg 기준
origin_name = '000001.jpg'
if origin_name not in name2c2w:
    origin_name = sorted(name2c2w.keys())[0]
    print(f'WARNING: 000001.jpg not found, using {origin_name} as origin')

origin_c2w  = name2c2w[origin_name]
inv_origin  = np.linalg.inv(origin_c2w)
print(f'Origin: {origin_name}')

# frames 생성
frames  = []
missing = 0

for frame_idx in range(1, 181):
    fname = f'{frame_idx:06d}.jpg'
    if fname not in name2c2w:
        print(f'  WARNING: {fname} not in SfM results')
        missing += 1
        continue

    c2w_world = name2c2w[fname]
    c2w_local = inv_origin @ c2w_world
    t_sec     = (frame_idx - 1) / CAM_FPS

    frames.append({
        'rgb_path':   f'./images/CAM_FRONT/{fname}',
        'camtoworld': c2w_local.tolist(),
        'intrinsics': K,
        'width':      W,
        'height':     H,
        'timestamp':  round(t_sec, 6),
        'dynamics':   {},
    })

print(f'\nframes: {len(frames)}, missing: {missing}')

# meta_data 조립
meta = {
    'camera_model': 'OPENCV',
    'verts':        {},
    'frames':       frames,
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nSaved: {OUT_JSON}')

# 샘플 확인
print('\n--- Sample frame[0] (000001.jpg) ---')
c2w = np.array(frames[0]['camtoworld'])
print(f'  c2w pos (should be ~[0,0,0]): {c2w[:3,3].round(6)}')
print(f'  intrinsics fx: {frames[0]["intrinsics"][0][0]:.4f}')

print('\n--- Sample frame[89] (000090.jpg) ---')
if len(frames) > 89:
    c2w = np.array(frames[89]['camtoworld'])
    print(f'  c2w pos: {c2w[:3,3].round(3)}')

import sys, json, numpy as np, sqlite3
sys.path.insert(0, '/home/ms/HUGSIM_N/HUGSIM/data')
from colmap.colmap import rotmat2qvec

with open('/home/ms/260308-KIST-Videos/kist_scene/meta_data.json') as f:
    meta = json.load(f)

# DB의 실제 image_id 읽기 (이름은 파일명만, 폴더명 없음)
db = sqlite3.connect('/home/ms/260308-KIST-Videos/kist_scene/database.db')
cur = db.cursor()
name2iid = {name: iid for iid, name in cur.execute('SELECT image_id, name FROM images')}
db.close()
print(f'DB images: {len(name2iid)}, sample: {list(name2iid.items())[:3]}')

lines = []
matched = 0
for frame in meta['frames']:
    cam_name = frame['rgb_path'].replace('./images/', '').split('/')[0]
    if cam_name != 'CAM_FRONT':
        continue
    # DB에는 파일명만 등록됨 (e.g. '000001.jpg')
    file_name = frame['rgb_path'].split('/')[-1]
    if file_name not in name2iid:
        print(f'WARNING: {file_name} not in DB')
        continue
    iid = name2iid[file_name]
    c2w = np.array(frame['camtoworld'])
    w2c = np.linalg.inv(c2w)
    q = rotmat2qvec(w2c[:3,:3])
    t = w2c[:3, 3]
    lines.append(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {file_name}\n\n")
    matched += 1

with open('/home/ms/260308-KIST-Videos/kist_scene/prior/images.txt', 'w') as f:
    f.writelines(lines)
print(f'Written {matched} frames')

import os 
import sys
import json
import argparse
import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM' #HUGSIM_ROOT라는 변수 설정
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data')) #python이 모듈을 찾는 경로에 HUGSIM_ROOT/data 추가 (0은 가장 높은 우선순위)
from colmap.colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat #qvec2rotmat : quaternoin 회전을 rotation matrix로 변환

CAMERAS =['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
          'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'] #CAMERAS라는 리스트 , 리스트는 여러 값을 순서대로 담는 자료형 EX) CAMERAS[0] -> 'CAM_BACK'

VIDEO_FPS = 12.5 # 비디오 FPS 를 저장한 변수

# 자료형은 float, int, str, list, dict 등등이 있다. 자료형은 변수에 저장된 값의 종류를 나타내며, 각 자료형은 특정한 연산과 기능을 지원한다. 예를 들어, int는 정수형 숫자를 나타내며, float는 소수점이 있는 숫자를 나타낸다. str은 문자열을 나타내며, list는 여러 값을 순서대로 담는 자료형이다. dict는 키-값 쌍으로 데이터를 저장하는 자료형이다. 

CHUNK_FRAMES = {
    'chunk_00' : (1,   525),
    'chunk_01' : (526, 987),
    'chunk_02' : (988, 1450),
    'chunk_03' : (1451, 1912),
    'chunk_04' : (1913, 2788),
}

#dict = { 'key1' : 'value1', 'key2' : 'value2'} / ()는 tuple 쉼표로 나타내고 변경하지 않는 값 list와 달리 메모리를 덜쓴다!

#main이라는 함수를 정의, 함수는 코드 묶음 으로 나중에 main() 호출해야 실행
def main():
    parser = argparse.ArgumentParser() #argparse 모듈의 ArgumentParser 클래스의 인스턴스 생성, 명령줄 인자를 처리하는 객체
    #argparse 모듈 안에 있는 ARgumentParser 클래스는 명령줄 인자를 처리하는 객체를 생성하는데 사용된다. 이 객체를 사용하여 명령줄에서 프로그램에 전달되는 인자를 정의하고, 프로그램이 실행될 때 해당 인자들 파싱
    #argparse.ArgumentParser() : 이 클래스를 실제로 실행해서 객체를 만든다 () 괄호가 붙이면 이 클래스로 실제 물건 하나 만들어줘 ! 라는 뜻 -> parser라는 변수에 저장
    parser.add_argument('--chunk', required=True, help='e.g. chunk_00') #명령줄 인자 --chunk를 정의, 필수 인자이며 도움말 메시지 제공 / required=True는 이 인자가 명령줄에서 반드시 제공되어야 함을 의미한다. help='e.g. chunk_00'는 이 인자에 대한 설명을 제공하는 도움말 메시지이다.
    parser.add_argument('--colmap_path', default=None, help='sparse BA 결과 경로 (default: chunks/<chunk>/colmap/sparse/0_ba)') #명령줄 인자 --colmap_path를 정의, default=None 선택적 인자이며 기본값은 None, 도움말 메시지 제공
    parser.add_argument('--out_dir', default=None, help='출력 디렉토리 (default: chunks/<chunk>/recon)') #명령줄 인자 --out_dir를 정의, 선택적 인자이며 기본값은 None, 도움말 메시지 제공
    args = parser.parse_args() #add_argument로 정의된 명령줄 인자를 실제로 파싱하여 args 라는 변수에 저장 즉 터미널에서 입력한 --chunk, --colmap_path, --out_dir 인자들을 실제로 읽고 검사하고 args 라는 객체에 저장하는 코드

    chunk = args.chunk #args 객체에서 chunk 속성 값을 chunk 변수에 저장
    base = '/home/ms/260308-KIST-Videos/KIST_ALL_FULL/chunks' #base라는 변수에 경로 문자열 저장

    colmap_path = args.colmap_path or os.path.join(base, chunk, 'colmap/sparse/0_ba') #colmap_path 변수에 args.colmap_path 값이 있으면 그 값을 사용하고, 그렇지 않으면 os.path.join(base, chunk, 'colmap/sparse/0_ba') 경로를 사용
    out_dir     = args.out_dir     or os.path.join(base, chunk, 'recon') #out_dir 변수에 args.out_dir 값이 있으면 그 값을 사용하고, 그렇지 않으면 os.path.join(base, chunk, 'recon') 경로를 사용

    if chunk not in CHUNK_FRAMES: #chunk가 CHUNK_FRAMES 딕셔너리에 없는 경우 if chunk_100 not in CHUNK_FRAMES: -> chunk_100이 CHUNK_FRAMES 딕셔너리에 없으면 
        print(f'ERROR: unknown chunk {chunk}. choices: {list(CHUNK_FRAMES.keys())}') #오류 메시지 출력, CHUNK_FRAMES 딕셔너리의 키 목록을 보여줌
        sys.exit(1) #프로그램 종료, 1은 비정상 종료를 나타냄

    frame_start, frame_end = CHUNK_FRAMES[chunk] #CHUNK_FRAMES 딕셔너리에서 chunk 키에 해당하는 값(튜플)을 frame_start와 frame_end 변수에 각각 저장 : tuple unpacking / frame_start = 1 , frame_end = 525
    n_frames = frame_end - frame_start + 1

    print(f'chunk      : {chunk}') #chunk 변수 출력
    print(f'frames     : {frame_start} ~ {frame_end}  ({n_frames} frames)') #frame_start, frame_end, n_frames 변수 출력
    print(f'colmap_path: {colmap_path}') #colmap_path 변수 출력
    print(f'out_dir    : {out_dir}') #out_dir 변수 출력 

    cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_path, 'images.bin')) #colmap_path/images.bin 파일에서 카메라 외부 파라미터 읽어서 cam_extrinsics 변수에 저장
    cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_path, 'cameras.bin')) #colmap_path/cameras.bin 파일에서 카메라 내부 파라미터 읽어서 cam_intrinsics 변수에 저장
    print(f'Loaded {len(cam_extrinsics)} images, {len(cam_intrinsics)} cameras') #cam_extrinsics와 cam_intrinsics의 길이 출력, 각각 이미지와 카메라의 수를 나타냄
    # len()은 파이썬 내장 함수로, 객체의 길이를 반환한다. 예를 들어, 리스트의 경우 len()은 리스트에 포함된 요소의 수를 반환한다. 딕셔너리의 경우 len()은 딕셔너리에 포함된 키-값 쌍의 수를 반환한다. 문자열의 경우 len()은 문자열의 문자 수를 반환한다. () : 함수 호출 연산자, len()은 객체의 길이를 반환하는 함수이다. {} : 딕셔너리를 정의하는 데 사용되는 중괄호, 예를 들어 my_dict = {'key1': 'value1', 'key2': 'value2'}

    name2pose  = {} #name2pose라는 빈 딕셔너리 생성, 나중에 카메라 이름과 포즈 정보를 저장하는 데 사용될 것으로 예상
    name2camid = {} #name2camid라는 빈 딕셔너리 생성, 나중에 카메라 이름과 카메라 ID 정보를 저장하는 데 사용될 것으로 예상
    for iid, image in cam_extrinsics.items(): #items() method -> key + value 를 같이 꺼낸다 / ex) iid = image_id , image 는 image의 정보 객체 (외부파라미터?)
        w2c = np.eye(4) #world to camera 좌표 변환 행렬 w2c를 4x4 단위 행렬로 초기화
        w2c[:3, :3] = qvec2rotmat(image.qvec) #w2c의 상위 3x3 부분에 image.qvec를 회전 행렬로 변환한 값을 저장, qvec2rotmat 함수는 쿼터니언을 회전 행렬로 변환하는 함수
        w2c[:3,  3] = image.tvec #w2c의 상위 3x1 부분에 image.tvec 값을 저장, tvec는 translation vector로, 카메라의 위치를 나타내는 벡터이다. w2c는 world to camera 좌표 변환 행렬로, 카메라의 위치와 방향을 나타내는 4x4 행렬이다. w2c[:3, :3]는 회전 행렬 부분을 나타내고, w2c[:3, 3]는 translation vector 부분을 나타낸다. 따라서 이 코드는 image.qvec를 회전 행렬로 변환하여 w2c의 회전 부분에 저장하고, image.tvec를 w2c의 translation 부분에 저장하여 카메라의 위치와 방향을 w2c 행렬로 표현하는 것이다.
        c2w = np.linalg.inv(w2c) #w2c 행렬의 역행렬을 계산하여 c2w 변수에 저장, c2w는 camera to world 좌표 변환 행렬로, 카메라에서 월드 좌표로 변환하는 행렬이다. w2c는 world to camera 좌표 변환 행렬이므로, 그 역행렬을 계산하면 camera to world 좌표 변환 행렬이 된다.
        name = image.name #image 객체의 name 속성 값을 name 변수에 저장
        if '/' not in name: #name에 '/' 문자가 없는 경우
            name = f'CAM_FRONT/{name}' #name 변수에 'CAM_FRONT/' 접두사를 추가하여 새로운 name 문자열 생성
        name2pose[name] = c2w
        name2camid[name] = image.camera_id #name2camid 딕셔너리에 name을 키로, image.camera_id를 값으로 저장, image.camera_id는 카메라의 ID를 나타내는 값이다. 따라서 이 코드는 name2camid 딕셔너리에 카메라 이름과 해당 카메라의 ID를 저장하는 것이다.
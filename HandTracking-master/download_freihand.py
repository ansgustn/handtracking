import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    # Streaming 다운로드 및 진행률 표시
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 Megabyte
    
    print(f"다운로드 시작: {url} -> {dest_path}")
    print(f"총 파일 크기: {total_size_in_bytes / (1024*1024*1024):.2f} GB")
    
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, 다운로드 중 문제가 발생했습니다.")
        return False
    return True

def extract_zip(zip_path, extract_dir):
    print(f"\n압축 해제 중: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 압축 파일 내의 총 파일 수
        total_files = len(zip_ref.namelist())
        for file in tqdm(iterable=zip_ref.namelist(), total=total_files, desc="압축 해제 진행률"):
            zip_ref.extract(member=file, path=extract_dir)
    print("압축 해제 완료!")

def main():
    # FreiHAND 공식 다운로드 링크 (v2)
    dataset_url = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
    
    # 다운로드 및 압축 해제할 기본 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_file_path = os.path.join(base_dir, "FreiHAND_pub_v2.zip")
    
    # tqdm 라이브러리가 없는 경우 설치 안내
    try:
        import tqdm
    except ImportError:
        print("tqdm 모듈이 설치되어 있지 않습니다. 'pip install tqdm'을 실행해주세요.")
        return

    # 1. 파일이 없는 경우 다운로드
    if not os.path.exists(zip_file_path):
        success = download_file(dataset_url, zip_file_path)
        if not success:
            return
    else:
        print(f"이미 파일이 존재합니다: {zip_file_path}")

    # 2. 압축 해제
    # dataset.py가 root_dir로 현재 디렉토리('.')를 바라보므로, 
    # 현재 디렉토리 안에 바로 training 폴더와 training_xyz.json 파일이 풀려야 합니다.
    extract_zip(zip_file_path, base_dir)

    print("\n✅ FreiHAND 데이터셋 다운로드 및 세팅이 모두 완료되었습니다!")
    print("이제 'python train.py'를 실행하여 모델을 학습시킬 수 있습니다.")

if __name__ == "__main__":
    main()

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples=None):
        """
        Args:
            root_dir (string): FreiHAND 데이터셋 최상위 경로 (예: c:/Users/user/Desktop/FreiHAND_pub-v2)
            transform (callable, optional): 이미지 변환(정규화 등)
            num_samples (int, optional): 테스트용으로 전체 데이터 중 일부만 사용할 때 지정
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "training", "rgb")
        self.xyz_json_path = os.path.join(root_dir, "training_xyz.json")
        self.transform = transform
        
        # 3D 좌표 JSON 로드
        with open(self.xyz_json_path, 'r') as f:
            self.xyz_data = json.load(f)
            
        # 총 이미지 개수 설정 (최대 32560개)
        if num_samples is not None:
            self.xyz_data = self.xyz_data[:num_samples]
            
        self.length = len(self.xyz_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 파일명은 00000000.jpg 형태
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # 이미지 로드 (RGB)
        image = Image.open(img_path).convert('RGB')
        
        # 3D 좌표 로드 및 (21, 3) 배열 변환
        xyz = np.array(self.xyz_data[idx], dtype=np.float32)
        
        # 모델의 출력 뉴런과 맞추기 위해 1D 배열(크기 63)로 평탄화
        xyz_flat = xyz.flatten()
        
        if self.transform:
            image = self.transform(image)
            
        # 텐서로 변환
        labels = torch.from_numpy(xyz_flat)
        
        return image, labels

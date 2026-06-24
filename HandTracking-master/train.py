import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FreiHANDDataset
from model import FreiHANDModel
import time

def train():
    # 1. 하이퍼파라미터 설정
    batch_size = 32
    num_epochs = 5  # 초기 테스트를 위해 5번만 반복 (나중에 50~100으로 늘릴 수 있음)
    learning_rate = 0.001
    
    # 2. 디바이스(GPU) 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device}] 장치를 사용하여 학습을 시작합니다.")

    # 3. 데이터 로더 준비
    # ResNet 입력 크기인 224x224로 변환 및 정규화
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FreiHANDDataset(root_dir='.', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"총 {len(dataset)}개의 이미지를 로드했습니다.")

    # 4. 모델, 손실 함수, 최적화 기법 초기화
    model = FreiHANDModel().to(device)
    criterion = nn.MSELoss()  # 3D 좌표 회귀(Regression) 문제이므로 평균제곱오차(MSE) 사용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 학습 루프
    print("=================== 학습 시작 ===================")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")
                
        epoch_loss = running_loss / len(dataloader)
        end_time = time.time()
        print(f"==> Epoch [{epoch+1}/{num_epochs}] 완료 | 평균 오차(Loss): {epoch_loss:.6f} | 소요 시간: {end_time - start_time:.1f}초")

    # 6. 학습된 모델 가중치 저장
    save_path = "freihand_custom_model.pth"
    torch.save(model.state_dict(), save_path)
    print("=================== 학습 완료 ===================")
    print(f"학습된 모델이 '{save_path}' 파일로 저장되었습니다!")

if __name__ == '__main__':
    train()

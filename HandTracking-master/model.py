import torch
import torch.nn as nn
from torchvision import models

class FreiHANDModel(nn.Module):
    def __init__(self, num_keypoints=21):
        super(FreiHANDModel, self).__init__()
        # 가벼운 구조인 ResNet18을 백본(Backbone)으로 사용
        # pretrained=True로 지정해 ImageNet 사전학습 가중치를 가져오면 학습이 더 빠릅니다.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 마지막 출력 층을 손 관절 개수(21개 * X,Y,Z 3좌표 = 63개)에 맞게 수정
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_keypoints * 3)

    def forward(self, x):
        return self.backbone(x)

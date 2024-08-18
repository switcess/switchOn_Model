import torch
import torch.nn as nn
import torch.nn.functional as F
"""
1. 모델 구조
모델 구조를 단순화하고, 필요 이상으로 깊은 네트워크를 사용하지 않도록 설계합니다. 대신, 적절한 크기의 필터와 풀링 레이어를 사용하여 실시간 성능을 유지합니다.

2. 실시간 성능 최적화
배치 크기 조절: 실시간 성능을 위해 작은 배치 크기를 사용합니다. 일반적으로 배치 크기는 1로 설정해 프레임마다 예측을 수행합니다.
모델 경량화: 복잡한 네트워크 구조를 피하고, 필요한 최소한의 컨볼루션 레이어를 사용합니다.
Dropout: 오버피팅을 방지하기 위해 드롭아웃 레이어를 추가합니다.
Mixed Precision Training: PyTorch의 torch.cuda.amp를 사용해 혼합 정밀도 학습을 통해 연산 속도를 높이고, 메모리 사용을 줄일 수 있습니다.

추가 최적화 제안:
프레임 수 및 해상도 조절: 입력 비디오의 프레임 수와 해상도를 조절하여 모델의 입력 크기를 최적화합니다.
혼합 정밀도 학습(Amp): PyTorch의 torch.cuda.amp를 사용하여 추론 속도를 높이고 메모리 사용량을 줄일 수 있습니다.
TensorRT 변환: 학습된 PyTorch 모델을 TensorRT로 변환하여 추론 최적화를 달성할 수 있습니다.
"""

class AnomalyDetection3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnomalyDetection3DCNN, self).__init__()
        # 3D convolutional layers
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.3)  # Dropout layer to reduce overfitting
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Permute the input to match the expected input format for 3D conv layers:
        # From [batch_size, num_frames, height, width, channels]
        # To [batch_size, channels, num_frames, height, width]
        x = x.permute(0, 4, 1, 2, 3)
        print(f"Input shape after permute: {x.shape}")

        x = self.pool(F.relu(self.conv1(x)))
        print(f"Shape after conv1 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Shape after conv2 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv3(x)))
        print(f"Shape after conv3 and pool: {x.shape}")

        # Flatten the tensor for the fully connected layers
        # Ensure this matches the output size after conv/pool layers
        x = x.view(-1, 64 * 4 * 4 * 4)
        print(f"Shape after flattening: {x.shape}")
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# AnomalyDetection3DCNN(num_classes=2)
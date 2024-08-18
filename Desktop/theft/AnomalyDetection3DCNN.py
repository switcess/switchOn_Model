import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(AnomalyDetectionModel, self).__init__()
        # 3D convolutional layers
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.3)  # Dropout layer to reduce overfitting
        
        # Fully connected layers
        # fc1의 입력 크기를 정확한 크기로 자동 계산하기 위해서는 placeholder로 설정 후 forward에서 결정
        self.fc1 = nn.Linear(90112, 128)  # 실제 입력 크기 (Conv3D 후 계산된 값)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward through Conv3D layers
        x = self.pool(F.relu(self.conv1(x)))
        print(f"Shape after conv1 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Shape after conv2 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv3(x)))
        print(f"Shape after conv3 and pool: {x.shape}")

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]
        print(f"Shape after flattening: {x.shape}")

        # Forward through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

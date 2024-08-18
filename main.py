import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import AnomalyDetection3DCNN as ad
import opendata as op

train_videos, train_labels = op.load_data("/Users/shinseohyunn/Desktop/Switch/aihub/Training")
val_videos, val_labels = op.load_data("/Users/shinseohyunn/Desktop/Switch/aihub/Validation")

print(train_videos.shape)

# 데이터를 PyTorch 텐서로 변환
X_train = torch.tensor(train_videos, dtype=torch.float32).permute(0, 1, 2, 3, 4)  # Conv3D 형식으로 변환
y_train = torch.tensor(train_labels, dtype=torch.long)

X_val = torch.tensor(val_videos, dtype=torch.float32).permute(0, 1, 2, 3, 4)
y_val = torch.tensor(val_labels, dtype=torch.long)

# 모델, 손실 함수, 최적화 도구 초기화
model = ad.AnomalyDetection3DCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

# 학습 및 검증 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 학습 데이터로 모델에 입력
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # 검증 단계
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

torch.save(model.state_dict(), "anomaly_detection_3dcnn.pth")

import os
import cv2
import torch
import torch.nn as nn
import AnomalyDetection3DCNN as ad  # 모델 파일 임포트
import opendata as op  # 데이터 로딩 파일 임포트

# 데이터 로드
train_videos, train_labels = op.load_data("/Users/eunjilee/Desktop/theft/Training")
X_train = torch.tensor(train_videos, dtype=torch.float32).permute(0, 4, 1, 2, 3).contiguous()

# 모델 초기화 및 예측
model = ad.AnomalyDetectionModel(num_classes=2)
model.eval()

with torch.no_grad():
    outputs = model(X_train)

# 비디오 작성기 설정 (원본 해상도 유지, RGB 색상 공간 유지)
height, width = train_videos.shape[2:4]  # 원본 해상도 가져오기
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 원본 해상도 그대로 사용하여 비디오 저장 (RGB 유지)
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))  # 원본 해상도와 FPS 설정

# 각 프레임을 예측 결과와 함께 비디오로 저장
for i in range(X_train.shape[2]):  # num_frames 만큼 반복
    frame = train_videos[0, i]  # 원본 프레임 가져오기
    frame = (frame * 255).astype('uint8')  # 0~1로 정규화된 값을 0~255로 변환 (RGB 값 유지)
    
    # 예측된 클래스 가져오기
    pred_class = torch.argmax(outputs, dim=1).item()  # 비디오 단위로 예측 결과 사용
    
    # 예측 결과를 프레임에 텍스트로 추가 (작은 크기와 적당한 색상으로 조정)
    frame_with_text = cv2.putText(frame, f"Pred: {pred_class}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # 하얀색과 작은 크기

    # 비디오 작성기 객체에 추가
    out.write(cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR을 사용하므로 변환 후 저장

# 비디오 저장 완료
out.release()
print(f"Video saved as {output_video_path}")

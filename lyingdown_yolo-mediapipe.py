import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# YOLO 설정
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
names_path = 'coco.names'

yolo_net = cv2.dnn.readNet(weights_path, config_path)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

with open(names_path, 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture("전도.mp4")

def detect_posture(landmarks):
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    hip_y = (left_hip_y + right_hip_y) / 2

    if abs(shoulder_y - hip_y) > 0.3:
        return "Standing"
    else:
        return "Lying down"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 원본 프레임 크기
    height, width = frame.shape[:2]

    # 비율을 유지하면서 높이를 400으로 조정
    new_height = 400
    new_width = int((new_height / height) * width)
    frame = cv2.resize(frame, (new_width, new_height))

    # 시점 조정: 확대/축소 및 이동
    scale = 1.0  # 확대/축소 비율
    translation_x = 0  # x축 이동
    translation_y = 0  # y축 이동

    M = np.float32([
        [scale, 0, translation_x],
        [0, scale, translation_y]
    ])
    frame = cv2.warpAffine(frame, M, (new_width, new_height))

    # YOLO 객체 감지
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 사람이 감지된 경우
                center_x = int(detection[0] * new_width)
                center_y = int(detection[1] * new_height)
                w = int(detection[2] * new_width)
                h = int(detection[3] * new_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    lying_count = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            roi_rgb = frame[y:y+h, x:x+w]
            results = pose.process(cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(roi_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                posture = detect_posture(results.pose_landmarks.landmark)
                cv2.putText(frame, posture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                
                if posture == "Lying down":
                    lying_count += 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 왼쪽 상단에 누워 있는 사람 수 표시
    cv2.putText(frame, f'Lying down: {lying_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

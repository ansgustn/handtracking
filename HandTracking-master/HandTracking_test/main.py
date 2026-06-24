import cv2
import mediapipe as mp
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import time
import numpy as np
import math

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

# ==========================================
# 1. 모델 초기화
# ==========================================
yolo_model = YOLO("yolo11n.pt")

# 최신 MediaPipe Tasks API 사용 (파이썬 버전에 따른 구버전 에러 해결)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_time = 0

print("파이프라인 HUD 가동! 화면 왼쪽 위 데이터 상태를 확인하세요.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    hud_yolo_box = "None"
    hud_mp_input = "None"
    hud_mp_output = "None"

    # ==========================================
    # 2. YOLO 추론
    # ==========================================
    yolo_results = yolo_model.predict(frame, verbose=False)

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        hud_yolo_box = f"({x1}, {y1}) to ({x2}, {y2})"

        margin = 30
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(frame_w, x2 + margin), min(frame_h, y2 + margin)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 10 or crop_h < 10:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ==========================================
        # 3. 데이터 전처리 (MediaPipe)
        # ==========================================
        cropped_img = frame[y1:y2, x1:x2]
        rgb_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        hud_mp_input = f"{rgb_cropped.shape}"

        # ==========================================
        # 4. MediaPipe 추론
        # ==========================================
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cropped)
        detection_result = landmarker.detect(mp_image)

        # ==========================================
        # 5. 후처리 및 각도 계산
        # ==========================================
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                pts = []
                for landmark in hand_landmarks:
                    local_x = landmark.x * crop_w
                    local_y = landmark.y * crop_h
                    global_x = local_x + x1
                    global_y = local_y + y1
                    pts.append(np.array([global_x, global_y, landmark.z * crop_w]))

                idx = {"Thumb": (2,3,4), "Index": (5,6,7), "Middle": (9,10,11), "Ring": (13,14,15), "Pinky": (17,18,19)}
                angles = {}
                for finger, (a, b, c) in idx.items():
                    v1 = pts[b] - pts[a]
                    v2 = pts[c] - pts[b]
                    angles[finger] = calculate_angle(v1, v2)

                hud_mp_output = f"Idx:{angles['Index']:.1f} Mid:{angles['Middle']:.1f} Rng:{angles['Ring']:.1f}"

                # 관절 직접 그리기
                for pt in pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                
                # 선 그리기
                connections = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)]
                for connection in connections:
                    start_idx, end_idx = connection
                    pt1 = (int(pts[start_idx][0]), int(pts[start_idx][1]))
                    pt2 = (int(pts[end_idx][0]), int(pts[end_idx][1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    # ==========================================
    # 🌟 화면 출력
    # ==========================================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    cv2.rectangle(frame, (5, 5), (460, 140), (0, 0, 0), -1)

    cv2.putText(frame, f'FPS: {int(fps)}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f'[1. YOLO Out] Box: {hud_yolo_box}', (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
    cv2.putText(frame, f'[2. MP Input] Crop Shape: {hud_mp_input}', (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(frame, f'[3. MP Out] Angle: {hud_mp_output}', (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    cv2.imshow('Pipeline Data HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
import cv2
import mediapipe as mp
import math
import numpy as np
import pykinect_azure as pykinect

# 1. MediaPipe 손 추적 초기화 (Tasks API 활용)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 모델 경로는 프로젝트 내의 파일 사용
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1, # 한 손만 집중해서 분석
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# 뼈대 연결 정보 (커스텀 그리기 함수 사용)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 소지
]

def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

# 카메라 캡처 초기화 (Azure Kinect)
pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF # 손가락 각도만 측정하므로 Depth는 OFF
device = pykinect.start_device(config=device_config)

# 기준 각도를 위한 변수
baseline_angle = None
absolute_angle = None  # 오류 방지를 위해 명시적 초기화

print("['s' 키]를 누르면 현재 손가락 각도를 0도(기준점)로 설정합니다.")
print("['q' 키]를 누르면 종료합니다.")

while True:
    capture = device.update()
    ret, frame_bgra = capture.get_color_image()
    if not ret:
        continue

    # BGRA -> BGR 변환 (MediaPipe 및 cv2.imshow 호환성)
    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # 화면 좌우 반전 (거울 모드) 및 BGR -> RGB 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Image 생성 (Tasks API 요구사항)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # MediaPipe로 손 랜드마크 추출
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # 뼈대 그리기
            draw_custom_landmarks(frame, hand_landmarks)

            # 화면 해상도 가져오기
            h, w, c = frame.shape

            # 2. 핵심 랜드마크 픽셀 좌표 추출 (4: 엄지 끝, 8: 검지 끝)
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # 엄지와 검지 연결하는 선 그리기 (시각적 확인용)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)

            # 3. atan2를 이용한 절대 각도 계산
            dx = index_x - thumb_x
            dy = index_y - thumb_y
            
            # math.atan2는 라디안을 반환하므로 degree로 변환
            # y축이 화면 아래로 향하므로 -dy를 사용하여 직관적인 각도로 맞춤
            absolute_angle = math.degrees(math.atan2(-dy, dx))

            # 기준 각도(baseline)가 설정되어 있다면, 상대적 회전 각도 계산
            if baseline_angle is not None:
                # 얼마나 돌아갔는지(Rotation) 계산
                rotation = absolute_angle - baseline_angle
                
                # -180 ~ 180도 사이로 값 정규화
                if rotation > 180: rotation -= 360
                elif rotation < -180: rotation += 360
                
                cv2.putText(frame, f"Rotation: {rotation:.1f} deg", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # 현재 절대 각도 표시
            cv2.putText(frame, f"Abs Angle: {absolute_angle:.1f} deg", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Finger Rotation Tracker', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 's'를 누르면 현재 절대 각도를 기준점으로 저장
        if absolute_angle is not None:
            baseline_angle = absolute_angle
            print(f"기준 각도가 설정되었습니다: {baseline_angle:.1f}도")

# 리소스 해제
device.close()
cv2.destroyAllWindows()
landmarker.close()
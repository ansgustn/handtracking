import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 손가락 마디 연결선 정의
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 소지
]

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    points = []
    # 그려질 좌표 계산
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        # 랜드마크 점 그리기
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    
    # 랜드마크 연결선 그리기
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# 기본 카메라(웹캠 0번) 사용
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("웹캠이 켜졌습니다. 종료하려면 키보드의 'q' 키 또는 'ESC' 키를 누르세요.")
    
    pTime = 0
    last_timestamp_ms = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라 프레임을 불러올 수 없습니다.")
            continue

        # MediaPipe 이미지 객체로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 현재 타임스탬프 계산 (밀리초) - MediaPipe는 엄격하게 증가하는 타임스탬프를 요구함
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms
        
        # mediapipe로 손 랜드마크 추출
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 이미지에 손 랜드마크 그리기
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(image, hand_landmarks)
                    
        # FPS 계산
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime > 0 else 0
        pTime = cTime
        
        # 화면 좌측 상단에 FPS 표시
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # 거울 모드처럼 보이도록 화면 좌우 반전
        cv2.imshow('MediaPipe Hand Tracking Test', cv2.flip(image, 1))
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

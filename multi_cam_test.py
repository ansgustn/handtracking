import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np
import sys

# ==========================================
# MediaPipe 디바이스 및 모델 옵션 설정
# ==========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 소지
]

def draw_custom_landmarks(image, landmarks):
    """ MediaPipe가 예측한 손 랜드마크를 화면에 그려주는 함수 """
    h, w, _ = image.shape
    points = []
    
    # 각 점의 좌푯값 계산
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    
    # 뼈대(연결선) 그리기
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

# ==========================================
# 설정 및 초기화
# ==========================================
print("[INFO] YOLOv11 모델을 로드하는 중 (첫 실행시 모델 자동 다운로드 진행될 수 있음)...")
try:
    yolo_model = YOLO('yolo11n.pt') 
except Exception as e:
    print(f"[ERROR] YOLO 모델 로드 실패: {e}")
    sys.exit(1)

# 사용할 카메라 인덱스 지정 (0: 기본 웹캠, 1: 추가 웹캠 등)
camera_indices = [0, 1]
caps = [cv2.VideoCapture(idx) for idx in camera_indices]

# 정상 접속된 카메라 선별
valid_caps = []
for i, cap in enumerate(caps):
    if cap.isOpened():
        valid_caps.append((camera_indices[i], cap))
        print(f"[INFO] {camera_indices[i]}번 카메라 오픈 성공 🎉")
    else:
        print(f"[WARNING] {camera_indices[i]}번 카메라를 열 수 없습니다.")

if not valid_caps:
    print("[ERROR] 사용 가능한 카메라가 없어 프로그램을 종료합니다.")
    sys.exit(1)

# 손 인식기(Landmarker) 설정
# 여러 대의 카메라가 서로 다른 영상을 보여주므로, Tracking 최적화를 위해
# 각각의 카메라 당 별도의 HandLandmarker 객체를 할당합니다.
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

landmarkers = [HandLandmarker.create_from_options(options) for _ in valid_caps]

# 각 HandLandmarker에 전달할 이전 프레임 타임스탬프 기록
last_timestamp_ms = [0] * len(valid_caps)

print("[INFO] 멀티 카메라 테스트를 시작합니다. 🚀")
print("[INFO] 종료하려면 화면이 눌린 상태에서 키보드의 'q' 키를 누르세요.")

pTime = 0

try:
    while True:
        frames = []
        process_successful = False
        
        cTime = time.perf_counter()
        
        # ----------------------------------------------------
        # 1. 모든 카메라 돌면서 프레임 가져오고 처리
        # ----------------------------------------------------
        for i, (cam_idx, cap) in enumerate(valid_caps):
            success, image = cap.read()
            if not success:
                # 에러나면 검은색 빈 화면이라도 표시
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                image = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(image, "Camera Disconnected", (50, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                process_successful = True
                
                # 사용자가 거울을 보는 것처럼 느낄 수 있게 좌우 반전
                image = cv2.flip(image, 1)

                # ==========================
                # Task A: YOLO 객체 탐지
                # ==========================
                # stream=False, verbose=False로 콘솔 출력 안뜨게 최적화 옵션 부여
                results = yolo_model(image, stream=False, verbose=False)
                # 모델이 찾은 객체들의 바운딩박스가 입혀진(rendered) 넘파이 배열 가져오기
                image = results[0].plot()

                # ==========================
                # Task B: MediaPipe 손 랜드마크
                # ==========================
                # 내부 연산을 위해 BGR(OpenCV 방식)을 RGB(MediaPipe 방식)으로 변환
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # MediaPipe의 VIDEO 모드는 매 프레임마다 증가하는 타임스탬프를 요구합니다
                timestamp_ms = int(time.perf_counter() * 1000)
                if timestamp_ms <= last_timestamp_ms[i]:
                    timestamp_ms = last_timestamp_ms[i] + 1
                last_timestamp_ms[i] = timestamp_ms

                mp_result = landmarkers[i].detect_for_video(mp_image, timestamp_ms)
                
                # 손이 발견된 경우 오버레이 그리기
                if mp_result.hand_landmarks:
                    for hand_landmarks in mp_result.hand_landmarks:
                        draw_custom_landmarks(image, hand_landmarks)

                # 어떤 카메라 화면인지 표시
                cv2.putText(image, f'Cam {cam_idx}', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                
            frames.append(image)
            
        if not process_successful:
            print("[WARNING] 모든 카메라에서 프레임을 읽어오지 못했습니다. 잠시 대기합니다.")
            time.sleep(1)
            continue
            
        # ----------------------------------------------------
        # 2. 결과 출력 창 구성 및 FPS 처리
        # ----------------------------------------------------
        fps = 1 / (cTime - pTime) if pTime > 0 else 0
        pTime = cTime
        
        # 카메라 두 개가 있을 경우 화면을 좌우로 붙이기(Concatenation)
        if len(frames) == 1:
            display_img = frames[0]
        else:
            h1, w1 = frames[0].shape[:2]
            for idx in range(1, len(frames)):
                h_next, w_next = frames[idx].shape[:2]
                if h1 != h_next:
                    # 세로 해상도가 다를 경우 높이를 첫번째 영상 기준으로 맞춤
                    frames[idx] = cv2.resize(frames[idx], (w_next * h1 // h_next, h1))
            
            # 리스트에 있는 모든 프레임들을 가로로 나란히 이어붙이기
            display_img = cv2.hconcat(frames)

        cv2.putText(display_img, f'Overall FPS: {int(fps)}', (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        # 윈도우 창에 출력해주기
        cv2.imshow('Multi-Camera: MediaPipe & YOLOv11', display_img)
        
        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] 사용자에 의해 강제 종료됩니다.")
finally:
    # ----------------------------------------------------
    # 3. 모든 자원 해제
    # ----------------------------------------------------
    print("[INFO] 사용된 자원을 해제합니다.")
    for _, cap in valid_caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()
    
    for landmarker in landmarkers:
        if landmarker is not None:
            landmarker.close()
    print("[INFO] 프로그램이 종료되었습니다.")

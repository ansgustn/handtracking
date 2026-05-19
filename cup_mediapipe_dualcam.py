import cv2
import mediapipe as mp
import math
import numpy as np
import threading
import time

"""
[실험 환경 정의 (Environment Setup & Controlled Variables)]
측정 실험을 진행할 때 반드시 통제되어야 하는 물리적, 소프트웨어적 세팅 값들입니다. 
이 기준을 벗어나면 데이터 오차가 커집니다.

1. 물리적 환경 통제
- 대상 객체 (Target Object): 상단 컵(고정축)과 하단 컵(회전축)으로 구성된 2단 종이컵. 마커 부착 금지 (Markerless 조건).
- 카메라 배치 (Camera Topology):
  * Cam 1: 작업자의 정면 기준 좌측 45도, 위에서 아래로(Top-down) 비스듬히 내려다보는 앵글. (객체와의 거리 30~50cm 고정)
  * Cam 2: 작업자의 정면 기준 우측 45도, 위에서 아래로 비스듬히 내려다보는 앵글. (Cam 1과 대칭 구조)
  ※ 두 카메라가 90도 각도를 이루며 객체의 360도 전방위를 교차 감시하여 사각지대를 없앰.
- 조명 및 배경 (Lighting & Background): 하얀색 종이컵과 손의 대비를 극대화하기 위해 바닥에 무광 검은색 매트 사용. 그림자 최소화를 위해 확산광(간접 조명) 사용.
- 작업자 동작 통제 (Actor Pose): 엄지와 검지만 사용하여 객체를 파지하는 정밀 그립(Precision Pinch Grip) 상태 유지.

2. 소프트웨어 파라미터 세팅 (코드에 적용됨)
- 해상도 (Resolution): 640 x 480 고정 (연산 속도와 픽셀 분해능의 타협점)
- MediaPipe 신뢰도: min_detection_confidence = 0.7, min_tracking_confidence = 0.7 
- 이중 EMA 필터 강도 (alpha): alpha_lm = 0.15 (랜드마크 스무딩), alpha_angle = 0.15 (각도 스무딩)
"""

# 1. MediaPipe 손 추적 초기화 (Tasks API 활용)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 뼈대 연결 정보 (커스텀 그리기 함수 사용)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 소지
]

def draw_custom_landmarks(image, landmarks_xy):
    h, w, _ = image.shape
    points = []
    # landmarks_xy는 (x, y, z) 형태의 튜플 리스트입니다.
    for pt in landmarks_xy:
        cx, cy = int(pt[0] * w), int(pt[1] * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

# 카메라 처리를 위한 멀티스레드 클래스 정의
class CameraThread(threading.Thread):
    def __init__(self, src, cap_idx):
        super().__init__()
        # 카메라 초기화
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_idx = cap_idx
        
        # 각 스레드마다 독립적인 MediaPipe Landmarker 모델 로드
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO, # 끊김 방지를 위해 VIDEO 모드 사용
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.baseline_angle = None
        self.current_frame = None
        self.abs_angle = None
        self.rotation = None # 현재 상대 회전 각도 저장용
        
        # 떨림 방지를 위한 변수들
        self.smoothed_angle = None
        self.alpha_angle = 0.15 # 각도 스무딩 강도
        
        self.smoothed_landmarks = None
        self.alpha_lm = 0.15 # 뼈대(랜드마크) 스무딩 강도 (낮을수록 부드러움)
        
        self.last_timestamp_ms = -1
        
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 독립된 객체로 추적 실행 (VIDEO 모드는 타임스탬프가 필요함)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            h, w, c = frame.shape
            cv2.putText(frame, f"Cam {self.cap_idx}", (w - 150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            abs_angle = None
            if results.hand_landmarks:
                # 첫 번째 인식된 손만 사용
                hand_landmarks = results.hand_landmarks[0]
                
                # --- 1. 뼈대(랜드마크) 3D 좌표 스무딩 (떨림 원천 차단) ---
                if self.smoothed_landmarks is None:
                    self.smoothed_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                else:
                    new_smoothed = []
                    for i, lm in enumerate(hand_landmarks):
                        # 이전 프레임 좌표와 현재 좌표를 혼합 (X, Y, Z 모두 적용)
                        sx = self.alpha_lm * lm.x + (1 - self.alpha_lm) * self.smoothed_landmarks[i][0]
                        sy = self.alpha_lm * lm.y + (1 - self.alpha_lm) * self.smoothed_landmarks[i][1]
                        sz = self.alpha_lm * lm.z + (1 - self.alpha_lm) * self.smoothed_landmarks[i][2]
                        new_smoothed.append((sx, sy, sz))
                    
                    # 손이 너무 빨리 움직여서 좌표 차이가 크면 스무딩 리셋 (잔상 방지)
                    # 0번(손목) 랜드마크 기준으로 차이 계산
                    if abs(hand_landmarks[0].x - self.smoothed_landmarks[0][0]) > 0.05:
                        self.smoothed_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                    else:
                        self.smoothed_landmarks = new_smoothed

                # 스무딩된 좌표로 뼈대 그리기
                draw_custom_landmarks(frame, self.smoothed_landmarks)

                # 2. 핵심 랜드마크 픽셀 좌표 추출 (4: 엄지 끝, 8: 검지 끝)
                thumb_tip = self.smoothed_landmarks[4]
                index_tip = self.smoothed_landmarks[8]

                thumb_x, thumb_y = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
                index_x, index_y = int(index_tip[0] * w), int(index_tip[1] * h)

                # 엄지와 검지 연결하는 선 그리기 (시각적 확인용)
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 2)
                cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)

                # --- 핵심 변경점: 2D(Y축) 대신 3D(Z축) 깊이 데이터를 사용하여 공중(Yaw) 회전 계산 ---
                # MediaPipe의 정규화된 3D 좌표계를 사용합니다.
                dx_3d = index_tip[0] - thumb_tip[0] # 좌우
                dz_3d = index_tip[2] - thumb_tip[2] # 앞뒤(깊이)
                
                # Z값이 작을수록 카메라에 가깝습니다. 직관적인 각도 계산을 위해 부호를 맞춥니다.
                raw_abs_angle = math.degrees(math.atan2(-dz_3d, dx_3d))
                
                # EMA(지수 이동 평균) 필터를 적용하여 각도 떨림 완화
                if self.smoothed_angle is None:
                    self.smoothed_angle = raw_abs_angle
                else:
                    # 179도에서 -179도로 넘어갈 때 값이 튀는 것을 방지
                    diff = raw_abs_angle - self.smoothed_angle
                    if diff > 180: raw_abs_angle -= 360
                    elif diff < -180: raw_abs_angle += 360
                    
                    # 필터 적용
                    self.smoothed_angle = self.alpha_angle * raw_abs_angle + (1 - self.alpha_angle) * self.smoothed_angle
                    
                    # 다시 -180 ~ 180 범위로 정규화
                    if self.smoothed_angle > 180: self.smoothed_angle -= 360
                    elif self.smoothed_angle < -180: self.smoothed_angle += 360
                    abs_angle = self.smoothed_angle

                    if self.baseline_angle is not None:
                        rotation = abs_angle - self.baseline_angle
                        if rotation > 180: rotation -= 360
                        elif rotation < -180: rotation += 360
                        self.rotation = rotation
                        
                        cv2.putText(frame, f"Cam Rot: {rotation:.1f} deg", (20, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        self.rotation = None
                    
                    cv2.putText(frame, f"Cam Abs: {abs_angle:.1f} deg", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                self.smoothed_angle = None # 손을 놓치면 스무딩 초기화
                self.smoothed_landmarks = None
                self.rotation = None
                cv2.putText(frame, "Hand Not Detected", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 메인 스레드에서 화면을 그리기 위해 최신 프레임 업데이트
            self.current_frame = frame
            self.abs_angle = abs_angle

    def set_baseline(self):
        if self.abs_angle is not None:
            self.baseline_angle = self.abs_angle
            print(f"Cam {self.cap_idx} 기준 각도 설정: {self.baseline_angle:.1f}도")

    def stop(self):
        self.running = False
        self.cap.release()
        self.landmarker.close()

def main():
    print("카메라를 준비 중입니다. 잠시만 기다려주세요...")
    # 스레드 2개 생성 및 시작
    cam1_thread = CameraThread(0, 1)
    cam2_thread = CameraThread(1, 2)
    
    cam1_thread.start()
    cam2_thread.start()
    
    print("['s' 키]를 누르면 두 카메라에서 보이는 현재 손가락 각도를 0도(기준점)로 설정합니다.")
    print("['q' 키]를 누르면 종료합니다.")
    
    # 두 카메라 프레임이 모두 한 번은 들어올 때까지 대기
    while cam1_thread.current_frame is None and cam2_thread.current_frame is None:
        time.sleep(0.1)

    while True:
        frame1 = cam1_thread.current_frame
        frame2 = cam2_thread.current_frame
        
        display_frame = None
        
        # 두 영상 가로로 병합
        if frame1 is not None and frame2 is not None:
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            if h1 != h2:
                frame2 = cv2.resize(frame2, (int(w2 * h1 / h2), h1))
            display_frame = np.hstack((frame1, frame2))
            
            dh, dw = display_frame.shape[:2]
            if dw > 1920:
                display_frame = cv2.resize(display_frame, (dw // 2, dh // 2))

        elif frame1 is not None:
            display_frame = frame1
        elif frame2 is not None:
            display_frame = frame2
            
        if display_frame is not None:
            # --- 종합 각도(Combined Rotation) 계산 로직 ---
            rot1 = cam1_thread.rotation
            rot2 = cam2_thread.rotation
            
            combined_rotation = None
            if rot1 is not None and rot2 is not None:
                # 두 카메라 모두에서 손이 보이면, 각도의 원형 평균(Circular Mean) 계산
                rad1, rad2 = math.radians(rot1), math.radians(rot2)
                sin_sum = math.sin(rad1) + math.sin(rad2)
                cos_sum = math.cos(rad1) + math.cos(rad2)
                combined_rotation = math.degrees(math.atan2(sin_sum, cos_sum))
            elif rot1 is not None:
                combined_rotation = rot1 # 카메라 1만 보일 때
            elif rot2 is not None:
                combined_rotation = rot2 # 카메라 2만 보일 때
                
            # 화면 가장 아래 중앙에 종합 각도 크게 표시
            if combined_rotation is not None:
                dh, dw = display_frame.shape[:2]
                text = f"Total Rotation: {combined_rotation:.1f} deg"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                text_x = (dw - text_size[0]) // 2
                text_y = dh - 40
                
                # 가독성을 위해 검은색 배경 박스 그리기
                cv2.rectangle(display_frame, (text_x - 10, text_y - text_size[1] - 10), 
                              (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), cv2.FILLED)
                cv2.putText(display_frame, text, (text_x, text_y), 
                            font, 1.5, (0, 255, 255), 3)

            cv2.imshow('Dual Camera Finger Rotation Tracker', display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cam1_thread.set_baseline()
            cam2_thread.set_baseline()

    # 안전하게 스레드 종료
    cam1_thread.stop()
    cam2_thread.stop()
    cam1_thread.join()
    cam2_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import mediapipe as mp
import time
import numpy as np
import threading

# ==========================================
# 백그라운드 프레임 스트리밍 클래스 (에러 방지용)
# ==========================================
class WebCamStreamer:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.opened = self.cap.isOpened()
        self.running = self.opened
        self.latest_frame = None
        self.lock = threading.Lock()
        
        if self.opened:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 거울 모드
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def isOpened(self):
        return self.opened

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.opened:
            try: self.thread.join(timeout=1.0)
            except: pass
            self.cap.release()

class KinectStreamer:
    def __init__(self, device):
        self.device = device
        self.running = True
        self.latest_color = None
        self.latest_depth = None
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                # 타임아웃 1000ms, 내부 버퍼가 안 차게 무한 반복
                capture = self.device.update(timeout_in_ms=1000)
                if capture is None:
                    continue
                
                ret_color, kinect_color = capture.get_color_image()
                ret_depth, aligned_depth = capture.get_transformed_depth_image()
                
                if ret_color:
                    bgr_img = cv2.cvtColor(kinect_color, cv2.COLOR_BGRA2BGR)
                    with self.lock:
                        self.latest_color = bgr_img
                        if ret_depth:
                            self.latest_depth = aligned_depth
            except Exception as e:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            c_ret = self.latest_color is not None
            c_frame = self.latest_color.copy() if c_ret else None
            d_ret = self.latest_depth is not None
            # Depth 이미지도 copy하여 반환
            d_frame = self.latest_depth.copy() if d_ret else None
            return c_ret, c_frame, d_ret, d_frame

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            try: self.thread.join(timeout=1.0)
            except: pass


# ==========================================
# MediaPipe 설정
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
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

def main():
    print("[INFO] Azure Kinect 초기화 중...")
    pykinect.initialize_libraries()

    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.synchronized_images_only = True
    device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE

    try:
        device = pykinect.start_device(config=device_config)
        kinect_streamer = KinectStreamer(device)
        kinect_available = True
        print("[INFO] Azure Kinect 연결 성공 🎉")
    except Exception as e:
        print(f"[ERROR] Azure Kinect 연결 실패: {e}")
        kinect_available = False

    print("[INFO] 웹캠(0번) 초기화 중...")
    webcam_streamer = WebCamStreamer(0)
    webcam_available = webcam_streamer.isOpened()
    if webcam_available:
         print("[INFO] 웹캠 연결 성공 🎉")
    else:
         print("[ERROR] 웹캠 연결 실패")

    # 신규 MediaPipe Task API로 Hand Landmarker 객체 생성
    print("[INFO] MediaPipe 객체 초기화 중...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    # 카메라 각각에 독립적인 객체를 사용할 수도 있지만 IMAGE 모드이므로 하나만 사용해도 무방합니다.
    landmarker = HandLandmarker.create_from_options(options)

    print("[INFO] MediaPipe 다중 카메라 실험 시작! 종료하려면 'q'를 누르세요.")
    
    pTime = 0
    try:
        while True:
            cTime = time.perf_counter()
            frames_to_display = []

            # =======================================
            # 1. Webcam 처리
            # =======================================
            if webcam_available:
                ret, web_frame = webcam_streamer.get_frame()
                if ret:
                    rgb_web = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_web)
                    results_web = landmarker.detect(mp_image)
                    
                    if results_web.hand_landmarks:
                        for hand_landmarks in results_web.hand_landmarks:
                            draw_custom_landmarks(web_frame, hand_landmarks)
                            
                            # 검지 끝(8번) 좌표 대략적으로 표시 (Pixel)
                            h, w, c = web_frame.shape
                            lm8 = hand_landmarks[8]
                            cx, cy = int(lm8.x * w), int(lm8.y * h)
                            cv2.putText(web_frame, f"Idx: ({cx},{cy})", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    cv2.putText(web_frame, 'WebCam (MediaPipe)', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    frames_to_display.append(web_frame)
                else:
                    frames_to_display.append(np.zeros((480, 640, 3), dtype=np.uint8))

            # =======================================
            # 2. Azure Kinect 처리
            # =======================================
            if kinect_available:
                ret_color, kinect_bgr, ret_depth, aligned_depth = kinect_streamer.get_frame()
                
                if ret_color:
                    rgb_kinect = cv2.cvtColor(kinect_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_kinect)
                    results_kinect = landmarker.detect(mp_image)
                    
                    if results_kinect.hand_landmarks:
                        for hand_landmarks in results_kinect.hand_landmarks:
                            draw_custom_landmarks(kinect_bgr, hand_landmarks)
                            
                            h, w, c = kinect_bgr.shape
                            lm8 = hand_landmarks[8] # 검지 끝
                            cx, cy = int(lm8.x * w), int(lm8.y * h)
                            
                            if ret_depth and 0 <= cx < w and 0 <= cy < h:
                                depth_val = aligned_depth[cy, cx]
                                if depth_val > 0:
                                    source2d = _k4a.k4a_float2_t()
                                    source2d.xy.x = float(cx)
                                    source2d.xy.y = float(cy)
                                    
                                    try:
                                        point_3d = device.calibration.convert_2d_to_3d(
                                            source_point2d=source2d, 
                                            source_depth=float(depth_val), 
                                            source_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR, 
                                            target_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR
                                        )
                                        # 밀리미터 Z값을 추적
                                        coord_text = f"Z: {point_3d.xyz.z:.0f}mm"
                                        cv2.putText(kinect_bgr, coord_text, (cx + 10, cy - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    except Exception as e:
                                        # 2D -> 3D 투영 실패 시 좌표 출력 생략
                                        pass

                    cv2.putText(kinect_bgr, 'Kinect (MediaPipe)', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    frames_to_display.append(kinect_bgr)
                else:
                    frames_to_display.append(np.zeros((720, 1280, 3), dtype=np.uint8))

            if not frames_to_display:
                print("[ERROR] 표시할 프레임이 없습니다.")
                break

            # =======================================
            # 3. 화면 합치기 및 출력
            # =======================================
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            if len(frames_to_display) == 1:
                display_img = frames_to_display[0]
            else:
                h1, w1 = frames_to_display[0].shape[:2]
                for idx in range(1, len(frames_to_display)):
                    h_next, w_next = frames_to_display[idx].shape[:2]
                    if h1 != h_next:
                        frames_to_display[idx] = cv2.resize(frames_to_display[idx], (w_next * h1 // h_next, h1))
                display_img = cv2.hconcat(frames_to_display)

            cv2.putText(display_img, f'FPS: {int(fps)}', (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe Multi-Cam Separate Test', display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] 자원 해제 중...")
        if webcam_available:
            webcam_streamer.release()
        if kinect_available:
            kinect_streamer.release()
            device.close()
        cv2.destroyAllWindows()
        landmarker.close()

if __name__ == "__main__":
    main()

import cv2
import pykinect_azure as pykinect
from ultralytics import YOLO
import time
import numpy as np
import threading
import sys

# 카메라 프레임을 백그라운드에서 계속 읽어오는 클래스 (Webcam용)
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
            try:
                self.thread.join(timeout=1.0)
            except:
                pass
            self.cap.release()

# 카메라 프레임을 백그라운드에서 계속 읽어오는 클래스 (Kinect용)
class KinectStreamer:
    def __init__(self, device):
        self.device = device
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                # 백그라운드에서 큐가 차지 않게 계속 update
                capture = self.device.update(timeout_in_ms=1000)
                if capture is None:
                    continue
                ret_color, kinect_color = capture.get_color_image()
                if ret_color:
                    bgr_img = cv2.cvtColor(kinect_color, cv2.COLOR_BGRA2BGR)
                    with self.lock:
                        self.latest_frame = bgr_img
            except Exception as e:
                # pykinect에서 예외가 발생할 경우 무시 (SDK timeout 등 보호)
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            try:
                self.thread.join(timeout=1.0)
            except:
                pass

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

    print("[INFO] YOLOv8-Hand-Pose 모델 로드 중...")
    # YOLOv8-Hand-Pose 모델 로드 (다운로드 받은 손 관절 전용 모델)
    # 손의 상세 키포인트를 추출합니다.
    yolo_model = YOLO('yolov8-hand-pose.pt') 

    print("[INFO] YOLOv8-Hand-Pose 다중 카메라 실험 시작! 추출되는 손 관절을 확인합니다. 종료: 'q'")
    
    pTime = 0
    try:
        while True:
            cTime = time.perf_counter()
            frames_to_display = []

            # 1. Webcam 프레임 읽기 및 YOLO-Pose 처리
            if webcam_available:
                ret, web_frame = webcam_streamer.get_frame()
                if ret:
                    # stream=False로 동기적 예측 
                    results_web = yolo_model(web_frame, stream=False, verbose=False)
                    
                    # 결과를 통해 프레임 자체에 Bounding Box + Keypoints(뼈대)를 그립니다.
                    web_res_frame = results_web[0].plot()
                    
                    # (선택) 개별 관절 좌표(Keypoints) 정보 출력 예시
                    # for r in results_web:
                    #     if r.keypoints is not None:
                    #         keypoints = r.keypoints.xy.cpu().numpy() # [Nx17x2]
                    #         # 여기서 각 관절 데이터를 추가적으로 활용 가능합니다.
                    
                    cv2.putText(web_res_frame, 'WebCam (YOLOv8-Hand-Pose)', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                    frames_to_display.append(web_res_frame)
                else:
                    frames_to_display.append(np.zeros((480, 640, 3), dtype=np.uint8))

            # 2. Azure Kinect 프레임 읽기 및 YOLO-Pose 처리
            if kinect_available:
                ret_color, kinect_bgr = kinect_streamer.get_frame()
                if ret_color:
                    results_kinect = yolo_model(kinect_bgr, stream=False, verbose=False)
                    
                    # Bounding Box와 Keypoints를 함께 렌더링
                    kinect_res_frame = results_kinect[0].plot()
                    
                    cv2.putText(kinect_res_frame, 'Kinect (YOLOv8-Hand-Pose)', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    frames_to_display.append(kinect_res_frame)
                else:
                    frames_to_display.append(np.zeros((720, 1280, 3), dtype=np.uint8))

            if not frames_to_display:
                print("[ERROR] 표시할 프레임이 없습니다.")
                break

            # FPS 연산
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            # 여러 프레임 좌우로 붙이기
            if len(frames_to_display) == 1:
                display_img = frames_to_display[0]
            else:
                h1, w1 = frames_to_display[0].shape[:2]
                for idx in range(1, len(frames_to_display)):
                    h_next, w_next = frames_to_display[idx].shape[:2]
                    if h1 != h_next:
                        # 첫번째 영상 세로 높이에 맞추기
                        frames_to_display[idx] = cv2.resize(frames_to_display[idx], (w_next * h1 // h_next, h1))
                display_img = cv2.hconcat(frames_to_display)

            cv2.putText(display_img, f'FPS: {int(fps)}', (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('YOLOv8-Hand-Pose Multi-Cam Test', display_img)

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

if __name__ == "__main__":
    main()

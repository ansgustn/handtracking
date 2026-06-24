import cv2
import math
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

import pykinect_azure as pykinect
import mediapipe as mp
from rtmlib import Hand

import torch
from torchvision import transforms
from PIL import Image

# FreiHAND 모델 경로 등록
sys.path.append(os.path.abspath('HandTracking-master'))
from model import FreiHANDModel  # type: ignore

def calc_angle(pt1, pt2):
    """ 두 점 사이의 Z축 회전(기울기) 계산 """
    return math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

class CameraThread(threading.Thread):
    def __init__(self, cam_type, device_index, cam_name, freihand_model=None, freihand_transform=None, torch_device='cpu'):
        super().__init__()
        self.cam_type = cam_type
        self.device_index = device_index
        self.cam_name = cam_name
        self.running = True
        
        self.current_frame = None
        self.angles = {"MP": np.nan, "RTM": np.nan, "Frei": np.nan}
        self.camera_ready = False
        
        # [NEW] 현재 작동할 모델
        self.active_model = "MP" 
        
        # ── 1. 카메라 초기화 ──
        try:
            if self.cam_type == 'webcam':
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼를 1로 설정하여 밀림 현상(프레임 드랍) 방지
                if not self.cap.isOpened():
                    raise ValueError("웹캠을 열 수 없습니다.")
            elif self.cam_type == 'kinect':
                device_config = pykinect.default_configuration
                device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
                
                # [NEW] 다중 키넥트 연결 시 USB 대역폭 폭발(충돌)을 막기 위한 극한의 최적화
                device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF # 우리는 컬러 영상만 쓰므로 Depth 센서 전원을 아예 끕니다 (대역폭 절반 이상 감소)
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15 # 프레임레이트를 15fps로 낮춰 대역폭 추가 확보
                
                self.kinect = pykinect.start_device(device_index=self.device_index, config=device_config)
            self.camera_ready = True
        except Exception as e:
            print(f"[{self.cam_name}] 초기화 실패: {e}")
            self.running = False
            return
            
        # ── 2. 모델 초기화 ──
        # MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.mp_detector = HandLandmarker.create_from_options(options)
        
        # RTMPose
        rtm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rtm_detector = Hand(to_openpose=False, backend='onnxruntime', device=rtm_device)
        
        # FreiHAND
        self.freihand_model = freihand_model
        self.freihand_transform = freihand_transform
        self.torch_device = torch_device

    def run(self):
        while self.running:
            if not self.camera_ready:
                time.sleep(0.1)
                continue
                
            frame = None
            if self.cam_type == 'webcam':
                ret, frame_bgr = self.cap.read()
                if ret: 
                    frame = cv2.flip(frame_bgr, 1)
            elif self.cam_type == 'kinect':
                capture = self.kinect.update()
                ret_color, frame_bgra = capture.get_color_image()
                if ret_color:
                    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    frame = cv2.flip(frame, 1)

            if frame is None:
                time.sleep(0.01)
                continue
                
            # 비율 맞추기 (키넥트 720p를 웹캠 480p 수준으로 리사이즈)
            if frame.shape[0] != 480:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * 480 / h), 480))
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            mp_angle = np.nan
            rtm_angle = np.nan
            frei_angle = np.nan
            
            # 선택된 모델만 연산 수행
            if self.active_model == "MP":
                # --- 1. MediaPipe Inference ---
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                mp_result = self.mp_detector.detect(mp_image)
                if mp_result.hand_landmarks:
                    lms = mp_result.hand_landmarks[0]
                    # 손목(0)과 중지 뿌리(9)를 사용하여 중심축 회전 각도 계산
                    mp_angle = calc_angle((lms[0].x * w, lms[0].y * h), (lms[9].x * w, lms[9].y * h))
                    cv2.circle(frame, (int(lms[0].x*w), int(lms[0].y*h)), 5, (255,0,0), -1)
                    cv2.circle(frame, (int(lms[9].x*w), int(lms[9].y*h)), 5, (255,0,0), -1)
                    
            elif self.active_model == "RTM":
                # --- 2. RTMPose Inference ---
                keypoints_all, scores_all = self.rtm_detector(frame)
                if keypoints_all is not None and len(keypoints_all) > 0:
                    kps = keypoints_all[0]
                    scs = scores_all[0]
                    # 손목(0)과 중지 뿌리(9)를 사용하여 중심축 회전 각도 계산
                    if scs[0] > 0.3 and scs[9] > 0.3:
                        rtm_angle = calc_angle(kps[0], kps[9])
                        cv2.circle(frame, (int(kps[0][0]), int(kps[0][1])), 6, (0,255,0), 2)
                        cv2.circle(frame, (int(kps[9][0]), int(kps[9][1])), 6, (0,255,0), 2)
                        
            elif self.active_model == "Frei":
                # --- 3. FreiHAND Inference ---
                if self.freihand_model is not None:
                    # 간단한 중심점 추정
                    keypoints_all, scores_all = self.rtm_detector(frame)
                    center_x, center_y = None, None
                    if keypoints_all is not None and len(keypoints_all) > 0:
                        center_x = int((keypoints_all[0][0][0] + keypoints_all[0][9][0])/2)
                        center_y = int((keypoints_all[0][0][1] + keypoints_all[0][9][1])/2)
                    else:
                        center_x, center_y = w//2, h//2
                        
                    if center_x is not None:
                        box_size = 100
                        x1, y1 = max(0, center_x - box_size), max(0, center_y - box_size)
                        x2, y2 = min(w, center_x + box_size), min(h, center_y + box_size)
                        if x2 - x1 > 20 and y2 - y1 > 20:
                            cropped = rgb_frame[y1:y2, x1:x2]
                            try:
                                pil_img = Image.fromarray(cropped)
                                input_tensor = self.freihand_transform(pil_img).unsqueeze(0).to(self.torch_device)
                                with torch.no_grad():
                                    outputs = self.freihand_model(input_tensor)
                                lm_3d = outputs.view(21, 3).cpu().numpy()
                                lm_3d[:, 1] = -lm_3d[:, 1] # Y 반전
                                frei_angle = calc_angle((lm_3d[0,0], lm_3d[0,1]), (lm_3d[9,0], lm_3d[9,1]))
                            except:
                                pass
            
            # --- 텍스트 오버레이 ---
            y_offset = 30
            cv2.putText(frame, f"[{self.cam_name}]", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            if self.active_model == "MP":
                cv2.putText(frame, f"MP: {mp_angle:.1f}" if not np.isnan(mp_angle) else "MP: NaN", (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,50,50), 2)
            elif self.active_model == "RTM":
                cv2.putText(frame, f"RTM: {rtm_angle:.1f}" if not np.isnan(rtm_angle) else "RTM: NaN", (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
            elif self.active_model == "Frei" and self.freihand_model is not None:
                cv2.putText(frame, f"Frei: {frei_angle:.1f}" if not np.isnan(frei_angle) else "Frei: NaN", (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,255), 2)

            self.angles = {"MP": mp_angle, "RTM": rtm_angle, "Frei": frei_angle}
            self.current_frame = frame
            
        # 스레드 종료 시 카메라 자원 안전하게 해제 (메인 스레드에서 강제 종료 시 충돌 방지)
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'kinect'):
            self.kinect.close()

    def stop(self):
        self.running = False

def main():
    print("========================================")
    print("시스템에 연결된 카메라(UVC)를 탐색합니다...")
    available_cams = []
    # 0번부터 4번까지 카메라 인덱스를 스캔합니다.
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cams.append(i)
            cap.release()
            
    print(f"발견된 카메라 인덱스: {available_cams} (총 {len(available_cams)}대)")
    if len(available_cams) == 0:
        print("연결된 카메라가 없습니다. 종료합니다.")
        return

    print("========================================")
    print("모델 초기화 중입니다...")
    
    freihand_model = None
    freihand_transform = None
    model_path = os.path.join("HandTracking-master", "freihand_custom_model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(model_path):
        try:
            freihand_model = FreiHANDModel(num_keypoints=21).to(device)
            freihand_model.load_state_dict(torch.load(model_path, map_location=device))
            freihand_model.eval()
            freihand_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ FreiHAND 가중치 로드 완료!")
        except Exception as e:
            print(f"⚠️ FreiHAND 로드 실패: {e}")
    else:
        print("⚠️ FreiHAND 모델 가중치가 없어 측정에서 제외됩니다.")

    cam_configs = []
    for idx, cam_index in enumerate(available_cams):
        name = f"Camera_{idx+1}"
        cam_configs.append({'type': 'webcam', 'idx': cam_index, 'name': name})
    
    threads = []
    for cfg in cam_configs:
        t = CameraThread(cfg['type'], cfg['idx'], cfg['name'], freihand_model, freihand_transform, device)
        threads.append(t)
        
    for t in threads:
        t.start()
        
    print("\n[안내] 3개의 카메라 준비를 기다립니다. (창이 뜰 때까지 대기)")
    time.sleep(3) # 카메라 워밍업

    stages = ["MP", "RTM"]
    if freihand_model is not None:
        stages.append("Frei")
        
    target_angles = [90, 180, 270, 360]
    # 스냅샷 저장소: snapshot_data[model][target][cam]
    snapshot_data = {
        model: {
            target: {cfg['name']: np.nan for cfg in cam_configs} 
            for target in target_angles
        } for model in stages
    }

    for stage_idx, stage_model in enumerate(stages):
        print(f"\n========================================")
        print(f"[Stage {stage_idx+1}/{len(stages)}] '{stage_model}' 모델 측정을 준비합니다.")
        print(">>> 's' 키: 카메라 영점(0도) 맞추기")
        print(">>> '1', '2', '3', '4' 키: 각각 90, 180, 270, 360도에 손을 맞추고 누르면 스냅샷 저장!")
        print(">>> 'n' 키: 다음 모델로 넘어가기")
        
        # 스레드에 현재 연산할 모델 지정
        for t in threads:
            t.active_model = stage_model
            
        baseline_angles = {t.cam_name: 0.0 for t in threads}
        quit_requested = False
        
        while True:
            frames = []
            
            for t in threads:
                if t.current_frame is not None:
                    frames.append(t.current_frame)
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    
            if frames:
                display = np.hstack(frames)
                if display.shape[1] > 1920:
                    display = cv2.resize(display, (display.shape[1]//2, display.shape[0]//2))
                    
                # 상단 안내 텍스트
                cv2.putText(display, f"Stage {stage_idx+1}/{len(stages)} : {stage_model} Only", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(display, "Press 's' to Zero, 'n' to Next Stage", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, "[Keys] 1:90  2:180  3:270  4:360", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 스냅샷 저장 현황 표시
                y_pos = 160
                for tgt in target_angles:
                    cam1_val = snapshot_data[stage_model][tgt][cam_configs[0]['name']]
                    status = "Recorded" if not np.isnan(cam1_val) else "Empty"
                    color = (0, 255, 0) if status == "Recorded" else (0, 0, 255)
                    cv2.putText(display, f"{tgt} deg: {status}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_pos += 30
                
                cv2.imshow('Multi-Cam Model-Sequential Evaluator', display)

            key = cv2.waitKey(1) & 0xFF
            
            target_to_record = None
            if key == ord('1'): target_to_record = 90
            elif key == ord('2'): target_to_record = 180
            elif key == ord('3'): target_to_record = 270
            elif key == ord('4'): target_to_record = 360
            elif key == ord('q'):
                quit_requested = True
                break
            elif key == ord('n'):
                break
            elif key == ord('s'):
                for t in threads:
                    baseline_angles[t.cam_name] = t.angles[stage_model] if not np.isnan(t.angles[stage_model]) else 0.0
                print(f">>> {stage_model} 모델의 영점이 조절되었습니다.")
                
            if target_to_record is not None:
                for t in threads:
                    raw_angle = t.angles[stage_model]
                    # 각도 언랩 및 영점 적용
                    if not np.isnan(raw_angle):
                        angle = raw_angle - baseline_angles[t.cam_name]
                        # 음수 역방향 회전도 쉽게 보기 위해 절댓값 취할 수 있지만 원본 각도를 우선 저장
                        snapshot_data[stage_model][target_to_record][t.cam_name] = angle
                    else:
                        snapshot_data[stage_model][target_to_record][t.cam_name] = np.nan
                print(f">>> {stage_model} 모델 - {target_to_record}도 스냅샷 측정 완료!")

        if quit_requested:
            break

    print("\n측정을 종료합니다. 스레드를 닫습니다...")
    for t in threads:
        t.stop()
    for t in threads:
        t.join()
    cv2.destroyAllWindows()

    # ── CSV 데이터 생성 ──
    csv_rows = []
    for model in stages:
        for target in target_angles:
            row = {"Model": model, "TargetAngle": target}
            for cfg in cam_configs:
                cam = cfg['name']
                row[cam] = snapshot_data[model][target][cam]
            csv_rows.append(row)
            
    final_df = pd.DataFrame(csv_rows)
    final_df.to_csv("multicam_evaluation_snapshot.csv", index=False)
    print("✅ 데이터가 'multicam_evaluation_snapshot.csv'에 저장되었습니다.")

    # ── 통합 막대 그래프 그리기 (오차 기준, 카메라 평균) ──
    # 한글 폰트 깨짐 방지
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 7))
    x = np.arange(len(target_angles))
    width = 0.25  # 막대 3개 (모델별 1개)
    
    bar_colors = {"MP": "#1f77b4", "RTM": "#ff7f0e", "Frei": "#2ca02c"}
    
    multiplier = 0
    for model in stages:
        bars = []
        for target in target_angles:
            errors = []
            for cfg in cam_configs:
                val = snapshot_data[model][target][cfg['name']]
                if not np.isnan(val):
                    # 오차 계산: 절댓값 오차 (음수 오차도 양수로 변환)
                    error = abs(abs(val) - target)
                    errors.append(error)
                    
            if errors:
                bars.append(np.mean(errors)) # 카메라들의 오차 평균
            else:
                bars.append(np.nan) # 모든 카메라 측정 실패
                
        offset = width * multiplier
        
        # NaN은 그려지지 않음
        rects = plt.bar(x + offset, bars, width, 
                        label=model, 
                        color=bar_colors.get(model),
                        edgecolor='white', alpha=0.9)
        
        # 막대 위에 오차 라벨 표시
        labels = ["Fail" if np.isnan(v) else f"{v:.1f}°" for v in bars]
        plt.bar_label(rects, labels=labels, padding=3, fontsize=10)
        
        multiplier += 1

    # 기준선 (오차 0) 강조
    plt.axhline(0, color='black', linewidth=2)

    plt.title("측정값 (Measurement Error)", fontsize=16, fontweight='bold')
    plt.xlabel("Target Angle")
    plt.ylabel("Absolute Measurement Error (Degree)")
    # X축 눈금을 중앙(가운데 막대)으로 맞춤
    plt.xticks(x + width, [f"{t}°" for t in target_angles])
    
    # 범례 위치
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Model")
    plt.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    
    # y축 최소값을 0으로 설정 (모든 오차가 양수이므로)
    ax = plt.gca()
    ax.set_ylim(0, max(20, ax.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    plt.savefig("multicam_evaluation_snapshot_error_bar.png", dpi=300)
    print("✅ 카메라 평균 오차 분석 그래프가 'multicam_evaluation_snapshot_error_bar.png'에 저장되었습니다.")

if __name__ == "__main__":
    main()

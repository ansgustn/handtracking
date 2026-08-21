import cv2
import math
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import socket
import json

from pythonosc.udp_client import SimpleUDPClient

import pykinect_azure as pykinect
import mediapipe as mp
from rtmlib import Hand

import torch
from torchvision import transforms
from PIL import Image

# =========================================================================
# [Unreal Engine OSC / UDP 실시간 통신 설정 (mediapipe_knob_osc.py 기반)]
# =========================================================================
ENABLE_OSC = True
UE_IP = "20.30.173.208"               # 👈 옆 컴퓨터(언리얼 엔진 실행 PC) IPv4 주소 (로컬이면 127.0.0.1)
UE_PORT = 8000                        # 언리얼 엔진 OSC 플러그인 수신 포트
OSC_ADDRESS = "/mediapipe/knob/angle" # BP_Knob이 수신하는 메인 OSC Address
PINCH_WHEN_NO_HAND = 1.5              # 손이 감지되지 않을 때 전송할 핀치값 (언리얼 Release 유도)

# FreiHAND 모델 경로 등록
sys.path.append(os.path.abspath('HandTracking-master'))
from model import FreiHANDModel  # type: ignore

def calc_angle(pt1, pt2):
    """ 두 점 사이의 Z축 회전(기울기) 계산 """
    return math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

def pinch_ratio_landmarks(pt_thumb, pt_index, pt_wrist, pt_middle):
    """
    엄지 끝 - 검지 끝 거리를 손목 - 중지관절 기준 길이로 정규화 (mediapipe_knob_osc.py 기반)
    0.2 ~ 0.3 : Grab(잡음) / 0.7 이상 : Release(놓음)
    """
    dist = math.hypot(pt_index[0] - pt_thumb[0], pt_index[1] - pt_thumb[1])
    ref = math.hypot(pt_middle[0] - pt_wrist[0], pt_middle[1] - pt_wrist[1])
    if ref < 1e-6:
        return PINCH_WHEN_NO_HAND
    return dist / ref


class CameraThread(threading.Thread):
    def __init__(self, cam_type, device_index, cam_name):
        super().__init__()
        self.cam_type = cam_type
        self.device_index = device_index
        self.cam_name = cam_name
        self.running = True
        
        self.current_frame = None
        self.camera_ready = False
        
        # ── 1. 카메라 초기화 ──
        try:
            if self.cam_type == 'webcam':
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                if self.device_index == 1:
                    # Azure Kinect 720p
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    raise ValueError(f"카메라(인덱스 {self.device_index})를 열 수 없습니다.")
            elif self.cam_type == 'kinect':
                device_config = pykinect.default_configuration
                device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
                device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
                
                try:
                    self.kinect = pykinect.start_device(device_index=self.device_index, config=device_config)
                    print(f"[{self.cam_name}] Azure Kinect SDK 장치 시작 성공!")
                except Exception as kinect_err:
                    print(f"⚠️ [{self.cam_name}] Kinect SDK 시작 실패({kinect_err}), OpenCV UVC(인덱스 1)로 자동 전환합니다...")
                    self.cam_type = 'webcam'
                    self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not self.cap.isOpened():
                        raise ValueError("Kinect UVC 웹캠 모드로도 열 수 없습니다.")
            self.camera_ready = True
        except Exception as e:
            print(f"❌ [{self.cam_name}] 초기화 실패: {e}")
            self.running = False
            return
            
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
                try:
                    capture = self.kinect.update()
                    ret_color, frame_bgra = capture.get_color_image()
                    if ret_color:
                        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                        frame = cv2.flip(frame, 1)
                except Exception as e:
                    time.sleep(0.05)
                    continue

            if frame is None:
                time.sleep(0.01)
                continue
                
            # 비율 맞추기 (키넥트 720p를 웹캠 480p 수준으로 리사이즈)
            if frame.shape[0] != 480:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * 480 / h), 480))
                
            self.current_frame = frame
            
        # 스레드 종료 시 카메라 자원 안전하게 해제
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'kinect') and self.kinect is not None:
            try:
                self.kinect.close()
            except:
                pass

    def stop(self):
        self.running = False

def main():
    print("========================================")
    cam_configs = []
    
    # =========================================================================
    # [카메라 인덱스 설정]
    # - Azure Kinect 컬러 카메라는 UVC 인덱스 1 (1280x720)로 안정적으로 인식됩니다.
    # - 일반 웹캠은 UVC 인덱스 0 (640x480)으로 인식됩니다.
    # =========================================================================
    # 키넥트 카메라 (UVC 인덱스 1)
    cam_configs.append({'type': 'webcam', 'idx': 1, 'name': 'Camera_1 (Kinect)'})
    # 일반 웹캠 (UVC 인덱스 0)
    cam_configs.append({'type': 'webcam', 'idx': 0, 'name': 'Camera_2 (Webcam)'})
            
    print(f"최종 할당된 카메라 목록:")
    for cfg in cam_configs:
        print(f" - {cfg['name']}: {cfg['type']} (idx={cfg['idx']})")

    if len(cam_configs) == 0:
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

    # cam_configs는 위에서 이미 최적화된 상태로 설정됨
    
    print("단일 추론기(MediaPipe, RTMPose)를 초기화합니다...")
    # MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    mp_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    mp_detector = HandLandmarker.create_from_options(mp_options)
    
    # RTMPose
    rtm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rtm_detector = Hand(to_openpose=False, backend='onnxruntime', device=rtm_device)

    threads = []
    for cfg in cam_configs:
        t = CameraThread(cfg['type'], cfg['idx'], cfg['name'])
        threads.append(t)
        
    for t in threads:
        t.start()
        
    print("\n[안내] 3개의 카메라 준비를 기다립니다. (창이 뜰 때까지 대기)")
    time.sleep(3) # 카메라 워밍업

    # Unreal Engine OSC 클라이언트 준비 (mediapipe_knob_osc.py 기반)
    osc_client = None
    if ENABLE_OSC:
        try:
            osc_client = SimpleUDPClient(UE_IP, UE_PORT)
            print(f"📡 Unreal Engine OSC 스트리밍 활성화 -> {UE_IP}:{UE_PORT} (Address: {OSC_ADDRESS})")
        except Exception as e:
            print(f"⚠️ OSC 클라이언트 생성 실패: {e}")

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
        
        baseline_angles = {t.cam_name: 0.0 for t in threads}
        current_angles = {t.cam_name: {stage_model: np.nan} for t in threads}
        last_sent_angle = 0.0
        quit_requested = False
        
        while True:
            frames = []
            primary_detected_angle = None
            primary_detected_pinch = PINCH_WHEN_NO_HAND
            
            for t in threads:
                if t.current_frame is not None:
                    frame = t.current_frame.copy()
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    
                    angle = np.nan
                    pinch = PINCH_WHEN_NO_HAND
                    
                    if stage_model == "MP":
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        mp_result = mp_detector.detect(mp_image)
                        if mp_result.hand_landmarks:
                            lms = mp_result.hand_landmarks[0]
                            pt_wrist = (lms[0].x * w, lms[0].y * h)
                            pt_thumb = (lms[4].x * w, lms[4].y * h)
                            pt_index = (lms[8].x * w, lms[8].y * h)
                            pt_middle = (lms[9].x * w, lms[9].y * h)
                            
                            angle = calc_angle(pt_wrist, pt_middle)
                            pinch = pinch_ratio_landmarks(pt_thumb, pt_index, pt_wrist, pt_middle)
                            
                            # 시각화 (손목, 중지, 엄지-검지 핀치 라인)
                            cv2.circle(frame, (int(pt_wrist[0]), int(pt_wrist[1])), 5, (255,0,0), -1)
                            cv2.circle(frame, (int(pt_middle[0]), int(pt_middle[1])), 5, (255,0,0), -1)
                            cv2.line(frame, (int(pt_thumb[0]), int(pt_thumb[1])), (int(pt_index[0]), int(pt_index[1])), (0, 255, 255), 2)
                            cv2.circle(frame, (int(pt_thumb[0]), int(pt_thumb[1])), 5, (0, 255, 255), -1)
                            cv2.circle(frame, (int(pt_index[0]), int(pt_index[1])), 5, (0, 255, 255), -1)
                            
                    elif stage_model == "RTM":
                        keypoints_all, scores_all = rtm_detector(frame)
                        if keypoints_all is not None and len(keypoints_all) > 0:
                            kps = keypoints_all[0]
                            scs = scores_all[0]
                            if scs[0] > 0.3 and scs[9] > 0.3:
                                pt_wrist, pt_middle = kps[0], kps[9]
                                pt_thumb, pt_index = kps[4], kps[8]
                                angle = calc_angle(pt_wrist, pt_middle)
                                pinch = pinch_ratio_landmarks(pt_thumb, pt_index, pt_wrist, pt_middle)
                                
                                cv2.circle(frame, (int(kps[0][0]), int(kps[0][1])), 6, (0,255,0), 2)
                                cv2.circle(frame, (int(kps[9][0]), int(kps[9][1])), 6, (0,255,0), 2)
                                cv2.line(frame, (int(pt_thumb[0]), int(pt_thumb[1])), (int(pt_index[0]), int(pt_index[1])), (0, 255, 255), 2)
                                
                    elif stage_model == "Frei":
                        if freihand_model is not None:
                            keypoints_all, scores_all = rtm_detector(frame)
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
                                        input_tensor = freihand_transform(pil_img).unsqueeze(0).to(device)
                                        with torch.no_grad():
                                            outputs = freihand_model(input_tensor)
                                        lm_3d = outputs.view(21, 3).cpu().numpy()
                                        lm_3d[:, 1] = -lm_3d[:, 1]
                                        angle = calc_angle((lm_3d[0,0], lm_3d[0,1]), (lm_3d[9,0], lm_3d[9,1]))
                                        pinch = pinch_ratio_landmarks(
                                            (lm_3d[4,0], lm_3d[4,1]), (lm_3d[8,0], lm_3d[8,1]),
                                            (lm_3d[0,0], lm_3d[0,1]), (lm_3d[9,0], lm_3d[9,1])
                                        )
                                    except:
                                        pass
                    
                    # --- 텍스트 오버레이 ---
                    y_offset = 30
                    cv2.putText(frame, f"[{t.cam_name}]", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    color = (255,255,255)
                    if stage_model == "MP": color = (255,50,50)
                    elif stage_model == "RTM": color = (50,255,50)
                    elif stage_model == "Frei": color = (50,50,255)
                    
                    cv2.putText(frame, f"{stage_model}: {angle:.1f}" if not np.isnan(angle) else f"{stage_model}: NaN", (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Pinch: {pinch:.2f}", (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    current_angles[t.cam_name][stage_model] = angle
                    frames.append(frame)
                    
                    # 첫 번째로 손이 잡힌 카메라(또는 Camera_1)의 값을 언리얼 전송용 메인 데이터로 사용
                    if not np.isnan(angle) and primary_detected_angle is None:
                        calibrated_main = angle - baseline_angles.get(t.cam_name, 0.0)
                        primary_detected_angle = calibrated_main
                        primary_detected_pinch = pinch
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    
            # ── [Unreal Engine OSC 메시지 전송 (mediapipe_knob_osc.py 호환)] ──
            if osc_client is not None:
                try:
                    if primary_detected_angle is not None:
                        # 손이 감지되었을 때: [각도(0..360), 핀치 비율]
                        send_angle = float(primary_detected_angle % 360)
                        send_pinch = float(primary_detected_pinch)
                        last_sent_angle = send_angle
                        osc_client.send_message(OSC_ADDRESS, [send_angle, send_pinch])
                    else:
                        # 손이 감지되지 않았을 때: 마지막 각도 유지 + 언리얼 Release용 핀치(1.5)
                        osc_client.send_message(OSC_ADDRESS, [float(last_sent_angle), float(PINCH_WHEN_NO_HAND)])
                except Exception as e:
                    pass
                    
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
                    baseline_angles[t.cam_name] = current_angles[t.cam_name][stage_model] if not np.isnan(current_angles[t.cam_name][stage_model]) else 0.0
                print(f">>> {stage_model} 모델의 영점이 조절되었습니다.")
                
            if target_to_record is not None:
                for t in threads:
                    raw_angle = current_angles[t.cam_name][stage_model]
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

    # ── 통합 막대 그래프 그리기 (오차 기준, 카메라 중앙값) ──
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
                bars.append(np.median(errors)) # 카메라들의 오차 중앙값
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

"""
cup_rtmpose_dualcam.py
MediaPipe(Tasks API) → RTMPose(rtmlib) 변환 버전 - 듀얼 카메라

[변경 사항]
- MediaPipe HandLandmarker → rtmlib Hand (RTMPose 기반)
- mp.Image 래퍼 / 타임스탬프 처리 불필요 → BGR 프레임 직접 입력
- 랜드마크 좌표: 정규화(0~1) → 픽셀 좌표 직접 사용
- 각도 계산: Z depth(MediaPipe 전용) 사용 불가
  → palm line(kp5 검지뿌리 → kp17 소지뿌리) 2D 각도로 교체
    (cup_mediapipe_wholehand.py 와 동일한 방식 / 손끝보다 안정적)
- EMA 스무딩: 정규화 좌표 → 픽셀 좌표 기준으로 리셋 임계값 조정 (0.05 → 32px)
- cv2.CAP_DSHOW 유지 (Windows 팀원 호환)

[설치]
  pip install rtmlib onnxruntime
"""

import cv2
import math
import numpy as np
import threading
import time

from rtmlib import Hand

# ── 뼈대 연결 정보 ──────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_custom_landmarks(image, keypoints, scores, score_thr=0.3):
    """RTMPose 키포인트 그리기 (pixel 좌표 직접 사용)"""
    pts = [(int(kp[0]), int(kp[1])) for kp in keypoints]
    for pt, sc in zip(pts, scores):
        if sc > score_thr:
            cv2.circle(image, pt, 3, (0, 0, 255), cv2.FILLED)
    for a, b in HAND_CONNECTIONS:
        if scores[a] > score_thr and scores[b] > score_thr:
            cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)


# ── 카메라 스레드 ───────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, src, cap_idx):
        super().__init__()
        # 카메라 초기화 (Windows: CAP_DSHOW / Mac: 인자 제거)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_idx = cap_idx

        # 스레드별 독립 RTMPose 인스턴스
        self.hand_detector = Hand(
            to_openpose=False,
            backend='onnxruntime',
            device='cpu'
        )

        self.baseline_angle = None
        self.current_frame  = None
        self.abs_angle      = None
        self.rotation       = None

        # EMA 스무딩 파라미터
        self.alpha_angle = 0.15   # 각도 스무딩 강도
        self.alpha_lm    = 0.15   # 랜드마크 스무딩 강도
        self.smoothed_angle     = None
        self.smoothed_kps       = None  # shape: (21, 2) — 픽셀 좌표

        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            cv2.putText(frame, f"Cam {self.cap_idx}", (w - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # ── RTMPose 추론 ──────────────────────────────────────────────────
            keypoints_all, scores_all = self.hand_detector(frame)

            abs_angle = None

            if keypoints_all is not None and len(keypoints_all) > 0:
                kps = keypoints_all[0]   # (21, 2) — 픽셀 좌표
                scs = scores_all[0]      # (21,)

                # ── 1. 랜드마크 EMA 스무딩 ───────────────────────────────────
                if self.smoothed_kps is None:
                    self.smoothed_kps = kps.copy().astype(float)
                else:
                    new_kps = self.alpha_lm * kps + (1 - self.alpha_lm) * self.smoothed_kps
                    # 손이 빠르게 이동하면 스무딩 리셋 (0번 손목 기준, 32px ≈ 정규화 0.05)
                    if np.linalg.norm(kps[0] - self.smoothed_kps[0]) > 32:
                        self.smoothed_kps = kps.copy().astype(float)
                    else:
                        self.smoothed_kps = new_kps

                draw_custom_landmarks(frame, self.smoothed_kps, scs)

                # ── 2. 기준 랜드마크: 5(검지 뿌리) → 17(소지 뿌리) palm line ─
                #    Z depth 없이 2D로 손등 방향을 가장 안정적으로 표현
                index_mcp_x = int(self.smoothed_kps[5][0])
                index_mcp_y = int(self.smoothed_kps[5][1])
                pinky_mcp_x = int(self.smoothed_kps[17][0])
                pinky_mcp_y = int(self.smoothed_kps[17][1])

                cv2.line(frame,
                         (index_mcp_x, index_mcp_y),
                         (pinky_mcp_x, pinky_mcp_y),
                         (255, 0, 255), 4)
                cv2.circle(frame, (index_mcp_x, index_mcp_y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (pinky_mcp_x, pinky_mcp_y), 8, (0, 255, 255), -1)

                # ── 3. 2D Palm 각도 계산 ─────────────────────────────────────
                dx = self.smoothed_kps[5][0] - self.smoothed_kps[17][0]
                dy = self.smoothed_kps[5][1] - self.smoothed_kps[17][1]
                raw_abs_angle = math.degrees(math.atan2(-dy, dx))

                # ── 4. 각도 EMA 스무딩 ───────────────────────────────────────
                if self.smoothed_angle is None:
                    self.smoothed_angle = raw_abs_angle
                else:
                    diff = raw_abs_angle - self.smoothed_angle
                    if diff > 180:   raw_abs_angle -= 360
                    elif diff < -180: raw_abs_angle += 360

                    self.smoothed_angle = (self.alpha_angle * raw_abs_angle
                                          + (1 - self.alpha_angle) * self.smoothed_angle)

                    if self.smoothed_angle > 180:   self.smoothed_angle -= 360
                    elif self.smoothed_angle < -180: self.smoothed_angle += 360

                abs_angle = self.smoothed_angle

                # ── 5. 기준 각도 대비 회전량 계산 ────────────────────────────
                if self.baseline_angle is not None:
                    rotation = abs_angle - self.baseline_angle
                    if rotation > 180:   rotation -= 360
                    elif rotation < -180: rotation += 360
                    self.rotation = rotation
                    cv2.putText(frame, f"Cam Rot: {rotation:.1f} deg", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    self.rotation = None

                cv2.putText(frame, f"Palm Abs: {abs_angle:.1f} deg", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else:
                # 손 미탐지 시 스무딩 초기화
                self.smoothed_angle = None
                self.smoothed_kps   = None
                self.rotation       = None
                cv2.putText(frame, "Hand Not Detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.current_frame = frame
            self.abs_angle     = abs_angle

    def set_baseline(self):
        if self.abs_angle is not None:
            self.baseline_angle = self.abs_angle
            print(f"Cam {self.cap_idx} 기준 각도 설정: {self.baseline_angle:.1f}도")

    def stop(self):
        self.running = False
        self.cap.release()


# ── CSV 저장 & 그래프 출력 ──────────────────────────────────────────────────────
def save_and_plot_results(trials):
    import csv

    if not trials:
        print("[오류] 저장할 실험 데이터가 없습니다!")
        return

    csv_filename   = "experiment_results.csv"
    graph_filename = "experiment_graph.png"

    # CSV 저장
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Trial", "Elapsed_Time_Sec", "Angle_Deg"])
            for trial_idx, data in sorted(trials.items()):
                for t, angle in data:
                    writer.writerow([trial_idx, f"{t:.3f}", f"{angle:.2f}"])
        print(f"[성공] '{csv_filename}'에 저장되었습니다.")
    except Exception as e:
        print(f"[오류] CSV 저장 실패: {e}")

    # 그래프 저장
    try:
        import matplotlib.pyplot as plt
        import platform
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
            plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 7))
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        print("\n--- 실험 통계 요약 ---")
        for i, (trial_idx, data) in enumerate(sorted(trials.items())):
            if not data:
                continue
            times  = [d[0] for d in data]
            angles = [d[1] for d in data]
            max_a, min_a = max(angles), min(angles)
            print(f"실험 {trial_idx}: {len(data)}개 | 최대 {max_a:.1f}° | 최소 {min_a:.1f}° | 변화폭 {max_a - min_a:.1f}°")
            color = colors[(trial_idx - 1) % len(colors)]
            plt.plot(times, angles,
                     label=f'실험 {trial_idx} (최대 {max_a:.1f}°)',
                     color=color, linewidth=2, alpha=0.85)

        plt.title("10회 회전 실험 각도 측정 결과", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("경과 시간 (초)", fontsize=12, labelpad=10)
        plt.ylabel("회전 각도 (도)",  fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                   borderaxespad=0, frameon=True, shadow=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        print(f"[성공] '{graph_filename}'에 그래프가 저장되었습니다.")
        plt.show()
    except ImportError:
        print("[알림] matplotlib 미설치 — pip install matplotlib 후 재시도")
    except Exception as e:
        print(f"[오류] 그래프 생성 실패: {e}")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    print("카메라를 준비 중입니다. 잠시만 기다려주세요...")
    cam1 = CameraThread(0, 1)
    cam2 = CameraThread(1, 2)
    cam1.start()
    cam2.start()

    print("['s']  두 카메라 기준 각도 0도 설정")
    print("['Space']  실험 녹화 시작/중지")
    print("['g']  CSV 저장 + 그래프 출력")
    print("['c']  데이터 초기화")
    print("['q']  종료")

    trials             = {}
    current_trial      = 1
    MAX_TRIALS         = 10
    is_recording       = False
    recording_start    = 0.0

    # 첫 프레임 대기
    while cam1.current_frame is None and cam2.current_frame is None:
        time.sleep(0.1)

    while True:
        f1 = cam1.current_frame
        f2 = cam2.current_frame

        display_frame = None
        if f1 is not None and f2 is not None:
            h1, w1 = f1.shape[:2]
            h2, w2 = f2.shape[:2]
            if h1 != h2:
                f2 = cv2.resize(f2, (int(w2 * h1 / h2), h1))
            display_frame = np.hstack((f1, f2))
            dh, dw = display_frame.shape[:2]
            if dw > 1920:
                display_frame = cv2.resize(display_frame, (dw // 2, dh // 2))
        elif f1 is not None:
            display_frame = f1
        elif f2 is not None:
            display_frame = f2

        if display_frame is not None:
            rot1, rot2 = cam1.rotation, cam2.rotation

            # 두 카메라 각도 원형 평균(Circular Mean)
            combined_rotation = None
            if rot1 is not None and rot2 is not None:
                r1, r2 = math.radians(rot1), math.radians(rot2)
                combined_rotation = math.degrees(
                    math.atan2(math.sin(r1) + math.sin(r2),
                               math.cos(r1) + math.cos(r2))
                )
            elif rot1 is not None:
                combined_rotation = rot1
            elif rot2 is not None:
                combined_rotation = rot2

            # 녹화
            elapsed = 0.0
            if is_recording:
                elapsed = time.time() - recording_start
                if combined_rotation is not None:
                    trials.setdefault(current_trial, []).append((elapsed, combined_rotation))

            # Total Rotation 표시
            if combined_rotation is not None:
                dh, dw = display_frame.shape[:2]
                text      = f"Total Rotation: {combined_rotation:.1f} deg"
                font      = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                text_x    = (dw - text_size[0]) // 2
                text_y    = dh - 40
                cv2.rectangle(display_frame,
                              (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(display_frame, text, (text_x, text_y),
                            font, 1.5, (0, 255, 255), 3)

            # ── HUD 오버레이 ──────────────────────────────────────────────────
            dh, dw = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (20, 20), (420, 160), (30, 30, 30), cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
            cv2.rectangle(display_frame, (20, 20), (420, 160), (100, 100, 100), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_recording:
                dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (50, 50, 150)
                cv2.circle(display_frame, (40, 55), 8, dot_color, -1)
                cv2.putText(display_frame, f"REC - Trial {current_trial}/{MAX_TRIALS}",
                            (65, 65), font, 0.75, (0, 0, 255), 2)
                data_count = len(trials.get(current_trial, []))
                cv2.putText(display_frame, f"Time: {elapsed:.1f}s | Pts: {data_count}",
                            (40, 105), font, 0.65, (255, 255, 255), 1)
                cv2.putText(display_frame, "Press SPACE to STOP Recording",
                            (40, 140), font, 0.55, (0, 255, 255), 1)
            else:
                if len(trials) >= MAX_TRIALS:
                    cv2.circle(display_frame, (40, 55), 8, (0, 215, 255), -1)
                    cv2.putText(display_frame, "COMPLETED 10/10 Trials",
                                (65, 65), font, 0.75, (0, 215, 255), 2)
                    cv2.putText(display_frame, "Press 'g' to PLOT | 'c' to RESET",
                                (40, 110), font, 0.65, (0, 255, 255), 2)
                else:
                    cv2.circle(display_frame, (40, 55), 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"READY - Trial {current_trial}/{MAX_TRIALS}",
                                (65, 65), font, 0.75, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press SPACE to START Recording",
                                (40, 110), font, 0.6, (200, 200, 200), 1)
                    cv2.putText(display_frame, "Press 'c' to RESET data",
                                (40, 140), font, 0.5, (150, 150, 150), 1)

            cv2.imshow('Dual Camera Rotation Tracker (RTMPose)', display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cam1.set_baseline()
            cam2.set_baseline()
        elif key == 32:  # Space
            if not is_recording:
                if current_trial <= MAX_TRIALS:
                    is_recording    = True
                    recording_start = time.time()
                    trials[current_trial] = []
                    print(f"\n>>> [실험 {current_trial}/{MAX_TRIALS}] 녹화 시작")
                else:
                    print(f"[경고] {MAX_TRIALS}회 완료. 'g'로 저장 또는 'c'로 초기화")
            else:
                is_recording = False
                pts = len(trials.get(current_trial, []))
                print(f">>> [실험 {current_trial}/{MAX_TRIALS}] 완료 — {pts}개 수집")
                if current_trial < MAX_TRIALS:
                    current_trial += 1
                else:
                    current_trial = MAX_TRIALS + 1
                    print("🎉 10회 완료! 'g'를 눌러 저장/그래프 출력")
        elif key == ord('g'):
            if trials:
                save_and_plot_results(trials)
            else:
                print("[오류] 기록된 데이터가 없습니다.")
        elif key == ord('c'):
            trials.clear()
            current_trial = 1
            is_recording  = False
            print("🧹 데이터 초기화 완료")

    cam1.stop()
    cam2.stop()
    cam1.join()
    cam2.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

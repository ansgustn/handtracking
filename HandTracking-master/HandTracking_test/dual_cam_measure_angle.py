import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import csv

"""
[듀얼캠 실험용 - 손가락(SVD) 버전]
measure_angle.py의 SVD 기반 3D 회전 추정(엄지/검지/중지) 로직을 듀얼카메라 멀티스레드 구조로 확장한 파일.

- 각 카메라마다 독립된 MediaPipe HandLandmarker(VIDEO 모드) 스레드 실행
- 'G' 키: 두 카메라 모두 현재 손가락 포즈를 0도 기준(grab)으로 저장
- 'R' 키: 기준 해제
- SPACE 키: 실험(Trial) 녹화 시작/종료 (최대 10회)
- 'P' 키: 수집한 데이터 CSV 저장 + 그래프 시각화
- 'C' 키: 실험 데이터 초기화
- 'Q' / ESC: 종료

두 카메라의 총 회전각(Total Rotation)은 원형 평균(Circular Mean)으로 합성.
"""

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SMOOTH_ALPHA = 0.4  # 랜드마크(3점) EMA 스무딩 강도
MAX_TRIALS = 10


def estimate_rotation(initial_points, current_points):
    """SVD를 이용한 3D 회전 행렬 계산 (measure_angle.py와 동일)"""
    centroid_init = np.mean(initial_points, axis=0)
    centroid_curr = np.mean(current_points, axis=0)
    A = initial_points - centroid_init
    B = current_points - centroid_curr
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def get_total_rotation_angle(R):
    """회전 행렬에서 전체 회전량(0~180도)을 계산"""
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def rotation_matrix_to_euler_angles(R):
    """회전 행렬을 Roll, Pitch, Yaw(degree)로 변환"""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z]))


class CameraThread(threading.Thread):
    def __init__(self, src, cap_idx):
        super().__init__()
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_idx = cap_idx

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.grabbed_points_3d = None
        self.smoothed_points_3d = None
        self.smoothed_points_2d = None

        self.current_frame = None
        self.total_angle = None  # 메인 루프에서 합성에 사용
        self.angles = None

        self.last_timestamp_ms = -1
        self.running = True

    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms

            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            h, w, _ = frame.shape

            cv2.putText(frame, f"Cam {self.cap_idx}", (w - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            local_total_angle = None
            local_angles = None

            if results.hand_world_landmarks and results.hand_landmarks:
                lm_3d = results.hand_world_landmarks[0]
                current_points_3d = np.array([
                    [lm_3d[4].x, lm_3d[4].y, lm_3d[4].z],
                    [lm_3d[8].x, lm_3d[8].y, lm_3d[8].z],
                    [lm_3d[12].x, lm_3d[12].y, lm_3d[12].z]
                ])

                lm_2d = results.hand_landmarks[0]
                points_2d = [(int(lm_2d[idx].x * w), int(lm_2d[idx].y * h)) for idx in [4, 8, 12]]

                # EMA 스무딩 (3점 좌표)
                if self.smoothed_points_3d is None:
                    self.smoothed_points_3d = current_points_3d.copy()
                    self.smoothed_points_2d = np.array(points_2d, dtype=float)
                else:
                    self.smoothed_points_3d = SMOOTH_ALPHA * current_points_3d + (1 - SMOOTH_ALPHA) * self.smoothed_points_3d
                    self.smoothed_points_2d = SMOOTH_ALPHA * np.array(points_2d) + (1 - SMOOTH_ALPHA) * self.smoothed_points_2d

                current_points_3d = self.smoothed_points_3d
                points_2d = [(int(p[0]), int(p[1])) for p in self.smoothed_points_2d]

                for pt in points_2d:
                    cv2.circle(frame, pt, 6, (255, 0, 255), -1)

                if self.grabbed_points_3d is not None:
                    R = estimate_rotation(self.grabbed_points_3d, current_points_3d)
                    local_angles = rotation_matrix_to_euler_angles(R)
                    local_total_angle = get_total_rotation_angle(R)

                    cv2.putText(frame, f"Total Rot: {local_total_angle:.1f} deg", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"R/P/Y: {local_angles[0]:.0f}/{local_angles[1]:.0f}/{local_angles[2]:.0f}",
                                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Press 'G' to set baseline", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                self._latest_points_3d = current_points_3d
            else:
                self.smoothed_points_3d = None
                self.smoothed_points_2d = None
                cv2.putText(frame, "Hand Not Detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.current_frame = frame
            self.total_angle = local_total_angle
            self.angles = local_angles

    def set_baseline(self):
        if getattr(self, "_latest_points_3d", None) is not None:
            self.grabbed_points_3d = self._latest_points_3d.copy()
            print(f"Cam {self.cap_idx} 기준 포즈 설정 완료")

    def clear_baseline(self):
        self.grabbed_points_3d = None
        self.total_angle = None

    def stop(self):
        self.running = False
        self.cap.release()
        self.landmarker.close()


def circular_mean_deg(angles):
    """여러 각도(degree) 값을 원형 평균으로 합성"""
    radians = np.radians(angles)
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    return np.degrees(np.arctan2(sin_sum, cos_sum))


def save_and_plot_results(trials):
    if not trials:
        print("[오류] 저장할 실험 데이터가 없습니다!")
        return

    csv_filename = "dual_cam_experiment_results.csv"
    graph_filename = "dual_cam_experiment_graph.png"

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "Elapsed_Time_Sec", "Angle_Deg"])
        for trial_idx, data in sorted(trials.items()):
            for t, angle in data:
                writer.writerow([trial_idx, f"{t:.3f}", f"{angle:.2f}"])
    print(f"[성공] 실험 데이터가 '{csv_filename}'에 저장되었습니다.")

    try:
        import matplotlib.pyplot as plt
        import platform
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
            plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        print("\n--- 실험 통계 요약 ---")
        for i, (trial_idx, data) in enumerate(sorted(trials.items())):
            if not data:
                continue
            times = [item[0] for item in data]
            angles = [item[1] for item in data]
            max_ang, min_ang, mean_ang = max(angles), min(angles), np.mean(angles)
            print(f"실험 {trial_idx}: 데이터 {len(data)}개 | 최대 {max_ang:.1f}° | 최소 {min_ang:.1f}° | 변화폭 {max_ang - min_ang:.1f}°")

            color = colors[(trial_idx - 1) % len(colors)]
            plt.plot(times, angles, label=f'실험 {trial_idx} (최대 {max_ang:.1f}°)', color=color, linewidth=2, alpha=0.85)

        plt.title("듀얼캠 손가락(SVD) 회전 실험 결과", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("경과 시간 (초)", fontsize=12)
        plt.ylabel("총 회전각 (도)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, shadow=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        print(f"[성공] 그래프가 '{graph_filename}'에 저장되었습니다.")
        plt.show()
    except ImportError:
        print("[알림] matplotlib이 없어 그래프 생성을 건너뜁니다. (pip install matplotlib)")


def main():
    print("카메라를 준비 중입니다...")
    cam1 = CameraThread(0, 1)
    cam2 = CameraThread(1, 2)
    cam1.start()
    cam2.start()

    print("['G'] 기준 포즈 설정  ['R'] 기준 해제  [SPACE] 녹화 시작/종료")
    print("['P'] 결과 저장+그래프  ['C'] 데이터 초기화  ['Q'/ESC] 종료")

    trials = {}
    current_trial = 1
    is_recording = False
    recording_start_time = 0.0

    while cam1.current_frame is None and cam2.current_frame is None:
        time.sleep(0.1)

    while True:
        frame1 = cam1.current_frame
        frame2 = cam2.current_frame
        display_frame = None

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
            angle_values = [a for a in (cam1.total_angle, cam2.total_angle) if a is not None]
            combined_angle = circular_mean_deg(angle_values) if angle_values else None

            elapsed_time = 0.0
            if is_recording:
                elapsed_time = time.time() - recording_start_time
                if combined_angle is not None:
                    trials.setdefault(current_trial, []).append((elapsed_time, combined_angle))

            if combined_angle is not None:
                dh, dw = display_frame.shape[:2]
                text = f"Total Rotation: {combined_angle:.1f} deg"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                text_x = (dw - text_size[0]) // 2
                text_y = dh - 40
                cv2.rectangle(display_frame, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), cv2.FILLED)
                cv2.putText(display_frame, text, (text_x, text_y), font, 1.5, (0, 255, 255), 3)

            # HUD 패널
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (20, 20), (420, 160), (30, 30, 30), cv2.FILLED)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
            cv2.rectangle(display_frame, (20, 20), (420, 160), (100, 100, 100), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_recording:
                dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (50, 50, 150)
                cv2.circle(display_frame, (40, 55), 8, dot_color, -1)
                cv2.putText(display_frame, f"REC - Trial {current_trial}/{MAX_TRIALS}", (65, 65), font, 0.75, (0, 0, 255), 2)
                data_count = len(trials.get(current_trial, []))
                cv2.putText(display_frame, f"Time: {elapsed_time:.1f}s | Pts: {data_count}", (40, 105), font, 0.65, (255, 255, 255), 1)
                cv2.putText(display_frame, "Press SPACE to STOP Recording", (40, 140), font, 0.55, (0, 255, 255), 1)
            else:
                if len(trials) >= MAX_TRIALS:
                    cv2.circle(display_frame, (40, 55), 8, (0, 215, 255), -1)
                    cv2.putText(display_frame, f"COMPLETED {MAX_TRIALS}/{MAX_TRIALS} Trials", (65, 65), font, 0.75, (0, 215, 255), 2)
                    cv2.putText(display_frame, "Press 'P' to SAVE | 'C' to RESET", (40, 110), font, 0.65, (0, 255, 255), 2)
                else:
                    cv2.circle(display_frame, (40, 55), 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"READY - Trial {current_trial}/{MAX_TRIALS}", (65, 65), font, 0.75, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press SPACE to START Recording", (40, 110), font, 0.6, (200, 200, 200), 1)
                    cv2.putText(display_frame, "Press 'C' to RESET data", (40, 140), font, 0.5, (150, 150, 150), 1)

            cv2.imshow('Dual Camera SVD Rotation Tracker', display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('g'):
            cam1.set_baseline()
            cam2.set_baseline()
        elif key == ord('r'):
            cam1.clear_baseline()
            cam2.clear_baseline()
            print("기준 포즈 해제")
        elif key == 32:  # SPACE
            if not is_recording:
                if current_trial <= MAX_TRIALS:
                    is_recording = True
                    recording_start_time = time.time()
                    trials[current_trial] = []
                    print(f"\n>>> [실험 {current_trial}/{MAX_TRIALS}] 녹화 시작")
                else:
                    print(f"\n[경고] 이미 {MAX_TRIALS}번의 실험을 완료했습니다. 'P'로 저장하거나 'C'로 초기화하세요.")
            else:
                is_recording = False
                data_points = len(trials.get(current_trial, []))
                print(f">>> [실험 {current_trial}/{MAX_TRIALS}] 녹화 완료! 데이터 {data_points}개")
                if current_trial < MAX_TRIALS:
                    current_trial += 1
                else:
                    current_trial = MAX_TRIALS + 1
                    print("\n모든 실험이 완료되었습니다! 'P' 키로 저장하세요.")
        elif key == ord('p'):
            if trials:
                save_and_plot_results(trials)
            else:
                print("\n[오류] 기록된 데이터가 없습니다.")
        elif key == ord('c'):
            trials.clear()
            current_trial = 1
            is_recording = False
            cam1.clear_baseline()
            cam2.clear_baseline()
            print("\n실험 데이터가 초기화되었습니다.")

    cam1.stop()
    cam2.stop()
    cam1.join()
    cam2.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

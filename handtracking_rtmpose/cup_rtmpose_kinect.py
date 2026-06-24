"""
cup_rtmpose_kinect.py
RTMPose + Azure Kinect Depth 기반 컵/나사 회전 각도 측정

[기존 MediaPipe 버전 대비 핵심 변경]
- MediaPipe → RTMPose (rtmlib)
- MediaPipe 추정 Z(노이즈 심함) → Kinect 실제 Depth → 진짜 3D 좌표
- 2D 투영각 → 3D 공간에서 Palm Normal / Fingertip Centroid 계산

[3D 각도 계산 전략 — 신뢰도 기반 자동 전환]
  1순위 — 3D Fingertip Centroid  (스크류 동작 최적화)
      kp4(엄지끝) + kp8(검지끝) + kp12(중지끝) → 3D 무게중심
      무게중심 → kp4 벡터의 X-Z 평면 각도 (yaw)
      나사 위에서 손끝이 돌면 이 각도가 변함

  2순위 — 3D Palm Normal         (손바닥 방향 yaw)
      kp0(손목), kp5(검지뿌리), kp17(소지뿌리) → 외적 → 법선 벡터
      법선의 X-Z 평면 각도 = 손바닥이 향하는 방향

  3순위 — 2D fallback            (depth 값 없을 때)
      depth=0 픽셀이 많을 때 자동 전환

[Kinect 데이터 흐름]
  Color(BGRA) + Depth(NFOV_UNBINNED)
      → get_transformed_depth_image() : Depth를 Color 시점으로 정렬
      → RTMPose : Color 이미지에서 2D 키포인트 (px, py) 추출
      → convert_2d_to_3d() : (px, py, depth_mm) → (X, Y, Z) mm 단위 3D 좌표

[설치]
  pip install rtmlib onnxruntime pykinect-azure
"""

import cv2
import math
import numpy as np
import pykinect_azure as pykinect
from rtmlib import Hand

# ── 파라미터 ────────────────────────────────────────────────────────────────────
TIP_CONF_THR  = 0.5    # 손끝 키포인트 최소 신뢰도
ALPHA_LM      = 0.15   # 랜드마크 EMA 스무딩 강도
ALPHA_ANGLE   = 0.15   # 각도 EMA 스무딩 강도
RESET_PX      = 32     # 급이동 시 스무딩 리셋 임계값 (픽셀)

# ── 뼈대 연결 정보 ──────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks(image, kps, scs, score_thr=0.3):
    pts = [(int(k[0]), int(k[1])) for k in kps]
    for pt, sc in zip(pts, scs):
        if sc > score_thr:
            cv2.circle(image, pt, 3, (0, 0, 255), cv2.FILLED)
    for a, b in HAND_CONNECTIONS:
        if scs[a] > score_thr and scs[b] > score_thr:
            cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)

def normalize_angle(a):
    while a > 180:  a -= 360
    while a < -180: a += 360
    return a

def get_3d_point(device, px, py, depth_image):
    """
    픽셀 좌표 + depth → 3D 좌표 (mm)
    pykinect_azure의 convert_2d_to_3d 사용 (카메라 내부 파라미터 자동 적용)
    """
    h, w = depth_image.shape[:2]
    px, py = int(px), int(py)
    if not (0 <= px < w and 0 <= py < h):
        return None
    depth_val = depth_image[py, px]
    if depth_val <= 0:
        return None
    valid, point_3d = device.calibration.convert_2d_to_3d(
        source_point2d=[px, py],
        source_depth=depth_val,
        source_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR,
        target_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR
    )
    if not valid:
        return None
    return np.array(point_3d, dtype=float)  # [X, Y, Z] mm

def get_all_3d(device, kps, scs, depth_image, score_thr=0.3):
    """
    21개 키포인트 전체를 3D로 변환
    신뢰도 낮거나 depth 없으면 None
    """
    pts_3d = []
    for i, (kp, sc) in enumerate(zip(kps, scs)):
        if sc > score_thr:
            pts_3d.append(get_3d_point(device, kp[0], kp[1], depth_image))
        else:
            pts_3d.append(None)
    return pts_3d

def calc_fingertip_centroid_3d(pts_3d):
    """
    3D Fingertip Centroid — 스크류 동작 최적화
    kp4(엄지끝), kp8(검지끝), kp12(중지끝) 무게중심
    무게중심 → kp4 벡터의 X-Z 평면 yaw 각도
    """
    p4, p8, p12 = pts_3d[4], pts_3d[8], pts_3d[12]
    valid = [p for p in [p4, p8, p12] if p is not None]
    if len(valid) < 2 or p4 is None:
        return None
    centroid = np.mean(valid, axis=0)
    vec = p4 - centroid
    return math.degrees(math.atan2(vec[0], vec[2]))  # X-Z 평면 yaw

def calc_palm_normal_3d(pts_3d):
    """
    3D Palm Normal — 손바닥 yaw 방향
    kp0(손목), kp5(검지뿌리), kp17(소지뿌리) 외적 → 법선 벡터
    법선의 X-Z 평면 각도 = 손바닥이 바라보는 방향
    """
    p0, p5, p17 = pts_3d[0], pts_3d[5], pts_3d[17]
    if p0 is None or p5 is None or p17 is None:
        return None
    vec1 = p5  - p0
    vec2 = p17 - p0
    normal = np.cross(vec1, vec2)
    return math.degrees(math.atan2(normal[0], normal[2]))  # X-Z 평면 yaw

def calc_fingertip_centroid_2d(kps):
    """2D fallback — depth 없을 때"""
    pts = [kps[i] for i in [4, 8, 12]]
    centroid = np.mean(pts, axis=0)
    dx = kps[4][0] - centroid[0]
    dy = kps[4][1] - centroid[1]
    return math.degrees(math.atan2(-dy, dx))


# ── Azure Kinect 초기화 ─────────────────────────────────────────────────────────
pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format          = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution      = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode            = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
device_config.camera_fps            = pykinect.K4A_FRAMES_PER_SECOND_30
device_config.synchronized_images_only = True   # Color + Depth 시간 동기화 필수

print("Azure Kinect 시작 중...")
device = pykinect.start_device(config=device_config)

# ── RTMPose 초기화 ─────────────────────────────────────────────────────────────
hand_detector = Hand(
    to_openpose=False,
    backend='onnxruntime',
    device='cpu'
)

# ── 상태 변수 ───────────────────────────────────────────────────────────────────
baseline_angle = None
smoothed_angle = None
smoothed_kps   = None
absolute_angle = None

print("['s'] 현재 각도를 0도 기준점으로 설정")
print("['d'] depth 시각화 창 on/off")
print("['q'] 종료")

show_depth = True

# ── 메인 루프 ───────────────────────────────────────────────────────────────────
try:
    while True:
        capture = device.update()

        # ── Kinect 프레임 획득 ─────────────────────────────────────────────────
        ret_color, frame_bgra = capture.get_color_image()
        if not ret_color:
            continue

        # Depth → Color 시점 정렬 (같은 픽셀 = 같은 물리 위치 보장)
        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth:
            continue

        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        frame = cv2.flip(frame, 1)
        # depth도 좌우 반전 (frame과 맞추기)
        depth_image = cv2.flip(depth_image, 1)

        # ── RTMPose 추론 ────────────────────────────────────────────────────────
        keypoints_all, scores_all = hand_detector(frame)

        if keypoints_all is not None and len(keypoints_all) > 0:
            kps = keypoints_all[0].astype(float)   # (21, 2) 픽셀 좌표
            scs = scores_all[0]                    # (21,)

            # ── 랜드마크 EMA 스무딩 ────────────────────────────────────────────
            if smoothed_kps is None:
                smoothed_kps = kps.copy()
            else:
                if np.linalg.norm(kps[0] - smoothed_kps[0]) > RESET_PX:
                    smoothed_kps = kps.copy()
                else:
                    smoothed_kps = ALPHA_LM * kps + (1 - ALPHA_LM) * smoothed_kps

            draw_landmarks(frame, smoothed_kps, scs)

            # ── 21개 키포인트 → 3D 변환 ────────────────────────────────────────
            pts_3d = get_all_3d(device, smoothed_kps, scs, depth_image)

            # ── 각도 계산: 3D 우선, 2D fallback ───────────────────────────────
            sc4, sc8, sc12 = scs[4], scs[8], scs[12]
            tips_ok = (sc4 > TIP_CONF_THR and sc8 > TIP_CONF_THR
                       and sc12 > TIP_CONF_THR)

            raw_angle  = None
            palm_angle = calc_palm_normal_3d(pts_3d)   # 참고용 항상 계산

            if tips_ok:
                raw_angle_3d = calc_fingertip_centroid_3d(pts_3d)
                if raw_angle_3d is not None:
                    raw_angle = raw_angle_3d
                    method    = "3D Fingertip"
                    color     = (0, 255, 255)   # 노란색
                else:
                    # depth 없으면 2D fallback
                    raw_angle = calc_fingertip_centroid_2d(smoothed_kps)
                    method    = "2D Fingertip (no depth)"
                    color     = (255, 165, 0)   # 주황색
            else:
                if palm_angle is not None:
                    raw_angle = palm_angle
                    method    = "3D Palm Normal"
                    color     = (255, 0, 255)   # 보라색
                else:
                    raw_angle = calc_fingertip_centroid_2d(smoothed_kps)
                    method    = "2D fallback"
                    color     = (180, 100, 255)

            # ── 시각화 ─────────────────────────────────────────────────────────
            if "Fingertip" in method:
                cx = int(np.mean([smoothed_kps[i][0] for i in [4, 8, 12]]))
                cy = int(np.mean([smoothed_kps[i][1] for i in [4, 8, 12]]))
                for tip_idx in [4, 8, 12]:
                    tx = int(smoothed_kps[tip_idx][0])
                    ty = int(smoothed_kps[tip_idx][1])
                    cv2.line(frame, (cx, cy), (tx, ty), color, 2)
                    cv2.circle(frame, (tx, ty), 8, color, -1)
                    # 3D 좌표 표시
                    p3d = pts_3d[tip_idx]
                    if p3d is not None:
                        cv2.putText(frame, f"Z:{p3d[2]:.0f}mm",
                                    (tx + 5, ty - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                cv2.line(frame, (cx, cy),
                         (int(smoothed_kps[4][0]), int(smoothed_kps[4][1])),
                         (0, 140, 255), 3)

            elif "Palm" in method:
                cv2.line(frame,
                         (int(smoothed_kps[0][0]),  int(smoothed_kps[0][1])),
                         (int(smoothed_kps[5][0]),  int(smoothed_kps[5][1])),
                         color, 2)
                cv2.line(frame,
                         (int(smoothed_kps[0][0]),  int(smoothed_kps[0][1])),
                         (int(smoothed_kps[17][0]), int(smoothed_kps[17][1])),
                         color, 2)
                cv2.circle(frame, (int(smoothed_kps[5][0]),  int(smoothed_kps[5][1])),  8, color, -1)
                cv2.circle(frame, (int(smoothed_kps[17][0]), int(smoothed_kps[17][1])), 8, color, -1)

            # ── 각도 EMA 스무딩 ────────────────────────────────────────────────
            if raw_angle is not None:
                if smoothed_angle is None:
                    smoothed_angle = raw_angle
                else:
                    diff = normalize_angle(raw_angle - smoothed_angle)
                    smoothed_angle = normalize_angle(
                        smoothed_angle + ALPHA_ANGLE * diff
                    )
                absolute_angle = smoothed_angle

                # ── 회전량 표시 ────────────────────────────────────────────────
                if baseline_angle is not None:
                    rotation = normalize_angle(absolute_angle - baseline_angle)
                    cv2.putText(frame, f"Rotation: {rotation:.1f} deg", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                cv2.putText(frame, f"Abs: {absolute_angle:.1f} deg", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Palm Normal 참고값 같이 표시
                if palm_angle is not None and "Palm" not in method:
                    cv2.putText(frame, f"Palm ref: {palm_angle:.1f} deg", (20, 185),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame,
                        f"[{method}]  4:{scs[4]:.2f} 8:{scs[8]:.2f} 12:{scs[12]:.2f}",
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            smoothed_kps   = None
            smoothed_angle = None
            cv2.putText(frame, "Hand Not Detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Cup/Screw Rotation Tracker (RTMPose + Kinect)', frame)

        # depth 시각화 (옵션)
        if show_depth:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Aligned Depth', depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if absolute_angle is not None:
                baseline_angle = absolute_angle
                print(f"기준 각도 설정: {baseline_angle:.1f}도")
        elif key == ord('d'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow('Aligned Depth')

except Exception as e:
    print(f"오류 발생: {e}")
    raise
finally:
    print("종료 중...")
    device.close()
    cv2.destroyAllWindows()

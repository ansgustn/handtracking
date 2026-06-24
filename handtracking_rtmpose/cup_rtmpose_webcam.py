"""
cup_rtmpose_webcam.py
RTMPose 기반 컵/나사 회전 각도 측정 - 단일 웹캠

[각도 계산 전략 — 신뢰도 기반 자동 전환]
  1순위 — Fingertip Centroid (3점)
      엄지(4) + 검지(8) + 중지(12) 무게중심 기준
      무게중심 → 엄지끝 벡터의 회전각 추적
      스크류/컵 파지 동작에 최적화

  2순위 — Fingertip Centroid (2점, fallback)
      중지(12)가 가려질 때 엄지(4) + 검지(8) 중점 사용

  3순위 — Wrist Roll (최후 fallback)
      손목(0) → 중지뿌리(9) 벡터 기울기
      손끝이 모두 가려졌을 때

[설치]
  pip install rtmlib onnxruntime
"""

import cv2
import math
import numpy as np
from rtmlib import Hand

# ── 파라미터 ────────────────────────────────────────────────────────────────────
TIP_CONF_THR   = 0.5    # 손끝 키포인트 최소 신뢰도
ALPHA_LM       = 0.15   # 랜드마크 EMA 스무딩 강도
ALPHA_ANGLE    = 0.15   # 각도 EMA 스무딩 강도
RESET_PX       = 32     # 급이동 시 스무딩 리셋 임계값 (픽셀)

# ── RTMPose 초기화 ─────────────────────────────────────────────────────────────
hand_detector = Hand(
    to_openpose=False,
    backend='onnxruntime',
    device='cpu'            # GPU: 'cuda' / Mac M4: 'cpu'
)

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

def calc_centroid_angle(kps, tip_indices):
    """
    손끝 무게중심 → 엄지끝(kp4) 벡터의 방향각
    스크류 회전 시 이 각도가 단조롭게 변함
    """
    centroid = np.mean([kps[i] for i in tip_indices], axis=0)
    dx = kps[4][0] - centroid[0]
    dy = kps[4][1] - centroid[1]
    return math.degrees(math.atan2(-dy, dx)), centroid

def calc_wrist_roll_angle(kps):
    dx = kps[9][0] - kps[0][0]
    dy = kps[9][1] - kps[0][1]
    return math.degrees(math.atan2(-dy, dx))


# ── 상태 변수 ───────────────────────────────────────────────────────────────────
baseline_angle = None
smoothed_angle = None
smoothed_kps   = None
absolute_angle = None

# ── 카메라 초기화 ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Mac: cv2.CAP_DSHOW 제거

print("['s'] 현재 각도를 0도 기준점으로 설정")
print("['q'] 종료")

if not cap.isOpened():
    print("경고: 카메라를 열 수 없습니다!")

# ── 메인 루프 ───────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    keypoints_all, scores_all = hand_detector(frame)

    if keypoints_all is not None and len(keypoints_all) > 0:
        kps = keypoints_all[0].astype(float)  # (21, 2) 픽셀 좌표
        scs = scores_all[0]                   # (21,)

        # ── 1. 랜드마크 EMA 스무딩 ─────────────────────────────────────────
        if smoothed_kps is None:
            smoothed_kps = kps.copy()
        else:
            if np.linalg.norm(kps[0] - smoothed_kps[0]) > RESET_PX:
                smoothed_kps = kps.copy()
            else:
                smoothed_kps = ALPHA_LM * kps + (1 - ALPHA_LM) * smoothed_kps

        draw_landmarks(frame, smoothed_kps, scs)

        # ── 2. 신뢰도 기반 방법 선택 ───────────────────────────────────────
        sc4  = scs[4]   # 엄지 끝
        sc8  = scs[8]   # 검지 끝
        sc12 = scs[12]  # 중지 끝

        if sc4 > TIP_CONF_THR and sc8 > TIP_CONF_THR and sc12 > TIP_CONF_THR:
            # ── 1순위: 엄지 + 검지 + 중지 세 점 ──────────────────────────
            raw_angle, centroid = calc_centroid_angle(smoothed_kps, [4, 8, 12])
            method = "3-Finger"
            color  = (0, 255, 255)   # 노란색

            # 시각화: 무게중심 + 각 손끝 → 무게중심 선
            cx, cy = int(centroid[0]), int(centroid[1])
            for tip_idx in [4, 8, 12]:
                tx, ty = int(smoothed_kps[tip_idx][0]), int(smoothed_kps[tip_idx][1])
                cv2.line(frame, (cx, cy), (tx, ty), color, 2)
                cv2.circle(frame, (tx, ty), 8, color, -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)  # 무게중심 흰 점
            # 기준 벡터 (무게중심 → 엄지끝) 강조
            cv2.line(frame, (cx, cy),
                     (int(smoothed_kps[4][0]), int(smoothed_kps[4][1])),
                     (0, 140, 255), 3)

        elif sc4 > TIP_CONF_THR and sc8 > TIP_CONF_THR:
            # ── 2순위: 엄지 + 검지 두 점 ─────────────────────────────────
            raw_angle, centroid = calc_centroid_angle(smoothed_kps, [4, 8])
            method = "2-Finger"
            color  = (255, 165, 0)   # 주황색

            cx, cy = int(centroid[0]), int(centroid[1])
            for tip_idx in [4, 8]:
                tx, ty = int(smoothed_kps[tip_idx][0]), int(smoothed_kps[tip_idx][1])
                cv2.line(frame, (cx, cy), (tx, ty), color, 2)
                cv2.circle(frame, (tx, ty), 8, color, -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)

        else:
            # ── 3순위: Wrist Roll ─────────────────────────────────────────
            raw_angle = calc_wrist_roll_angle(smoothed_kps)
            method    = "Wrist Roll"
            color     = (180, 100, 255)  # 보라색

            cv2.line(frame,
                     (int(smoothed_kps[0][0]), int(smoothed_kps[0][1])),
                     (int(smoothed_kps[9][0]), int(smoothed_kps[9][1])),
                     color, 3)
            cv2.circle(frame, (int(smoothed_kps[9][0]), int(smoothed_kps[9][1])), 8, color, -1)

        # ── 3. 각도 EMA 스무딩 ─────────────────────────────────────────────
        if smoothed_angle is None:
            smoothed_angle = raw_angle
        else:
            diff = normalize_angle(raw_angle - smoothed_angle)
            smoothed_angle = normalize_angle(
                smoothed_angle + ALPHA_ANGLE * diff
            )

        absolute_angle = smoothed_angle

        # ── 4. 기준 대비 회전량 ────────────────────────────────────────────
        if baseline_angle is not None:
            rotation = normalize_angle(absolute_angle - baseline_angle)
            cv2.putText(frame, f"Rotation: {rotation:.1f} deg", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # ── 5. 상태 표시 ───────────────────────────────────────────────────
        cv2.putText(frame, f"Abs: {absolute_angle:.1f} deg", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"[{method}]  4:{sc4:.2f} 8:{sc8:.2f} 12:{sc12:.2f}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:
        smoothed_kps   = None
        smoothed_angle = None
        cv2.putText(frame, "Hand Not Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Cup/Screw Rotation Tracker (RTMPose)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if absolute_angle is not None:
            baseline_angle = absolute_angle
            print(f"기준 각도 설정: {baseline_angle:.1f}도")

cap.release()
cv2.destroyAllWindows()

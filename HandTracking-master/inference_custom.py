import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import FreiHANDModel
import mediapipe as mp

# =====================================================
# 핵심 변경사항:
# 1. 화면 마크 표시 → MediaPipe 2D 좌표 직접 사용
# 2. FreiHAND → 회전 계산에만 사용
# 3. Y축 반전 문제 해결
# 4. 회전각도 정확도 개선 (21개 관절 전부 사용)
# =====================================================

# ── 손가락 연결선 정의 ────────────────────────────────
CONNECTIONS = {
    "thumb":  ([0,1,2,3,4],     (255, 0,   0  )),
    "index":  ([0,5,6,7,8],     (0,   128, 255)),
    "middle": ([0,9,10,11,12],  (0,   255, 0  )),
    "ring":   ([0,13,14,15,16], (255, 0,   128)),
    "pinky":  ([0,17,18,19,20], (0,   0,   255)),
}

# ── 회전 추정 함수들 ──────────────────────────────────

def estimate_rotation(initial_points, current_points):
    """
    SVD 기반 회전 행렬 추정
    21개 관절 전부 사용 → 정확도 향상
    """
    centroid_init = np.mean(initial_points, axis=0)
    centroid_curr = np.mean(current_points, axis=0)
    A = initial_points - centroid_init
    B = current_points - centroid_curr
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)

    # 수치 불안정 체크
    if S[-1] < 1e-6:
        return np.eye(3)

    R = Vt.T @ U.T

    # 반사 행렬 보정
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def rotation_matrix_to_euler_angles(R):
    """회전 행렬 → Roll/Pitch/Yaw (도 단위)"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2( R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2( R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def get_total_rotation_angle(R):
    """전체 회전각도 (도 단위)"""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ── MediaPipe 2D 좌표 → 화면 픽셀 변환 ───────────────

def landmarks_to_pixels(landmarks, w, h):
    """
    MediaPipe 정규화 좌표(0~1) → 픽셀 좌표
    MediaPipe가 직접 주는 2D 좌표 사용
    → 항상 정확하게 손을 따라감
    """
    pts = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts.append((px, py))
    return pts  # list of (x, y)


def draw_hand_skeleton(img, pts_2d):
    """
    MediaPipe 2D 좌표로 스켈레톤 그리기
    pts_2d: list of (x, y) 21개
    """
    h, w = img.shape[:2]
    for finger, (joints, color) in CONNECTIONS.items():
        for i in range(len(joints) - 1):
            pt1 = pts_2d[joints[i]]
            pt2 = pts_2d[joints[i+1]]
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(img, pt1, pt2, color, 2)
    for i, pt in enumerate(pts_2d):
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, pt, 5, (255, 255, 255), -1)
            cv2.circle(img, pt, 5, (0,   0,   0  ),  1)
            # 관절 번호 표시 (디버깅용, 필요없으면 주석처리)
            # cv2.putText(img, str(i), pt,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)


# ── 가상 물체 렌더링 ──────────────────────────────────

def draw_virtual_object(img, R, center_2d, scale=60):
    """
    회전 행렬 반영한 3D 큐브 + 좌표축
    원근감(Z) 적용
    """
    vertices = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
    ], dtype=float)

    rotated  = vertices @ R.T
    z_offset = 3.0

    def proj(v):
        denom = max(v[2] + z_offset, 1e-4)
        return (
            int(v[0] / denom * scale + center_2d[0]),
            int(v[1] / denom * scale + center_2d[1])
        )

    pts = [proj(v) for v in rotated]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for p1, p2 in edges:
        cv2.line(img, pts[p1], pts[p2], (0, 255, 255), 2)

    # 좌표축 (X=빨강, Y=초록, Z=파랑)
    axis = np.array([[0,0,0],[1.5,0,0],[0,1.5,0],[0,0,1.5]]) @ R.T
    o = proj(axis[0])
    cv2.line(img, o, proj(axis[1]), (0,   0,   255), 3)
    cv2.line(img, o, proj(axis[2]), (0,   255,   0), 3)
    cv2.line(img, o, proj(axis[3]), (255,   0,   0), 3)


def draw_info_panel(img, angles, total_angle, is_grabbing):
    """정보 패널"""
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (360, 220), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    color  = (0, 255, 0) if is_grabbing else (150, 150, 150)
    status = "GRABBING" if is_grabbing else "IDLE - Press G to Grab"
    cv2.putText(img, f"STATUS: {status}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if is_grabbing and angles is not None:
        cv2.putText(img, f"Roll  (X): {angles[0]:>7.1f} deg",
                    (20,  85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, f"Pitch (Y): {angles[1]:>7.1f} deg",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, f"Yaw   (Z): {angles[2]:>7.1f} deg",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img, f"Total Rot: {total_angle:>7.1f} deg",
                    (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (0,255,255),   2)
    else:
        cv2.putText(img, "[G] Grab  [R] Release  [ESC] Exit",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)


# =====================================================
# 모델 초기화
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

model = FreiHANDModel(num_keypoints=21).to(device)
model.load_state_dict(
    torch.load("freihand_custom_model.pth", map_location=device)
)
model.eval()
print("✅ FreiHAND 모델 로드 완료")

# =====================================================
# MediaPipe 초기화
# =====================================================
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
mp_detector = HandLandmarker.create_from_options(options)
print("✅ MediaPipe 초기화 완료")

# =====================================================
# 이미지 전처리 (FreiHAND 모델 입력용)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

# =====================================================
# 메인 루프
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

grabbed_points_3d = None   # G키로 캡처한 기준 3D 좌표
current_points_3d = None   # 현재 프레임 3D 좌표
hand_detected     = False  # 현재 프레임 손 감지 여부

print("✅ 웹캠 시작!")
print("   [G] 손 잡기  [R] 놓기  [ESC] 종료")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠 읽기 실패")
        break

    image     = cv2.flip(image, 1)
    h, w, _   = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── 매 프레임 초기화 ─────────────────────────────
    hand_detected     = False
    current_points_3d = None

    # ── MediaPipe 손 감지 ────────────────────────────
    mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=rgb_image)
    detection_result = mp_detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_detected = True
        landmarks     = detection_result.hand_landmarks[0]

        # ── 핵심 변경 1: MediaPipe 2D 좌표로 스켈레톤 표시
        # → 항상 손을 정확하게 따라감
        pts_2d = landmarks_to_pixels(landmarks, w, h)
        draw_hand_skeleton(image, pts_2d)

        # 손목 위치를 가상 물체 중심으로 사용
        center_2d = pts_2d[0]  # 손목(관절 0)

        # 바운딩 박스 계산 (FreiHAND 입력용 크롭)
        x_coords = [p[0] for p in pts_2d]
        y_coords = [p[1] for p in pts_2d]
        margin = 40
        x1 = max(0, min(x_coords) - margin)
        y1 = max(0, min(y_coords) - margin)
        x2 = min(w, max(x_coords) + margin)
        y2 = min(h, max(y_coords) + margin)

        # 바운딩 박스 표시
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if x2 - x1 > 10 and y2 - y1 > 10:
            # ── FreiHAND 모델 추론 (회전 계산용) ────────
            cropped      = image[y1:y2, x1:x2]
            rgb_cropped  = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_img      = Image.fromarray(rgb_cropped)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)

            # 핵심 변경 2: mm 단위 변환 + Y축 반전 보정
            lm_3d = outputs.view(21, 3).cpu().numpy()
            lm_3d = lm_3d * 1000   # m → mm

            # Y축 반전 (FreiHAND Y+= 위쪽, 화면 Y+= 아래쪽)
            lm_3d[:, 1] = -lm_3d[:, 1]

            # 손목 기준 정규화 (손목을 원점으로)
            lm_3d = lm_3d - lm_3d[0]

            # 핵심 변경 3: 21개 관절 전부 회전 계산에 사용
            current_points_3d = lm_3d.copy()

            # ── 회전 계산 및 가상 물체 렌더링 ───────────
            if grabbed_points_3d is not None:
                R           = estimate_rotation(grabbed_points_3d,
                                                current_points_3d)
                angles      = rotation_matrix_to_euler_angles(R)
                total_angle = get_total_rotation_angle(R)
                draw_virtual_object(image, R, center_2d)
                draw_info_panel(image, angles, total_angle,
                                is_grabbing=True)
            else:
                draw_virtual_object(image, np.eye(3), center_2d)
                draw_info_panel(image, None, None,
                                is_grabbing=False)

    else:
        # 손 감지 안 됨
        draw_info_panel(image, None, None, is_grabbing=False)

    # 안내 텍스트
    cv2.putText(image,
                "[G] Grab   [R] Release   [ESC] Exit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow('FreiHAND Hand Rotation Tracker v2', image)

    # ── 키 입력 ──────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC
        print("종료")
        break

    elif key == ord('g') or key == ord('G'):
        # 핵심 변경 4: 현재 프레임 감지 여부로 판단
        if hand_detected and current_points_3d is not None:
            grabbed_points_3d = current_points_3d.copy()
            print("✅ Grab! 기준 자세 캡처 완료")
        else:
            print("⚠️  손이 감지되지 않아 Grab 불가")

    elif key == ord('r') or key == ord('R'):
        grabbed_points_3d = None
        print("🔄 Released")

cap.release()
cv2.destroyAllWindows()
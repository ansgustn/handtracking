import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 최신 Tasks API로 초기화 (구버전 solutions 에러 해결)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
hands = HandLandmarker.create_from_options(options)

def estimate_rotation(initial_points, current_points, root_init, root_curr):
    """SVD를 이용한 3D 회전 행렬 계산 (Root를 외부에서 주입받음)"""
    # 손가락 버전(centroid)과 다르게, 명시적으로 전달받은 Root(손목)를 기준으로 삼습니다.
    A = initial_points - root_init # 0도에 대한 포인트 정보 (Root 기준)
    B = current_points - root_curr # 현재 포인트 정보 (Root 기준)
    
    H = A.T @ B # 상관관계 행렬 -> 어느 방향으로 얼마나 옮겨갔는가에 대한 값
    U, S, Vt = np.linalg.svd(H) # SVD 알고리즘 적용 -> 공간이 꺽인 방향 및 회전에 대한 정보 추출
    R = Vt.T @ U.T # 회전행렬
    
    if np.linalg.det(R) < 0: # SVD 알고리즘 거울 반사 방지
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def get_total_rotation_angle(R):
    """
    회전 행렬에서 물체가 전체적으로 회전한 단일 총량 각도를 구합니다.
    (0도 ~ 180도 사이의 값 반환)
    """
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def rotation_matrix_to_euler_angles(R):
    """회전 행렬을 Roll, Pitch, Yaw 각도(Degree)로 변환합니다."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6  # 짐벌락 방지

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], sy)  # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))


def draw_virtual_object(img, R, center_2d, scale=60):
    """회전 행렬(R)을 적용하여 가상의 3D 정육면체와 축을 그립니다."""
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=float)

    rotated_vertices = vertices @ R.T

    points_2d = []
    for v in rotated_vertices:
        x = int(v[0] * scale + center_2d[0])
        y = int(v[1] * scale + center_2d[1])
        points_2d.append((x, y))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 뒷면
        (4, 5), (5, 6), (6, 7), (7, 4),  # 앞면
        (0, 4), (1, 5), (2, 6), (3, 7)  # 기둥
    ]
    # 노란색 정육면체(네모박스) 렌더링 비활성화
    # for p1, p2 in edges:
    #     cv2.line(img, points_2d[p1], points_2d[p2], (0, 255, 255), 2)

    # XYZ 3D 좌표축 그리기
    axis_pts = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]) @ R.T
    o = (int(axis_pts[0, 0] * scale + center_2d[0]), int(axis_pts[0, 1] * scale + center_2d[1]))
    x = (int(axis_pts[1, 0] * scale + center_2d[0]), int(axis_pts[1, 1] * scale + center_2d[1]))
    y = (int(axis_pts[2, 0] * scale + center_2d[0]), int(axis_pts[2, 1] * scale + center_2d[1]))
    z = (int(axis_pts[3, 0] * scale + center_2d[0]), int(axis_pts[3, 1] * scale + center_2d[1]))

    # XYZ 축 선 렌더링 비활성화
    # cv2.line(img, o, x, (0, 0, 255), 3)  # X: Red
    # cv2.line(img, o, y, (0, 255, 0), 3)  # Y: Green
    # cv2.line(img, o, z, (255, 0, 0), 3)  # Z: Blue


def draw_info_panel(img, angles, total_angle, is_grabbing):
    """화면 좌측 상단에 각도 정보를 표시하는 반투명 UI를 그립니다."""
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 200), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    color = (0, 255, 0) if is_grabbing else (150, 150, 150)
    status_text = "GRABBING VIRTUAL OBJ (WRIST ROOT)" if is_grabbing else "IDLE"

    cv2.putText(img, f"STATUS: {status_text}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if is_grabbing and angles is not None:
        cv2.putText(img, f"Roll  (X): {angles[0]:>6.1f} deg", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Pitch (Y): {angles[1]:>6.1f} deg", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Yaw   (Z): {angles[2]:>6.1f} deg", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 총 회전량 표시
        cv2.putText(img, f"Total Rot: {total_angle:>6.1f} deg", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(img, "Press 'G' to Capture", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# 웹캠 열기
cap = cv2.VideoCapture(0)
grabbed_points_3d = None
grabbed_root_3d = None

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # MediaPipe Tasks API 추론
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = hands.detect(mp_image)
    h, w, c = image.shape

    # 하단 조작 안내 문구
    cv2.putText(image, "[G] Grab Object   [R] Release Object   [ESC] Exit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if results.hand_world_landmarks and results.hand_landmarks:
        # 3D 랜드마크 추출
        lm_3d = results.hand_world_landmarks[0]
        
        # 손목과 손바닥의 뼈대(0, 5, 17)를 사용하여 손목의 회전을 측정합니다.
        current_points_3d = np.array([
            [lm_3d[0].x, lm_3d[0].y, lm_3d[0].z],
            [lm_3d[5].x, lm_3d[5].y, lm_3d[5].z],
            [lm_3d[17].x, lm_3d[17].y, lm_3d[17].z]
        ])
        
        # 추가: 손목을 Root로 사용하기 위해 손목 좌표(0) 추출
        current_root_3d = np.array([lm_3d[0].x, lm_3d[0].y, lm_3d[0].z])

        # 2D 랜드마크 추출 (시각화용)
        lm_2d = results.hand_landmarks[0]
        points_2d = [(int(lm_2d[idx].x * w), int(lm_2d[idx].y * h)) for idx in [0, 5, 17]]

        # 손목 및 손바닥 마디 점 시각화
        for pt in points_2d:
            cv2.circle(image, pt, 6, (0, 255, 255), -1) # 노란색으로 강조

        # 중심점 계산
        center_2d = (
            int(np.mean([p[0] for p in points_2d])),
            int(np.mean([p[1] for p in points_2d]))
        )

        if grabbed_points_3d is not None and grabbed_root_3d is not None:
            # 회전 행렬 및 각도 계산 (차이점: 손목 좌표를 Root로 넘겨줌)
            R = estimate_rotation(grabbed_points_3d, current_points_3d, grabbed_root_3d, current_root_3d)
            angles = rotation_matrix_to_euler_angles(R)
            total_angle = get_total_rotation_angle(R)

            # 렌더링 및 UI 업데이트
            draw_virtual_object(image, R, center_2d)
            draw_info_panel(image, angles, total_angle, is_grabbing=True)
        else:
            # 잡지 않았을 때
            draw_virtual_object(image, np.eye(3), center_2d)
            draw_info_panel(image, None, None, is_grabbing=False)
    else:
        # 손이 화면에 없을 때
        draw_info_panel(image, None, None, is_grabbing=False)

    cv2.imshow('Virtual Object Rotation Tracker (Wrist Root)', image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('g'):
        if 'current_points_3d' in locals() and 'current_root_3d' in locals():
            grabbed_points_3d = current_points_3d.copy()
            grabbed_root_3d = current_root_3d.copy()
    elif key == ord('r'):
        grabbed_points_3d = None
        grabbed_root_3d = None

cap.release()
cv2.destroyAllWindows()
hands.close()

import cv2
import cv2.aruco as aruco
import math
import numpy as np

# ArUco 딕셔너리 설정 (4x4 마커 사용 권장)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# 이전 각도를 저장하여 미세한 변화량(Delta)을 계산하기 위한 변수
prev_center_angle = None

print("Tracking Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 마커 탐지
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) >= 2:
        points = {}
        marker_angles = {}
        
        # 탐지된 마커들의 정보 추출
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            
            # 4개의 코너 중심점 계산 (float 형태로 계산하여 더 정밀하게 위치 파악)
            center_x = c[:, 0].mean()
            center_y = c[:, 1].mean()
            points[marker_id] = (center_x, center_y)
            
            # 마커 자체의 회전 각도 계산
            # 코너 순서: 0:좌상, 1:우상, 2:우하, 3:좌하
            # 좌상단과 우상단 코너의 기울기를 통해 마커의 자체 각도를 구함
            top_left = c[0]
            top_right = c[1]
            marker_angle = math.degrees(math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
            marker_angles[marker_id] = marker_angle

        # ID 0(고정컵)과 ID 1(회전컵)이 모두 있을 때 조작
        if 0 in points and 1 in points:
            p0 = points[0]
            p1 = points[1]
            
            # 1. 두 마커의 중심점을 이은 선의 각도 (제공해주신 코드의 방식)
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            center_angle = math.degrees(math.atan2(dy, dx))
            
            # 2. 중심점 각도의 미세 변화량 (초당 프레임 간의 각도 변화)
            delta_angle = 0.0
            if prev_center_angle is not None:
                delta_angle = center_angle - prev_center_angle
            prev_center_angle = center_angle
            
            # 3. 마커 자체의 방향(회전) 차이 (두 컵에 마커가 나란히 붙어있을 때 유용함)
            # 기준 마커 대비 회전 마커의 자체 각도 차이를 보여줌 (0~360도를 넘어가는 부분 보정 가능)
            relative_marker_angle = marker_angles[1] - marker_angles[0]
            
            # 시각화: 두 마커 연결 선 그리기
            p0_int = (int(p0[0]), int(p0[1]))
            p1_int = (int(p1[0]), int(p1[1]))
            cv2.line(frame, p0_int, p1_int, (255, 0, 0), 2)
            
            # 각 마커의 중심에 포인트 표시
            cv2.circle(frame, p0_int, 4, (0, 0, 255), -1)
            cv2.circle(frame, p1_int, 4, (0, 0, 255), -1)
            
            # 결과 텍스트 출력
            cv2.putText(frame, f"Absolute Angle: {center_angle:.2f} deg", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Micro Movement (Delta): {delta_angle:.3f} deg", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Relative Marker Rotation: {relative_marker_angle:.2f} deg", (30, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Cup Rotation Tracker', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import math

# 웹캠 연결 및 해상도 설정 (고해상도로 선을 또렷이 인식하기 위함)
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Line Tracking Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break
    
    # 1. 흑백 변환 및 블러링 (노이즈 제거)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny 알고리즘으로 윤곽선(Edge) 추출
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. 허프 변환(HoughLinesP)으로 직선 검출
    # threshold, minLineLength 값은 선의 길이 및 탐지 민감도에 따라 조절할 수 있습니다.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=15)
    
    drawn_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 선의 길이와 기울기 각도 계산
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 각도가 수평에 가까운 쓸데없는 배경 선은 제외 (수직선에 가까운 선만 필터링)
            if abs(angle) > 45 and abs(angle) < 135:
                drawn_lines.append((length, angle, x1, y1, x2, y2))
                
        # 선을 길이 순으로 내림차순 정렬 (가장 뚜렷하고 긴 선 2개를 찾기 위함)
        drawn_lines.sort(key=lambda x: x[0], reverse=True)
        
        # 가장 뚜렷한 선 2개가 모두 인식된 경우 (위쪽 종이컵의 선과 아래쪽 종이컵의 선)
        if len(drawn_lines) >= 2:
            line1 = drawn_lines[0]
            line2 = drawn_lines[1]
            
            # 두 선 사이의 틀어진 상대 각도 (어긋난 정도) 계산
            relative_angle_diff = abs(line1[1] - line2[1])
            
            # 화면에 두 선을 그리기 (녹색 과 파란색)
            cv2.line(frame, (line1[2], line1[3]), (line1[4], line1[5]), (0, 255, 0), 3)
            cv2.line(frame, (line2[2], line2[3]), (line2[4], line2[5]), (255, 0, 0), 3)
            
            cv2.putText(frame, f"Top Line Angle: {line1[1]:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Bottom Line Angle: {line2[1]:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Difference (Misalignment): {relative_angle_diff:.2f} deg", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 디버깅: 인식된 가장 긴 선 1개만 있을 때
        elif len(drawn_lines) == 1:
            line1 = drawn_lines[0]
            cv2.line(frame, (line1[2], line1[3]), (line1[4], line1[5]), (0, 255, 255), 3)
            cv2.putText(frame, f"Waiting for 2nd line.. Angle: {line1[1]:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 윤곽선 검출 결과를 보여주는 Edge 화면과 원본 화면을 같이 띄움
    cv2.imshow('Edge Filter View', edges)
    cv2.imshow('Line Rotation Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

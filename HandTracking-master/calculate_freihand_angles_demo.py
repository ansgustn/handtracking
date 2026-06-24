import json
import numpy as np
import math

def calculate_angle(v1, v2):
    """두 벡터(v1, v2) 사이의 각도를 도(degree) 단위로 계산합니다."""
    # 벡터의 내적
    dot_product = np.dot(v1, v2)
    # 벡터의 크기(Norm)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 코사인 값 계산 (부동소수점 오차 방지를 위해 -1.0 ~ 1.0으로 클리핑)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 아크코사인을 이용해 라디안 각도 계산 후 도(degree)로 변환
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def main():
    # 1. FreiHAND 데이터셋의 3D 좌표 JSON 파일 로드
    json_path = 'training_xyz.json'
    print(f"데이터셋 로딩 중: {json_path}")
    
    with open(json_path, 'r') as f:
        xyz_data = json.load(f)
        
    print(f"총 {len(xyz_data)}개의 손 데이터가 있습니다.\n")
    
    # 2. 첫 번째 이미지(인덱스 0)의 3D 손 관절 21개 좌표 가져오기
    hand_landmarks = np.array(xyz_data[0])
    
    # MediaPipe와 FreiHAND의 관절 인덱스는 동일합니다.
    # 각 손가락별로 구부러지는 핵심 관절(PIP 또는 유사 위치)의 각도를 계산해 봅니다.
    # 구조: 손목(0), 
    # 엄지: 1-2-3-4
    # 검지: 5-6-7-8
    # 중지: 9-10-11-12
    # 약지: 13-14-15-16
    # 소지: 17-18-19-20
    
    # 각 손가락별로 3개의 점(A, B, C)을 잡아 B에서 꺾이는 각도를 계산
    # 예: 검지의 경우 점 5(MCP), 점 6(PIP), 점 7(DIP)을 사용
    # 벡터 v1 = B - A (5번에서 6번으로 향하는 뼈)
    # 벡터 v2 = C - B (6번에서 7번으로 향하는 뼈)
    
    finger_indices = {
        "엄지(Thumb)": (2, 3, 4),    # 엄지는 관절 구조가 조금 다르지만 대략적인 구부러짐 계산
        "검지(Index)": (5, 6, 7),
        "중지(Middle)": (9, 10, 11),
        "약지(Ring)": (13, 14, 15),
        "소지(Pinky)": (17, 18, 19)
    }
    
    print("=== 첫 번째 손(데이터 인덱스 0)의 손가락 구부러짐 각도 ===")
    print("(0도에 가까울수록 곧게 펴진 상태, 각도가 클수록 많이 구부러진 상태를 의미합니다)")
    for finger_name, (idx_a, idx_b, idx_c) in finger_indices.items():
        pt_a = hand_landmarks[idx_a]
        pt_b = hand_landmarks[idx_b]
        pt_c = hand_landmarks[idx_c]
        
        # 벡터 생성
        vec1 = pt_b - pt_a
        vec2 = pt_c - pt_b
        
        # 각도 계산
        angle = calculate_angle(vec1, vec2)
        print(f"- {finger_name} 각도: {angle:.1f} 도")

if __name__ == "__main__":
    main()

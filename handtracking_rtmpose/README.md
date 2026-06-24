# 컵 회전 각도 측정 — RTMPose 버전

MediaPipe 기반 코드를 RTMPose(rtmlib)로 변환한 버전입니다.

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `cup_rtmpose_webcam.py` | 단일 웹캠 버전 (Palm Normal + Wrist Roll 혼용) |
| `cup_rtmpose_dualcam.py` | 듀얼 카메라 버전 (녹화 / CSV 저장 / 그래프 출력 포함) |
| `cup_rtmpose_kinect.py` | Azure Kinect 버전 (실제 Depth 기반 3D 각도 측정) |

---

## 설치 방법

### 1단계 — Python 가상환경 만들기 (권장)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac / Linux**
```bash
source venv/bin/activate
```

### 2단계 — 패키지 설치

```bash
pip install -r requirements.txt
```

> **첫 실행 시** rtmlib이 RTMPose 모델 가중치를 자동으로 다운로드합니다.
> 인터넷 연결 필요 / 수십 초 소요 (이후 실행부터는 바로 시작)

---

## 실행 방법

### 단일 웹캠

```bash
python cup_rtmpose_webcam.py
```

### 듀얼 카메라

```bash
python cup_rtmpose_dualcam.py
```

### Azure Kinect (Kinect 연결 시)

```bash
python cup_rtmpose_kinect.py
```

---

## 키 조작

### 공통

| 키 | 동작 |
|---|---|
| `s` | 현재 각도를 0도 기준점으로 설정 |
| `q` | 종료 |

### 듀얼 카메라 전용

| 키 | 동작 |
|---|---|
| `Space` | 실험 녹화 시작 / 중지 |
| `g` | CSV 저장 + 그래프 출력 |
| `c` | 데이터 초기화 |

### Kinect 전용

| 키 | 동작 |
|---|---|
| `d` | Depth 시각화 창 on/off |

---

## 각도 측정 방식

### webcam / dualcam — 신뢰도 기반 자동 전환

| 방법 | 조건 | 색상 |
|---|---|---|
| 3-Finger Centroid | kp4, kp8, kp12 신뢰도 > 0.5 | 노란색 |
| 2-Finger Centroid | kp4, kp8만 > 0.5 | 주황색 |
| Wrist Roll | 손끝 가려짐 | 보라색 |

### Kinect — 실제 Depth 활용 3D 각도

| 방법 | 조건 | 색상 |
|---|---|---|
| 3D Fingertip Centroid | 손끝 신뢰도 높고 depth 유효 | 노란색 |
| 3D Palm Normal | 손끝 신뢰도 낮을 때 | 보라색 |
| 2D fallback | depth 없을 때 | 주황색 |

---

## 카메라 인덱스 변경

카메라가 안 잡히면 코드 상단의 인덱스를 수정하세요.

```python
# cup_rtmpose_webcam.py
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 → 1, 2 ... 로 변경

# cup_rtmpose_dualcam.py
cam1 = CameraThread(0, 1)   # 첫 번째 인자가 카메라 인덱스
cam2 = CameraThread(1, 2)
```

> **Mac 사용자**: `cv2.CAP_DSHOW` 인자를 제거하세요.
> ```python
> cap = cv2.VideoCapture(0)
> ```

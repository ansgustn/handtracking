# 3D 손 관절 추적 및 SVD 기반 객체 회전 추정 시스템 설계 및 분석

본 문서는 현재 구현된 손 관절 추적 파이프라인(통제군)의 기술적 원리, 사용된 사전 학습 모델의 구조, 그리고 가상 객체의 3D 회전(Screw) 각도를 추정하기 위한 수학적 근거를 논문 형식으로 상술합니다.

---

## 1. 시스템 아키텍처 및 모델 구성 (Model Architecture)

현재 시스템은 복잡한 배경 속에서도 정확도 높은 3D 관절 좌표를 추출하기 위해 **YOLOv11**과 **MediaPipe Hand Landmarker**를 결합한 파이프라인을 채택하고 있습니다.

### 1.1. 1단계: 관심 영역(ROI) 추출 (YOLOv11)
*   **사용 모델**: `yolo11n.pt` (Nano 버전)
*   **활용 및 이유**: 전체 화면에서 곧바로 손 관절을 찾으면 연산량이 많고 정확도가 떨어집니다. 따라서 객체 탐지에 특화된 YOLO 모델을 선행적으로 가동하여 '손' 객체에 대한 바운딩 박스(Bounding Box)를 검출하고, 해당 영역만 크롭(Crop)하여 2단계 모델로 넘깁니다.

### 1.2. 2단계: 3D 관절 랜드마크 추출 (MediaPipe Hand Landmarker)
*   **참고 문헌**: [MediaPipe Hand Landmarker Official Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=ko)
*   **모델 특징**: 구글이 수만 장의 실제 이미지와 합성 3D 모델 데이터를 통해 사전 학습(Pre-training)한 모델입니다. 크롭된 RGB 이미지를 입력받아 21개 손 관절의 상대적 3D 좌표($x, y, z$)를 출력합니다.
*   **MediaPipe의 한계와 본 연구의 접근 방식**: 
    MediaPipe는 엣지 디바이스에서의 실시간 추론을 위해 가중치가 고정된 직렬화 포맷(`.task` 또는 `.tflite`)으로 제공됩니다. 따라서 파이썬 환경에서 **FreiHAND 데이터셋을 주입하여 가중치를 재업데이트(Fine-tuning/Backpropagation)하는 것이 구조적으로 불가능**합니다. 
    *결론적으로 본 실험에서 MediaPipe는 'FreiHAND 학습 모델(실험군)'과 성능을 비교하기 위한 **'절대적 성능을 가진 통제군(Baseline)'**으로서, API(`HandLandmarker.create_from_options`)를 호출하여 순수하게 3D 좌표를 추출하는(Off-the-shelf feature extractor) 용도로만 사용됩니다.*

---

## 2. 손의 회전(Screw) 각도 추정 알고리즘 (SVD 기반 Kabsch 알고리즘)

웹캠에서 얻은 21개의 관절 데이터 중 특정 시점(예: 사용자가 버튼을 누른 시점)을 기준으로, 현재 손(또는 손에 쥔 가상 객체)이 **3차원 공간상에서 얼마나 스크류(회전)했는지**를 계산하기 위해 `measure_angle.py`에서 **특이값 분해(SVD, Singular Value Decomposition)** 알고리즘을 사용했습니다.

### 2.1. 왜 이 알고리즘을 사용했는가? (Rationale)
손가락은 개별적으로 굽혀지는 '비강체(관절형)' 특성을 지닙니다. 단순히 손목 좌표 하나만으로 회전을 구하면 각 손가락의 미세한 비틀림이나 물체를 쥐고 돌리는 모션을 정확히 캡처할 수 없습니다. 따라서 **가장 안정적인 3개의 끝점(엄지, 검지, 중지 끝)**을 선택하여, 두 3D 점군(Point Cloud) 사이의 최적의 강체 회전 행렬(Rigid Rotation Matrix)을 오차 없이 구하는 **직교 프로크루스테스 문제(Orthogonal Procrustes Problem)**의 해법인 SVD를 도입했습니다.

### 2.2. 수학적 유도 과정 (Mathematical Derivation)

1.  **중심점(Centroid) 정렬**:
    버튼을 누른 초기 상태의 3개 관절 좌표를 $P_{init}$, 현재 상태의 3개 관절 좌표를 $P_{curr}$라고 합니다. 회전만을 순수하게 구하기 위해, 각 점군의 무게중심($\mu$)을 빼서 원점(0,0,0)으로 이동시킵니다.
    $$A = P_{init} - \mu_{init}$$
    $$B = P_{curr} - \mu_{curr}$$

2.  **공분산 행렬(Covariance Matrix) 계산**:
    두 점군이 어느 방향으로 상관관계를 가지며 변형되었는지 나타내는 행렬 $H$를 계산합니다.
    $$H = A^T B$$

3.  **특이값 분해(SVD)**:
    행렬 $H$에 SVD를 적용하여 회전 성분을 분리합니다.
    $$U, S, V^T = SVD(H)$$

4.  **최적 회전 행렬(Rotation Matrix) $R$ 도출**:
    $$R = V U^T$$
    이때, $R$의 행렬식(Determinant)이 음수($<0$)가 나올 경우 회전이 아닌 '거울 반사(Reflection)'가 일어난 것이므로, 수학적 보정($V$의 마지막 열 부호 반전)을 통해 올바른 회전 행렬로 강제 조정합니다. (코드 내 짐벌락 방지 로직)

5.  **오일러 각(Euler Angles) 변환**:
    구해진 $3 \times 3$ 회전 행렬 $R$을 삼각함수(`arctan2`)를 이용해 우리가 이해할 수 있는 **Roll(X축 회전), Pitch(Y축 회전), Yaw(Z축 회전)** 각도(Degree)로 변환하여 화면에 출력합니다.

---

## 3. 요약 및 향후 실험 방향

현재 파이프라인은 MediaPipe의 압도적인 3D 추론 성능과 SVD 기반의 수학적 회전 변환 공식을 통해, 가상 객체의 정밀한 스크류 각도를 실시간으로 계산하고 있습니다. 

진행될 **[실험군 학습]**에서는, MediaPipe를 걷어내고 **"FreiHAND 데이터셋으로 학습된 ResNet 기반 자체 모델"**을 위치시킵니다. 이후 **정확히 동일한 SVD 회전 공식**을 통과시켰을 때, MediaPipe와 자체 모델 간의 회전 각도 오차율(MSE 등)을 비교하는 것이 본 연구의 최종 목표가 됩니다.

# 🖐️ VR 환경 다중 카메라 핑거 트래킹 시스템 (VR Multi-Camera Finger Tracking)

> 3대의 카메라 비전 데이터를 융합하여 정밀한 손가락 움직임을 추적하고, 이를 Unity VR 환경에 실시간으로 연동하는 시스템입니다.

## 1. 프로젝트 개요 (Overview)
기존 단일 카메라 기반 핸드 트래킹의 사각지대 한계를 극복하기 위해 기획되었습니다. 3각 구도의 다중 카메라 데이터를 종합하여 손가락 관절의 인식 정확도를 대폭 향상시켰으며, 이를 통해 인간의 섬세한 수작업을 VR 환경에서 정밀하게 모방하고 나아가 자동화 및 시뮬레이션에 기여할 수 있는 기반을 마련했습니다.

## 2. 기술 스택 (Tech Stack)
* **Core:** Python, C#
* **AI & Vision:** MediaPipe, OpenCV
* **Engine:** Unity 3D

## 3. 핵심 기능 (Key Features)
* **다중 카메라 데이터 동기화:** 3대의 카메라에서 입력되는 영상 스트림의 딜레이를 최소화하고 프레임을 동기화
* **정밀 핑거 트래킹 로직:** MediaPipe 랜드마크 데이터를 추출하여 3차원 공간 좌표로 변환
* **VR 환경 실시간 렌더링:** 추출된 좌표 데이터를 Unity 엔진으로 전송하여 가상 손 객체와 동기화

## 4. 기술적 문제 해결 (Troubleshooting) ⭐️
*(포트폴리오에서 가장 중요한 섹션입니다. 개발하며 겪은 문제를 어떻게 해결했는지 구체적으로 적어주세요.)*

**문제 1: 3대의 카메라 랜드마크 데이터 충돌 및 오차 발생**
* **상황:** 각 카메라의 시점에 따라 특정 손가락 랜드마크가 가려지거나, 좌표 값이 튀는 현상 발생.
* **해결:** (여기에 본인이 직접 해결한 방식 작성. 예: 3대의 데이터 중 신뢰도가 가장 높은 프레임 데이터를 가중 평균하는 필터링 로직 구현)

**문제 2: 파이썬(비전 인식)과 Unity(VR) 간의 통신 지연**
* **상황:** 프레임 단위로 실시간 전송 시 병목 현상으로 인해 VR 화면에서 끊김 발생.
* **해결:** (여기에 해결 방식 작성. 예: 통신 프로토콜 최적화 및 비동기 처리 적용)

## 5. 실행 화면 (Demo / Screenshots)
<img width="4032" height="3024" alt="세로" src="https://github.com/user-attachments/assets/71fd74fb-623e-41f1-9b62-b84f4a1d9a34" />
<img width="3600" height="2100" alt="세로" src="https://github.com/user-attachments/assets/a0d1a9ac-7526-4c21-88bf-47c9a99e68a8" />


## 6. 설치 및 실행 방법 (Getting Started)
```bash
# 1. 저장소 클론
$ git clone [https://github.com/ansgustn/handtracking.git](https://github.com/ansgustn/handtracking.git)

# 2. 파이썬 의존성 패키지 설치
$ pip install -r requirements.txt

# 3. 비전 추적 모듈 실행
$ python main_tracker.py

import cv2

def main():
    print("OpenCV 카메라 스캔 시작...")
    available_cams = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"카메라 인덱스 {i} 정상 작동 확인! (해상도: {frame.shape[1]}x{frame.shape[0]})")
                available_cams.append(i)
            else:
                print(f"카메라 인덱스 {i} 열렸으나 프레임 읽기 실패")
            cap.release()
        else:
            print(f"카메라 인덱스 {i} 접근 불가")
            
    print(f"\n총 {len(available_cams)}대의 UVC 카메라가 발견되었습니다: {available_cams}")

if __name__ == "__main__":
    main()

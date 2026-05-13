import cv2
import pykinect_azure as pykinect
import mediapipe as mp

def main():
    # Azure Kinect SDK (dll 파일 등) 초기화
    # 만약 경로 오류가 나면 initialize_libraries()에 dll 파일 경로를 명시해야 할 수 있습니다.
    pykinect.initialize_libraries()

    # 카메라 설정
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    
    # Step 1: RGB와 Depth 프레임 동시 획득
    # 동기화 옵션을 켜서 같은 시점의 Color와 Depth를 가져오도록 합니다.
    device_config.synchronized_images_only = True

    print("Azure Kinect 기기를 시작합니다...")
    device = pykinect.start_device(config=device_config)

    # MediaPipe Hands 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    print("영상을 캡처 중입니다. 종료하려면 'q'를 누르세요.")

    try:
        while True:
            # 프레임 캡처 대기 및 업데이트
            capture = device.update()

            # Color 이미지 가져오기
            ret_color, color_image = capture.get_color_image()
            if not ret_color:
                continue

            # Step 2: 시점 일치 (Alignment)
            # 가장 중요한 단계입니다. Depth 이미지를 Color 카메라 렌즈의 시점으로 변환.
            # 이 과정을 거쳐야만 Color 이미지의 (x,y) 픽셀과 Depth의 (x,y) 픽셀이 같은 물리적 위치를 참고하게 됩니다.
            ret_aligned_depth, aligned_depth_image = capture.get_transformed_depth_image()
            if not ret_aligned_depth:
                continue

            # MediaPipe 처리를 위해 BGRA -> RGB 변환
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            
            # Step 3: MediaPipe로 2D 랜드마크 추출
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Color 이미지 위에 랜드마크 뼈대 그리기
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 21개의 손 관절(랜드마크)에 대해 처리
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = color_image.shape
                        
                        # MediaPipe의 정규화된 좌표(0.0~1.0)를 실제 픽셀 좌표(u, v)로 변환
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        # 픽셀 좌표가 이미지 범위를 벗어나지 않는지 확인
                        if 0 <= cx < w and 0 <= cy < h:
                            # Step 4: 2D 픽셀을 3D 공간 좌표로 역투영 (Deprojection)
                            # Align된 Depth 맵에서 해당 (cy, cx) 픽셀의 깊이(d) 값(밀리미터)을 읽습니다.
                            depth_val = aligned_depth_image[cy, cx]
                            
                            # 0보다 큰 유효한 깊이 값일 경우에만 계산 진행
                            if depth_val > 0:
                                # 카메라 내부 파라미터(Intrinsic parameters)를 이용해
                                # 2D 픽셀과 깊이를 3D 공간 좌표 (X, Y, Z)로 변환합니다.
                                # pykinect_azure의 convert_2d_to_3d 함수를 사용합니다.
                                # 기준 카메라 뷰를 Color로 설정했으므로 source/target camera를 COLOR로 맞춰줍니다.
                                valid, point_3d = device.calibration.convert_2d_to_3d(
                                    source_point2d=[cx, cy], 
                                    source_depth=depth_val, 
                                    source_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR, 
                                    target_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR
                                )
                                
                                if valid:
                                    # point_3d는 [X, Y, Z] 형태의 리스트/배열 (단위: 밀리미터 mm)
                                    
                                    # 예시: 검지 손가락 끝(id: 8)에 대해서만 값 표시
                                    if id == 8:
                                        # 화면에 Text 출력
                                        coord_text = f"X:{point_3d[0]:.0f} Y:{point_3d[1]:.0f} Z:{point_3d[2]:.0f}mm"
                                        cv2.putText(color_image, coord_text, (cx + 10, cy - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # 터미널에 디버깅용 출력
                                        # print(f"[검지 손가락 끝] 2D 픽셀: ({cx}, {cy}) | 깊이: {depth_val}mm | 3D(X,Y,Z): {point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}")

            # 화면에 보여주기
            cv2.imshow("Azure Kinect & MediaPipe Hand Tracking", color_image)
            
            # 깊이 맵(Depth map) 시각화 - 옵션
            # 거리에 따라 색상을 다르게 보여줌
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Aligned Depth", depth_colormap)

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 안전한 종료를 위해 디바이스 닫기 및 창 닫기
        print("프로그램을 종료합니다.")
        if 'device' in locals():
            device.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

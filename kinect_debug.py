import pykinect_azure as pykinect
import sys

def test_kinect_config(sync_only, mode):
    print(f"\n--- [TEST] synchronized_images_only: {sync_only}, wired_sync_mode: {mode} ---")
    pykinect.initialize_libraries()
    
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.synchronized_images_only = sync_only
    if mode is not None:
        device_config.wired_sync_mode = mode
        
    try:
        device = pykinect.start_device(config=device_config)
        print(">> 성공! 카메라가 정상적으로 켜졌습니다.")
        device.close()
        return True
    except SystemExit:
        print(">> 실패! (Start K4A cameras failed!)")
        return False
    except Exception as e:
        print(f">> 예외 발생: {e}")
        return False

if __name__ == "__main__":
    print("[INFO] Azure Kinect 연결 진단 스크립트")
    # 1. 이전 작동했던 기존 설정 그대로
    test_1 = test_kinect_config(True, None)
    
    # 2. Sync 해제 (Color MCU 부하 감소)
    if not test_1:
         test_2 = test_kinect_config(False, None)
    
    # 3. Standalone 명시
    if not test_1 and not test_2:
         test_3 = test_kinect_config(True, pykinect.K4A_WIRED_SYNC_MODE_STANDALONE)

    print("\n[INFO] 모든 테스트 완료.")

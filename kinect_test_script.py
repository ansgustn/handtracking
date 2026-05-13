import pykinect_azure as pykinect
import cv2

pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
device = pykinect.start_device(config=device_config)

try:
    capture = device.update(timeout_in_ms=1000)
    print("Capture:", capture)
    ret_color, color_img = capture.get_color_image()
    print("Color:", ret_color)
    ret_depth, aligned_depth = capture.get_transformed_depth_image()
    print("Depth:", ret_depth)
except Exception as e:
    print("Exception:", e)

device.close()

cmake_minimum_required(VERSION 3.16)
project(CupMediaPipeDualCam CXX)

# Force C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# MediaPipe & Dependencies Configuration (Windows Setup Guide)
#
# Note: MediaPipe is built using Bazel by default. To link MediaPipe in a CMake 
# project, you must point to your built MediaPipe C++ library artifacts.
# ---------------------------------------------------------------------------

# Path to MediaPipe repository installation or built directories
# Set this to your local MediaPipe build output directory
set(MEDIAPIPE_DIR "" CACHE PATH "Path to MediaPipe workspace")

include_directories(
    ${MEDIAPIPE_DIR}
    ${MEDIAPIPE_DIR}/bazel-bin
    # Add other dependency paths if necessary (e.g. protobuf, abseil, flatbuffers)
)

# Define executable
add_executable(cup_mediapipe_dualcam cup_mediapipe_dualcam.cpp)

# Link OpenCV
target_link_libraries(cup_mediapipe_dualcam PRIVATE ${OpenCV_LIBS})

# Link MediaPipe and dependencies
# Un-comment and adjust library names depending on how you compiled MediaPipe:
# target_link_libraries(cup_mediapipe_dualcam PRIVATE
#     ${MEDIAPIPE_DIR}/bazel-bin/mediapipe/tasks/cc/vision/hand_landmarker/libhand_landmarker.a
#     # Add other dependency libraries like abseil, protobuf, glog, etc.
# )

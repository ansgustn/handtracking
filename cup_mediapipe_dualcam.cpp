#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mutex>
#include <thread>
#include <memory>
#include <fstream>
#include <iomanip>
#include <map>
#include <atomic>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// MediaPipe Headers (Tasks API)
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

using namespace mediapipe::tasks::vision::hand_landmarker;
using namespace mediapipe::tasks::core;
using namespace mediapipe::tasks::vision;

// Custom 2D coordinates for custom hand drawing
const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},        // Thumb
    {0, 5}, {5, 6}, {6, 7}, {7, 8},        // Index
    {5, 9}, {9, 10}, {10, 11}, {11, 12},   // Middle
    {9, 13}, {13, 14}, {14, 15}, {15, 16}, // Ring
    {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20} // Pinky
};

// Struct to represent a 3D point (compatible with MediaPipe float3)
struct Point3D {
    float x;
    float y;
    float z;
};

// Struct for experimental trials data point
struct TrialPoint {
    double elapsed_time;
    double angle;
};

void drawCustomLandmarks(cv::Mat& image, const std::vector<Point3D>& landmarks) {
    int h = image.rows;
    int w = image.cols;
    std::vector<cv::Point> points;
    points.reserve(landmarks.size());

    // Draw landmark joints
    for (const auto& pt : landmarks) {
        int cx = static_cast<int>(pt.x * w);
        int cy = static_cast<int>(pt.y * h);
        points.push_back(cv::Point(cx, cy));
        cv::circle(image, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    // Draw connection lines
    for (const auto& connection : HAND_CONNECTIONS) {
        if (connection.first < static_cast<int>(points.size()) && 
            connection.second < static_cast<int>(points.size())) {
            cv::line(image, points[connection.first], points[connection.second], cv::Scalar(0, 255, 0), 2);
        }
    }
}

// CameraThread class for handling camera streams, MediaPipe landmarking, and calculations
class CameraThread {
private:
    int cap_device;
    int cap_idx;
    std::thread thread_obj;
    std::mutex data_mutex;
    std::atomic<bool> running{false};

    // Shared data members (protected by data_mutex)
    cv::Mat current_frame;
    double abs_angle = 0.0;
    bool has_abs_angle = false;
    double rotation = 0.0;
    bool has_rotation = false;
    double baseline_angle = 0.0;
    bool has_baseline = false;

    // Filter states (local to thread execution)
    double smoothed_angle = 0.0;
    bool has_smoothed_angle = false;
    double alpha_angle = 0.15; // Angle smoothing strength

    std::vector<Point3D> smoothed_landmarks;
    bool has_smoothed_landmarks = false;
    double alpha_lm = 0.15; // Landmark coordinate smoothing strength
    int64_t last_timestamp_ms = -1;

    void run() {
        cv::VideoCapture cap(cap_device, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            std::cerr << "[오류] Cam " << cap_idx << " 장치를 열 수 없습니다." << std::endl;
            running = false;
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // Initialize MediaPipe HandLandmarker C++ Options
        auto options = std::make_unique<HandLandmarkerOptions>();
        options->base_options.model_asset_path = "hand_landmarker.task";
        options->running_mode = RunningMode::VIDEO;
        options->num_hands = 1;
        options->min_hand_detection_confidence = 0.7f;
        options->min_tracking_confidence = 0.7f;

        auto landmarker_status = HandLandmarker::Create(std::move(options));
        if (!landmarker_status.ok()) {
            std::cerr << "[오류] MediaPipe HandLandmarker 생성 실패 (Cam " << cap_idx << "): " 
                      << landmarker_status.status().message() << std::endl;
            running = false;
            cap.release();
            return;
        }
        auto landmarker = std::move(landmarker_status.value());

        while (running) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::flip(frame, frame, 1);
            cv::Mat rgb_frame;
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

            // Construct MediaPipe Image from OpenCV Mat
            auto image_frame = std::make_shared<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows,
                rgb_frame.step, rgb_frame.data, [](uint8_t*) {} // Non-owning deleter
            );
            mediapipe::Image mp_image(image_frame);

            // Generate monotonically increasing timestamp
            auto now = std::chrono::steady_clock::now();
            int64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            if (timestamp_ms <= last_timestamp_ms) {
                timestamp_ms = last_timestamp_ms + 1;
            }
            last_timestamp_ms = timestamp_ms;

            // Detect landmarks
            auto detect_result = landmarker->DetectForVideo(mp_image, timestamp_ms);

            int h = frame.rows;
            int w = frame.cols;

            cv::putText(frame, "Cam " + std::to_string(cap_idx), cv::Point(w - 150, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);

            bool hand_detected = false;
            double local_abs_angle = 0.0;
            bool local_has_abs_angle = false;
            double local_rotation = 0.0;
            bool local_has_rotation = false;

            if (detect_result.ok()) {
                const auto& results = detect_result.value();
                if (!results.hand_landmarks.empty()) {
                    hand_detected = true;
                    // First detected hand landmarks
                    const auto& raw_landmarks = results.hand_landmarks[0].landmarks;

                    // 1. Double EMA Landmark coordinate smoothing to eliminate jitter
                    if (!has_smoothed_landmarks || smoothed_landmarks.size() != raw_landmarks.size()) {
                        smoothed_landmarks.resize(raw_landmarks.size());
                        for (size_t i = 0; i < raw_landmarks.size(); ++i) {
                            smoothed_landmarks[i] = { raw_landmarks[i].x, raw_landmarks[i].y, raw_landmarks[i].z };
                        }
                        has_smoothed_landmarks = true;
                    } else {
                        std::vector<Point3D> new_smoothed(raw_landmarks.size());
                        for (size_t i = 0; i < raw_landmarks.size(); ++i) {
                            new_smoothed[i].x = alpha_lm * raw_landmarks[i].x + (1.0f - alpha_lm) * smoothed_landmarks[i].x;
                            new_smoothed[i].y = alpha_lm * raw_landmarks[i].y + (1.0f - alpha_lm) * smoothed_landmarks[i].y;
                            new_smoothed[i].z = alpha_lm * raw_landmarks[i].z + (1.0f - alpha_lm) * smoothed_landmarks[i].z;
                        }

                        // Reset smoothing filter if hand wrist moves too fast (prevents ghosting)
                        if (std::abs(raw_landmarks[0].x - smoothed_landmarks[0].x) > 0.05f) {
                            for (size_t i = 0; i < raw_landmarks.size(); ++i) {
                                smoothed_landmarks[i] = { raw_landmarks[i].x, raw_landmarks[i].y, raw_landmarks[i].z };
                            }
                        } else {
                            smoothed_landmarks = new_smoothed;
                        }
                    }

                    // Render skeleton with smoothed coordinates
                    drawCustomLandmarks(frame, smoothed_landmarks);

                    // 2. Extract keypoints (4: Thumb Tip, 8: Index Tip)
                    if (smoothed_landmarks.size() > 8) {
                        Point3D thumb_tip = smoothed_landmarks[4];
                        Point3D index_tip = smoothed_landmarks[8];

                        int thumb_x = static_cast<int>(thumb_tip.x * w);
                        int thumb_y = static_cast<int>(thumb_tip.y * h);
                        int index_x = static_cast<int>(index_tip.x * w);
                        int index_y = static_cast<int>(index_tip.y * h);

                        // Draw thumb-to-index link and tip circles
                        cv::line(frame, cv::Point(thumb_x, thumb_y), cv::Point(index_x, index_y), cv::Scalar(0, 255, 255), 2);
                        cv::circle(frame, cv::Point(thumb_x, thumb_y), 5, cv::Scalar(0, 0, 255), -1);
                        cv::circle(frame, cv::Point(index_x, index_y), 5, cv::Scalar(255, 0, 0), -1);

                        // Compute 3D rotation using Z-axis depth data (Yaw)
                        double dx_3d = index_tip.x - thumb_tip.x;
                        double dz_3d = index_tip.z - thumb_tip.z;

                        // Calculate raw angle (Z values smaller as object moves closer to camera)
                        double raw_abs_angle = std::atan2(-dz_3d, dx_3d) * 180.0 / M_PI;

                        // Apply EMA to smooth absolute angle and handle circular wrappings (-180 to 180)
                        if (!has_smoothed_angle) {
                            smoothed_angle = raw_abs_angle;
                            has_smoothed_angle = true;
                        } else {
                            double diff = raw_abs_angle - smoothed_angle;
                            if (diff > 180.0) raw_abs_angle -= 360.0;
                            else if (diff < -180.0) raw_abs_angle += 360.0;

                            smoothed_angle = alpha_angle * raw_abs_angle + (1.0 - alpha_angle) * smoothed_angle;

                            if (smoothed_angle > 180.0) smoothed_angle -= 360.0;
                            else if (smoothed_angle < -180.0) smoothed_angle += 360.0;
                        }

                        local_abs_angle = smoothed_angle;
                        local_has_abs_angle = true;

                        // Lock data to read baseline configuration safely
                        std::lock_guard<std::mutex> lock(data_mutex);
                        if (has_baseline) {
                            double rotation_diff = local_abs_angle - baseline_angle;
                            if (rotation_diff > 180.0) rotation_diff -= 360.0;
                            else if (rotation_diff < -180.0) rotation_diff += 360.0;

                            rotation = rotation_diff;
                            has_rotation = true;

                            local_rotation = rotation;
                            local_has_rotation = true;
                        }
                    }
                }
            }

            if (!hand_detected) {
                // Clear smoothing states when hand tracker loses detection
                has_smoothed_angle = false;
                has_smoothed_landmarks = false;
                cv::putText(frame, "Hand Not Detected", cv::Point(20, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            } else {
                // Text overlay display on camera frames
                char buf[100];
                if (local_has_rotation) {
                    std::snprintf(buf, sizeof(buf), "Cam Rot: %.1f deg", local_rotation);
                    cv::putText(frame, buf, cv::Point(20, 100),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                }
                std::snprintf(buf, sizeof(buf), "Cam Abs: %.1f deg", local_abs_angle);
                cv::putText(frame, buf, cv::Point(20, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            }

            // Thread-safe update of variables shared with the main loop
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                current_frame = frame.clone();
                abs_angle = local_abs_angle;
                has_abs_angle = local_has_abs_angle;
                if (!hand_detected) {
                    has_rotation = false;
                } else {
                    has_rotation = local_has_rotation;
                }
            }
        }

        cap.release();
        landmarker->Close();
    }

public:
    CameraThread(int src, int idx) : cap_device(src), cap_idx(idx) {}

    ~CameraThread() {
        stop();
    }

    void start() {
        if (!running) {
            running = true;
            thread_obj = std::thread(&CameraThread::run, this);
        }
    }

    void stop() {
        if (running) {
            running = false;
            if (thread_obj.joinable()) {
                thread_obj.join();
            }
        }
    }

    void setBaseline() {
        std::lock_guard<std::mutex> lock(data_mutex);
        if (has_abs_angle) {
            baseline_angle = abs_angle;
            has_baseline = true;
            std::cout << "Cam " << cap_idx << " 기준 각도 설정: " << std::fixed << std::setprecision(1) << baseline_angle << "도" << std::endl;
        }
    }

    void clearBaseline() {
        std::lock_guard<std::mutex> lock(data_mutex);
        has_baseline = false;
        has_rotation = false;
    }

    struct ThreadOutput {
        cv::Mat frame;
        double rotation;
        bool has_rotation;
        double abs_angle;
        bool has_abs_angle;
    };

    ThreadOutput getOutput() {
        std::lock_guard<std::mutex> lock(data_mutex);
        ThreadOutput out;
        if (!current_frame.empty()) {
            out.frame = current_frame.clone();
        }
        out.rotation = rotation;
        out.has_rotation = has_rotation;
        out.abs_angle = abs_angle;
        out.has_abs_angle = has_abs_angle;
        return out;
    }
};

// Function to calculate statistical metrics and save data points to CSV
void saveResults(const std::map<int, std::vector<TrialPoint>>& trials) {
    if (trials.empty()) {
        std::cout << "[오류] 저장할 실험 데이터가 없습니다!" << std::endl;
        return;
    }

    std::string csv_filename = "experiment_results.csv";
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "[오류] CSV 저장 중 문제가 발생했습니다: 파일을 열 수 없습니다." << std::endl;
        return;
    }

    csv_file << "Trial,Elapsed_Time_Sec,Angle_Deg\n";
    std::cout << "\n--- 실험 통계 요약 ---" << std::endl;

    for (const auto& pair : trials) {
        int trial_idx = pair.first;
        const auto& data = pair.second;
        if (data.empty()) continue;

        double max_ang = -9999.0;
        double min_ang = 9999.0;
        double sum_ang = 0.0;

        for (const auto& pt : data) {
            csv_file << trial_idx << "," 
                     << std::fixed << std::setprecision(3) << pt.elapsed_time << ","
                     << std::fixed << std::setprecision(2) << pt.angle << "\n";

            if (pt.angle > max_ang) max_ang = pt.angle;
            if (pt.angle < min_ang) min_ang = pt.angle;
            sum_ang += pt.angle;
        }

        double mean_ang = sum_ang / data.size();
        double range_ang = max_ang - min_ang;

        std::cout << "실험 " << trial_idx << ": 데이터 " << data.size() << "개 | "
                  << "최대 " << std::fixed << std::setprecision(1) << max_ang << "° | "
                  << "최소 " << min_ang << "° | "
                  << "변화폭 " << range_ang << "°" << std::endl;
    }

    csv_file.close();
    std::cout << "\n[성공] 실험 데이터가 '" << csv_filename << "'에 성공적으로 저장되었습니다." << std::endl;
    std::cout << "[알림] C++는 자체 그래프 기능을 내장하고 있지 않습니다. '" << csv_filename 
              << "' 데이터를 시각화하려면 python 스크립트나 외부 분석 도구를 활용해 주세요." << std::endl;
}

int main() {
    std::cout << "카메라를 준비 중입니다. 잠시만 기다려주세요..." << std::endl;

    // Create threads for Cam 1 (Index 0) and Cam 2 (Index 1)
    CameraThread cam1_thread(0, 1);
    CameraThread cam2_thread(1, 2);

    cam1_thread.start();
    cam2_thread.start();

    std::cout << "\n['s' 키]를 누르면 두 카메라에서 보이는 현재 손가락 각도를 0도(기준점)로 설정합니다." << std::endl;
    std::cout << "['q' 키]를 누르면 종료합니다." << std::endl;
    std::cout << "[Space 키]를 누르면 현재 실험의 실시간 녹화를 시작/중지합니다." << std::endl;
    std::cout << "['g' 키]를 누르면 수집된 데이터를 CSV 파일로 저장합니다." << std::endl;
    std::cout << "['c' 키]를 누르면 수집된 모든 실험 데이터를 초기화합니다.\n" << std::endl;

    // Experiment trial tracking variables
    std::map<int, std::vector<TrialPoint>> trials;
    int current_trial = 1;
    const int MAX_TRIALS = 10;
    bool is_recording = false;
    std::chrono::steady_clock::time_point recording_start_time;

    // Block until at least one frame is received
    while (true) {
        auto d1 = cam1_thread.getOutput();
        auto d2 = cam2_thread.getOutput();
        if (!d1.frame.empty() || !d2.frame.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (true) {
        auto d1 = cam1_thread.getOutput();
        auto d2 = cam2_thread.getOutput();

        cv::Mat display_frame;

        // Horizontally stack cameras
        if (!d1.frame.empty() && !d2.frame.empty()) {
            cv::Mat f1 = d1.frame;
            cv::Mat f2 = d2.frame;
            if (f1.rows != f2.rows) {
                int new_w = static_cast<int>(f2.cols * static_cast<double>(f1.rows) / f2.rows);
                cv::resize(f2, f2, cv::Size(new_w, f1.rows));
            }
            cv::hconcat(f1, f2, display_frame);

            // Resize if the horizontal window exceeds standard 1080p width
            if (display_frame.cols > 1920) {
                cv::resize(display_frame, display_frame, cv::Size(display_frame.cols / 2, display_frame.rows / 2));
            }
        } else if (!d1.frame.empty()) {
            display_frame = d1.frame;
        } else if (!d2.frame.empty()) {
            display_frame = d2.frame;
        }

        if (!display_frame.empty()) {
            // Circular Mean calculations for combined rotation
            double combined_rotation = 0.0;
            bool has_combined_rotation = false;

            if (d1.has_rotation && d2.has_rotation) {
                double rad1 = d1.rotation * M_PI / 180.0;
                double rad2 = d2.rotation * M_PI / 180.0;
                double sin_sum = std::sin(rad1) + std::sin(rad2);
                double cos_sum = std::cos(rad1) + std::cos(rad2);
                combined_rotation = std::atan2(sin_sum, cos_sum) * 180.0 / M_PI;
                has_combined_rotation = true;
            } else if (d1.has_rotation) {
                combined_rotation = d1.rotation;
                has_combined_rotation = true;
            } else if (d2.has_rotation) {
                combined_rotation = d2.rotation;
                has_combined_rotation = true;
            }

            // Recording functionality
            double elapsed_time = 0.0;
            if (is_recording) {
                auto now_time = std::chrono::steady_clock::now();
                elapsed_time = std::chrono::duration<double>(now_time - recording_start_time).count();
                if (has_combined_rotation) {
                    trials[current_trial].push_back({elapsed_time, combined_rotation});
                }
            }

            // Display Combined Rotation on screen
            if (has_combined_rotation) {
                int dh = display_frame.rows;
                int dw = display_frame.cols;
                char buf[100];
                std::snprintf(buf, sizeof(buf), "Total Rotation: %.1f deg", combined_rotation);
                std::string text(buf);
                int font = cv::FONT_HERSHEY_SIMPLEX;
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(text, font, 1.5, 3, &baseline);
                int text_x = (dw - text_size.width) / 2;
                int text_y = dh - 40;

                // Draw background box for text legibility
                cv::rectangle(display_frame, 
                              cv::Point(text_x - 10, text_y - text_size.height - 10),
                              cv::Point(text_x + text_size.width + 10, text_y + 10), 
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(display_frame, text, cv::Point(text_x, text_y),
                            font, 1.5, cv::Scalar(0, 255, 255), 3);
            }

            // Render HUD box panel (alpha blended semi-transparent overlay)
            int panel_x1 = 20, panel_y1 = 20;
            int panel_x2 = 420, panel_y2 = 160;
            cv::Mat overlay = display_frame.clone();
            cv::rectangle(overlay, cv::Point(panel_x1, panel_y1), cv::Point(panel_x2, panel_y2), cv::Scalar(30, 30, 30), cv::FILLED);
            cv::addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame);
            cv::rectangle(display_frame, cv::Point(panel_x1, panel_y1), cv::Point(panel_x2, panel_y2), cv::Scalar(100, 100, 100), 2);

            int font = cv::FONT_HERSHEY_SIMPLEX;
            if (is_recording) {
                // Blink REC dot every 500ms
                auto now_time = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now_time.time_since_epoch()).count();
                cv::Scalar dot_color = (static_cast<int>(sec * 2) % 2 == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(50, 50, 150);
                cv::circle(display_frame, cv::Point(40, 55), 8, dot_color, -1);

                std::string status_text = "REC - Trial " + std::to_string(current_trial) + "/" + std::to_string(MAX_TRIALS);
                cv::putText(display_frame, status_text, cv::Point(65, 65), font, 0.75, cv::Scalar(0, 0, 255), 2);

                size_t data_count = trials[current_trial].size();
                char time_buf[100];
                std::snprintf(time_buf, sizeof(time_buf), "Time: %.1fs | Pts: %zu", elapsed_time, data_count);
                cv::putText(display_frame, time_buf, cv::Point(40, 105), font, 0.65, cv::Scalar(255, 255, 255), 1);

                cv::putText(display_frame, "Press SPACE to STOP Recording", cv::Point(40, 140), font, 0.55, cv::Scalar(0, 255, 255), 1);
            } else {
                if (trials.size() >= static_cast<size_t>(MAX_TRIALS)) {
                    cv::circle(display_frame, cv::Point(40, 55), 8, cv::Scalar(0, 215, 255), -1); // Gold dot
                    std::string status_text = "COMPLETED " + std::to_string(MAX_TRIALS) + "/" + std::to_string(MAX_TRIALS) + " Trials";
                    cv::putText(display_frame, status_text, cv::Point(65, 65), font, 0.75, cv::Scalar(0, 215, 255), 2);
                    cv::putText(display_frame, "Press 'g' to SAVE | 'c' to RESET", cv::Point(40, 110), font, 0.65, cv::Scalar(0, 255, 255), 2);
                } else {
                    cv::circle(display_frame, cv::Point(40, 55), 8, cv::Scalar(0, 255, 0), -1); // Green dot
                    std::string status_text = "READY - Trial " + std::to_string(current_trial) + "/" + std::to_string(MAX_TRIALS);
                    cv::putText(display_frame, status_text, cv::Point(65, 65), font, 0.75, cv::Scalar(0, 255, 0), 2);
                    cv::putText(display_frame, "Press SPACE to START Recording", cv::Point(40, 110), font, 0.6, cv::Scalar(200, 200, 200), 1);
                    cv::putText(display_frame, "Press 'c' to RESET data", cv::Point(40, 140), font, 0.5, cv::Scalar(150, 150, 150), 1);
                }
            }

            cv::imshow("Dual Camera Finger Rotation Tracker (C++)", display_frame);
        }

        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 'Q') {
            break;
        } else if (key == 's' || key == 'S') {
            cam1_thread.setBaseline();
            cam2_thread.setBaseline();
        } else if (key == 32) { // Space key
            if (!is_recording) {
                if (current_trial <= MAX_TRIALS) {
                    is_recording = true;
                    recording_start_time = std::chrono::steady_clock::now();
                    trials[current_trial] = std::vector<TrialPoint>();
                    std::cout << "\n>>> [실험 " << current_trial << "/" << MAX_TRIALS << "] 녹화를 시작합니다. (Space 키를 누르면 종료)" << std::endl;
                } else {
                    std::cout << "\n[경고] 이미 " << MAX_TRIALS << "번의 실험을 완료했습니다. 'g' 키로 저장하거나 'c' 키로 초기화하세요." << std::endl;
                }
            } else {
                is_recording = false;
                size_t data_points = trials[current_trial].size();
                std::cout << ">>> [실험 " << current_trial << "/" << MAX_TRIALS << "] 녹화 완료! 수집된 데이터 포인트: " << data_points << "개" << std::endl;

                if (current_trial < MAX_TRIALS) {
                    current_trial++;
                } else {
                    current_trial = MAX_TRIALS + 1; // Mark completed
                    std::cout << "\n🎉 모든 10회의 실험 데이터 수집이 완료되었습니다!" << std::endl;
                    std::cout << ">>> 'g' 키를 눌러 CSV 데이터 저장을 수행하세요." << std::endl;
                }
            }
        } else if (key == 'g' || key == 'G') {
            if (!trials.empty()) {
                std::cout << "\n[작업] 실험 데이터 저장 및 통계 계산을 진행합니다..." << std::endl;
                saveResults(trials);
            } else {
                std::cout << "\n[오류] 기록된 실험 데이터가 없습니다. 먼저 실험을 시작해 주세요!" << std::endl;
            }
        } else if (key == 'c' || key == 'C') {
            trials.clear();
            current_trial = 1;
            is_recording = false;
            cam1_thread.clearBaseline();
            cam2_thread.clearBaseline();
            std::cout << "\n🧹 모든 실험 데이터가 초기화되었습니다. 처음부터 다시 시작합니다." << std::endl;
        }
    }

    // Stop threads gracefully
    cam1_thread.stop();
    cam2_thread.stop();
    cv::destroyAllWindows();

    return 0;
}

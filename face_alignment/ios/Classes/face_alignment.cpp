#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <vector>

cv::Mat align_face(const double* landmarks, int landmark_count, const std::string& filepath) {
    if (landmark_count != 68) {
        throw std::runtime_error("Expected 68 landmarks, got " + std::to_string(landmark_count));
    }

    // Load image
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    // Convert landmarks to Eigen vectors
    std::vector<Eigen::Vector2d> lm;
    for (int i = 0; i < landmark_count; ++i) {
        lm.emplace_back(landmarks[i * 2], landmarks[i * 2 + 1]);
    }

    // Geometric calculations (translated from Python)
    std::vector<Eigen::Vector2d> lm_chin(lm.begin(), lm.begin() + 17);
    std::vector<Eigen::Vector2d> lm_eye_left(lm.begin() + 36, lm.begin() + 42);
    std::vector<Eigen::Vector2d> lm_eye_right(lm.begin() + 42, lm.begin() + 48);

    Eigen::Vector2d eye_left = Eigen::Vector2d::Zero();
    for (const auto& pt : lm_eye_left) eye_left += pt;
    eye_left /= lm_eye_left.size();

    Eigen::Vector2d eye_right = Eigen::Vector2d::Zero();
    for (const auto& pt : lm_eye_right) eye_right += pt;
    eye_right /= lm_eye_right.size();

    Eigen::Vector2d eye_avg = (eye_left + eye_right) * 0.5;
    Eigen::Vector2d eye_to_eye = eye_right - eye_left;
    Eigen::Vector2d mouth_left = lm[48];
    Eigen::Vector2d mouth_right = lm[54];
    Eigen::Vector2d mouth_avg = (mouth_left + mouth_right) * 0.5;
    Eigen::Vector2d eye_to_mouth = mouth_avg - eye_avg;

    // Choose oriented crop rectangle
    Eigen::Vector2d x = eye_to_eye - Eigen::Vector2d(-eye_to_mouth[1], eye_to_mouth[0]);
    double x_norm = x.norm();
    if (x_norm > 0) x /= x_norm;
    x *= std::max(eye_to_eye.norm() * 2.0, eye_to_mouth.norm() * 1.8);
    Eigen::Vector2d y(-x[1], x[0]);
    Eigen::Vector2d c = eye_avg + eye_to_mouth * 0.1;
    std::vector<cv::Point2f> quad = {
            cv::Point2f(c[0] - x[0] - y[0], c[1] - x[1] - y[1]),
            cv::Point2f(c[0] - x[0] + y[0], c[1] - x[1] + y[1]),
            cv::Point2f(c[0] + x[0] + y[0], c[1] + x[1] + y[1]),
            cv::Point2f(c[0] + x[0] - y[0], c[1] + x[1] - y[1])
    };
    double qsize = x.norm() * 2;

    // Shrink
    cv::Mat resized = img;
    int shrink = static_cast<int>(std::floor(qsize / 1024 * 0.5));
    if (shrink > 1) {
        cv::resize(img, resized, cv::Size(img.cols / shrink, img.rows / shrink), 0, 0, cv::INTER_AREA);
        for (auto& pt : quad) pt /= shrink;
        qsize /= shrink;
    }

    // Crop
    float min_x = quad[0].x, max_x = quad[0].x, min_y = quad[0].y, max_y = quad[0].y;
    for (const auto& pt : quad) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    int border = std::max(static_cast<int>(std::round(qsize * 0.1)), 3);
    cv::Rect crop(
            std::max(static_cast<int>(std::floor(min_x)) - border, 0),
            std::max(static_cast<int>(std::floor(min_y)) - border, 0),
            std::min(static_cast<int>(std::ceil(max_x)) + border, resized.cols) - std::max(static_cast<int>(std::floor(min_x)) - border, 0),
            std::min(static_cast<int>(std::ceil(max_y)) + border, resized.rows) - std::max(static_cast<int>(std::floor(min_y)) - border, 0)
    );
    cv::Mat cropped = resized(crop);
    for (auto& pt : quad) pt -= cv::Point2f(crop.x, crop.y);

    // Perspective transform
    std::vector<cv::Point2f> dst_pts = { {0, 0}, {1024, 0}, {1024, 1024}, {0, 1024} };
    cv::Mat M = cv::getPerspectiveTransform(quad, dst_pts);
    cv::Mat aligned;
    cv::warpPerspective(cropped, aligned, M, cv::Size(1024, 1024));

    return aligned;
}

extern "C" {
char* align_face_ffi(const double* landmarks, int landmark_count, const char* filepath) {
    try {
        cv::Mat aligned = align_face(landmarks, landmark_count, filepath);
        std::string output_path = "aligned_image.jpg";
        cv::imwrite(output_path, aligned);
        return strdup(output_path.c_str());
    } catch (const std::exception& e) {
        char* error = strdup(e.what());
        return error;
    }
}

void free_string(char* str) {
    free(str);
}
}
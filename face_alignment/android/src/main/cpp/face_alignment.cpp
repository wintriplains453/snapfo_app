#include <opencv2/opencv.hpp>
#include <glm.hpp>
#include <string>
#include <stdexcept>
#include <vector>

cv::Mat align_face(const double* landmarks, int landmark_count, const std::string& filepath) {
    // Validate landmark count
    if (landmark_count < 5) {
        throw std::runtime_error("Expected at least 5 landmarks, got " + std::to_string(landmark_count));
    }

    // Load image
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    // Resize image to max 512px immediately
    cv::Mat resized;
    double scale = std::min(512.0 / img.cols, 512.0 / img.rows);
    cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    img.release(); // Release original image early

    // Convert landmarks to GLM vectors
    std::vector<glm::dvec2> lm;
    lm.reserve(landmark_count);
    for (int i = 0; i < landmark_count; ++i) {
        lm.emplace_back(landmarks[i * 2] * scale, landmarks[i * 2 + 1] * scale);
    }

    // Use available landmarks (assuming order: left eye, right eye, nose base, left mouth, right mouth)
    glm::dvec2 eye_left = lm[0];   // Left eye
    glm::dvec2 eye_right = lm[1];  // Right eye
    glm::dvec2 mouth_left = lm[3]; // Left mouth
    glm::dvec2 mouth_right = lm[4]; // Right mouth

    glm::dvec2 eye_avg = (eye_left + eye_right) * 0.5;
    glm::dvec2 eye_to_eye = eye_right - eye_left;
    glm::dvec2 mouth_avg = (mouth_left + mouth_right) * 0.5;
    glm::dvec2 eye_to_mouth = mouth_avg - eye_avg;

    // Choose oriented crop rectangle
    glm::dvec2 x = eye_to_eye - glm::dvec2(-eye_to_mouth.y, eye_to_mouth.x);
    double x_norm = glm::length(x);
    if (x_norm > 0) x /= x_norm;
    x *= std::max(glm::length(eye_to_eye) * 2.0, glm::length(eye_to_mouth) * 1.8);
    glm::dvec2 y(-x.y, x.x);
    glm::dvec2 c = eye_avg + eye_to_mouth * 0.1;
    std::vector<cv::Point2f> quad = {
            cv::Point2f(static_cast<float>(c.x - x.x - y.x), static_cast<float>(c.y - x.y - y.y)),
            cv::Point2f(static_cast<float>(c.x - x.x + y.x), static_cast<float>(c.y - x.y + y.y)),
            cv::Point2f(static_cast<float>(c.x + x.x + y.x), static_cast<float>(c.y + x.y + y.y)),
            cv::Point2f(static_cast<float>(c.x + x.x - y.x), static_cast<float>(c.y + x.y - y.y))
    };
    double qsize = glm::length(x) * 2;

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
    cv::Mat cropped;
    if (crop.width > 0 && crop.height > 0) {
        cropped = resized(crop);
    } else {
        cropped = resized.clone(); // Clone to avoid referencing released memory
    }
    resized.release(); // Release resized image early

    // Adjust quad points for crop
    for (auto& pt : quad) pt -= cv::Point2f(crop.x, crop.y);

    // Perspective transform
    std::vector<cv::Point2f> dst_pts = {
        {512, 0},   // Adjust for 90-degree clockwise rotation
        {512, 512},
        {0, 512},
        {0, 0}
    };
    cv::Mat M = cv::getPerspectiveTransform(quad, dst_pts);
    cv::Mat aligned;
    cv::warpPerspective(cropped, aligned, M, cv::Size(512, 512));
    cropped.release(); // Release cropped image early

    return aligned;
}

extern "C" {
char* align_face_ffi(const double* landmarks, int landmark_count, const char* filepath) {
    try {
        cv::Mat aligned = align_face(landmarks, landmark_count, filepath);
        std::string output_path = std::string(filepath) + "_aligned.jpg";
        cv::imwrite(output_path, aligned);
        aligned.release();
        return strdup(output_path.c_str());
    } catch (const std::exception& e) {
        return strdup(("Error: " + std::string(e.what())).c_str());
    }
}

void free_string(char* str) {
    free(str);
}
}
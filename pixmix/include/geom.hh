#pragma once

#include "config.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace pixmix {

inline cv::Mat ensure_bgr(const cv::Mat& image) {
    if (image.empty() || image.type() == CV_8UC3) {
        return image;
    }
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
        image.convertTo(bgr, CV_8UC3);
    }
    return bgr;
}

inline cv::Mat normalize_mask(const cv::Mat& mask, const cv::Size& size) {
    cv::Mat gray;
    if (mask.empty()) {
        return cv::Mat(size, CV_8U, cv::Scalar(0));
    }
    if (mask.channels() == 3) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = mask;
    }
    if (gray.size() != size) {
        cv::resize(gray, gray, size, 0, 0, cv::INTER_NEAREST);
    }
    if (gray.type() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    return binary;
}

inline bool in_bounds(int x, int y, const cv::Size& size) {
    return x >= 0 && y >= 0 && x < size.width && y < size.height;
}

inline cv::Mat mask_from_contour(const cv::Size& size, const std::vector<cv::Point>& contour) {
    cv::Mat mask(size, CV_8U, cv::Scalar(0));
    if (contour.size() >= 3) {
        const std::vector<std::vector<cv::Point>> contours{contour};
        cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);
    }
    return mask;
}

inline std::vector<cv::Point> contour_from_mask(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return {};
    }
    const auto best = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    return *best;
}

inline std::vector<cv::Point> sample_closed_contour(const std::vector<cv::Point>& hull, int count) {
    if (static_cast<int>(hull.size()) < 3 || count < 3) {
        return hull;
    }
    const double perimeter = cv::arcLength(hull, true);
    if (perimeter < 1.0) {
        return hull;
    }
    std::vector<cv::Point> sampled;
    sampled.reserve(static_cast<size_t>(count));
    const double step = perimeter / static_cast<double>(count);
    double consumed = 0.0;
    double next = 0.0;
    for (size_t i = 0; i < hull.size() && static_cast<int>(sampled.size()) < count; ++i) {
        const cv::Point a = hull[i];
        const cv::Point b = hull[(i + 1) % hull.size()];
        const double length = cv::norm(b - a);
        if (length < 1e-6) {
            continue;
        }
        while (next <= consumed + length + 1e-6 && static_cast<int>(sampled.size()) < count) {
            const double t = std::clamp((next - consumed) / length, 0.0, 1.0);
            sampled.emplace_back(
                cvRound(a.x + t * (b.x - a.x)),
                cvRound(a.y + t * (b.y - a.y)));
            next += step;
        }
        consumed += length;
    }
    return sampled;
}

inline std::vector<cv::Point> scale_contour(const std::vector<cv::Point>& contour, double scale) {
    std::vector<cv::Point> scaled;
    scaled.reserve(contour.size());
    for (const cv::Point& point : contour) {
        scaled.emplace_back(cvRound(point.x * scale), cvRound(point.y * scale));
    }
    return scaled;
}

// Image-plane rotation of a homography. PixMix resets the keyframe at 45 degrees
// of concatenated rotation; the thesis does not specify a calibration matrix,
// so this is the polar rotation of the homography's linear 2x2 block.
inline float homography_rotation_degrees(const cv::Mat& homography) {
    if (homography.empty() || homography.rows < 2 || homography.cols < 2) {
        return 0.f;
    }
    cv::Mat H;
    homography.convertTo(H, CV_64F);
    cv::Mat linear = H(cv::Rect(0, 0, 2, 2)).clone();
    cv::SVD svd(linear, cv::SVD::FULL_UV);
    cv::Mat rotation = svd.u * svd.vt;
    if (cv::determinant(rotation) < 0.0) {
        cv::Mat u = svd.u.clone();
        u.col(1) *= -1.0;
        rotation = u * svd.vt;
    }
    const double angle = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0));
    return static_cast<float>(std::abs(angle) * 180.0 / CV_PI);
}

inline cv::Mat normalized_homography(const cv::Mat& homography) {
    cv::Mat H;
    homography.convertTo(H, CV_64F);
    if (H.rows == 3 && H.cols == 3) {
        const double scale = H.at<double>(2, 2);
        if (std::abs(scale) > 1e-8) {
            H /= scale;
        }
    }
    return H;
}

}  // namespace pixmix

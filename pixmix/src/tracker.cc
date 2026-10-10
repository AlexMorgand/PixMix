#include "tracker.hh"

#include "features.hh"
#include "geom.hh"

#include <algorithm>
#include <cmath>
#include <random>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace pixmix {
namespace {

cv::Point clip_point(const cv::Point2f& point, const cv::Size& size) {
    return cv::Point(
        std::clamp(cvRound(point.x), 0, std::max(0, size.width - 1)),
        std::clamp(cvRound(point.y), 0, std::max(0, size.height - 1)));
}

std::vector<cv::Point2f> to_float(const std::vector<cv::Point>& points) {
    std::vector<cv::Point2f> out;
    out.reserve(points.size());
    for (const cv::Point& point : points) {
        out.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
    return out;
}

}  // namespace

PlanarTracker::PlanarTracker(PixMixConfig config) : config_(std::move(config)) {}

TrackResult PlanarTracker::track(const cv::Mat& previous_bgr,
                                 const cv::Mat& current_bgr,
                                 const std::vector<cv::Point>& previous_contour) const {
    TrackResult result;
    const cv::Mat previous = ensure_bgr(previous_bgr);
    const cv::Mat current = ensure_bgr(current_bgr);
    if (previous.empty() || previous.size() != current.size() || previous_contour.size() < 4) {
        return result;
    }
    cv::Mat previous_gray;
    cv::Mat current_gray;
    cv::cvtColor(previous, previous_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(current, current_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> inlier_points;
    std::vector<cv::Point2f> outlier_points;
    auto split_points = [&](const std::vector<cv::Point2f>& destination, const cv::Mat& inlier_mask) {
        inlier_points.clear();
        outlier_points.clear();
        for (size_t i = 0; i < destination.size(); ++i) {
            const bool inlier = !inlier_mask.empty() && inlier_mask.at<uchar>(static_cast<int>(i)) != 0;
            (inlier ? inlier_points : outlier_points).push_back(destination[i]);
        }
    };

    auto estimate = [&](bool strict, cv::Mat& homography, int& inliers, int& tracked) {
        cv::Mat band(previous_gray.size(), CV_8U, cv::Scalar(0));
        const std::vector<std::vector<cv::Point>> contours{previous_contour};
        cv::drawContours(band, contours, 0, cv::Scalar(255), strict ? 9 : 21);
        std::vector<cv::Point2f> points;
        cv::goodFeaturesToTrack(previous_gray, points, strict ? 300 : 500, strict ? 0.01 : 0.001, 4.0, band, 3,
                                true, 0.04);
        if (points.size() < 4) {
            return false;
        }
        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> error;
        const cv::Size window = strict ? cv::Size(21, 21) : cv::Size(41, 41);
        cv::calcOpticalFlowPyrLK(previous_gray, current_gray, points, next_points, status, error, window,
                                 strict ? 3 : 4,
                                 cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));
        std::vector<cv::Point2f> source;
        std::vector<cv::Point2f> destination;
        for (size_t i = 0; i < points.size(); ++i) {
            if (!status[i] || error[i] > (strict ? 25.f : 40.f)) {
                continue;
            }
            if (!in_bounds(cvRound(next_points[i].x), cvRound(next_points[i].y), current_gray.size())) {
                continue;
            }
            source.push_back(points[i]);
            destination.push_back(next_points[i]);
        }
        tracked = static_cast<int>(source.size());
        if (source.size() < 4) {
            return false;
        }
        cv::Mat inlier_mask;
        homography = cv::findHomography(source, destination, cv::RANSAC, 3.0, inlier_mask, 2000, 0.995);
        if (homography.empty()) {
            return false;
        }
        inliers = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
        split_points(destination, inlier_mask);
        return inliers >= 4;
    };

    auto estimate_grid = [&](cv::Mat& homography, int& inliers, int& tracked) {
        std::vector<cv::Point2f> points;
        constexpr int cells_x = 8;
        constexpr int cells_y = 6;
        const int cell_w = std::max(1, previous_gray.cols / cells_x);
        const int cell_h = std::max(1, previous_gray.rows / cells_y);
        for (int cy = 0; cy < cells_y; ++cy) {
            for (int cx = 0; cx < cells_x; ++cx) {
                cv::Rect cell(cx * cell_w, cy * cell_h, std::min(cell_w, previous_gray.cols - cx * cell_w),
                              std::min(cell_h, previous_gray.rows - cy * cell_h));
                if (cell.width < 4 || cell.height < 4) {
                    continue;
                }
                cv::Mat cell_mask(previous_gray.size(), CV_8U, cv::Scalar(0));
                cell_mask(cell).setTo(255);
                std::vector<cv::Point2f> local;
                cv::goodFeaturesToTrack(previous_gray, local, 1, 0.001, 3.0, cell_mask, 3, true, 0.04);
                if (!local.empty()) {
                    points.push_back(local.front());
                }
            }
        }
        if (points.size() < 4) {
            return false;
        }
        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(previous_gray, current_gray, points, next_points, status, error, cv::Size(31, 31), 4);
        std::vector<cv::Point2f> source;
        std::vector<cv::Point2f> destination;
        for (size_t i = 0; i < points.size(); ++i) {
            if (status[i] && error[i] < 40.f &&
                in_bounds(cvRound(next_points[i].x), cvRound(next_points[i].y), current_gray.size())) {
                source.push_back(points[i]);
                destination.push_back(next_points[i]);
            }
        }
        tracked = static_cast<int>(source.size());
        if (source.size() < 4) {
            return false;
        }
        cv::Mat inlier_mask;
        homography = cv::findHomography(source, destination, cv::RANSAC, 3.0, inlier_mask, 2000, 0.995);
        if (homography.empty()) {
            return false;
        }
        inliers = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
        split_points(destination, inlier_mask);
        return inliers >= 4;
    };

    cv::Mat homography;
    int inliers = 0;
    int tracked = 0;
    const bool strict_ok = estimate(true, homography, inliers, tracked) && inliers >= config_.min_inliers;
    const bool loose_ok = !strict_ok && estimate(false, homography, inliers, tracked) && inliers >= config_.min_inliers;
    const bool grid_ok = !strict_ok && !loose_ok && estimate_grid(homography, inliers, tracked) &&
                         inliers >= config_.min_inliers;
    result.inlier_points = inlier_points;
    result.outlier_points = outlier_points;
    result.inliers = inliers;
    result.tracked_points = tracked;
    if (!strict_ok && !loose_ok && !grid_ok) {
        return result;
    }

    std::vector<cv::Point2f> warped_float;
    cv::perspectiveTransform(to_float(previous_contour), warped_float, homography);
    std::vector<cv::Point> warped;
    warped.reserve(warped_float.size());
    for (const cv::Point2f& point : warped_float) {
        warped.push_back(clip_point(point, current.size()));
    }

    std::vector<cv::Point> corrected = warped;
    if (config_.correct_contour && warped.size() >= 3) {
        cv::Point2f centroid(0.f, 0.f);
        for (const cv::Point& point : warped) {
            centroid.x += static_cast<float>(point.x);
            centroid.y += static_cast<float>(point.y);
        }
        centroid.x /= static_cast<float>(warped.size());
        centroid.y /= static_cast<float>(warped.size());

        std::vector<cv::Point2f> inward(warped.size());
        std::vector<cv::Point> fingerprint_samples;
        for (size_t i = 0; i < warped.size(); ++i) {
            const cv::Point2f previous(warped[(i + warped.size() - 1) % warped.size()]);
            const cv::Point2f next(warped[(i + 1) % warped.size()]);
            const cv::Point2f current_point(warped[i]);
            cv::Point2f tangent = next - previous;
            const float length = std::hypot(tangent.x, tangent.y);
            cv::Point2f normal = length < 1e-3f ? cv::Point2f(0.f, 1.f)
                                                : cv::Point2f(-tangent.y / length, tangent.x / length);
            const cv::Point2f toward = centroid - current_point;
            if (normal.dot(toward) < 0.f) {
                normal *= -1.f;
            }
            inward[i] = normal;
            const cv::Point2f outside = current_point - normal * 5.f;
            const cv::Point sample = clip_point(outside, current.size());
            if (in_bounds(sample.x, sample.y, current.size())) {
                fingerprint_samples.push_back(sample);
            }
        }
        std::mt19937 rng(config_.seed + static_cast<unsigned>(warped.size()) * 17u);
        FingerprintModel model;
        const int sample_count = std::min(config_.fingerprint_count, static_cast<int>(fingerprint_samples.size()));
        const std::vector<cv::Point> samples = sample_closed_contour(fingerprint_samples, std::max(8, sample_count));
        if (model.fit(current, samples, config_.large_radius, config_.small_radius, 4, rng)) {
            IntegralBGR integral(current);
            std::vector<cv::Point> adjusted;
            adjusted.reserve(warped.size());
            for (size_t i = 0; i < warped.size(); ++i) {
                const cv::Point2f start = cv::Point2f(warped[i]) - inward[i] * 4.f;
                int run = 0;
                cv::Point landmark = warped[i];
                bool found = false;
                for (int step = 0; step <= 28; ++step) {
                    const cv::Point2f probe = start + inward[i] * static_cast<float>(step);
                    const int x = cvRound(probe.x);
                    const int y = cvRound(probe.y);
                    if (!in_bounds(x, y, current.size())) {
                        break;
                    }
                    const bool rejected =
                        model.rejection_ratio(model.feature_at(integral, x, y)) >= config_.selection_gamma;
                    if (rejected) {
                        if (run == 0) {
                            landmark = cv::Point(x, y);
                        }
                        ++run;
                        if (run >= 3) {
                            found = true;
                            break;
                        }
                    } else {
                        run = 0;
                    }
                }
                if (found) {
                    const cv::Point2f placed = cv::Point2f(landmark) - inward[i] * 1.f;
                    adjusted.push_back(clip_point(placed, current.size()));
                } else {
                    adjusted.push_back(warped[i]);
                }
            }
            const double before = std::abs(cv::contourArea(warped));
            const double after = std::abs(cv::contourArea(adjusted));
            if (before > 1.0 && after > 0.4 * before && after < 2.5 * before) {
                corrected = std::move(adjusted);
            }
        }
    }

    result.ok = corrected.size() >= 3;
    result.homography = normalized_homography(homography);
    result.warped_contour = warped;
    result.contour = std::move(corrected);
    return result;
}

}  // namespace pixmix

#pragma once

#include "config.hh"

#include <vector>

#include <opencv2/core.hpp>

namespace pixmix {

struct TrackResult {
    bool ok = false;
    cv::Mat homography;
    std::vector<cv::Point> contour;
    int inliers = 0;
    int tracked_points = 0;
    // Current-frame positions of the matched features, split by the RANSAC verdict.
    std::vector<cv::Point2f> inlier_points;
    std::vector<cv::Point2f> outlier_points;
    // Contour after the homography warp, before fingerprint correction.
    std::vector<cv::Point> warped_contour;
};

// Homography from the previous frame to the current frame, followed by a
// fingerprint correction of the warped contour.
class PlanarTracker {
  public:
    explicit PlanarTracker(PixMixConfig config = {});

    TrackResult track(const cv::Mat& previous_bgr,
                      const cv::Mat& current_bgr,
                      const std::vector<cv::Point>& previous_contour) const;

  private:
    PixMixConfig config_;
};

}  // namespace pixmix

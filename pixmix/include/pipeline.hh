#pragma once

#include "config.hh"
#include "inpainter.hh"
#include "selector.hh"
#include "tracker.hh"

#include <vector>

#include <opencv2/core.hpp>

namespace pixmix {

struct FrameStats {
    double track_ms = 0.0;
    double reference_ms = 0.0;
    double inpaint_ms = 0.0;
    double composite_ms = 0.0;
    double total_ms = 0.0;
    int hole_pixels = 0;
    int inliers = 0;
    bool tracking_ok = true;
    bool keyframe_reset = false;
    float rotation_deg = 0.f;
};

// Intermediate state kept for debug visualization. Cheap to fill: the
// matrices share data with the pipeline's own buffers.
struct FrameDebug {
    std::vector<cv::Point2f> inlier_points;
    std::vector<cv::Point2f> outlier_points;
    std::vector<cv::Point> warped_contour;
    // Previous frame to current frame; empty when tracking failed.
    cv::Mat homography;
    cv::Mat reference;
    cv::Mat mapping;
};

struct FrameOutput {
    cv::Mat image;
    cv::Mat mask;
    std::vector<cv::Point> contour;
    FrameStats stats;
    FrameDebug debug;
};

// Forwards a previous full-resolution offset map through the prev->current
// homography: f_hat = H^-1 o f o H. Invalid pixels keep kInvalidOffset.
cv::Mat forward_pixel_map(const cv::Mat& previous_map,
                          const cv::Mat& previous_mask,
                          const cv::Mat& current_mask,
                          const cv::Mat& homography_prev_to_current);

class PixMixPipeline {
  public:
    explicit PixMixPipeline(PixMixConfig config = {});

    void reset();
    bool start_with_mask(const cv::Mat& bgr, const cv::Mat& hole_mask, FrameOutput* first = nullptr);
    bool start_with_polygon(const cv::Mat& bgr, const std::vector<cv::Point>& polygon, FrameOutput* first = nullptr);
    // Returns false only when the pipeline was not started or the frame is unusable.
    // A lost track still returns true, with stats.tracking_ok == false.
    bool process(const cv::Mat& bgr, FrameOutput& output);

  private:
    bool finish_start(const cv::Mat& bgr,
                      const cv::Mat& mask,
                      const std::vector<cv::Point>& contour,
                      FrameOutput* first);
    cv::Mat composite(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat& synthesized) const;

    PixMixConfig config_;
    PixMixInpainter inpainter_;
    FingerprintSelector selector_;
    PlanarTracker tracker_;
    cv::Mat keyframe_;
    cv::Mat homography_key_to_current_;
    cv::Mat previous_bgr_;
    cv::Mat previous_mask_;
    cv::Mat previous_map_;
    std::vector<cv::Point> contour_;
    bool ready_ = false;
};

}  // namespace pixmix

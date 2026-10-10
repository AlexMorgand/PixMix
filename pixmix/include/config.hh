#pragma once

#include <opencv2/core.hpp>

namespace pixmix {

// Sentinel stored in CV_32SC2 pixel maps. Real offsets stay within the frame.
constexpr int kInvalidOffset = -1000000;

inline bool offset_valid(const cv::Vec2i& offset) {
    return offset[0] > kInvalidOffset / 2 && offset[1] > kInvalidOffset / 2;
}

// Unpublished PixMix thresholds are explicit here so they can be tuned without
// hunting through the solver. Comments mark values that the thesis states.
struct PixMixConfig {
    // 5x5 SSD neighborhood (thesis).
    int patch_radius = 2;
    // alpha * spatial + (1 - alpha) * appearance, alpha = 0.2 (thesis).
    float alpha = 0.2f;
    // d_s = min(|f(p)+v - f(p+v)|^2, tau_s) in full-resolution pixels (thesis
    // 4.12). Costs are raw: appearance is a mean per-pixel 3-channel SSD, so
    // tau_s sets how much a seam between two source blocks may cost. The thesis
    // does not publish the value.
    float tau_s = 2000.f;
    // Border samples whose neighborhood touches the source are up-weighted.
    // The thesis says tau_a >> 1 but does not give the constant.
    float tau_a = 8.f;
    int max_pyramid_levels = 8;
    // Coarsest level keeps every hole pixel within this many pixels of the border.
    float coarse_border_px = 3.f;
    int still_passes = 5;
    // Still layers keep iterating until fewer than 1% of mappings change.
    int max_still_passes = 20;
    // Forwarding the previous map is reported to cut iterations about 4x.
    int video_passes = 3;
    int video_search_radius = 12;
    int blend_radius = 4;
    // Extra pixels added around the selected hole so a cast shadow is removed
    // with the object. Zero keeps the fingerprint contour tight.
    int hole_margin = 0;
    float selection_gamma = 0.95f;
    int fingerprint_count = 200;
    int fingerprint_restarts = 8;
    int large_radius = 8;
    int small_radius = 2;
    int min_inliers = 15;
    float keyframe_rotation_deg = 45.f;
    // Fingerprint boundary correction. Turn off when the contour already
    // contains a margin and the background is cluttered enough to pull it open.
    bool correct_contour = true;
    unsigned seed = 1;
};

}  // namespace pixmix

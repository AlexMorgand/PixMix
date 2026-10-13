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
    // 5x5 SSD neighborhood on every pyramid level (thesis 4.16).
    int patch_radius = 2;
    // alpha * spatial + (1 - alpha) * appearance, alpha = 0.2 (thesis).
    float alpha = 0.2f;
    // The implemented spatial term is the minimum clamped neighbor
    // disagreement, then weighted by 1/4 (thesis 4.61 and 4.12). Distances are
    // full-resolution pixels. The thesis does not publish tau_s.
    float tau_s = 4000.f;
    // Border samples whose neighborhood touches the source are up-weighted.
    // The thesis says tau_a >> 1 but does not give the constant.
    float tau_a = 8.f;
    // Tie-break visually similar initialization patches toward local source
    // material, which better preserves perspective scale.
    float source_locality = 0.05f;
    // 2010 paper section 5.3: when two source patches have similar appearance,
    // keep the spatially nearer one so the fill does not jump to a distant
    // copy. The value is a mean per-pixel SSD. Zero disables the preference.
    float similar_appearance = 12.f;
    int max_pyramid_levels = 8;
    // Coarsest level keeps every hole pixel within this many pixels of the border.
    float coarse_border_px = 3.f;
    int still_passes = 5;
    // The thesis reports convergence below 3% after ten iterations.
    int max_still_passes = 10;
    // Optional full-resolution restart for thin grids. It is not part of the
    // thesis pyramid, which upsamples the coarse mapping and refines it.
    bool structural_candidate = false;
    // Forwarding the previous map is reported to cut iterations about 4x.
    int video_passes = 3;
    // Zero follows PatchMatch/thesis 4.3 and starts each random search over
    // the whole layer. A positive value is an explicit speed/quality override.
    int video_search_radius = 0;
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

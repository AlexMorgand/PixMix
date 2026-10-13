#pragma once

#include "config.hh"

#include <vector>

namespace pixmix {

// One pyramid level, coarse levels first. `before` is the upsampled guess
// (or the erosion initialization on the coarsest level). `after` is that
// level once its own refinement has finished.
struct ScaleView {
    cv::Mat before;
    cv::Mat after;
    // Paper Fig. 4/5 neighboring blocks after this level's refinement.
    // Gray: every in-hole neighbor shares this offset (summed spatial cost 0).
    // Black: at least one neighbor maps somewhere else.
    cv::Mat blocks;
};

// Coarse-to-fine PixMix pixel-map solver. Hole pixels are copied from source
// pixels in the current frame; patches are never blended inside the hole.
class PixMixInpainter {
  public:
    explicit PixMixInpainter(PixMixConfig config = {});

    // initial_map is a full-resolution CV_32SC2 offset map. An empty map selects
    // first-frame initialization. An empty reference disables temporal coherence.
    // passes < 0 uses the still or video default from the configuration.
    cv::Mat run(const cv::Mat& bgr,
                const cv::Mat& hole_mask,
                const cv::Mat& initial_map = cv::Mat(),
                const cv::Mat& reference = cv::Mat(),
                int passes = -1);

    const cv::Mat& mapping() const { return mapping_; }
    int levels_used() const { return levels_used_; }
    const std::vector<ScaleView>& scales() const { return scales_; }

  private:
    PixMixConfig config_;
    cv::Mat mapping_;
    int levels_used_ = 0;
    std::vector<ScaleView> scales_;
};

}  // namespace pixmix

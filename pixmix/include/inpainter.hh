#pragma once

#include "config.hh"

namespace pixmix {

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

  private:
    PixMixConfig config_;
    cv::Mat mapping_;
};

}  // namespace pixmix

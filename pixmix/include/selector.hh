#pragma once

#include "config.hh"

#include <vector>

#include <opencv2/core.hpp>

namespace pixmix {

struct Selection {
    cv::Mat mask;
    std::vector<cv::Point> contour;
};

// Refines a rough contour, drawn around the object with some background inside
// it, into a tight hole mask. Fingerprints on the contour vote a pixel into the
// hole when at least gamma of them reject it.
class FingerprintSelector {
  public:
    explicit FingerprintSelector(PixMixConfig config = {});

    Selection refine(const cv::Mat& bgr, const std::vector<cv::Point>& rough) const;

  private:
    PixMixConfig config_;
};

}  // namespace pixmix

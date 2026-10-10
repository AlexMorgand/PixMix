#include "selector.hh"

#include "features.hh"
#include "geom.hh"

#include <iterator>
#include <random>

#include <opencv2/imgproc.hpp>

namespace pixmix {

FingerprintSelector::FingerprintSelector(PixMixConfig config) : config_(std::move(config)) {}

Selection FingerprintSelector::refine(const cv::Mat& bgr, const std::vector<cv::Point>& rough) const {
    Selection selection;
    const cv::Mat frame = ensure_bgr(bgr);
    if (frame.empty() || rough.size() < 3) {
        return selection;
    }

    // One full-resolution pass. A coarse label that marks background as object cannot
    // be repaired by a border-only upsample, and this vote runs once per selection.
    std::mt19937 rng(config_.seed);
    const std::vector<cv::Point> samples = sample_closed_contour(rough, config_.fingerprint_count);
    FingerprintModel model;
    if (!model.fit(frame, samples, config_.large_radius, config_.small_radius, config_.fingerprint_restarts, rng)) {
        selection.mask = mask_from_contour(frame.size(), rough);
        selection.contour = rough;
        return selection;
    }

    IntegralBGR integral(frame);
    const cv::Mat inside = mask_from_contour(frame.size(), rough);
    cv::Mat mask(frame.size(), CV_8U, cv::Scalar(0));
    const float gamma = config_.selection_gamma;
#pragma omp parallel for schedule(static) if (mask.rows > 32)
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* inside_row = inside.ptr<uchar>(y);
        uchar* out_row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (!inside_row[x]) {
                continue;
            }
            const float ratio = model.rejection_ratio(model.feature_at(integral, x, y));
            out_row[x] = ratio >= gamma ? static_cast<uchar>(255) : static_cast<uchar>(0);
        }
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) {
        return selection;
    }
    const auto best = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    selection.mask = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
    cv::drawContours(selection.mask, contours, static_cast<int>(std::distance(contours.begin(), best)),
                     cv::Scalar(255), cv::FILLED);
    cv::dilate(selection.mask, selection.mask, kernel, cv::Point(-1, -1), 1);
    selection.contour = contour_from_mask(selection.mask);
    return selection;
}

}  // namespace pixmix

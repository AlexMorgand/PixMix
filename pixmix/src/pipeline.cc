#include "pipeline.hh"

#include "geom.hh"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace pixmix {
namespace {

double milliseconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void apply_lighting(cv::Mat& reference, const cv::Mat& frame, const cv::Mat& warped_keyframe, const cv::Mat& mask) {
    if (reference.empty() || warped_keyframe.empty()) {
        return;
    }
    // Sample just outside the hole so the color differences come from the
    // background in both frames, never from the object being removed.
    cv::Mat ring;
    cv::dilate(mask, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    const std::vector<cv::Point> outer = contour_from_mask(ring);
    if (outer.size() < 3) {
        return;
    }
    const std::vector<cv::Point> samples = sample_closed_contour(outer, 48);
    const cv::Rect bounds = cv::boundingRect(outer) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bounds.width <= 0 || bounds.height <= 0) {
        return;
    }
    constexpr int step = 16;
    const int grid_cols = std::max(1, (bounds.width + step - 1) / step);
    const int grid_rows = std::max(1, (bounds.height + step - 1) / step);
    cv::Mat delta(grid_rows, grid_cols, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int gy = 0; gy < grid_rows; ++gy) {
        cv::Vec3f* row = delta.ptr<cv::Vec3f>(gy);
        for (int gx = 0; gx < grid_cols; ++gx) {
            const float x = static_cast<float>(bounds.x + gx * step);
            const float y = static_cast<float>(bounds.y + gy * step);
            cv::Vec3f accum(0, 0, 0);
            float weight_sum = 0.f;
            for (const cv::Point& sample : samples) {
                if (!in_bounds(sample.x, sample.y, frame.size())) {
                    continue;
                }
                const float distance = std::hypot(x - static_cast<float>(sample.x), y - static_cast<float>(sample.y));
                const float weight = std::exp(-std::sqrt(distance));
                const cv::Vec3b live = frame.ptr<cv::Vec3b>(sample.y)[sample.x];
                const cv::Vec3b key = warped_keyframe.ptr<cv::Vec3b>(sample.y)[sample.x];
                accum[0] += weight * (float(live[0]) - float(key[0]));
                accum[1] += weight * (float(live[1]) - float(key[1]));
                accum[2] += weight * (float(live[2]) - float(key[2]));
                weight_sum += weight;
            }
            if (weight_sum > 1e-4f) {
                row[gx] = accum * (1.f / weight_sum);
            }
        }
    }

    for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        cv::Vec3b* ref_row = reference.ptr<cv::Vec3b>(y);
        const float gy = static_cast<float>(y - bounds.y) / static_cast<float>(step);
        const int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0, grid_rows - 1);
        const int y1 = std::min(grid_rows - 1, y0 + 1);
        const float ty = std::clamp(gy - static_cast<float>(y0), 0.f, 1.f);
        for (int x = bounds.x; x < bounds.x + bounds.width; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const float gx = static_cast<float>(x - bounds.x) / static_cast<float>(step);
            const int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0, grid_cols - 1);
            const int x1 = std::min(grid_cols - 1, x0 + 1);
            const float tx = std::clamp(gx - static_cast<float>(x0), 0.f, 1.f);
            const cv::Vec3f d00 = delta.ptr<cv::Vec3f>(y0)[x0];
            const cv::Vec3f d10 = delta.ptr<cv::Vec3f>(y0)[x1];
            const cv::Vec3f d01 = delta.ptr<cv::Vec3f>(y1)[x0];
            const cv::Vec3f d11 = delta.ptr<cv::Vec3f>(y1)[x1];
            cv::Vec3b& pixel = ref_row[x];
            for (int channel = 0; channel < 3; ++channel) {
                const float top = d00[channel] + tx * (d10[channel] - d00[channel]);
                const float bottom = d01[channel] + tx * (d11[channel] - d01[channel]);
                const float value = static_cast<float>(pixel[channel]) + top + ty * (bottom - top);
                pixel[channel] = cv::saturate_cast<uchar>(value);
            }
        }
    }
}

}  // namespace

cv::Mat forward_pixel_map(const cv::Mat& previous_map,
                          const cv::Mat& previous_mask,
                          const cv::Mat& current_mask,
                          const cv::Mat& homography_prev_to_current) {
    cv::Mat forwarded(current_mask.size(), CV_32SC2, cv::Scalar(kInvalidOffset, kInvalidOffset));
    if (previous_map.empty() || previous_mask.empty() || current_mask.empty() ||
        previous_map.type() != CV_32SC2 || previous_map.size() != previous_mask.size()) {
        return forwarded;
    }
    const cv::Mat homography = normalized_homography(homography_prev_to_current);
    if (homography.rows != 3 || homography.cols != 3) {
        return forwarded;
    }
    cv::Mat inverse;
    if (!cv::invert(homography, inverse)) {
        return forwarded;
    }

    std::vector<cv::Point> holes;
    for (int y = 0; y < current_mask.rows; ++y) {
        const uchar* row = current_mask.ptr<uchar>(y);
        for (int x = 0; x < current_mask.cols; ++x) {
            if (row[x]) {
                holes.emplace_back(x, y);
            }
        }
    }
    if (holes.empty()) {
        return forwarded;
    }

    std::vector<cv::Point2f> current_points;
    current_points.reserve(holes.size());
    for (const cv::Point& hole : holes) {
        current_points.emplace_back(static_cast<float>(hole.x), static_cast<float>(hole.y));
    }
    std::vector<cv::Point2f> previous_points;
    cv::perspectiveTransform(current_points, previous_points, inverse);

    std::vector<cv::Point2f> source_previous;
    std::vector<int> valid;
    source_previous.reserve(holes.size());
    for (size_t i = 0; i < holes.size(); ++i) {
        const int x = cvRound(previous_points[i].x);
        const int y = cvRound(previous_points[i].y);
        if (!in_bounds(x, y, previous_mask.size()) || previous_mask.ptr<uchar>(y)[x] == 0) {
            continue;
        }
        const cv::Vec2i offset = previous_map.ptr<cv::Vec2i>(y)[x];
        if (!offset_valid(offset)) {
            continue;
        }
        source_previous.emplace_back(previous_points[i].x + static_cast<float>(offset[0]),
                                     previous_points[i].y + static_cast<float>(offset[1]));
        valid.push_back(static_cast<int>(i));
    }
    if (source_previous.empty()) {
        return forwarded;
    }
    std::vector<cv::Point2f> source_current;
    cv::perspectiveTransform(source_previous, source_current, homography);
    for (size_t k = 0; k < valid.size(); ++k) {
        const cv::Point target = holes[static_cast<size_t>(valid[k])];
        const int sx = cvRound(source_current[k].x);
        const int sy = cvRound(source_current[k].y);
        if (!in_bounds(sx, sy, current_mask.size()) || current_mask.ptr<uchar>(sy)[sx] != 0) {
            continue;
        }
        forwarded.ptr<cv::Vec2i>(target.y)[target.x] = cv::Vec2i(sx - target.x, sy - target.y);
    }
    return forwarded;
}

PixMixPipeline::PixMixPipeline(PixMixConfig config)
    : config_(config),
      inpainter_(config),
      selector_(config),
      tracker_(config),
      homography_key_to_current_(cv::Mat::eye(3, 3, CV_64F)) {}

void PixMixPipeline::reset() {
    ready_ = false;
    keyframe_.release();
    previous_bgr_.release();
    previous_mask_.release();
    previous_map_.release();
    contour_.clear();
    homography_key_to_current_ = cv::Mat::eye(3, 3, CV_64F);
}

cv::Mat expand_hole(const cv::Mat& mask, int margin) {
    if (margin <= 0 || mask.empty()) {
        return mask;
    }
    cv::Mat source;
    cv::bitwise_not(mask, source);
    cv::Mat reachable = source.clone();
    const cv::Point seeds[] = {cv::Point(0, 0), cv::Point(source.cols - 1, 0), cv::Point(0, source.rows - 1),
                               cv::Point(source.cols - 1, source.rows - 1)};
    for (const cv::Point& seed : seeds) {
        if (reachable.at<uchar>(seed) == 255) {
            cv::floodFill(reachable, seed, 128);
        }
    }
    cv::Mat islands;
    cv::compare(reachable, 255, islands, cv::CMP_EQ);
    cv::Mat closed;
    cv::bitwise_or(mask, islands, closed);
    const int diameter = margin * 2 + 1;
    cv::Mat expanded;
    cv::dilate(closed, expanded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(diameter, diameter)));
    return expanded;
}

bool PixMixPipeline::start_with_polygon(const cv::Mat& bgr,
                                        const std::vector<cv::Point>& polygon,
                                        FrameOutput* first) {
    const cv::Mat frame = ensure_bgr(bgr);
    const Selection selection = selector_.refine(frame, polygon);
    if (selection.mask.empty() || selection.contour.size() < 3 || cv::countNonZero(selection.mask) == 0) {
        return false;
    }
    return finish_start(frame, selection.mask, selection.contour, first);
}

bool PixMixPipeline::start_with_mask(const cv::Mat& bgr, const cv::Mat& hole_mask, FrameOutput* first) {
    const cv::Mat frame = ensure_bgr(bgr);
    const cv::Mat mask = normalize_mask(hole_mask, frame.size());
    std::vector<cv::Point> contour = contour_from_mask(mask);
    if (contour.size() > 800) {
        contour = sample_closed_contour(contour, 400);
    }
    if (contour.size() < 3 || cv::countNonZero(mask) == 0) {
        return false;
    }
    return finish_start(frame, mask, contour, first);
}

bool PixMixPipeline::finish_start(const cv::Mat& bgr,
                                  const cv::Mat& mask,
                                  const std::vector<cv::Point>& contour,
                                  FrameOutput* first) {
    const auto started = std::chrono::steady_clock::now();
    const cv::Mat hole = expand_hole(mask, config_.hole_margin);
    const cv::Mat synthesized = inpainter_.run(bgr, hole);
    if (synthesized.empty() || inpainter_.mapping().empty()) {
        return false;
    }
    keyframe_ = composite(bgr, hole, synthesized);
    previous_bgr_ = bgr.clone();
    previous_mask_ = hole.clone();
    previous_map_ = inpainter_.mapping().clone();
    contour_ = contour;
    homography_key_to_current_ = cv::Mat::eye(3, 3, CV_64F);
    ready_ = true;
    if (first != nullptr) {
        *first = FrameOutput{};
        first->image = keyframe_.clone();
        first->mask = hole.clone();
        first->contour = contour_;
        first->stats.hole_pixels = cv::countNonZero(hole);
        first->stats.inpaint_ms = milliseconds_since(started);
        first->stats.total_ms = first->stats.inpaint_ms;
        first->stats.tracking_ok = true;
        first->debug.mapping = previous_map_;
        first->debug.reference = keyframe_;
    }
    return true;
}

bool PixMixPipeline::process(const cv::Mat& bgr, FrameOutput& output) {
    const auto started = std::chrono::steady_clock::now();
    output = FrameOutput{};
    const cv::Mat frame = ensure_bgr(bgr);
    if (!ready_ || frame.empty() || frame.size() != previous_bgr_.size()) {
        return false;
    }

    const auto track_started = std::chrono::steady_clock::now();
    const TrackResult tracked = tracker_.track(previous_bgr_, frame, contour_);
    output.stats.track_ms = milliseconds_since(track_started);
    output.stats.tracking_ok = tracked.ok;
    output.stats.inliers = tracked.inliers;
    output.debug.inlier_points = tracked.inlier_points;
    output.debug.outlier_points = tracked.outlier_points;
    output.debug.warped_contour = tracked.warped_contour;
    if (!tracked.ok) {
        output.image = frame.clone();
        output.mask = previous_mask_.clone();
        output.contour = contour_;
        output.stats.hole_pixels = cv::countNonZero(previous_mask_);
        output.stats.total_ms = milliseconds_since(started);
        return true;
    }

    cv::Mat homography = normalized_homography(tracked.homography);
    output.debug.homography = homography;
    const cv::Mat previous_key = homography_key_to_current_.clone();
    const std::vector<cv::Point> previous_contour = contour_;
    homography_key_to_current_ = normalized_homography(homography * homography_key_to_current_);
    contour_ = tracked.contour;
    cv::Mat mask = expand_hole(mask_from_contour(frame.size(), contour_), config_.hole_margin);
    if (cv::countNonZero(mask) == 0) {
        homography_key_to_current_ = previous_key;
        contour_ = previous_contour;
        output.stats.tracking_ok = false;
        output.image = frame.clone();
        output.mask = previous_mask_.clone();
        output.contour = contour_;
        output.stats.total_ms = milliseconds_since(started);
        return true;
    }

    const auto reference_started = std::chrono::steady_clock::now();
    const cv::Mat initial = forward_pixel_map(previous_map_, previous_mask_, mask, homography);
    cv::Mat warped_keyframe;
    cv::warpPerspective(keyframe_, warped_keyframe, homography_key_to_current_, frame.size(), cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    cv::Mat reference = frame.clone();
    warped_keyframe.copyTo(reference, mask);
    apply_lighting(reference, frame, warped_keyframe, mask);
    output.stats.reference_ms = milliseconds_since(reference_started);
    output.debug.reference = reference;

    const auto inpaint_started = std::chrono::steady_clock::now();
    const cv::Mat synthesized = inpainter_.run(frame, mask, initial, reference, config_.video_passes);
    output.stats.inpaint_ms = milliseconds_since(inpaint_started);

    const auto composite_started = std::chrono::steady_clock::now();
    output.image = composite(frame, mask, synthesized);
    output.stats.composite_ms = milliseconds_since(composite_started);

    previous_bgr_ = frame.clone();
    previous_mask_ = mask;
    previous_map_ = inpainter_.mapping().clone();
    output.debug.mapping = previous_map_;
    output.mask = mask;
    output.contour = contour_;
    output.stats.hole_pixels = cv::countNonZero(mask);
    output.stats.rotation_deg = homography_rotation_degrees(homography_key_to_current_);
    if (output.stats.rotation_deg >= config_.keyframe_rotation_deg) {
        keyframe_ = output.image.clone();
        homography_key_to_current_ = cv::Mat::eye(3, 3, CV_64F);
        output.stats.keyframe_reset = true;
        output.stats.rotation_deg = 0.f;
    }
    output.stats.total_ms = milliseconds_since(started);
    return true;
}

cv::Mat PixMixPipeline::composite(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat& synthesized) const {
    cv::Mat output = frame.clone();
    const cv::Mat& mapping = inpainter_.mapping();
    if (mapping.empty() || mapping.size() != frame.size()) {
        if (!synthesized.empty()) {
            synthesized.copyTo(output, mask);
        }
        return output;
    }
    // The hole is grown past the object (hole_margin), so the pixels just inside
    // the border are background and can be blended with the synthesis (3-5 px).
    cv::Mat distance;
    cv::distanceTransform(mask, distance, cv::DIST_L2, 3);
    const float blend = static_cast<float>(std::max(1, config_.blend_radius));
    for (int y = 0; y < frame.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const float* distance_row = distance.ptr<float>(y);
        const cv::Vec2i* map_row = mapping.ptr<cv::Vec2i>(y);
        cv::Vec3b* out_row = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const cv::Vec2i offset = map_row[x];
            cv::Vec3b filled = synthesized.empty() ? frame.ptr<cv::Vec3b>(y)[x] : synthesized.ptr<cv::Vec3b>(y)[x];
            const int sx = x + offset[0];
            const int sy = y + offset[1];
            if (offset_valid(offset) && in_bounds(sx, sy, frame.size()) && mask.ptr<uchar>(sy)[sx] == 0) {
                filled = frame.ptr<cv::Vec3b>(sy)[sx];
            }
            const float depth = distance_row[x];
            if (config_.hole_margin > 0 && depth < blend) {
                const float alpha = std::clamp(depth / blend, 0.f, 1.f);
                const cv::Vec3b original = frame.ptr<cv::Vec3b>(y)[x];
                for (int channel = 0; channel < 3; ++channel) {
                    filled[channel] = cv::saturate_cast<uchar>((1.f - alpha) * original[channel] + alpha * filled[channel]);
                }
            }
            out_row[x] = filled;
        }
    }
    return output;
}

}  // namespace pixmix

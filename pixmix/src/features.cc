#include "features.hh"

#include "geom.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace pixmix {
namespace {

cv::Vec6f stddev_of(const std::vector<cv::Vec6f>& values, const std::vector<int>& indices) {
    cv::Vec6f mean = cv::Vec6f::all(0);
    cv::Vec6f second = cv::Vec6f::all(0);
    const int count = static_cast<int>(indices.size());
    if (count <= 0) {
        return cv::Vec6f::all(0);
    }
    for (const int index : indices) {
        const cv::Vec6f& value = values[static_cast<size_t>(index)];
        mean += value;
        for (int channel = 0; channel < 6; ++channel) {
            second[channel] += value[channel] * value[channel];
        }
    }
    cv::Vec6f deviation;
    const float inverse = 1.f / static_cast<float>(count);
    for (int channel = 0; channel < 6; ++channel) {
        mean[channel] *= inverse;
        const float variance = std::max(0.f, second[channel] * inverse - mean[channel] * mean[channel]);
        deviation[channel] = std::sqrt(variance);
    }
    return deviation;
}

}  // namespace

IntegralBGR::IntegralBGR(const cv::Mat& bgr) : cols_(bgr.cols), rows_(bgr.rows) {
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);
    for (int channel = 0; channel < 3 && channel < static_cast<int>(channels.size()); ++channel) {
        cv::integral(channels[static_cast<size_t>(channel)], sum_[channel], CV_32S);
    }
}

cv::Vec3f IntegralBGR::box_mean(int x, int y, int radius) const {
    const int x0 = std::clamp(x - radius, 0, cols_);
    const int y0 = std::clamp(y - radius, 0, rows_);
    const int x1 = std::clamp(x + radius + 1, 0, cols_);
    const int y1 = std::clamp(y + radius + 1, 0, rows_);
    const int area = (x1 - x0) * (y1 - y0);
    cv::Vec3f mean(0, 0, 0);
    if (area <= 0) {
        return mean;
    }
    for (int channel = 0; channel < 3; ++channel) {
        const cv::Mat& sum = sum_[channel];
        const int value = sum.at<int>(y1, x1) - sum.at<int>(y0, x1) - sum.at<int>(y1, x0) + sum.at<int>(y0, x0);
        mean[channel] = static_cast<float>(value) / static_cast<float>(area);
    }
    return mean;
}

cv::Vec6f FingerprintModel::feature_at(const IntegralBGR& integral, int x, int y) const {
    const cv::Vec3f large = integral.box_mean(x, y, large_radius_);
    const cv::Vec3f small = integral.box_mean(x, y, small_radius_);
    cv::Vec6f feature;
    for (int channel = 0; channel < 3; ++channel) {
        feature[channel] = large[channel];
        feature[channel + 3] = small[channel];
    }
    return feature;
}

float FingerprintModel::rejection_ratio(const cv::Vec6f& feature) const {
    if (prints_.empty() || clusters_.empty()) {
        return 1.f;
    }
    // A background mode accepts the pixel for every fingerprint in that cluster.
    // Gamma then stays meaningful when the contour contains more than one color.
    int largest = 0;
    for (const ColorCluster& cluster : clusters_) {
        largest = std::max(largest, cluster.count);
    }
    // When one background color is a majority, ignore smaller contour clusters.
    // A cable crossing the box would otherwise vote the object back in.
    const bool dominant = largest * 2 >= static_cast<int>(prints_.size());
    int matched = 0;
    for (const ColorCluster& cluster : clusters_) {
        if (dominant && cluster.count != largest) {
            continue;
        }
        bool near = true;
        for (int channel = 0; channel < 3; ++channel) {
            const float gate = std::max(3.f * cluster.deviation[channel], 8.f);
            if (std::fabs(feature[channel + 3] - cluster.mean[channel]) > gate) {
                near = false;
                break;
            }
        }
        if (near) {
            matched += cluster.count;
        }
    }
    const float ratio = 1.f - static_cast<float>(matched) / static_cast<float>(prints_.size());
    return std::clamp(ratio, 0.f, 1.f);
}

bool FingerprintModel::fit(const cv::Mat& bgr,
                           const std::vector<cv::Point>& samples,
                           int large_radius,
                           int small_radius,
                           int restarts,
                           std::mt19937& rng) {
    prints_.clear();
    vmax_ = cv::Vec6f::all(1.f);
    large_radius_ = std::max(1, large_radius);
    small_radius_ = std::max(1, small_radius);
    if (small_radius_ >= large_radius_) {
        small_radius_ = std::max(1, large_radius_ / 2);
    }
    if (bgr.empty() || samples.size() < 8) {
        return false;
    }

    IntegralBGR integral(bgr);
    std::vector<cv::Vec6f> features;
    features.reserve(samples.size());
    for (const cv::Point& sample : samples) {
        if (!in_bounds(sample.x, sample.y, bgr.size())) {
            continue;
        }
        features.push_back(feature_at(integral, sample.x, sample.y));
    }
    if (features.size() < 8) {
        return false;
    }

    const int count = static_cast<int>(features.size());
    std::vector<int> best_labels(static_cast<size_t>(count), 0);
    float best_worst = std::numeric_limits<float>::infinity();
    std::uniform_int_distribution<int> pick(0, count - 1);
    const int attempts = std::max(1, restarts);

    for (int attempt = 0; attempt < attempts; ++attempt) {
        std::vector<int> labels(static_cast<size_t>(count), -1);
        int remaining = count;
        int clusters = 0;
        float worst = 0.f;
        while (remaining > 0) {
            int center = pick(rng);
            for (int guard = 0; guard < count && labels[static_cast<size_t>(center)] >= 0; ++guard) {
                center = (center + 1) % count;
            }
            std::vector<int> unlabeled;
            unlabeled.reserve(static_cast<size_t>(remaining));
            for (int index = 0; index < count; ++index) {
                if (labels[static_cast<size_t>(index)] < 0) {
                    unlabeled.push_back(index);
                }
            }
            cv::Vec6f threshold = stddev_of(features, unlabeled);
            for (int channel = 0; channel < 6; ++channel) {
                threshold[channel] = std::max(threshold[channel], 1.f);
            }
            std::vector<int> members;
            members.push_back(center);
            labels[static_cast<size_t>(center)] = clusters;
            --remaining;
            for (const int index : unlabeled) {
                if (index == center) {
                    continue;
                }
                bool near = true;
                const cv::Vec6f& feature = features[static_cast<size_t>(index)];
                const cv::Vec6f& origin = features[static_cast<size_t>(center)];
                for (int channel = 3; channel < 6; ++channel) {
                    if (std::fabs(feature[channel] - origin[channel]) > threshold[channel]) {
                        near = false;
                        break;
                    }
                }
                if (near) {
                    labels[static_cast<size_t>(index)] = clusters;
                    members.push_back(index);
                    --remaining;
                }
            }
            const cv::Vec6f deviation = stddev_of(features, members);
            float magnitude = 0.f;
            for (int channel = 3; channel < 6; ++channel) {
                magnitude += deviation[channel];
            }
            worst = std::max(worst, magnitude);
            ++clusters;
            if (clusters > count) {
                break;
            }
        }
        if (worst < best_worst) {
            best_worst = worst;
            best_labels.swap(labels);
        }
    }

    int cluster_count = 0;
    for (const int label : best_labels) {
        cluster_count = std::max(cluster_count, label + 1);
    }
    vmax_ = cv::Vec6f::all(0);
    clusters_.clear();
    clusters_.reserve(static_cast<size_t>(cluster_count));
    for (int cluster = 0; cluster < cluster_count; ++cluster) {
        std::vector<int> members;
        cv::Vec3f mean(0, 0, 0);
        for (int index = 0; index < count; ++index) {
            if (best_labels[static_cast<size_t>(index)] == cluster) {
                members.push_back(index);
                const cv::Vec6f& feature = features[static_cast<size_t>(index)];
                mean[0] += feature[3];
                mean[1] += feature[4];
                mean[2] += feature[5];
            }
        }
        if (members.empty()) {
            continue;
        }
        mean *= 1.f / static_cast<float>(members.size());
        const cv::Vec6f deviation = stddev_of(features, members);
        for (int channel = 0; channel < 6; ++channel) {
            vmax_[channel] = std::max(vmax_[channel], deviation[channel]);
        }
        ColorCluster summary;
        summary.mean = mean;
        summary.deviation = cv::Vec3f(deviation[3], deviation[4], deviation[5]);
        summary.count = static_cast<int>(members.size());
        clusters_.push_back(summary);
    }
    for (int channel = 0; channel < 6; ++channel) {
        vmax_[channel] = std::max(vmax_[channel], 0.5f);
    }
    prints_ = std::move(features);
    return true;
}

}  // namespace pixmix

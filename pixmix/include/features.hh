#pragma once

#include "config.hh"

#include <random>
#include <vector>

#include <opencv2/core.hpp>

namespace pixmix {

class IntegralBGR {
  public:
    explicit IntegralBGR(const cv::Mat& bgr);

    cv::Vec3f box_mean(int x, int y, int radius) const;
    int cols() const { return cols_; }
    int rows() const { return rows_; }

  private:
    cv::Mat sum_[3];
    int cols_ = 0;
    int rows_ = 0;
};

class FingerprintModel {
  public:
    bool fit(const cv::Mat& bgr,
             const std::vector<cv::Point>& samples,
             int large_radius,
             int small_radius,
             int restarts,
             std::mt19937& rng);

    cv::Vec6f feature_at(const IntegralBGR& integral, int x, int y) const;
    float rejection_ratio(const cv::Vec6f& feature) const;
    bool empty() const { return prints_.empty(); }

  private:
    struct ColorCluster {
        cv::Vec3f mean = cv::Vec3f::all(0);
        cv::Vec3f deviation = cv::Vec3f::all(0);
        int count = 0;
    };

    int large_radius_ = 8;
    int small_radius_ = 2;
    std::vector<cv::Vec6f> prints_;
    cv::Vec6f vmax_ = cv::Vec6f::all(0);
    std::vector<ColorCluster> clusters_;
};

}  // namespace pixmix

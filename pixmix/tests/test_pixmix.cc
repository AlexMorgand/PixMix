#include "geom.hh"
#include "inpainter.hh"
#include "pipeline.hh"
#include "selector.hh"
#include "tracker.hh"

#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

int g_failures = 0;

void expect(bool condition, const char* text, int line) {
    if (!condition) {
        std::cerr << "FAIL " << line << ": " << text << "\n";
        ++g_failures;
    }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)

void paint_texture(cv::Mat& image) {
    for (int y = 0; y < image.rows; ++y) {
        cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x) {
            const int checker = ((x / 16) & 1) ^ ((y / 16) & 1);
            const int base = checker ? 210 : 36;
            row[x] = cv::Vec3b(static_cast<uchar>(base), static_cast<uchar>(base * 3 / 4),
                               static_cast<uchar>(40 + (y % 40)));
        }
    }
}

double mae(const cv::Mat& output, const cv::Mat& truth, const cv::Mat& region) {
    double error = 0.0;
    int count = 0;
    for (int y = 0; y < region.rows; ++y) {
        const uchar* mask = region.ptr<uchar>(y);
        for (int x = 0; x < region.cols; ++x) {
            if (!mask[x]) {
                continue;
            }
            const cv::Vec3b a = output.ptr<cv::Vec3b>(y)[x];
            const cv::Vec3b b = truth.ptr<cv::Vec3b>(y)[x];
            error += std::abs(int(a[0]) - int(b[0])) + std::abs(int(a[1]) - int(b[1])) + std::abs(int(a[2]) - int(b[2]));
            ++count;
        }
    }
    return count == 0 ? 1e9 : error / (3.0 * count);
}

double iou(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat intersection;
    cv::Mat either;
    cv::bitwise_and(a, b, intersection);
    cv::bitwise_or(a, b, either);
    const double denom = static_cast<double>(cv::countNonZero(either));
    return denom <= 0.0 ? 0.0 : static_cast<double>(cv::countNonZero(intersection)) / denom;
}

int invalid_sources(const cv::Mat& mask, const cv::Mat& mapping) {
    int bad = 0;
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const cv::Vec2i* map_row = mapping.ptr<cv::Vec2i>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const cv::Vec2i offset = map_row[x];
            const int sx = x + offset[0];
            const int sy = y + offset[1];
            if (!pixmix::offset_valid(offset) || !pixmix::in_bounds(sx, sy, mask.size()) ||
                mask.ptr<uchar>(sy)[sx] != 0) {
                ++bad;
            }
        }
    }
    return bad;
}

void test_rotation() {
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    rotation.at<double>(0, 0) = 0.0;
    rotation.at<double>(0, 1) = -1.0;
    rotation.at<double>(1, 0) = 1.0;
    rotation.at<double>(1, 1) = 0.0;
    const float angle = pixmix::homography_rotation_degrees(rotation);
    EXPECT(std::abs(angle - 90.f) < 1.f);

    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = 12.0;
    translation.at<double>(1, 2) = -4.0;
    EXPECT(pixmix::homography_rotation_degrees(translation) < 1.f);
}

void test_forward_translation() {
    cv::Mat previous_mask(100, 100, CV_8U, cv::Scalar(0));
    cv::rectangle(previous_mask, cv::Rect(20, 20, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::Mat previous_map(100, 100, CV_32SC2, cv::Scalar(pixmix::kInvalidOffset, pixmix::kInvalidOffset));
    for (int y = 20; y < 30; ++y) {
        for (int x = 20; x < 30; ++x) {
            previous_map.ptr<cv::Vec2i>(y)[x] = cv::Vec2i(15, 0);
        }
    }
    cv::Mat current_mask(100, 100, CV_8U, cv::Scalar(0));
    cv::rectangle(current_mask, cv::Rect(30, 20, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::Mat homography = cv::Mat::eye(3, 3, CV_64F);
    homography.at<double>(0, 2) = 10.0;
    const cv::Mat forwarded = pixmix::forward_pixel_map(previous_map, previous_mask, current_mask, homography);
    int mismatches = 0;
    int holes = 0;
    for (int y = 20; y < 30; ++y) {
        for (int x = 30; x < 40; ++x) {
            ++holes;
            const cv::Vec2i offset = forwarded.ptr<cv::Vec2i>(y)[x];
            if (offset != cv::Vec2i(15, 0)) {
                ++mismatches;
            }
        }
    }
    EXPECT(holes == 100);
    EXPECT(mismatches == 0);
}

void test_inpaint_and_selection() {
    constexpr int width = 320;
    constexpr int height = 240;
    cv::Mat clean(height, width, CV_8UC3);
    paint_texture(clean);
    cv::Mat shown = clean.clone();
    const cv::Rect object(120, 80, 80, 70);
    cv::rectangle(shown, object, cv::Scalar(0, 220, 0), cv::FILLED);
    cv::Mat truth_mask(clean.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(truth_mask, object, cv::Scalar(255), cv::FILLED);

    cv::Mat smooth(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        cv::Vec3b* row = smooth.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = cv::Vec3b(static_cast<uchar>(30 + (x / 64) * 25), static_cast<uchar>(90 + y / 6), 150);
        }
    }
    cv::Mat marked = smooth.clone();
    cv::rectangle(marked, object, cv::Scalar(0, 220, 0), cv::FILLED);
    const std::vector<cv::Point> polygon{
        cv::Point(object.x - 12, object.y - 12),
        cv::Point(object.x + object.width + 12, object.y - 12),
        cv::Point(object.x + object.width + 12, object.y + object.height + 12),
        cv::Point(object.x - 12, object.y + object.height + 12),
    };
    pixmix::PixMixConfig config;
    config.seed = 1;
    const pixmix::Selection selection = pixmix::FingerprintSelector(config).refine(marked, polygon);
    EXPECT(!selection.mask.empty());
    const double overlap = iou(selection.mask, truth_mask);
    std::cout << "selection iou=" << overlap << "\n";
    EXPECT(overlap > 0.8);

    pixmix::PixMixInpainter inpainter(config);
    const cv::Mat filled = inpainter.run(shown, truth_mask);
    EXPECT(invalid_sources(truth_mask, inpainter.mapping()) == 0);
    cv::Mat interior;
    cv::erode(truth_mask, interior, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
    const double error = mae(filled, clean, interior);
    std::cout << "inpaint mae=" << error << "\n";
    EXPECT(error < 25.0);
    if (error >= 25.0) {
        cv::imwrite("pixmix_test_inpaint.png", filled);
        cv::imwrite("pixmix_test_clean.png", clean);
    }
}

void test_video_coherence() {
    constexpr int width = 320;
    constexpr int height = 240;
    constexpr int shift = 6;
    cv::Mat clean(height, width + shift, CV_8UC3);
    paint_texture(clean);
    cv::Mat shown = clean.clone();
    const cv::Rect object(110, 70, 70, 60);
    cv::rectangle(shown, object, cv::Scalar(0, 220, 0), cv::FILLED);

    const cv::Mat frame0 = shown(cv::Rect(0, 0, width, height)).clone();
    const cv::Mat frame1 = shown(cv::Rect(shift, 0, width, height)).clone();
    const cv::Mat truth0 = clean(cv::Rect(0, 0, width, height)).clone();
    const cv::Mat truth1 = clean(cv::Rect(shift, 0, width, height)).clone();
    cv::Mat mask0(frame0.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask0, object, cv::Scalar(255), cv::FILLED);

    pixmix::PixMixConfig config;
    config.seed = 1;
    pixmix::PixMixPipeline pipeline(config);
    pixmix::FrameOutput first;
    EXPECT(pipeline.start_with_mask(frame0, mask0, &first));
    pixmix::FrameOutput second;
    EXPECT(pipeline.process(frame1, second));
    EXPECT(second.stats.tracking_ok);
    EXPECT(second.stats.inliers >= config.min_inliers);
    const cv::Rect moved(object.x - shift, object.y, object.width, object.height);
    cv::Mat truth_mask(frame1.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(truth_mask, moved, cv::Scalar(255), cv::FILLED);
    const double overlap = iou(second.mask, truth_mask);
    std::cout << "track iou=" << overlap << " inliers=" << second.stats.inliers << "\n";
    EXPECT(overlap > 0.7);
    cv::Mat interior;
    cv::erode(truth_mask, interior, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    const double error = mae(second.image, truth1, interior);
    std::cout << "video mae=" << error << " total_ms=" << second.stats.total_ms << "\n";
    EXPECT(error < 30.0);
    const double first_error = mae(first.image, truth0, mask0);
    std::cout << "first mae=" << first_error << "\n";
    EXPECT(first_error < 30.0);
    if (error >= 30.0 || overlap <= 0.7) {
        cv::imwrite("pixmix_test_frame1.png", second.image);
        cv::imwrite("pixmix_test_mask1.png", second.mask);
    }
}

void test_tracking_loss_recovers() {
    constexpr int width = 320;
    constexpr int height = 240;
    constexpr int shift = 6;
    cv::Mat clean(height, width + shift, CV_8UC3);
    paint_texture(clean);
    cv::Mat shown = clean.clone();
    const cv::Rect object(110, 70, 70, 60);
    cv::rectangle(shown, object, cv::Scalar(0, 220, 0), cv::FILLED);
    const cv::Mat frame0 = shown(cv::Rect(0, 0, width, height)).clone();
    const cv::Mat frame1 = shown(cv::Rect(shift, 0, width, height)).clone();
    cv::Mat mask0(frame0.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask0, object, cv::Scalar(255), cv::FILLED);

    pixmix::PixMixConfig config;
    config.seed = 1;
    pixmix::PixMixPipeline pipeline(config);
    EXPECT(pipeline.start_with_mask(frame0, mask0));
    pixmix::FrameOutput moved;
    EXPECT(pipeline.process(frame1, moved));
    EXPECT(moved.stats.tracking_ok);

    cv::Mat blank(height, width, CV_8UC3, cv::Scalar(18, 18, 18));
    pixmix::FrameOutput lost;
    EXPECT(pipeline.process(blank, lost));
    EXPECT(!lost.stats.tracking_ok);
    EXPECT(cv::norm(lost.image, blank, cv::NORM_INF) == 0.0);

    pixmix::FrameOutput recovered;
    EXPECT(pipeline.process(frame1, recovered));
    EXPECT(recovered.stats.tracking_ok);
    EXPECT(recovered.stats.inliers >= config.min_inliers);
    const cv::Rect moved_object(object.x - shift, object.y, object.width, object.height);
    cv::Mat truth_mask(frame1.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(truth_mask, moved_object, cv::Scalar(255), cv::FILLED);
    const double overlap = iou(recovered.mask, truth_mask);
    std::cout << "recovery iou=" << overlap << " inliers=" << recovered.stats.inliers << "\n";
    EXPECT(overlap > 0.7);
}

}  // namespace

int main() {
    test_rotation();
    test_forward_translation();
    test_inpaint_and_selection();
    test_video_coherence();
    test_tracking_loss_recovers();
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

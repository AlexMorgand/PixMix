#include "pipeline.hh"

#include "geom.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct Args {
    int camera = 0;
    std::string video;
    std::string still;
    std::string mask;
    int width = 640;
    int height = 480;
    unsigned seed = 1;
    bool benchmark = false;
    int bench_frames = 20;
    int bench_hole = 0;
    std::string polygon;
    std::string output;
    int max_side = 0;
    int rotate = 0;
    int dilate = 0;
    float alpha = -1.f;
    int video_passes = -1;
    int search_radius = -1;
    int max_frames = 0;
    bool exact_mask = false;
    bool correct_contour = true;
    float tau_s = -1.f;
    float tau_a = -1.f;
    float locality = -1.f;
    int patch_size = 0;
    bool structural_candidate = false;
    std::string debug_output;
    std::string scales;
};

void print_usage() {
    std::cout
        << "PixMix diminished reality\n"
        << "  pixmix [--camera N] [--video PATH] [--width 640] [--height 480] [--seed 1]\n"
        << "  pixmix --still IMAGE --mask MASK\n"
        << "  pixmix --still IMAGE --polygon x,y,x,y,... --output OUT [--max-side N] [--dilate N]\n"
        << "  pixmix --video PATH --polygon x,y,x,y,... --output OUT [--rotate 90] [--max-side N] [--dilate N]\n"
        << "         [--debug-output DEBUG.mp4] [--no-correct] [--exact] [--max-frames N]\n"
        << "  tuning: --alpha A --tau-s T --tau-a T --locality T --patch-size ODD --video-passes N --search-radius R --no-structure\n"
        << "  --scales DIR writes each pyramid level, coarse to fine\n"
        << "  pixmix --benchmark [--bench-hole PIXELS] [--bench-frames N]\n\n"
        << "Selection: click the frozen frame, z undo, c clear, Enter start, q quit.\n"
        << "Running: q quit, r reselect, d mask overlay, p pause.\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (key == "--camera") {
            args.camera = std::stoi(need("--camera"));
        } else if (key == "--video") {
            args.video = need("--video");
        } else if (key == "--still") {
            args.still = need("--still");
        } else if (key == "--mask") {
            args.mask = need("--mask");
        } else if (key == "--width") {
            args.width = std::stoi(need("--width"));
        } else if (key == "--height") {
            args.height = std::stoi(need("--height"));
        } else if (key == "--seed") {
            args.seed = static_cast<unsigned>(std::stoul(need("--seed")));
        } else if (key == "--benchmark") {
            args.benchmark = true;
        } else if (key == "--bench-frames") {
            args.bench_frames = std::stoi(need("--bench-frames"));
        } else if (key == "--bench-hole") {
            args.bench_hole = std::stoi(need("--bench-hole"));
        } else if (key == "--polygon") {
            args.polygon = need("--polygon");
        } else if (key == "--output") {
            args.output = need("--output");
        } else if (key == "--max-side") {
            args.max_side = std::stoi(need("--max-side"));
        } else if (key == "--rotate") {
            args.rotate = std::stoi(need("--rotate"));
        } else if (key == "--dilate") {
            args.dilate = std::stoi(need("--dilate"));
        } else if (key == "--alpha") {
            args.alpha = std::stof(need("--alpha"));
        } else if (key == "--video-passes") {
            args.video_passes = std::stoi(need("--video-passes"));
        } else if (key == "--search-radius") {
            args.search_radius = std::stoi(need("--search-radius"));
        } else if (key == "--max-frames") {
            args.max_frames = std::stoi(need("--max-frames"));
        } else if (key == "--exact") {
            args.exact_mask = true;
        } else if (key == "--tau-s") {
            args.tau_s = std::stof(need("--tau-s"));
        } else if (key == "--tau-a") {
            args.tau_a = std::stof(need("--tau-a"));
        } else if (key == "--locality") {
            args.locality = std::stof(need("--locality"));
        } else if (key == "--patch-size") {
            args.patch_size = std::stoi(need("--patch-size"));
        } else if (key == "--no-structure") {
            args.structural_candidate = false;
        } else if (key == "--debug-output") {
            args.debug_output = need("--debug-output");
        } else if (key == "--scales") {
            args.scales = need("--scales");
        } else if (key == "--no-correct") {
            args.correct_contour = false;
        } else if (key == "--help" || key == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument " + key);
        }
    }
    return args;
}

pixmix::PixMixConfig make_config(const Args& args) {
    pixmix::PixMixConfig config;
    config.seed = args.seed;
    config.hole_margin = std::max(0, args.dilate);
    config.structural_candidate = args.structural_candidate;
    if (args.alpha >= 0.f) {
        config.alpha = args.alpha;
    }
    if (args.video_passes > 0) {
        config.video_passes = args.video_passes;
    }
    if (args.search_radius > 0) {
        config.video_search_radius = args.search_radius;
    }
    if (args.tau_s > 0.f) {
        config.tau_s = args.tau_s;
    }
    if (args.tau_a > 0.f) {
        config.tau_a = args.tau_a;
    }
    if (args.locality >= 0.f) {
        config.source_locality = args.locality;
    }
    if (args.patch_size > 0) {
        if (args.patch_size % 2 == 0) {
            throw std::runtime_error("--patch-size must be odd");
        }
        config.patch_radius = args.patch_size / 2;
    }
    config.correct_contour = args.correct_contour;
    return config;
}

cv::Mat resize_frame(const cv::Mat& frame, int width, int height) {
    if (frame.empty() || (frame.cols == width && frame.rows == height)) {
        return frame;
    }
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    return resized;
}

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

void draw_stats(cv::Mat& image, const pixmix::FrameStats& stats, bool tracking) {
    char line[192];
    const double fps = 1000.0 / std::max(stats.total_ms, 0.1);
    std::snprintf(line, sizeof(line), "fps %4.1f   track %5.1f  ref %5.1f  inpaint %5.1f  comp %4.1f ms", fps,
                  stats.track_ms, stats.reference_ms, stats.inpaint_ms, stats.composite_ms);
    cv::putText(image, line, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, line, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::snprintf(line, sizeof(line), "hole %d   inliers %d   rot %4.1f deg%s", stats.hole_pixels, stats.inliers,
                  stats.rotation_deg, stats.keyframe_reset ? "   keyframe" : "");
    cv::putText(image, line, cv::Point(8, 42), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, line, cv::Point(8, 42), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if (!tracking) {
        cv::putText(image, "tracking lost", cv::Point(8, 66), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2,
                    cv::LINE_AA);
    }
}

struct UiState {
    cv::Mat* live = nullptr;
    cv::Mat frozen;
    std::vector<cv::Point> polygon;
};

void on_mouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    auto* ui = static_cast<UiState*>(userdata);
    if (ui->frozen.empty() && ui->live != nullptr && !ui->live->empty()) {
        ui->frozen = ui->live->clone();
    }
    if (!ui->frozen.empty()) {
        ui->polygon.emplace_back(x, y);
    }
}

std::vector<cv::Point> parse_polygon(const std::string& text) {
    std::vector<int> values;
    std::string token;
    for (const char character : text) {
        if (character == ',' || character == ' ' || character == ';') {
            if (!token.empty()) {
                values.push_back(std::stoi(token));
                token.clear();
            }
        } else {
            token.push_back(character);
        }
    }
    if (!token.empty()) {
        values.push_back(std::stoi(token));
    }
    if (values.size() < 6 || values.size() % 2 != 0) {
        throw std::runtime_error("polygon needs at least 3 x,y pairs");
    }
    std::vector<cv::Point> polygon;
    for (size_t i = 0; i < values.size(); i += 2) {
        polygon.emplace_back(values[i], values[i + 1]);
    }
    return polygon;
}

cv::Mat orient_frame(const cv::Mat& frame, int rotate) {
    if (rotate == 0 || frame.empty()) {
        return frame;
    }
    cv::Mat oriented;
    if (rotate == 90) {
        cv::rotate(frame, oriented, cv::ROTATE_90_CLOCKWISE);
    } else if (rotate == 180) {
        cv::rotate(frame, oriented, cv::ROTATE_180);
    } else if (rotate == 270 || rotate == -90) {
        cv::rotate(frame, oriented, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else {
        throw std::runtime_error("rotate must be 0, 90, 180, or 270");
    }
    return oriented;
}

cv::Mat fit_max_side(const cv::Mat& frame, int max_side) {
    if (frame.empty() || max_side <= 0) {
        return frame;
    }
    const int long_side = std::max(frame.cols, frame.rows);
    if (long_side <= max_side) {
        return frame;
    }
    const double scale = static_cast<double>(max_side) / static_cast<double>(long_side);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

std::vector<cv::Point> scale_polygon(const std::vector<cv::Point>& polygon, const cv::Size& from, const cv::Size& to) {
    std::vector<cv::Point> scaled;
    scaled.reserve(polygon.size());
    for (const cv::Point& point : polygon) {
        scaled.emplace_back(cvRound(point.x * (static_cast<double>(to.width) / from.width)),
                            cvRound(point.y * (static_cast<double>(to.height) / from.height)));
    }
    return scaled;
}

void label(cv::Mat& image, const std::string& text, int line) {
    const cv::Point origin(6, 18 + line * 18);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

// Offsets as color: hue is the direction of f(p) - p, brightness its length.
// Constant color regions are the joined source blocks of the thesis (Fig. 4.6).
cv::Mat mapping_view(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat& mapping) {
    cv::Mat view;
    cv::addWeighted(frame, 0.35, cv::Mat::zeros(frame.size(), frame.type()), 0.65, 0.0, view);
    if (mapping.empty() || mapping.size() != frame.size() || mask.empty()) {
        return view;
    }
    const float longest = 0.5f * static_cast<float>(std::max(frame.cols, frame.rows));
    cv::Mat hsv(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < frame.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const cv::Vec2i* map_row = mapping.ptr<cv::Vec2i>(y);
        cv::Vec3b* hsv_row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            if (!mask_row[x]) {
                continue;
            }
            const cv::Vec2i offset = map_row[x];
            if (!pixmix::offset_valid(offset)) {
                hsv_row[x] = cv::Vec3b(0, 0, 255);
                continue;
            }
            const float angle = std::atan2(static_cast<float>(offset[1]), static_cast<float>(offset[0]));
            const float length = std::hypot(static_cast<float>(offset[0]), static_cast<float>(offset[1]));
            const int hue = static_cast<int>((angle + static_cast<float>(CV_PI)) * 90.f / static_cast<float>(CV_PI)) % 180;
            const int value = 80 + static_cast<int>(175.f * std::min(1.f, length / longest));
            hsv_row[x] = cv::Vec3b(static_cast<uchar>(hue), 255, static_cast<uchar>(value));
        }
    }
    cv::Mat colored;
    cv::cvtColor(hsv, colored, cv::COLOR_HSV2BGR);
    colored.copyTo(view, mask);
    return view;
}

cv::Mat debug_panel(const cv::Mat& frame, const pixmix::FrameOutput& output, int index) {
    cv::Mat tracking = frame.clone();
    if (!output.mask.empty()) {
        cv::Mat tint = tracking.clone();
        tint.setTo(cv::Scalar(0, 0, 255), output.mask);
        cv::addWeighted(tint, 0.3, tracking, 0.7, 0.0, tracking);
    }
    if (output.debug.warped_contour.size() >= 3) {
        cv::polylines(tracking, std::vector<std::vector<cv::Point>>{output.debug.warped_contour}, true,
                      cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    if (output.contour.size() >= 3) {
        const cv::Scalar color = output.stats.tracking_ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::polylines(tracking, std::vector<std::vector<cv::Point>>{output.contour}, true, color, 2, cv::LINE_AA);
    }
    for (const cv::Point2f& point : output.debug.outlier_points) {
        cv::circle(tracking, point, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
    for (const cv::Point2f& point : output.debug.inlier_points) {
        cv::circle(tracking, point, 2, cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
    }
    char line[160];
    std::snprintf(line, sizeof(line), "frame %d  %s  inliers %d  outliers %zu", index,
                  output.stats.tracking_ok ? "track ok" : "LOST", output.stats.inliers,
                  output.debug.outlier_points.size());
    label(tracking, line, 0);
    std::snprintf(line, sizeof(line), "hole %d px  rot %.1f deg%s", output.stats.hole_pixels, output.stats.rotation_deg,
                  output.stats.keyframe_reset ? "  KEYFRAME" : "");
    label(tracking, line, 1);
    std::snprintf(line, sizeof(line), "track %.1f ref %.1f inpaint %.1f ms", output.stats.track_ms,
                  output.stats.reference_ms, output.stats.inpaint_ms);
    label(tracking, line, 2);

    cv::Mat result = output.image.empty() ? frame.clone() : output.image.clone();
    label(result, "output", 0);
    cv::Mat reference = output.debug.reference.empty() ? frame.clone() : output.debug.reference.clone();
    label(reference, "reference model (warped keyframe + lighting)", 0);
    cv::Mat mapping = mapping_view(frame, output.mask, output.debug.mapping);
    label(mapping, "offset map (hue = direction)", 0);

    cv::Mat top;
    cv::Mat bottom;
    cv::Mat panel;
    cv::hconcat(tracking, result, top);
    cv::hconcat(reference, mapping, bottom);
    cv::vconcat(top, bottom, panel);
    return panel;
}

bool open_writer(cv::VideoWriter& writer, const std::string& path, double fps, const cv::Size& size) {
    const int codecs[] = {cv::VideoWriter::fourcc('a', 'v', 'c', '1'), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G')};
    for (const int codec : codecs) {
        writer.open(path, codec, fps > 1.0 ? fps : 30.0, size);
        if (writer.isOpened()) {
            return true;
        }
    }
    return false;
}

int run_polygon_still(const Args& args) {
    cv::Mat image = orient_frame(cv::imread(args.still, cv::IMREAD_COLOR), args.rotate);
    if (image.empty()) {
        std::cerr << "could not read " << args.still << "\n";
        return 1;
    }
    const cv::Size oriented = image.size();
    image = fit_max_side(image, args.max_side);
    const std::vector<cv::Point> polygon = scale_polygon(parse_polygon(args.polygon), oriented, image.size());
    pixmix::PixMixPipeline pipeline(make_config(args));
    pixmix::FrameOutput output;
    const bool started = args.exact_mask
                             ? pipeline.start_with_mask(image, pixmix::mask_from_contour(image.size(), polygon), &output)
                             : pipeline.start_with_polygon(image, polygon, &output);
    if (!started) {
        std::cerr << "selection failed; using the polygon as the mask\n";
        if (!pipeline.start_with_mask(image, pixmix::mask_from_contour(image.size(), polygon), &output)) {
            std::cerr << "inpainting failed\n";
            return 1;
        }
    }
    const std::string path = args.output.empty() ? std::string("pixmix_still.png") : args.output;
    if (!cv::imwrite(path, output.image)) {
        std::cerr << "could not write " << path << "\n";
        return 1;
    }
    std::cout << "wrote " << path << " size=" << output.image.cols << "x" << output.image.rows
              << " hole=" << output.stats.hole_pixels << " inpaint_ms=" << output.stats.inpaint_ms << "\n";
    return 0;
}

int run_polygon_video(const Args& args) {
    cv::VideoCapture capture(args.video);
    if (!capture.isOpened()) {
        std::cerr << "could not open " << args.video << "\n";
        return 1;
    }
    const std::vector<cv::Point> source_polygon = parse_polygon(args.polygon);
    pixmix::PixMixPipeline pipeline(make_config(args));
    cv::VideoWriter writer;
    cv::VideoWriter debug_writer;
    cv::Size oriented_size;
    int index = 0;
    int lost = 0;
    double total_ms = 0.0;
    // Temporal flicker: mean absolute change inside the hole between the
    // motion-compensated previous output and the current output.
    double flicker_sum = 0.0;
    double background_sum = 0.0;
    int flicker_frames = 0;
    cv::Mat previous_output;
    while (true) {
        cv::Mat raw;
        if (!capture.read(raw) || raw.empty()) {
            break;
        }
        cv::Mat oriented = orient_frame(raw, args.rotate);
        if (oriented_size.empty()) {
            oriented_size = oriented.size();
        }
        cv::Mat frame = fit_max_side(oriented, args.max_side);
        pixmix::FrameOutput output;
        if (index == 0) {
            const std::vector<cv::Point> polygon = scale_polygon(source_polygon, oriented_size, frame.size());
            const bool started = args.exact_mask
                                     ? pipeline.start_with_mask(frame, pixmix::mask_from_contour(frame.size(), polygon), &output)
                                     : pipeline.start_with_polygon(frame, polygon, &output);
            if (!started) {
                std::cerr << "selection failed; using the polygon as the mask\n";
                if (!pipeline.start_with_mask(frame, pixmix::mask_from_contour(frame.size(), polygon), &output)) {
                    std::cerr << "inpainting failed\n";
                    return 1;
                }
            }
            if (args.output.empty() || !open_writer(writer, args.output, capture.get(cv::CAP_PROP_FPS), output.image.size())) {
                std::cerr << "could not open video writer for " << args.output << "\n";
                return 1;
            }
            const cv::Size panel_size(frame.cols * 2, frame.rows * 2);
            if (!args.debug_output.empty() &&
                !open_writer(debug_writer, args.debug_output, capture.get(cv::CAP_PROP_FPS), panel_size)) {
                std::cerr << "could not open debug writer for " << args.debug_output << "\n";
                return 1;
            }
        } else if (!pipeline.process(frame, output)) {
            std::cerr << "frame " << index << " failed\n";
            return 1;
        }
        if (!output.stats.tracking_ok) {
            ++lost;
        }
        writer.write(output.image);
        if (debug_writer.isOpened()) {
            debug_writer.write(debug_panel(frame, output, index));
        }
        if (index > 0) {
            total_ms += output.stats.total_ms;
        }
        double flicker = 0.0;
        if (!previous_output.empty() && !output.debug.homography.empty() && !output.mask.empty()) {
            cv::Mat warped;
            cv::warpPerspective(previous_output, warped, output.debug.homography, frame.size(), cv::INTER_LINEAR,
                                cv::BORDER_REPLICATE);
            cv::Mat inner;
            cv::erode(output.mask, inner, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
            if (cv::countNonZero(inner) > 0) {
                cv::Mat difference;
                cv::absdiff(warped, output.image, difference);
                const cv::Scalar mean = cv::mean(difference, inner);
                flicker = (mean[0] + mean[1] + mean[2]) / 3.0;
                cv::Mat ring;
                cv::dilate(output.mask, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41)));
                ring.setTo(0, output.mask);
                const cv::Scalar outside = cv::mean(difference, ring);
                flicker_sum += flicker;
                background_sum += (outside[0] + outside[1] + outside[2]) / 3.0;
                ++flicker_frames;
            }
        }
        previous_output = output.image;
        if (index % 15 == 0) {
            std::cout << "frame " << index << " hole=" << output.stats.hole_pixels
                      << " total_ms=" << output.stats.total_ms << " inliers=" << output.stats.inliers
                      << " ok=" << output.stats.tracking_ok << " flicker=" << flicker << "\n";
        }
        ++index;
        if (args.max_frames > 0 && index >= args.max_frames) {
            break;
        }
    }
    std::cout << "wrote " << args.output << " frames=" << index << " lost=" << lost
              << " mean_ms=" << (index > 1 ? total_ms / (index - 1) : 0.0)
              << " mean_flicker=" << (flicker_frames > 0 ? flicker_sum / flicker_frames : 0.0)
              << " background_flicker=" << (flicker_frames > 0 ? background_sum / flicker_frames : 0.0) << "\n";
    return index == 0 ? 1 : 0;
}

void write_scales(const std::string& directory, const std::vector<pixmix::ScaleView>& scales) {
    if (scales.empty()) {
        return;
    }
    std::filesystem::create_directories(directory);
    const int cell_h = 220;
    std::vector<cv::Mat> before_row;
    std::vector<cv::Mat> after_row;
    std::vector<cv::Mat> block_row;
    int sheet_w = 0;
    for (size_t index = 0; index < scales.size(); ++index) {
        const pixmix::ScaleView& scale = scales[index];
        cv::imwrite(directory + "/before_" + std::to_string(index) + ".png", scale.before);
        cv::imwrite(directory + "/after_" + std::to_string(index) + ".png", scale.after);
        if (!scale.blocks.empty()) {
            cv::imwrite(directory + "/blocks_" + std::to_string(index) + ".png", scale.blocks);
        }
        auto present = [&](const cv::Mat& image, const std::string& caption) {
            cv::Mat cell;
            const int cell_w = std::max(1, image.cols * cell_h / std::max(1, image.rows));
            cv::resize(image, cell, cv::Size(cell_w, cell_h), 0, 0, cv::INTER_NEAREST);
            cv::rectangle(cell, cv::Rect(0, 0, std::min(cell.cols, 280), 28), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(cell, caption, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1,
                        cv::LINE_AA);
            return cell;
        };
        const std::string size = std::to_string(scale.after.cols) + "x" + std::to_string(scale.after.rows);
        before_row.push_back(present(scale.before, "in " + size));
        after_row.push_back(present(scale.after, "out " + size));
        if (!scale.blocks.empty()) {
            int gray = 0;
            int hole = 0;
            for (int y = 0; y < scale.blocks.rows; ++y) {
                const cv::Vec3b* row = scale.blocks.ptr<cv::Vec3b>(y);
                for (int x = 0; x < scale.blocks.cols; ++x) {
                    if (row[x] == cv::Vec3b(255, 255, 255)) {
                        continue;
                    }
                    ++hole;
                    if (row[x] != cv::Vec3b(0, 0, 0)) {
                        ++gray;
                    }
                }
            }
            const int percent = hole > 0 ? (100 * gray) / hole : 0;
            block_row.push_back(present(scale.blocks, "blocks " + std::to_string(percent) + "% agree"));
        }
        sheet_w += before_row.back().cols + 6;
    }
    const int rows = block_row.empty() ? 2 : 3;
    cv::Mat sheet(cell_h * rows + 6 * (rows - 1), std::max(1, sheet_w), CV_8UC3, cv::Scalar(30, 30, 30));
    int x = 0;
    for (size_t index = 0; index < before_row.size(); ++index) {
        before_row[index].copyTo(sheet(cv::Rect(x, 0, before_row[index].cols, cell_h)));
        after_row[index].copyTo(sheet(cv::Rect(x, cell_h + 6, after_row[index].cols, cell_h)));
        if (index < block_row.size()) {
            block_row[index].copyTo(sheet(cv::Rect(x, cell_h * 2 + 12, block_row[index].cols, cell_h)));
        }
        x += before_row[index].cols + 6;
    }
    cv::imwrite(directory + "/scales.png", sheet);
    std::cout << "wrote " << directory << "/scales.png levels=" << scales.size() << "\n";
}

int run_still(const Args& args) {
    if (args.mask.empty()) {
        std::cerr << "--still requires --mask\n";
        return 1;
    }
    const cv::Mat image = cv::imread(args.still, cv::IMREAD_COLOR);
    const cv::Mat mask = cv::imread(args.mask, cv::IMREAD_GRAYSCALE);
    if (image.empty() || mask.empty()) {
        std::cerr << "could not read still image or mask\n";
        return 1;
    }
    pixmix::PixMixPipeline pipeline(make_config(args));
    pixmix::FrameOutput first;
    if (!pipeline.start_with_mask(image, mask, &first)) {
        std::cerr << "inpainting failed\n";
        return 1;
    }
    const std::string output = args.output.empty() ? std::string("pixmix_still.png") : args.output;
    if (!cv::imwrite(output, first.image)) {
        std::cerr << "could not write " << output << "\n";
        return 1;
    }
    std::cout << "wrote " << output << " hole=" << first.stats.hole_pixels
              << " inpaint_ms=" << first.stats.inpaint_ms << "\n";
    if (!args.scales.empty()) {
        write_scales(args.scales, pipeline.scales());
    }
    if (args.output.empty() && std::getenv("DISPLAY") != nullptr) {
        cv::imshow("PixMix", first.image);
        cv::waitKey(0);
    }
    return 0;
}

int benchmark_hole(const Args& args, int hole_pixels) {
    const int side = std::max(16, static_cast<int>(std::lround(std::sqrt(static_cast<double>(hole_pixels)))));
    const int frames = std::max(2, args.bench_frames);
    const int shift = 2;
    const cv::Size view(args.width, args.height);
    const cv::Rect object(80, std::max(20, (view.height - side) / 2), side, side);
    const int world_width = view.width + frames * shift + 4;
    cv::Mat clean(view.height, world_width, CV_8UC3);
    paint_texture(clean);
    cv::Mat shown = clean.clone();
    cv::rectangle(shown, object, cv::Scalar(0, 220, 0), cv::FILLED);

    pixmix::PixMixPipeline pipeline(make_config(args));
    std::vector<double> times;
    int lost = 0;
    double mae_sum = 0.0;
    int mae_frames = 0;
    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        const cv::Rect viewport(frame_index * shift, 0, view.width, view.height);
        const cv::Mat frame = shown(viewport).clone();
        const cv::Mat truth = clean(viewport).clone();
        const cv::Rect object_view(object.x - viewport.x, object.y, object.width, object.height);
        cv::Mat truth_mask(view, CV_8U, cv::Scalar(0));
        cv::rectangle(truth_mask, object_view, cv::Scalar(255), cv::FILLED);
        cv::Mat eroded;
        cv::erode(truth_mask, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));

        pixmix::FrameOutput output;
        if (frame_index == 0) {
            if (!pipeline.start_with_mask(frame, truth_mask, &output)) {
                std::cerr << "benchmark failed to start\n";
                return 1;
            }
        } else if (!pipeline.process(frame, output)) {
            std::cerr << "benchmark frame failed\n";
            return 1;
        } else {
            times.push_back(output.stats.total_ms);
            if (!output.stats.tracking_ok) {
                ++lost;
            }
        }
        if (!output.image.empty() && cv::countNonZero(eroded) > 0) {
            const cv::Mat scored = (frame_index == 0) ? truth_mask : eroded;
            double error = 0.0;
            int count = 0;
            for (int y = 0; y < scored.rows; ++y) {
                const uchar* region = scored.ptr<uchar>(y);
                for (int x = 0; x < scored.cols; ++x) {
                    if (!region[x]) {
                        continue;
                    }
                    const cv::Vec3b a = output.image.ptr<cv::Vec3b>(y)[x];
                    const cv::Vec3b b = truth.ptr<cv::Vec3b>(y)[x];
                    error += std::abs(int(a[0]) - int(b[0])) + std::abs(int(a[1]) - int(b[1])) +
                             std::abs(int(a[2]) - int(b[2]));
                    ++count;
                }
            }
            if (count > 0) {
                mae_sum += error / (3.0 * count);
                ++mae_frames;
            }
        }
        std::cout << "hole=" << hole_pixels << " frame=" << frame_index << " total_ms=" << output.stats.total_ms
                  << " inpaint_ms=" << output.stats.inpaint_ms << " track_ms=" << output.stats.track_ms
                  << " inliers=" << output.stats.inliers << " ok=" << output.stats.tracking_ok << "\n";
    }
    if (times.empty()) {
        return 1;
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    std::cout << "RESULT hole=" << side * side << " frames=" << times.size() << " median_ms=" << median
              << " fps=" << (1000.0 / median) << " lost=" << lost
              << " mae=" << (mae_frames ? mae_sum / mae_frames : -1.0) << "\n";
    return lost > static_cast<int>(times.size()) / 4 ? 1 : 0;
}

int run_benchmark(const Args& args) {
    const std::vector<int> holes = args.bench_hole > 0 ? std::vector<int>{args.bench_hole}
                                                       : std::vector<int>{13000, 60000};
    int status = 0;
    for (const int hole : holes) {
        status |= benchmark_hole(args, hole);
    }
    return status;
}

int run_capture(const Args& args) {
    cv::VideoCapture capture;
    if (!args.video.empty()) {
        capture.open(args.video);
    } else {
        capture.open(args.camera);
    }
    if (!capture.isOpened()) {
        std::cerr << "could not open " << (args.video.empty() ? "camera" : args.video) << "\n";
        return 1;
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);

    if (std::getenv("DISPLAY") == nullptr) {
        std::cerr << "no DISPLAY; use --benchmark or --still for headless runs\n";
        return 1;
    }

    UiState ui;
    cv::namedWindow("PixMix", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("PixMix", on_mouse, &ui);
    pixmix::PixMixPipeline pipeline(make_config(args));
    bool selecting = true;
    bool paused = false;
    bool show_mask = false;
    cv::Mat live;
    cv::Mat last_output;

    while (true) {
        if (!paused || live.empty()) {
            cv::Mat raw;
            if (!capture.read(raw) || raw.empty()) {
                if (!args.video.empty()) {
                    break;
                }
                continue;
            }
            live = resize_frame(raw, args.width, args.height);
            ui.live = &live;
        }

        if (selecting) {
            cv::Mat display = (ui.frozen.empty() ? live : ui.frozen).clone();
            for (size_t i = 0; i < ui.polygon.size(); ++i) {
                cv::circle(display, ui.polygon[i], 3, cv::Scalar(0, 255, 255), cv::FILLED);
                if (i > 0) {
                    cv::line(display, ui.polygon[i - 1], ui.polygon[i], cv::Scalar(0, 200, 255), 1);
                }
            }
            if (ui.polygon.size() > 2) {
                cv::line(display, ui.polygon.back(), ui.polygon.front(), cv::Scalar(0, 200, 255), 1);
            }
            const std::string help = ui.frozen.empty() ? "click to freeze and mark the object"
                                                       : "click contour, Enter starts, z undo, c clear";
            cv::putText(display, help, cv::Point(8, display.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(display, help, cv::Point(8, display.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::imshow("PixMix", display);
        }

        const int key = cv::waitKey(selecting || paused ? 30 : 1);
        if (key == 'q' || key == 27) {
            break;
        }
        if (selecting) {
            if ((key == 'z' || key == 8) && !ui.polygon.empty()) {
                ui.polygon.pop_back();
            } else if (key == 'c') {
                ui.polygon.clear();
                ui.frozen.release();
            } else if ((key == 13 || key == 10 || key == ' ') && ui.polygon.size() >= 3 && !ui.frozen.empty()) {
                if (!pipeline.start_with_polygon(ui.frozen, ui.polygon)) {
                    std::cerr << "selection produced an empty mask; draw around the object again\n";
                    ui.polygon.clear();
                    ui.frozen.release();
                    continue;
                }
                selecting = false;
                last_output = ui.frozen;
            }
            continue;
        }

        if (key == 'r') {
            pipeline.reset();
            ui.polygon.clear();
            ui.frozen.release();
            selecting = true;
            continue;
        }
        if (key == 'd') {
            show_mask = !show_mask;
        }
        if (key == 'p') {
            paused = !paused;
        }
        if (paused) {
            continue;
        }

        pixmix::FrameOutput output;
        if (!pipeline.process(live, output) || output.image.empty()) {
            continue;
        }
        cv::Mat display = output.image.clone();
        if (output.contour.size() >= 2) {
            const std::vector<std::vector<cv::Point>> contours{output.contour};
            cv::polylines(display, contours, true, output.stats.tracking_ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                          1, cv::LINE_AA);
        }
        if (show_mask && !output.mask.empty()) {
            cv::Mat red(display.size(), CV_8UC3, cv::Scalar(0, 0, 255));
            red.copyTo(display, output.mask);
        }
        draw_stats(display, output.stats, output.stats.tracking_ok);
        cv::imshow("PixMix", display);
        last_output = display;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.benchmark) {
            return run_benchmark(args);
        }
        if (!args.still.empty() && !args.polygon.empty()) {
            return run_polygon_still(args);
        }
        if (!args.still.empty()) {
            return run_still(args);
        }
        if (!args.video.empty() && !args.polygon.empty()) {
            return run_polygon_video(args);
        }
        return run_capture(args);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        print_usage();
        return 1;
    }
}

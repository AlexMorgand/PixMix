#include "inpainter.hh"

#include "geom.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pixmix {
namespace {

struct Level {
    cv::Mat original;
    cv::Mat image;
    cv::Mat mask;
    cv::Mat map;
    cv::Mat cost;
    cv::Mat weight;
    cv::Mat weight_sum;
    cv::Mat reference;
    cv::Mat distance;
    std::vector<int> row_holes;
    std::vector<cv::Point> holes;
    int scale = 1;
};

class Solver {
  public:
    Solver(const PixMixConfig& config, unsigned seed)
        : config_(config), seed_(seed), profile_(std::getenv("PIXMIX_PROFILE") != nullptr) {}

    cv::Mat run(const cv::Mat& bgr,
                const cv::Mat& hole_mask,
                const cv::Mat& initial_map,
                const cv::Mat& reference,
                int passes,
                cv::Mat& mapping_out,
                int& levels_out,
                std::vector<ScaleView>& scales_out) {
        levels_out = 0;
        scales_out.clear();
        const cv::Mat frame = ensure_bgr(bgr);
        const cv::Mat mask = normalize_mask(hole_mask, frame.size());
        if (frame.empty() || cv::countNonZero(mask) == 0) {
            mapping_out.release();
            return frame.clone();
        }
        auto record = [&](const Level& level, bool after) {
            if (!after) {
                scales_out.emplace_back();
                scales_out.back().before = level.image.clone();
                return;
            }
            scales_out.back().after = level.image.clone();
            scales_out.back().blocks = neighboring_blocks(level);
        };

        const bool video = !initial_map.empty() && initial_map.size() == frame.size() && initial_map.type() == CV_32SC2;
        const int pass_count = passes >= 0 ? passes : (video ? config_.video_passes : config_.still_passes);

        auto timed = [&](const char* phase, int scale, auto&& function) {
            const auto started = std::chrono::steady_clock::now();
            const uint64_t calls_before = cost_calls_.load(std::memory_order_relaxed);
            function();
            if (profile_) {
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                const uint64_t calls = cost_calls_.load(std::memory_order_relaxed) - calls_before;
                std::cerr << "profile phase=" << phase << " scale=" << scale << " ms=" << ms
                          << " cost_calls=" << calls << "\n";
            }
        };

        std::vector<Level> levels;
        timed("build_levels", 0, [&] { levels = build_levels(frame, mask, reference); });
        if (levels.empty()) {
            mapping_out.release();
            return frame.clone();
        }
        scales_out.reserve(levels.size());
        levels_out = static_cast<int>(levels.size());
        const int coarsest = static_cast<int>(levels.size()) - 1;
        Level& top = levels[static_cast<size_t>(coarsest)];
        timed("initialize_top", top.scale, [&] {
            if (video) {
                seed_from_full_map(initial_map, mask, top);
                paint_valid(top);
                recompute_costs(top);
                initialize_invalid(top, false);
            } else {
                // Thesis 4.4.3: one single-pixel patch-initialization pass on
                // the coarsest layer. Finer layers only refine that mapping.
                initialize_thesis_patch(top);
            }
        });
        timed("recompute", top.scale, [&] { recompute_costs(top); });
        record(top, false);
        const int top_radius = search_radius(top, video);
        timed("refine", top.scale,
              [&] { refine(top, pass_count, top_radius, video ? pass_count : config_.max_still_passes); });
        record(top, true);

        for (int index = coarsest - 1; index >= 0; --index) {
            Level& level = levels[static_cast<size_t>(index)];
            if (video) {
                seed_from_full_map(initial_map, mask, level);
                paint_valid(level);
                recompute_costs(level);
            }
            timed("upsample_initialize", level.scale, [&] {
                upsample_from(levels[static_cast<size_t>(index + 1)], level);
                initialize_invalid(level, false);
            });
            timed("recompute", level.scale, [&] { recompute_costs(level); });
            if (!video && index == 0) {
                const cv::Mat before = level.image.clone();
                const cv::Mat before_map = level.map.clone();
                const cv::Mat before_costs = level.cost.clone();
                const int radius = search_radius(level, false);

                // Candidate A keeps the normal upsampled map.
                timed("regular_refine", level.scale,
                      [&] { refine(level, pass_count, radius, config_.max_still_passes); });
                timed("regular_recompute", level.scale, [&] { recompute_costs(level); });
                const double regular_cost = mean_cost(level);
                const double regular_structure = structure_score(level, level.image);
                const cv::Mat regular_image = level.image.clone();
                const cv::Mat regular_map = level.map.clone();
                const cv::Mat regular_costs = level.cost.clone();
                const int hole_pixels = cv::countNonZero(level.mask);
                // Candidate B is specifically for regular line structures that
                // disappear in the pyramid. Smooth fills and weak edge fields
                // cannot benefit from its expensive full-resolution patch
                // initialization.
                const bool structural_evidence =
                    regular_cost > 10.0 && regular_structure > static_cast<double>(hole_pixels) * 1.25;
                if (!config_.structural_candidate || !structural_evidence) {
                    if (profile_) {
                        std::cerr << "profile candidate regular_cost=" << regular_cost
                                  << " regular_structure=" << regular_structure
                                  << " selected=regular structured_skipped=1\n";
                    }
                    scales_out.push_back({before, regular_image, neighboring_blocks(level)});
                    continue;
                }

                // Candidate B reconstructs visible structure at full
                // resolution. This recovers paving lines that disappeared on
                // coarse layers, but can unnecessarily perturb a texture that
                // was already exact.
                level.image = before.clone();
                level.map = before_map.clone();
                level.cost = before_costs.clone();
                timed("structured_initialize", level.scale, [&] {
                    initialize_coarse(level, std::max(10, config_.patch_radius * 5), 4096);
                });
                timed("structured_recompute", level.scale, [&] { recompute_costs(level); });
                timed("structured_refine", level.scale,
                      [&] { refine(level, pass_count, radius, config_.max_still_passes); });
                timed("structured_final_recompute", level.scale, [&] { recompute_costs(level); });
                const double structured_cost = mean_cost(level);
                const double structured_structure = structure_score(level, level.image);
                // The pixel cost prefers erasing edges, so a structured fill may
                // cost more while keeping the lines. Accept that premium only
                // when the candidate actually adds line evidence.
                const bool prefer_structure = structured_cost < regular_cost * 2.0 &&
                                              structured_structure > regular_structure * 1.15;
                if (profile_) {
                    std::cerr << "profile candidate regular_cost=" << regular_cost
                              << " structured_cost=" << structured_cost
                              << " regular_structure=" << regular_structure
                              << " structured_structure=" << structured_structure
                              << " selected=" << ((regular_cost <= structured_cost && !prefer_structure) ? "regular"
                                                                                                        : "structured")
                              << "\n";
                }
                if (regular_cost <= structured_cost && !prefer_structure) {
                    level.image = regular_image;
                    level.map = regular_map;
                    level.cost = regular_costs;
                }
                scales_out.push_back({before, level.image.clone(), neighboring_blocks(level)});
                continue;
            }
            record(level, false);
            const int radius = search_radius(level, video);
            timed("refine", level.scale,
                  [&] { refine(level, pass_count, radius, video ? pass_count : config_.max_still_passes); });
            record(level, true);
        }

        mapping_out = levels.front().map;
        return levels.front().image;
    }

  private:
    PixMixConfig config_;
    unsigned seed_;
    bool profile_ = false;
    mutable std::atomic<uint64_t> cost_calls_{0};

    int search_radius(const Level& level, bool video) const {
        if (video && config_.video_search_radius > 0) {
            return config_.video_search_radius;
        }
        // PatchMatch and thesis 4.3: each random search starts by covering the
        // entire source image, then halves the radius down to one pixel.
        return std::max(level.image.cols, level.image.rows);
    }

    std::vector<Level> build_levels(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat& reference) const {
        std::vector<Level> levels;
        levels.push_back(make_level(frame, mask, reference, 1));
        const int wanted = level_count(mask);
        while (static_cast<int>(levels.size()) < wanted) {
            cv::Mat smaller_image;
            cv::Mat smaller_mask;
            downsample(levels.back().original, levels.back().mask, smaller_image, smaller_mask);
            if (smaller_image.rows < 8 || smaller_image.cols < 8 || cv::countNonZero(smaller_mask) == 0) {
                break;
            }
            cv::Mat smaller_reference;
            if (!reference.empty()) {
                cv::resize(levels.back().reference, smaller_reference, smaller_image.size(), 0, 0, cv::INTER_CUBIC);
            }
            const int scale = std::max(1, frame.cols / smaller_image.cols);
            levels.push_back(make_level(smaller_image, smaller_mask, smaller_reference, scale));
        }
        return levels;
    }

    int level_count(const cv::Mat& mask) const {
        cv::Mat distance;
        cv::distanceTransform(mask, distance, cv::DIST_L2, 3);
        double farthest = 0.0;
        cv::minMaxLoc(distance, nullptr, &farthest);
        int levels = 1;
        while (farthest > config_.coarse_border_px && levels < config_.max_pyramid_levels) {
            farthest *= 0.5;
            ++levels;
        }
        // The distance rule can shrink a thin line or paving pattern below the
        // resolution at which patch initialization can still see it. The
        // published elephant pyramid stops at 48x32, so do not go below that.
        int limited = 1;
        int width = mask.cols;
        int height = mask.rows;
        while (width / 2 >= 48 && height / 2 >= 48 && limited < config_.max_pyramid_levels) {
            width /= 2;
            height /= 2;
            ++limited;
        }
        return std::min(levels, limited);
    }

    Level make_level(const cv::Mat& image, const cv::Mat& mask, const cv::Mat& reference, int scale) const {
        Level level;
        level.original = image.clone();
        level.image = image.clone();
        level.mask = mask.clone();
        level.scale = std::max(1, scale);
        level.map = cv::Mat(mask.size(), CV_32SC2);
        level.map.setTo(cv::Scalar(kInvalidOffset, kInvalidOffset));
        level.cost = cv::Mat(mask.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
        level.reference = reference.empty() ? cv::Mat() : reference;
        // Thesis (4.18): a pixel is weighted by tau_a when its appearance
        // neighborhood touches the source. That covers the known border and
        // the hole pixels within one patch of it. Deeper hole pixels stay at 1.
        level.weight = cv::Mat(mask.size(), CV_32F, cv::Scalar(1.f));
        const int weight_radius = config_.patch_radius;
        for (int y = 0; y < mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            float* weight_row = level.weight.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                bool touches_source = mask_row[x] == 0;
                for (int dy = -weight_radius; dy <= weight_radius && !touches_source; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= mask.rows) {
                        continue;
                    }
                    const uchar* neighbor_row = level.mask.ptr<uchar>(yy);
                    for (int dx = -weight_radius; dx <= weight_radius; ++dx) {
                        const int xx = x + dx;
                        if (xx >= 0 && xx < mask.cols && neighbor_row[xx] == 0) {
                            touches_source = true;
                            break;
                        }
                    }
                }
                if (touches_source) {
                    weight_row[x] = config_.tau_a;
                }
            }
        }
        cv::distanceTransform(level.mask, level.distance, cv::DIST_L2, 3);
        level.weight_sum = cv::Mat(mask.size(), CV_32F, cv::Scalar(0));
        level.row_holes.assign(static_cast<size_t>(mask.rows), 0);
        const int radius = config_.patch_radius;
        int sample_dx[25];
        int sample_dy[25];
        const int sample_count = patch_samples(radius, sample_dx, sample_dy, 25);
        for (int y = 0; y < mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            float* sum_row = level.weight_sum.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (mask_row[x]) {
                    ++level.row_holes[static_cast<size_t>(y)];
                    level.holes.emplace_back(x, y);
                }
                float sum = 0.f;
                for (int sample = 0; sample < sample_count; ++sample) {
                    const int yy = y + sample_dy[sample];
                    const int xx = x + sample_dx[sample];
                    if (yy >= 0 && yy < mask.rows && xx >= 0 && xx < mask.cols) {
                        sum += level.weight.ptr<float>(yy)[xx];
                    }
                }
                sum_row[x] = std::max(sum, 1.f);
            }
        }
        return level;
    }

    static void downsample(const cv::Mat& image, const cv::Mat& mask, cv::Mat& image_out, cv::Mat& mask_out) {
        const cv::Size size(std::max(1, image.cols / 2), std::max(1, image.rows / 2));
        // 2010 paper section 5.2: bicubic image levels, while the mask stays
        // binary. A 2x2 OR keeps a thin hole that nearest-neighbor sampling
        // would delete.
        cv::resize(image, image_out, size, 0, 0, cv::INTER_CUBIC);
        mask_out = cv::Mat(size, CV_8U, cv::Scalar(0));
        for (int y = 0; y < size.height; ++y) {
            uchar* out_row = mask_out.ptr<uchar>(y);
            for (int x = 0; x < size.width; ++x) {
                uchar hole = 0;
                for (int dy = 0; dy < 2 && !hole; ++dy) {
                    const int yy = std::min(y * 2 + dy, mask.rows - 1);
                    const uchar* in_row = mask.ptr<uchar>(yy);
                    for (int dx = 0; dx < 2; ++dx) {
                        const int xx = std::min(x * 2 + dx, mask.cols - 1);
                        if (in_row[xx]) {
                            hole = 255;
                            break;
                        }
                    }
                }
                out_row[x] = hole;
            }
        }
    }

    // Center-free 5x5 sample lattice used by both the appearance cost and its
    // maximum weight. Duplicate rounded positions are skipped.
    int patch_samples(int radius, int* dxs, int* dys, int capacity) const {
        const int steps = std::max(1, 2 * config_.patch_radius);
        int count = 0;
        for (int iy = 0; iy <= steps; ++iy) {
            const int dy = (iy * 2 * radius + steps / 2) / steps - radius;
            for (int ix = 0; ix <= steps; ++ix) {
                const int dx = (ix * 2 * radius + steps / 2) / steps - radius;
                bool duplicate = false;
                for (int previous = 0; previous < count; ++previous) {
                    if (dxs[previous] == dx && dys[previous] == dy) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate || count >= capacity) {
                    continue;
                }
                dxs[count] = dx;
                dys[count] = dy;
                ++count;
            }
        }
        return count;
    }

    bool source_ok(const Level& level, const cv::Point& point, const cv::Vec2i& offset) const {
        if (!offset_valid(offset)) {
            return false;
        }
        const int x = point.x + offset[0];
        const int y = point.y + offset[1];
        return in_bounds(x, y, level.mask.size()) && level.mask.ptr<uchar>(y)[x] == 0;
    }

    // Thesis implementation (4.61), not the summed definition (4.4). The
    // real-time solver keeps only the smallest clamped neighbor disagreement,
    // weighted by 1/|Ns|. A propagated offset therefore has spatial cost zero.
    // Offsets are scaled to full-resolution pixels so every pyramid layer
    // shares one tau_s, as required for a consistent spatial impact.
    // Paper Fig. 4 bottom row and Fig. 5: a joint block is a region whose
    // summed four-neighbor spatial cost is zero. The optimizer itself uses
    // the minimum (spatial_cost); this picture is the block segmentation the
    // paper draws, so seams between copied regions stay visible.
    cv::Mat neighboring_blocks(const Level& level) const {
        cv::Mat view(level.mask.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        const int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int y = 0; y < level.mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            const cv::Vec2i* map_row = level.map.ptr<cv::Vec2i>(y);
            cv::Vec3b* out_row = view.ptr<cv::Vec3b>(y);
            for (int x = 0; x < level.mask.cols; ++x) {
                if (mask_row[x] == 0) {
                    continue;
                }
                const cv::Vec2i offset = map_row[x];
                bool disagrees = !offset_valid(offset);
                for (const auto& neighbor : neighbors) {
                    const int nx = x + neighbor[0];
                    const int ny = y + neighbor[1];
                    if (!in_bounds(nx, ny, level.mask.size()) || level.mask.ptr<uchar>(ny)[nx] == 0) {
                        continue;
                    }
                    const cv::Vec2i other = level.map.ptr<cv::Vec2i>(ny)[nx];
                    if (!offset_valid(other) || other != offset) {
                        disagrees = true;
                    }
                }
                out_row[x] = disagrees ? cv::Vec3b(0, 0, 0) : cv::Vec3b(160, 160, 160);
            }
        }
        return view;
    }

    float spatial_cost(const Level& level, const cv::Point& point, const cv::Vec2i& offset) const {
        const int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        const float clamp = config_.tau_s;
        const float scale = static_cast<float>(level.scale);
        float best = clamp;
        bool any = false;
        for (const auto& neighbor : neighbors) {
            const int x = point.x + neighbor[0];
            const int y = point.y + neighbor[1];
            if (!in_bounds(x, y, level.mask.size()) || level.mask.ptr<uchar>(y)[x] == 0) {
                continue;
            }
            const cv::Vec2i other = level.map.ptr<cv::Vec2i>(y)[x];
            if (!offset_valid(other)) {
                continue;
            }
            any = true;
            const float dx = (offset[0] - other[0]) * scale;
            const float dy = (offset[1] - other[1]) * scale;
            best = std::min(best, std::min(dx * dx + dy * dy, clamp));
        }
        return any ? best * 0.25f : 0.f;
    }

    // Thesis (4.13) appearance plus (5.16) coherence, both as weighted mean
    // per-pixel SSD over the 5x5 window. Appearance compares the target
    // neighborhood with the candidate's source neighborhood and only counts
    // samples whose source lies in S. Coherence compares the candidate's
    // source neighborhood with the reference model, using the same center-free
    // neighborhood N_a.
    float cost_of(const Level& level, const cv::Point& point, const cv::Vec2i& offset, float best) const {
        if (profile_) {
            cost_calls_.fetch_add(1, std::memory_order_relaxed);
        }
        const float spatial = spatial_cost(level, point, offset);
        const float budget = best - config_.alpha * spatial;
        if (budget <= 0.f) {
            return std::numeric_limits<float>::infinity();
        }
        const int radius = config_.patch_radius;
        int sample_dx[25];
        int sample_dy[25];
        const int sample_count = patch_samples(radius, sample_dx, sample_dy, 25);
        const bool coherence = !level.reference.empty();
        const int rows = level.image.rows;
        const int cols = level.image.cols;
        const float appearance_scale = std::max(1e-6f, 1.f - config_.alpha);
        float appear_sse = 0.f;
        float cohere_sse = 0.f;
        int known_targets = 0;
        int appear_samples = 0;
        int cohere_samples = 0;
        auto squared_difference = [](const uchar* left, const uchar* right) {
            const int db = int(left[0]) - int(right[0]);
            const int dg = int(left[1]) - int(right[1]);
            const int dr = int(left[2]) - int(right[2]);
            return static_cast<float>(db * db + dg * dg + dr * dr);
        };
        for (int sample = 0; sample < sample_count; ++sample) {
            const int dy = sample_dy[sample];
            const int dx = sample_dx[sample];
            const int y = point.y + dy;
            if (y < 0 || y >= rows) {
                continue;
            }
            const int sy = y + offset[1];
            const bool source_row_ok = sy >= 0 && sy < rows;
            const float* weight_row = level.weight.ptr<float>(y);
            const uchar* target_row = level.image.ptr<uchar>(y);
            const uchar* target_mask = level.mask.ptr<uchar>(y);
            const cv::Vec2i* target_map = level.map.ptr<cv::Vec2i>(y);
            const uchar* source_row = source_row_ok ? level.image.ptr<uchar>(sy) : nullptr;
            const uchar* source_mask = source_row_ok ? level.mask.ptr<uchar>(sy) : nullptr;
            const uchar* reference_row = coherence ? level.reference.ptr<uchar>(y) : nullptr;
            {
                const int x = point.x + dx;
                if (static_cast<unsigned>(x) >= static_cast<unsigned>(cols)) {
                    continue;
                }
                const bool center = dx == 0 && dy == 0;
                // Every level is fully initialized before refinement, so real
                // source pixels and all mapped target neighbors are available.
                const bool target_known = !center && (target_mask[x] == 0 || offset_valid(target_map[x]));
                known_targets += target_known ? 1 : 0;
                const int sx = x + offset[0];
                if (!source_row_ok || static_cast<unsigned>(sx) >= static_cast<unsigned>(cols) || source_mask[sx] != 0) {
                    continue;
                }
                const uchar* source = source_row + sx * 3;
                const float weight = weight_row[x];
                if (target_known) {
                    // Border samples stay the original frame. The working image
                    // inside the hole is only the current synthesis.
                    const uchar* target = target_mask[x] == 0 ? level.original.ptr<uchar>(y) + x * 3
                                                              : target_row + x * 3;
                    appear_sse += weight * squared_difference(target, source);
                    ++appear_samples;
                }
                if (coherence && !center) {
                    const uchar* guide = reference_row + x * 3;
                    cohere_sse += weight * squared_difference(source, guide);
                    ++cohere_samples;
                }
                // Thesis (4.15) normalizes by the number of valid samples. A
                // uniform 1/|Na| factor would only rescale alpha, which the
                // thesis fixes at 0.2, so it is not applied again here.
                const int unseen = sample_count - sample - 1;
                const int max_samples = std::max(1, appear_samples + cohere_samples + unseen);
                if ((appear_sse + cohere_sse) / max_samples >= budget / appearance_scale) {
                    return std::numeric_limits<float>::infinity();
                }
            }
        }
        // A source window clipped by the image border or the hole must still
        // cover most of the known target neighborhood, otherwise a handful of
        // lucky samples would make the candidate look free.
        if (known_targets > 0 && appear_samples * 2 < known_targets) {
            return std::numeric_limits<float>::infinity();
        }
        if (appear_samples <= 0 && cohere_samples <= 0) {
            return std::numeric_limits<float>::infinity();
        }
        float appearance = appear_samples > 0 ? appear_sse / appear_samples : 0.f;
        if (cohere_samples > 0) {
            appearance += cohere_sse / cohere_samples;
        }
        return config_.alpha * spatial + (1.f - config_.alpha) * appearance;
    }

    bool consider(Level& level, const cv::Point& point, const cv::Vec2i& offset) const {
        if (!source_ok(level, point, offset)) {
            return false;
        }
        float& best = level.cost.ptr<float>(point.y)[point.x];
        const float cost = cost_of(level, point, offset, best);
        if (!(cost < best)) {
            return false;
        }
        level.map.ptr<cv::Vec2i>(point.y)[point.x] = offset;
        const int sx = point.x + offset[0];
        const int sy = point.y + offset[1];
        const uchar* source = level.original.ptr<uchar>(sy) + sx * 3;
        uchar* destination = level.image.ptr<uchar>(point.y) + point.x * 3;
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
        best = cost;
        return true;
    }

    void paint_valid(Level& level) const {
        for (int y = 0; y < level.mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            for (int x = 0; x < level.mask.cols; ++x) {
                if (!mask_row[x]) {
                    continue;
                }
                const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(y)[x];
                const cv::Point point(x, y);
                if (!source_ok(level, point, offset)) {
                    level.map.ptr<cv::Vec2i>(y)[x] = cv::Vec2i(kInvalidOffset, kInvalidOffset);
                    level.cost.ptr<float>(y)[x] = std::numeric_limits<float>::infinity();
                    continue;
                }
                const uchar* source = level.original.ptr<uchar>(y + offset[1]) + (x + offset[0]) * 3;
                uchar* destination = level.image.ptr<uchar>(y) + x * 3;
                destination[0] = source[0];
                destination[1] = source[1];
                destination[2] = source[2];
            }
        }
    }

    void seed_from_full_map(const cv::Mat& full_map, const cv::Mat& full_mask, Level& level) const {
        const double map_x = static_cast<double>(full_map.cols) / level.map.cols;
        const double map_y = static_cast<double>(full_map.rows) / level.map.rows;
        const double offset_x = static_cast<double>(level.map.cols) / full_map.cols;
        const double offset_y = static_cast<double>(level.map.rows) / full_map.rows;
        for (int y = 0; y < level.mask.rows; ++y) {
            if (level.row_holes[static_cast<size_t>(y)] == 0) {
                continue;
            }
            for (int x = 0; x < level.mask.cols; ++x) {
                if (level.mask.ptr<uchar>(y)[x] == 0) {
                    continue;
                }
                const int fy = std::clamp(cvRound((y + 0.5) * map_y - 0.5), 0, full_map.rows - 1);
                const int fx = std::clamp(cvRound((x + 0.5) * map_x - 0.5), 0, full_map.cols - 1);
                if (full_mask.ptr<uchar>(fy)[fx] == 0) {
                    continue;
                }
                const cv::Vec2i full_offset = full_map.ptr<cv::Vec2i>(fy)[fx];
                if (!offset_valid(full_offset)) {
                    continue;
                }
                const cv::Vec2i candidate(cvRound(full_offset[0] * offset_x), cvRound(full_offset[1] * offset_y));
                if (source_ok(level, cv::Point(x, y), candidate)) {
                    level.map.ptr<cv::Vec2i>(y)[x] = candidate;
                }
            }
        }
    }

    void upsample_from(const Level& coarse, Level& fine) const {
        const double mul_x = static_cast<double>(fine.original.cols) / coarse.original.cols;
        const double mul_y = static_cast<double>(fine.original.rows) / coarse.original.rows;
        for (int y = 0; y < fine.mask.rows; ++y) {
            if (fine.row_holes[static_cast<size_t>(y)] == 0) {
                continue;
            }
            const int cy = std::min(coarse.mask.rows - 1, y * coarse.mask.rows / fine.mask.rows);
            for (int x = 0; x < fine.mask.cols; ++x) {
                if (fine.mask.ptr<uchar>(y)[x] == 0) {
                    continue;
                }
                const int cx = std::min(coarse.mask.cols - 1, x * coarse.mask.cols / fine.mask.cols);
                cv::Vec2i offset(kInvalidOffset, kInvalidOffset);
                const int try_x[3] = {cx, std::max(0, cx - 1), std::min(coarse.mask.cols - 1, cx + 1)};
                const int try_y[3] = {cy, std::max(0, cy - 1), std::min(coarse.mask.rows - 1, cy + 1)};
                bool found = false;
                for (int iy = 0; iy < 3 && !found; ++iy) {
                    for (int ix = 0; ix < 3 && !found; ++ix) {
                        if (coarse.mask.ptr<uchar>(try_y[iy])[try_x[ix]] == 0) {
                            continue;
                        }
                        const cv::Vec2i parent = coarse.map.ptr<cv::Vec2i>(try_y[iy])[try_x[ix]];
                        if (!offset_valid(parent)) {
                            continue;
                        }
                        const cv::Vec2i candidate(cvRound(parent[0] * mul_x), cvRound(parent[1] * mul_y));
                        if (source_ok(fine, cv::Point(x, y), candidate)) {
                            offset = candidate;
                            found = true;
                        }
                    }
                }
                if (offset_valid(offset)) {
                    // Video levels may already contain the directly projected
                    // H^-1 f H map. Keep whichever initialization has lower
                    // cost instead of discarding either source of information.
                    consider(fine, cv::Point(x, y), offset);
                }
            }
        }
        paint_valid(fine);
    }

    // Thesis 4.4.3. Shrink the coarse hole one pixel at a time. The next pixel
    // is the border pixel whose known structure is strongest across the border,
    // prior(p) = |b(p) × o(p)|. Only that center pixel is copied.
    void initialize_thesis_patch(Level& level) const {
        const int radius = config_.patch_radius;
        cv::Mat guide = level.original.clone();
        cv::Mat known;
        cv::compare(level.mask, 0, known, cv::CMP_EQ);
        std::vector<cv::Point> sources;
        for (int y = radius; y < level.mask.rows - radius; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            for (int x = radius; x < level.mask.cols - radius; ++x) {
                if (mask_row[x] == 0) {
                    sources.emplace_back(x, y);
                }
            }
        }
        if (sources.empty()) {
            initialize_invalid(level, false);
            return;
        }
        auto luminance = [](const cv::Vec3b& pixel) {
            return 0.114f * pixel[0] + 0.587f * pixel[1] + 0.299f * pixel[2];
        };
        auto gradient = [&](int x, int y) {
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(level.mask.cols - 1, x + 1);
            const int y0 = std::max(0, y - 1);
            const int y1 = std::min(level.mask.rows - 1, y + 1);
            float gx = luminance(guide.ptr<cv::Vec3b>(y)[x1]) - luminance(guide.ptr<cv::Vec3b>(y)[x0]);
            float gy = luminance(guide.ptr<cv::Vec3b>(y1)[x]) - luminance(guide.ptr<cv::Vec3b>(y0)[x]);
            // Thesis (4.32): fold a gradient and its opposite onto one direction.
            if (gx < 0.f) {
                gx = -gx;
                gy = -gy;
            }
            return cv::Point2f(gx, gy);
        };

        int remaining = cv::countNonZero(level.mask);
        while (remaining > 0) {
            cv::Point target(-1, -1);
            float best_priority = -1.f;
            int best_support = -1;
            for (int y = 1; y < level.mask.rows - 1; ++y) {
                for (int x = 1; x < level.mask.cols - 1; ++x) {
                    if (known.ptr<uchar>(y)[x]) {
                        continue;
                    }
                    float border_x = 0.f;
                    float border_y = 0.f;
                    float orientation_x = 0.f;
                    float orientation_y = 0.f;
                    int support = 0;
                    bool contour = false;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        for (int dx = -radius; dx <= radius; ++dx) {
                            if (dx == 0 && dy == 0) {
                                continue;
                            }
                            const int xx = x + dx;
                            const int yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= level.mask.cols || yy >= level.mask.rows ||
                                !known.ptr<uchar>(yy)[xx]) {
                                continue;
                            }
                            border_x += static_cast<float>(dx);
                            border_y += static_cast<float>(dy);
                            const cv::Point2f direction = gradient(xx, yy);
                            orientation_x += direction.x;
                            orientation_y += direction.y;
                            ++support;
                            if (std::abs(dx) + std::abs(dy) == 1) {
                                contour = true;
                            }
                        }
                    }
                    if (!contour) {
                        continue;
                    }
                    const float priority = std::abs(border_x * orientation_y - border_y * orientation_x);
                    if (priority > best_priority || (priority == best_priority && support > best_support)) {
                        best_priority = priority;
                        best_support = support;
                        target = cv::Point(x, y);
                    }
                }
            }
            if (target.x < 0) {
                break;
            }

            struct Sample {
                int dx;
                int dy;
                cv::Vec3b color;
            };
            std::vector<Sample> samples;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = target.x + dx;
                    const int yy = target.y + dy;
                    if ((dx == 0 && dy == 0) || xx < 0 || yy < 0 || xx >= level.mask.cols || yy >= level.mask.rows ||
                        !known.ptr<uchar>(yy)[xx]) {
                        continue;
                    }
                    samples.push_back({dx, dy, guide.ptr<cv::Vec3b>(yy)[xx]});
                }
            }

            cv::Point best_source = sources.front();
            double best_cost = std::numeric_limits<double>::infinity();
            for (const cv::Point& source : sources) {
                double cost = 0.0;
                bool valid = !samples.empty();
                for (const Sample& sample : samples) {
                    const int xx = source.x + sample.dx;
                    const int yy = source.y + sample.dy;
                    if (!known.ptr<uchar>(yy)[xx]) {
                        valid = false;
                        break;
                    }
                    const cv::Vec3b other = guide.ptr<cv::Vec3b>(yy)[xx];
                    for (int channel = 0; channel < 3; ++channel) {
                        const int delta = int(sample.color[channel]) - int(other[channel]);
                        cost += delta * delta;
                    }
                }
                if (valid && cost < best_cost) {
                    best_cost = cost;
                    best_source = source;
                }
            }
            if (!std::isfinite(best_cost)) {
                int nearest = std::numeric_limits<int>::max();
                for (const cv::Point& source : sources) {
                    const int span = (source.x - target.x) * (source.x - target.x) +
                                     (source.y - target.y) * (source.y - target.y);
                    if (span < nearest) {
                        nearest = span;
                        best_source = source;
                    }
                }
            }
            guide.ptr<cv::Vec3b>(target.y)[target.x] = level.original.ptr<cv::Vec3b>(best_source.y)[best_source.x];
            level.map.ptr<cv::Vec2i>(target.y)[target.x] =
                cv::Vec2i(best_source.x - target.x, best_source.y - target.y);
            known.ptr<uchar>(target.y)[target.x] = 255;
            --remaining;
        }
        level.image = guide;
        initialize_invalid(level, false);
    }

    // 2010 paper section 5.2. On a coarsest layer whose hole is only a few
    // pixels deep, copy each hole pixel from the nearest mask-border pixel.
    // That is the single-source form of filling the erosion with border colors.
    void initialize_from_border(Level& level) const {
        for (const cv::Point& point : level.holes) {
            const int reach = static_cast<int>(std::ceil(level.distance.ptr<float>(point.y)[point.x])) + 1;
            cv::Point border(-1, -1);
            int best = std::numeric_limits<int>::max();
            for (int dy = -reach; dy <= reach; ++dy) {
                const int y = point.y + dy;
                if (y < 0 || y >= level.mask.rows) {
                    continue;
                }
                const uchar* mask_row = level.mask.ptr<uchar>(y);
                for (int dx = -reach; dx <= reach; ++dx) {
                    const int x = point.x + dx;
                    if (x < 0 || x >= level.mask.cols || mask_row[x] != 0) {
                        continue;
                    }
                    const int span = dx * dx + dy * dy;
                    if (span < best) {
                        best = span;
                        border = cv::Point(x, y);
                    }
                }
            }
            if (border.x < 0) {
                continue;
            }
            const cv::Vec2i offset(border.x - point.x, border.y - point.y);
            if (!source_ok(level, point, offset)) {
                continue;
            }
            level.map.ptr<cv::Vec2i>(point.y)[point.x] = offset;
            const uchar* source = level.original.ptr<uchar>(border.y) + border.x * 3;
            uchar* destination = level.image.ptr<uchar>(point.y) + point.x * 3;
            destination[0] = source[0];
            destination[1] = source[1];
            destination[2] = source[2];
        }
        initialize_invalid(level, false);
    }

    // Thesis 4.4.3 patch initialization. Grow the known region from the mask
    // border, prioritizing patches that contain strong known structure. Each
    // accepted source patch installs one coherent offset for all newly covered
    // pixels, instead of independently matching pixels after color erosion.
    void initialize_coarse(Level& level, int radius, size_t source_limit) const {
        cv::Mat guide = level.original.clone();
        cv::Mat known;
        cv::compare(level.mask, 0, known, cv::CMP_EQ);
        std::vector<cv::Point> sources;
        for (int y = radius; y < level.mask.rows - radius; ++y) {
            for (int x = radius; x < level.mask.cols - radius; ++x) {
                bool legal = true;
                for (int dy = -radius; dy <= radius && legal; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (level.mask.ptr<uchar>(y + dy)[x + dx] != 0) {
                            legal = false;
                            break;
                        }
                    }
                }
                if (legal) {
                    sources.emplace_back(x, y);
                }
            }
        }
        if (source_limit > 0 && sources.size() > source_limit) {
            std::vector<cv::Point> sampled;
            sampled.reserve(source_limit);
            const double step = static_cast<double>(sources.size()) / source_limit;
            for (size_t index = 0; index < source_limit; ++index) {
                sampled.push_back(sources[std::min(sources.size() - 1, static_cast<size_t>(index * step))]);
            }
            sources.swap(sampled);
        }
        if (sources.empty()) {
            initialize_invalid(level, false);
            return;
        }

        const int cols = level.mask.cols;
        std::set<int> frontier;
        for (int y = 0; y < level.mask.rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (known.ptr<uchar>(y)[x]) {
                    continue;
                }
                bool touches_known = false;
                for (int dy = -radius; dy <= radius && !touches_known; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= level.mask.rows) {
                        continue;
                    }
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = x + dx;
                        if (xx >= 0 && xx < cols && known.ptr<uchar>(yy)[xx]) {
                            touches_known = true;
                            break;
                        }
                    }
                }
                if (touches_known) {
                    frontier.insert(y * cols + x);
                }
            }
        }

        int remaining = cv::countNonZero(level.mask);
        while (remaining > 0) {
            cv::Point target(-1, -1);
            double best_priority = -1.0;
            for (auto iterator = frontier.begin(); iterator != frontier.end();) {
                const int index = *iterator;
                const int y = index / cols;
                const int x = index % cols;
                if (known.ptr<uchar>(y)[x]) {
                    iterator = frontier.erase(iterator);
                    continue;
                }
                ++iterator;
                    int known_count = 0;
                    double structure = 0.0;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        const int yy = y + dy;
                        if (yy <= 0 || yy >= level.mask.rows - 1) {
                            continue;
                        }
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int xx = x + dx;
                            if (xx <= 0 || xx >= level.mask.cols - 1 || !known.ptr<uchar>(yy)[xx]) {
                                continue;
                            }
                            ++known_count;
                            const cv::Vec3b left = guide.ptr<cv::Vec3b>(yy)[xx - 1];
                            const cv::Vec3b right = guide.ptr<cv::Vec3b>(yy)[xx + 1];
                            const cv::Vec3b up = guide.ptr<cv::Vec3b>(yy - 1)[xx];
                            const cv::Vec3b down = guide.ptr<cv::Vec3b>(yy + 1)[xx];
                            for (int channel = 0; channel < 3; ++channel) {
                                structure += std::abs(int(right[channel]) - int(left[channel])) +
                                             std::abs(int(down[channel]) - int(up[channel]));
                            }
                        }
                    }
                    if (known_count == 0) {
                        continue;
                    }
                    const double priority = structure * known_count + known_count;
                    if (priority > best_priority) {
                        best_priority = priority;
                        target = cv::Point(x, y);
                    }
            }
            if (target.x < 0) {
                break;
            }

            cv::Point best_source = sources.front();
            double best_ssd = std::numeric_limits<double>::infinity();
            struct KnownSample {
                int dx;
                int dy;
                cv::Vec3b color;
            };
            std::vector<KnownSample> samples;
            samples.reserve(static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)));
            for (int dy = -radius; dy <= radius; ++dy) {
                const int ty = target.y + dy;
                if (ty < 0 || ty >= level.mask.rows) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int tx = target.x + dx;
                    if ((dx != 0 || dy != 0) && tx >= 0 && tx < level.mask.cols && known.ptr<uchar>(ty)[tx]) {
                        samples.push_back({dx, dy, guide.ptr<cv::Vec3b>(ty)[tx]});
                    }
                }
            }
            auto search_sources = [&](size_t begin, size_t end, double& local_best, size_t& local_index) {
                for (size_t source_index = begin; source_index < end; ++source_index) {
                    const cv::Point& source = sources[source_index];
                    const double ox = source.x - target.x;
                    const double oy = source.y - target.y;
                    const double locality = config_.source_locality * (ox * ox + oy * oy);
                    if (locality >= local_best || samples.empty()) {
                        continue;
                    }
                    double ssd = 0.0;
                    const double limit = (local_best - locality) * samples.size();
                    for (const KnownSample& sample : samples) {
                        const cv::Vec3b b =
                            level.original.ptr<cv::Vec3b>(source.y + sample.dy)[source.x + sample.dx];
                        for (int channel = 0; channel < 3; ++channel) {
                            const int difference = int(sample.color[channel]) - int(b[channel]);
                            ssd += difference * difference;
                        }
                        if (ssd >= limit) {
                            break;
                        }
                    }
                    // Among visually similar paving patches, prefer nearby
                    // source material. This preserves perspective scale.
                    const double score = ssd / samples.size() + locality;
                    if (score < local_best) {
                        local_best = score;
                        local_index = source_index;
                    }
                }
            };

            size_t best_index = 0;
            search_sources(0, sources.size(), best_ssd, best_index);
            best_source = sources[best_index];

            bool filled = false;
            std::vector<cv::Point> newly_known;
            const cv::Vec2i offset(best_source.x - target.x, best_source.y - target.y);
            for (int dy = -radius; dy <= radius; ++dy) {
                const int ty = target.y + dy;
                const int sy = best_source.y + dy;
                if (ty < 0 || ty >= level.mask.rows) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int tx = target.x + dx;
                    if (tx < 0 || tx >= level.mask.cols || known.ptr<uchar>(ty)[tx]) {
                        continue;
                    }
                    guide.ptr<cv::Vec3b>(ty)[tx] = level.original.ptr<cv::Vec3b>(sy)[best_source.x + dx];
                    level.map.ptr<cv::Vec2i>(ty)[tx] = offset;
                    known.ptr<uchar>(ty)[tx] = 255;
                    newly_known.emplace_back(tx, ty);
                    --remaining;
                    filled = true;
                }
            }
            if (!filled) {
                known.ptr<uchar>(target.y)[target.x] = 255;
                newly_known.push_back(target);
                --remaining;
            }
            for (const cv::Point& point : newly_known) {
                frontier.erase(point.y * cols + point.x);
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int y = point.y + dy;
                    if (y < 0 || y >= level.mask.rows) {
                        continue;
                    }
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int x = point.x + dx;
                        if (x >= 0 && x < cols && !known.ptr<uchar>(y)[x]) {
                            frontier.insert(y * cols + x);
                        }
                    }
                }
            }
        }
        level.image = guide;
        initialize_invalid(level, false);
    }

    void initialize_invalid(Level& level, bool exhaustive) const {
        std::vector<cv::Point> sources;
        std::vector<cv::Point> holes;
        sources.reserve(static_cast<size_t>(level.mask.total() / 2));
        for (int y = 0; y < level.mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            for (int x = 0; x < level.mask.cols; ++x) {
                if (mask_row[x]) {
                    const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(y)[x];
                    if (!source_ok(level, cv::Point(x, y), offset)) {
                        holes.emplace_back(x, y);
                        level.cost.ptr<float>(y)[x] = std::numeric_limits<float>::infinity();
                    }
                } else {
                    sources.emplace_back(x, y);
                }
            }
        }
        if (sources.empty() || holes.empty()) {
            return;
        }
        const bool full_search = exhaustive && sources.size() <= 2500 && holes.size() <= 4000;
        if (full_search) {
            cv::Mat distance;
            cv::distanceTransform(level.mask, distance, cv::DIST_L2, 3);
            std::sort(holes.begin(), holes.end(), [&distance](const cv::Point& a, const cv::Point& b) {
                return distance.at<float>(a) < distance.at<float>(b);
            });
        }
        std::mt19937 rng(seed_ + static_cast<unsigned>(level.scale) * 17u);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(sources.size()) - 1);
        const int samples = 48;
        for (const cv::Point& hole : holes) {
            if (full_search) {
                for (const cv::Point& source : sources) {
                    consider(level, hole, cv::Vec2i(source.x - hole.x, source.y - hole.y));
                }
            } else {
                for (int sample = 0; sample < samples; ++sample) {
                    const cv::Point source = sources[static_cast<size_t>(pick(rng))];
                    consider(level, hole, cv::Vec2i(source.x - hole.x, source.y - hole.y));
                }
            }
        }
    }

    void recompute_costs(Level& level) const {
#pragma omp parallel for schedule(static) if (level.holes.size() > 256)
        for (int index = 0; index < static_cast<int>(level.holes.size()); ++index) {
            const cv::Point point = level.holes[static_cast<size_t>(index)];
            const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(point.y)[point.x];
            float cost = std::numeric_limits<float>::infinity();
            if (source_ok(level, point, offset)) {
                cost = cost_of(level, point, offset, std::numeric_limits<float>::infinity());
            }
            level.cost.ptr<float>(point.y)[point.x] = cost;
        }
    }

    double mean_cost(const Level& level) const {
        double sum = 0.0;
        int count = 0;
        for (int y = 0; y < level.mask.rows; ++y) {
            for (int x = 0; x < level.mask.cols; ++x) {
                if (!level.mask.ptr<uchar>(y)[x]) {
                    continue;
                }
                const float cost = level.cost.ptr<float>(y)[x];
                if (std::isfinite(cost)) {
                    sum += cost;
                    ++count;
                }
            }
        }
        return count > 0 ? sum / count : std::numeric_limits<double>::infinity();
    }

    double structure_score(const Level& level, const cv::Mat& image) const {
        cv::Mat gray;
        cv::Mat edges;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 30, 90);
        cv::Mat outside;
        cv::compare(level.mask, 0, outside, cv::CMP_EQ);
        edges.setTo(0, outside);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0, 12, 8.0, 3.0);
        double score = 0.0;
        for (const cv::Vec4i& line : lines) {
            const int x = (line[0] + line[2]) / 2;
            const int y = (line[1] + line[3]) / 2;
            if (!in_bounds(x, y, level.mask.size()) || !level.mask.ptr<uchar>(y)[x]) {
                continue;
            }
            const double dx = line[2] - line[0];
            const double dy = line[3] - line[1];
            score += dx * dx + dy * dy;
        }
        return score;
    }

    bool improve(Level& level, const cv::Point& point, bool forward, int radius, std::mt19937& rng, int probes) const {
        const cv::Vec2i before = level.map.ptr<cv::Vec2i>(point.y)[point.x];
        if (forward) {
            propagate_from(level, point, cv::Point(point.x - 1, point.y));
            propagate_from(level, point, cv::Point(point.x, point.y - 1));
        } else {
            propagate_from(level, point, cv::Point(point.x + 1, point.y));
            propagate_from(level, point, cv::Point(point.x, point.y + 1));
        }
        cv::Vec2i base = level.map.ptr<cv::Vec2i>(point.y)[point.x];
        if (!offset_valid(base)) {
            base = cv::Vec2i(0, 0);
        }
        for (int window = std::max(1, radius); window >= 1; window /= 2) {
            std::uniform_int_distribution<int> jitter(-window, window);
            for (int probe = 0; probe < probes; ++probe) {
                const cv::Vec2i candidate(base[0] + jitter(rng), base[1] + jitter(rng));
                if (consider(level, point, candidate)) {
                    base = candidate;
                }
            }
        }
        return level.map.ptr<cv::Vec2i>(point.y)[point.x] != before;
    }

    bool propagate_from(Level& level, const cv::Point& point, const cv::Point& neighbor) const {
        if (!in_bounds(neighbor.x, neighbor.y, level.mask.size())) {
            return false;
        }
        // Source pixels have no offset. Their colors already enter the
        // appearance cost as the known border; copying the adjacent border
        // pixel would smear that edge one step into the hole.
        if (level.mask.ptr<uchar>(neighbor.y)[neighbor.x] == 0) {
            return false;
        }
        const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(neighbor.y)[neighbor.x];
        if (!offset_valid(offset) || !source_ok(level, point, offset)) {
            return false;
        }
        consider(level, point, offset);
        return true;
    }

    int scan_rows(Level& level, int y0, int y1, bool forward, int radius, unsigned seed) const {
        std::mt19937 rng(seed);
        int changed = 0;
        for (int step = 0; step < y1 - y0; ++step) {
            const int y = forward ? y0 + step : y1 - 1 - step;
            if (level.row_holes[static_cast<size_t>(y)] == 0) {
                continue;
            }
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            for (int i = 0; i < level.mask.cols; ++i) {
                const int x = forward ? i : level.mask.cols - 1 - i;
                if (!mask_row[x]) {
                    continue;
                }
                // 2010 paper section 6.3: the first patch of a parallel subset
                // gets many more random probes so propagation starts near a
                // real match instead of at the subset boundary.
                if (improve(level, cv::Point(x, y), forward, radius, rng, 1)) {
                    ++changed;
                }
            }
        }
        return changed;
    }

    // Runs at least `passes` sweeps, alternating forward and backward scanline
    // order. With max_passes > passes, keeps iterating until fewer than 3% of
    // the hole pixels change, matching the thesis' reported convergence.
    // Row bands are processed in parallel with a guard gap of 2r+1 rows; the
    // band origin moves every pass so no seam stays fixed. Every pyramid level
    // refines its complete initialized map: after upsampling, all hole pixels
    // already have synthesized colors and are valid appearance neighbors.
    void refine(Level& level, int passes, int radius, int max_passes) const {
        if (passes <= 0) {
            return;
        }
        int holes = 0;
        for (const int count : level.row_holes) {
            holes += count;
        }
        const int limit = std::max(passes, max_passes);
        const int band = 32;
        const int gap = 2 * config_.patch_radius + 1;
        const int period = band + gap;
        struct Range {
            int y0;
            int y1;
        };
        int executed = 0;
        for (int pass = 0; pass < limit; ++pass) {
            executed = pass + 1;
            const bool forward = (pass % 2) == 0;
            const int shift = (pass * 13) % period;
            std::vector<Range> interiors;
            std::vector<Range> seams;
            for (int y = -shift; y < level.image.rows; y += period) {
                const int y0 = std::max(0, y);
                const int y1 = std::min(level.image.rows, y + band);
                if (y0 < y1) {
                    interiors.push_back({y0, y1});
                }
                const int seam0 = std::max(0, y1);
                const int seam1 = std::min(level.image.rows, y + period);
                if (seam0 < seam1) {
                    seams.push_back({seam0, seam1});
                }
            }
            int changed = 0;
            auto run = [&](const std::vector<Range>& ranges) {
#pragma omp parallel for schedule(static) reduction(+ : changed) if (ranges.size() > 1)
                for (int index = 0; index < static_cast<int>(ranges.size()); ++index) {
                    const Range& range = ranges[static_cast<size_t>(index)];
                    const unsigned seed = seed_ + static_cast<unsigned>(level.scale) * 10007u +
                                          static_cast<unsigned>(pass) * 97u + static_cast<unsigned>(range.y0) * 13u +
                                          (forward ? 1u : 2u);
                    changed += scan_rows(level, range.y0, range.y1, forward, radius, seed);
                }
            };
            run(interiors);
            run(seams);
            if (pass + 1 >= passes && changed * 100 < holes * 3) {
                break;
            }
        }
        if (profile_) {
            std::cerr << "profile refine_mode=banded scale=" << level.scale << " passes=" << executed << "\n";
        }
    }
};

}  // namespace

PixMixInpainter::PixMixInpainter(PixMixConfig config) : config_(std::move(config)) {}

cv::Mat PixMixInpainter::run(const cv::Mat& bgr,
                             const cv::Mat& hole_mask,
                             const cv::Mat& initial_map,
                             const cv::Mat& reference,
                             int passes) {
    Solver solver(config_, config_.seed);
    return solver.run(bgr, hole_mask, initial_map, reference, passes, mapping_, levels_used_, scales_);
}

}  // namespace pixmix

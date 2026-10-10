#include "inpainter.hh"

#include "geom.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
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
    std::vector<int> row_holes;
    int scale = 1;
};

class Solver {
  public:
    Solver(const PixMixConfig& config, unsigned seed) : config_(config), seed_(seed) {}

    cv::Mat run(const cv::Mat& bgr,
                const cv::Mat& hole_mask,
                const cv::Mat& initial_map,
                const cv::Mat& reference,
                int passes,
                cv::Mat& mapping_out) {
        const cv::Mat frame = ensure_bgr(bgr);
        const cv::Mat mask = normalize_mask(hole_mask, frame.size());
        if (frame.empty() || cv::countNonZero(mask) == 0) {
            mapping_out.release();
            return frame.clone();
        }

        const bool video = !initial_map.empty() && initial_map.size() == frame.size() && initial_map.type() == CV_32SC2;
        const int pass_count = passes >= 0 ? passes : (video ? config_.video_passes : config_.still_passes);

        if (video) {
            // Video frames start from the forwarded map H^-1 o f o H and only
            // refine the full-resolution layer; the coherence term against the
            // reference model keeps the result close to the previous frames.
            Level level = make_level(frame, mask, reference, 1);
            for (int y = 0; y < mask.rows; ++y) {
                const uchar* mask_row = mask.ptr<uchar>(y);
                const cv::Vec2i* in_row = initial_map.ptr<cv::Vec2i>(y);
                cv::Vec2i* out_row = level.map.ptr<cv::Vec2i>(y);
                for (int x = 0; x < mask.cols; ++x) {
                    if (mask_row[x]) {
                        out_row[x] = in_row[x];
                    }
                }
            }
            paint_valid(level);
            recompute_costs(level);
            initialize_invalid(level, false);
            refine(level, std::max(1, pass_count), config_.video_search_radius, 0);
            mapping_out = level.map;
            return level.image;
        }

        std::vector<Level> levels = build_levels(frame, mask, reference);
        if (levels.empty()) {
            mapping_out.release();
            return frame.clone();
        }
        const int coarsest = static_cast<int>(levels.size()) - 1;
        Level& top = levels[static_cast<size_t>(coarsest)];
        initialize_invalid(top, true);
        recompute_costs(top);
        refine(top, pass_count, std::max(top.image.cols, top.image.rows), config_.max_still_passes);

        for (int index = coarsest - 1; index >= 0; --index) {
            Level& level = levels[static_cast<size_t>(index)];
            upsample_from(levels[static_cast<size_t>(index + 1)], level);
            initialize_invalid(level, false);
            recompute_costs(level);
            const int radius = std::max(8, std::max(level.image.cols, level.image.rows) / 16);
            refine(level, pass_count, radius, config_.max_still_passes);
        }

        mapping_out = levels.front().map;
        return levels.front().image;
    }

  private:
    PixMixConfig config_;
    unsigned seed_;

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
                cv::resize(levels.back().reference, smaller_reference, smaller_image.size(), 0, 0, cv::INTER_AREA);
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
        int limited = 1;
        int width = mask.cols;
        int height = mask.rows;
        while (width / 2 >= 8 && height / 2 >= 8 && limited < config_.max_pyramid_levels) {
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
        level.weight = cv::Mat(mask.size(), CV_32F, cv::Scalar(1.f));
        cv::Mat source;
        cv::compare(level.mask, 0, source, cv::CMP_EQ);
        const int diameter = 2 * config_.patch_radius + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(diameter, diameter));
        cv::Mat near_source;
        cv::dilate(source, near_source, kernel);
        for (int y = 0; y < near_source.rows; ++y) {
            const uchar* near_row = near_source.ptr<uchar>(y);
            float* weight_row = level.weight.ptr<float>(y);
            for (int x = 0; x < near_source.cols; ++x) {
                if (near_row[x]) {
                    weight_row[x] = config_.tau_a;
                }
            }
        }
        level.weight_sum = cv::Mat(mask.size(), CV_32F, cv::Scalar(0));
        level.row_holes.assign(static_cast<size_t>(mask.rows), 0);
        const int radius = config_.patch_radius;
        for (int y = 0; y < mask.rows; ++y) {
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            float* sum_row = level.weight_sum.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (mask_row[x]) {
                    ++level.row_holes[static_cast<size_t>(y)];
                }
                float sum = 0.f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= mask.rows) {
                        continue;
                    }
                    const float* weight_row = level.weight.ptr<float>(yy);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int xx = x + dx;
                        if (xx >= 0 && xx < mask.cols) {
                            sum += weight_row[xx];
                        }
                    }
                }
                sum_row[x] = std::max(sum, 1.f);
            }
        }
        return level;
    }

    static void downsample(const cv::Mat& image, const cv::Mat& mask, cv::Mat& image_out, cv::Mat& mask_out) {
        const cv::Size size(std::max(1, image.cols / 2), std::max(1, image.rows / 2));
        cv::resize(image, image_out, size, 0, 0, cv::INTER_AREA);
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

    bool source_ok(const Level& level, const cv::Point& point, const cv::Vec2i& offset) const {
        if (!offset_valid(offset)) {
            return false;
        }
        const int x = point.x + offset[0];
        const int y = point.y + offset[1];
        return in_bounds(x, y, level.mask.size()) && level.mask.ptr<uchar>(y)[x] == 0;
    }

    // Thesis (4.61)/(4.12): minimum over the 4-neighborhood of the clamped
    // squared offset difference, uniform weight 1/|Ns|, distances measured in
    // full-resolution pixels so every pyramid layer has the same cost scale.
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
    // source neighborhood with the reference model, center included.
    float cost_of(const Level& level, const cv::Point& point, const cv::Vec2i& offset, float best) const {
        const float spatial = spatial_cost(level, point, offset);
        const float budget = best - config_.alpha * spatial;
        if (budget <= 0.f) {
            return best;
        }
        const int radius = config_.patch_radius;
        const float weight_max = level.weight_sum.ptr<float>(point.y)[point.x];
        const bool coherence = !level.reference.empty();
        const int rows = level.image.rows;
        const int cols = level.image.cols;
        const float limit = budget * weight_max / std::max(1e-6f, 1.f - config_.alpha);
        float appear_sse = 0.f;
        float appear_weight = 0.f;
        float cohere_sse = 0.f;
        float cohere_weight = 0.f;
        int known_targets = 0;
        int appear_samples = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
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
            for (int dx = -radius; dx <= radius; ++dx) {
                const int x = point.x + dx;
                if (static_cast<unsigned>(x) >= static_cast<unsigned>(cols)) {
                    continue;
                }
                const bool center = dx == 0 && dy == 0;
                // Unassigned target pixels still hold the object's colors.
                const bool target_known = !center && (target_mask[x] == 0 || offset_valid(target_map[x]));
                known_targets += target_known ? 1 : 0;
                const int sx = x + offset[0];
                if (!source_row_ok || static_cast<unsigned>(sx) >= static_cast<unsigned>(cols) || source_mask[sx] != 0) {
                    continue;
                }
                const uchar* source = source_row + sx * 3;
                const float weight = weight_row[x];
                if (target_known) {
                    const uchar* target = target_row + x * 3;
                    const int db = int(target[0]) - int(source[0]);
                    const int dg = int(target[1]) - int(source[1]);
                    const int dr = int(target[2]) - int(source[2]);
                    appear_sse += weight * static_cast<float>(db * db + dg * dg + dr * dr);
                    appear_weight += weight;
                    ++appear_samples;
                }
                if (coherence) {
                    const uchar* guide = reference_row + x * 3;
                    const int cb = int(source[0]) - int(guide[0]);
                    const int cg = int(source[1]) - int(guide[1]);
                    const int cr = int(source[2]) - int(guide[2]);
                    cohere_sse += weight * static_cast<float>(cb * cb + cg * cg + cr * cr);
                    cohere_weight += weight;
                }
                if (appear_sse + cohere_sse >= limit) {
                    return best;
                }
            }
        }
        // A source window clipped by the image border or the hole must still
        // cover most of the known target neighborhood, otherwise a handful of
        // lucky samples would make the candidate look free.
        if (known_targets > 0 && appear_samples * 2 < known_targets) {
            return std::numeric_limits<float>::infinity();
        }
        if (appear_weight <= 0.f && cohere_weight <= 0.f) {
            return std::numeric_limits<float>::infinity();
        }
        float appearance = appear_weight > 0.f ? appear_sse / appear_weight : 0.f;
        if (cohere_weight > 0.f) {
            appearance += cohere_sse / cohere_weight;
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

    void upsample_from(const Level& coarse, Level& fine) const {
        const int mul_x = std::max(1, fine.original.cols / coarse.original.cols);
        const int mul_y = std::max(1, fine.original.rows / coarse.original.rows);
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
                        const cv::Vec2i candidate(parent[0] * mul_x, parent[1] * mul_y);
                        if (source_ok(fine, cv::Point(x, y), candidate)) {
                            offset = candidate;
                            found = true;
                        }
                    }
                }
                fine.map.ptr<cv::Vec2i>(y)[x] = offset;
            }
        }
        paint_valid(fine);
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
        std::vector<cv::Point> holes;
        for (int y = 0; y < level.mask.rows; ++y) {
            if (level.row_holes[static_cast<size_t>(y)] == 0) {
                continue;
            }
            const uchar* mask_row = level.mask.ptr<uchar>(y);
            for (int x = 0; x < level.mask.cols; ++x) {
                if (mask_row[x]) {
                    holes.emplace_back(x, y);
                }
            }
        }
#pragma omp parallel for schedule(static) if (holes.size() > 256)
        for (int index = 0; index < static_cast<int>(holes.size()); ++index) {
            const cv::Point point = holes[static_cast<size_t>(index)];
            const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(point.y)[point.x];
            float cost = std::numeric_limits<float>::infinity();
            if (source_ok(level, point, offset)) {
                cost = cost_of(level, point, offset, std::numeric_limits<float>::infinity());
            }
            level.cost.ptr<float>(point.y)[point.x] = cost;
        }
    }

    bool improve(Level& level, const cv::Point& point, bool forward, int radius, std::mt19937& rng) const {
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
            const cv::Vec2i candidate(base[0] + jitter(rng), base[1] + jitter(rng));
            if (consider(level, point, candidate)) {
                base = candidate;
            }
        }
        return level.map.ptr<cv::Vec2i>(point.y)[point.x] != before;
    }

    void propagate_from(Level& level, const cv::Point& point, const cv::Point& neighbor) const {
        if (!in_bounds(neighbor.x, neighbor.y, level.mask.size())) {
            return;
        }
        if (level.mask.ptr<uchar>(neighbor.y)[neighbor.x] == 0) {
            consider(level, point, cv::Vec2i(neighbor.x - point.x, neighbor.y - point.y));
            return;
        }
        const cv::Vec2i offset = level.map.ptr<cv::Vec2i>(neighbor.y)[neighbor.x];
        if (offset_valid(offset)) {
            consider(level, point, offset);
        }
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
                if (mask_row[x] && improve(level, cv::Point(x, y), forward, radius, rng)) {
                    ++changed;
                }
            }
        }
        return changed;
    }

    // Runs at least `passes` sweeps, alternating forward and backward scanline
    // order. With max_passes > passes, keeps iterating until fewer than 1% of
    // the hole pixels change (the thesis reports <3% after ten iterations).
    // Row bands are processed in parallel with a guard gap of 2r+1 rows; the
    // band origin moves every pass so no seam stays fixed.
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
        for (int pass = 0; pass < limit; ++pass) {
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
            if (pass + 1 >= passes && changed * 100 < holes) {
                break;
            }
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
    return solver.run(bgr, hole_mask, initial_map, reference, passes, mapping_);
}

}  // namespace pixmix

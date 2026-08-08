#include "keyword_detection/dtw_matcher_core.h"

#include <algorithm>
#include <cmath>
#include <limits>

static constexpr float INF = std::numeric_limits<float>::infinity();


DTWMatcherCore::DTWMatcherCore() {}
DTWMatcherCore::~DTWMatcherCore() {}

void DTWMatcherCore::set_distance_metric(int metric) { _distance_metric = metric; }
int DTWMatcherCore::get_distance_metric() const { return _distance_metric; }
void DTWMatcherCore::set_band_width(float width) { _band_width = width; }
float DTWMatcherCore::get_band_width() const { return _band_width; }
void DTWMatcherCore::set_classify_method(int method) { _classify_method = method; }
int DTWMatcherCore::get_classify_method() const { return _classify_method; }

// ============================================================================
// Core DTW
// ============================================================================
float DTWMatcherCore::compute(const FeatureSequence& seq_a,
    const FeatureSequence& seq_b) const {
    const int N = seq_a.size();
    const int M = seq_b.size();

    if (N == 0 || M == 0) {
        return INF;
    }

    // Cost matrix (N x M), row-major
    std::vector<float> cost(N * M, INF);

    auto idx = [&](int i, int j) -> float& { return cost[i * M + j]; };

    const int max_len = std::max(N, M);
    const int band = static_cast<int>(std::ceil(_band_width * max_len));

    for (int i = 0; i < N; ++i) {
        // Expected position on the diagonal in sequence B
        const int diagonal = (N > 1)
            ? static_cast<int>(std::round(
                static_cast<float>(i) * (M - 1) / (N - 1)))
            : 0;

        const int j_begin = std::max(0, diagonal - band);
        const int j_end = std::min(M - 1, diagonal + band);

        for (int j = j_begin; j <= j_end; ++j) {
            const std::vector<float>& fa = seq_a[i];
            const std::vector<float>& fb = seq_b[j];
            float d = _frame_distance(fa, fb);

            float prev = INF;
            if (i > 0 && j > 0) {
                prev = std::min({ cost[(i - 1) * M + j],
                                  cost[i * M + j - 1],
                                  cost[(i - 1) * M + j - 1] });
            }
            else if (i > 0) {
                prev = cost[(i - 1) * M + j];
            }
            else if (j > 0) {
                prev = cost[i * M + j - 1];
            }
            else {
                prev = 0.0f; // (0,0) origin
            }

            idx(i, j) = (prev == INF) ? INF : d + prev;
        }
    }

    float raw = cost[(N - 1) * M + (M - 1)];

    if (raw == INF) {
        return INF;
    }

    // Normalize by path length so distance is comparable across sequence lengths
    return raw / static_cast<float>(N + M);
}

// ---- Frame-level distance --------------------------------------------------
float DTWMatcherCore::_frame_distance(const std::vector<float>& a,
    const std::vector<float>& b) const {
    switch (_distance_metric) {
    case COSINE:
        return _cosine(a, b);
    case EUCLIDEAN: // fall-through
    default:
        return _euclidean(a, b);
    }
}

float DTWMatcherCore::_euclidean(const std::vector<float>& a,
    const std::vector<float>& b) const
{
    if (a.size() != b.size()) {
        return INF;
    }

    const int len = std::min(a.size(), b.size());
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float diff = float(a[i]) - float(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float DTWMatcherCore::_cosine(const std::vector<float>& a,
    const std::vector<float>& b) const {
    const int len = std::min(a.size(), b.size());
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < len; ++i) {
        dot += float(a[i]) * float(b[i]);
        norm_a += float(a[i]) * float(a[i]);
        norm_b += float(b[i]) * float(b[i]);
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-8f)
        return 1.0f; // orthogonal by convention
    return 1.0f - (dot / denom); // distance ∈ [0, 2]
}


void DTWMatcherCore::add_template(const std::string& label,
    const FeatureSequence& features) {
    _templates[label].push_back(features);
}

void DTWMatcherCore::clear_templates() {
    _templates.clear();
}

std::string DTWMatcherCore::classify(const FeatureSequence& features) const {
    ClassifyResult result = classify_with_best_score(features);
    return result.label;
}

DTWMatcherCore::ClassifyResult DTWMatcherCore::classify_with_best_score(const FeatureSequence& features) const {
    ClassifyResult result;
    if (_templates.empty()) {
        return result;
    }

    float best_dist = INF;
    std::string best_label = "";

    for (const auto& [label, clipList] : _templates) {
        float distance = 0.0f;
        switch (_classify_method) {
        case AVERAGE_DISTANCE:
            distance = _computeAverageDistance(label, features);
            break;
        case BEST_MATCH:
            distance = _computeBestDistance(label, features);
            break;
        default:
            return result; // unknown classify method
        }
        if (distance < best_dist) {
            best_dist = distance;
            best_label = label;
        }
    }
    result.label = best_label;
    result.distance = best_dist;
    return result;
}

std::vector<DTWMatcherCore::ClassifyResult> DTWMatcherCore::classify_with_every_score(const FeatureSequence& features) const {
    std::vector<ClassifyResult> result;
    if (_templates.empty()) {
        return result;
    }

    for (const auto& [label, clipList] : _templates) {
        float distance = 0.0f;
        if (_classify_method == AVERAGE_DISTANCE) {
            distance = _computeAverageDistance(label, features);
        }
        else if (_classify_method == BEST_MATCH) {
            distance = _computeBestDistance(label, features);
        }
        else {
            return result; // unknown classify method
        }
        ClassifyResult res;
        res.label = label;
        res.distance = distance;
        result.push_back(res);
    }
    return result;
}

float DTWMatcherCore::_computeAverageDistance(const std::string& label, const std::vector<std::vector<float>>& features) const {
    auto it = _templates.find(label);
    if (it == _templates.end()) {
        return INF;
    }

    const auto& clipList = it->second;
    if (clipList.empty()) {
        return INF;
    }

    float totalDistance = 0.0f;
    for (const auto& clip : clipList) {
        totalDistance += compute(features, clip);
    }
    return totalDistance / clipList.size();
}

float DTWMatcherCore::_computeBestDistance(const std::string& label, const std::vector<std::vector<float>>& features) const {
    auto it = _templates.find(label);
    if (it == _templates.end()) {
        return INF;
    }

    const auto& clipList = it->second;
    if (clipList.empty()) {
        return INF;
    }

    float bestDistance = INF;
    for (const auto& clip : clipList) {
        float dist = compute(features, clip);
        if (dist < bestDistance) {
            bestDistance = dist;
        }
    }
    return bestDistance;
}

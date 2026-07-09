#include "dtw_matcher.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace godot;

static constexpr float INF = std::numeric_limits<float>::infinity();

// ============================================================================
DTWMatcher::DTWMatcher() {}
DTWMatcher::~DTWMatcher() {}

// ---- Godot bindings --------------------------------------------------------
void DTWMatcher::_bind_methods() {
    BIND_ENUM_CONSTANT(EUCLIDEAN);
    BIND_ENUM_CONSTANT(COSINE);

    BIND_ENUM_CONSTANT(AVERAGE_DISTANCE);
    BIND_ENUM_CONSTANT(BEST_MATCH);

    ClassDB::bind_method(D_METHOD("set_distance_metric", "metric"), &DTWMatcher::set_distance_metric);
    ClassDB::bind_method(D_METHOD("get_distance_metric"), &DTWMatcher::get_distance_metric);
    ClassDB::bind_method(D_METHOD("set_band_width", "width"), &DTWMatcher::set_band_width);
    ClassDB::bind_method(D_METHOD("get_band_width"), &DTWMatcher::get_band_width);
    ClassDB::bind_method(D_METHOD("set_classify_method", "method"), &DTWMatcher::set_classify_method);
    ClassDB::bind_method(D_METHOD("get_classify_method"), &DTWMatcher::get_classify_method);

    ClassDB::bind_method(D_METHOD("compute", "seq_a", "seq_b"), &DTWMatcher::compute);

    ClassDB::bind_method(D_METHOD("add_template", "label", "mfcc"), &DTWMatcher::add_template);
    ClassDB::bind_method(D_METHOD("clear_templates"), &DTWMatcher::clear_templates);
    ClassDB::bind_method(D_METHOD("classify", "mfcc"), &DTWMatcher::classify);
    ClassDB::bind_method(D_METHOD("classify_with_best_score", "mfcc"), &DTWMatcher::classify_with_best_score);
    ClassDB::bind_method(D_METHOD("classify_with_every_score", "mfcc"), &DTWMatcher::classify_with_every_score);


    ADD_PROPERTY(PropertyInfo(Variant::INT, "distance_metric", PROPERTY_HINT_ENUM, "Euclidean,Cosine"),
        "set_distance_metric", "get_distance_metric");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "band_width"),
        "set_band_width", "get_band_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "classify_method", PROPERTY_HINT_ENUM, "Average Distance,Best Match"),
        "set_classify_method", "get_classify_method");
}

// ---- Setters / Getters -----------------------------------------------------
void DTWMatcher::set_distance_metric(int p_metric) { _distance_metric = p_metric; }
int DTWMatcher::get_distance_metric() const { return _distance_metric; }
void DTWMatcher::set_band_width(int p_width) { _band_width = p_width; }
int DTWMatcher::get_band_width() const { return _band_width; }
void DTWMatcher::set_classify_method(int p_method) { _classify_method = p_method; }
int DTWMatcher::get_classify_method() const { return _classify_method; }

// ============================================================================
// Core DTW
// ============================================================================
float DTWMatcher::compute(const TypedArray<PackedFloat32Array> &p_seq_a,
    const TypedArray<PackedFloat32Array>& p_seq_b) const {
    const int N = p_seq_a.size();
    const int M = p_seq_b.size();

    if (N == 0 || M == 0) {
        UtilityFunctions::push_warning("DTWMatcher: one or both sequences are empty.");
        return INF;
    }

    // Cost matrix (N x M), row-major
    std::vector<float> cost(N * M, INF);

    auto idx = [&](int i, int j) -> float& { return cost[i * M + j]; };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            // Sakoe-Chiba band
            if (_band_width > 0 && std::abs(i - j) > _band_width) {
                continue;
            }

            PackedFloat32Array fa = p_seq_a[i];
            PackedFloat32Array fb = p_seq_b[j];
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

            idx(i, j) = d + (prev == INF ? 0.0f : prev);
        }
    }

    float raw = cost[(N - 1) * M + (M - 1)];
    // Normalize by path length so distance is comparable across sequence lengths
    return raw / float(N + M);
}

// ---- Frame-level distance --------------------------------------------------
float DTWMatcher::_frame_distance(const PackedFloat32Array &a,
    const PackedFloat32Array& b) const {
    switch (_distance_metric) {
    case COSINE:
        return _cosine(a, b);
    case EUCLIDEAN: // fall-through
    default:
        return _euclidean(a, b);
    }
}

float DTWMatcher::_euclidean(const PackedFloat32Array &a,
    const PackedFloat32Array& b) const
{
    if (a.size() != b.size()) {
        UtilityFunctions::push_warning("DTWMatcher: frame size mismatch: " + String::num_int64(a.size()) + " vs " + String::num_int64(b.size()));
    }

    const int len = std::min(a.size(), b.size());
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float diff = float(a[i]) - float(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float DTWMatcher::_cosine(const PackedFloat32Array &a,
    const PackedFloat32Array& b) const {
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

// ============================================================================
// Template matching
// ============================================================================
void DTWMatcher::add_template(const String &p_label,
    const TypedArray<PackedFloat32Array>& p_mfcc) {
    _templates[p_label].push_back(p_mfcc);
}

void DTWMatcher::clear_templates() {
    _templates.clear();
}

String DTWMatcher::classify(const TypedArray<PackedFloat32Array> &p_mfcc) const {
    Dictionary d = classify_with_best_score(p_mfcc);
    return d.get("label", String());
}

Dictionary DTWMatcher::classify_with_best_score(const TypedArray<PackedFloat32Array> &p_mfcc) const {
    Dictionary result;
    if (_templates.empty()) {
        UtilityFunctions::push_warning("DTWMatcher: no templates registered.");
        return result;
    }

    float best_dist = INF;
    String best_label = "";

    for (const auto& [label, clipList] : _templates) {
        float distance = 0.0f;
        if (_classify_method == AVERAGE_DISTANCE) {
            distance = _computeAverageDistance(label, p_mfcc);
        }
        else if (_classify_method == BEST_MATCH) {
            distance = _computeBestDistance(label, p_mfcc);
        }
        else {
            UtilityFunctions::push_warning("DTWMatcher: unknown classify method: " + String::num_int64(_classify_method));
            continue;
        }
        if (distance < best_dist) {
            best_dist = distance;
            best_label = label;
        }
#ifdef DEBUG_ENABLED
        UtilityFunctions::print("DTWMatcher: template=", label, " dist=", distance, " method=", _classify_method);
#endif
    }
    result["label"] = best_label;
    result["distance"] = best_dist;
    return result;
}

Dictionary DTWMatcher::classify_with_every_score(const TypedArray<PackedFloat32Array> &p_mfcc) const {
    Dictionary result;
    if (_templates.empty()) {
        UtilityFunctions::push_warning("DTWMatcher: no templates registered.");
        return result;
    }

    for (const auto& [label, clipList] : _templates) {
        float distance = 0.0f;
        if (_classify_method == AVERAGE_DISTANCE) {
            distance = _computeAverageDistance(label, p_mfcc);
        }
        else if (_classify_method == BEST_MATCH) {
            distance = _computeBestDistance(label, p_mfcc);
        }
        else {
            UtilityFunctions::push_warning("DTWMatcher: unknown classify method: " + String::num_int64(_classify_method));
            continue;
        }
        result[label] = distance;
#ifdef DEBUG_ENABLED
        UtilityFunctions::print("DTWMatcher: template=", label, " dist=", distance, " method=", _classify_method);
#endif
    }
    return result;
}

float DTWMatcher::_computeAverageDistance(const String& p_label, const TypedArray<PackedFloat32Array>& p_mfcc) const {
    auto it = _templates.find(p_label);
    if (it == _templates.end()) {
        UtilityFunctions::push_warning("DTWMatcher: template not found: " + p_label);
        return INF;
    }

    const auto& clipList = it->second;
    if (clipList.empty()) {
        UtilityFunctions::push_warning("DTWMatcher: template has no clips: " + p_label);
        return INF;
    }

    float totalDistance = 0.0f;
    for (const auto& clip : clipList) {
        totalDistance += compute(p_mfcc, clip);
    }
    return totalDistance / clipList.size();
}

float DTWMatcher::_computeBestDistance(const String& p_label, const TypedArray<PackedFloat32Array>& p_mfcc) const {
    auto it = _templates.find(p_label);
    if (it == _templates.end()) {
        UtilityFunctions::push_warning("DTWMatcher: template not found: " + p_label);
        return INF;
    }

    const auto& clipList = it->second;
    if (clipList.empty()) {
        UtilityFunctions::push_warning("DTWMatcher: template has no clips: " + p_label);
        return INF;
    }

    float bestDistance = INF;
    for (const auto& clip : clipList) {
        float dist = compute(p_mfcc, clip);
        if (dist < bestDistance) {
            bestDistance = dist;
        }
    }
    return bestDistance;
}

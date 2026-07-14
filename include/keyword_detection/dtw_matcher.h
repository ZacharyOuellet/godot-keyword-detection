#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include <vector>
#include <map>

using namespace godot;

// ---------------------------------------------------------------------------
// DTWMatcher
//
// Computes the Dynamic Time Warping distance between two MFCC sequences.
// Both sequences are Array of PackedFloat32Array (i.e. the output of
// MFCCProcessor::compute).
//
// Typical usage (GDScript):
//   var dtw = DTWMatcher.new()
//   dtw.distance_metric = DTWMatcher.EUCLIDEAN   # or COSINE
//   var dist: float = dtw.compute(mfcc_a, mfcc_b)
//
// For template matching:
//   dtw.add_template("hello", mfcc_hello)
//   var label: String = dtw.classify(mfcc_unknown)  # returns closest label
// ---------------------------------------------------------------------------
class DTWMatcher : public RefCounted {
    GDCLASS(DTWMatcher, RefCounted)

public:
    enum DistanceMetric {
        EUCLIDEAN = 0,
        COSINE    = 1,
    };

    enum ClassifyMethod {
        AVERAGE_DISTANCE = 0,
        BEST_MATCH = 1,
    };

    DTWMatcher();
    ~DTWMatcher();

    // --- Distance metric ---
    void set_distance_metric(int p_metric);
    int  get_distance_metric() const;

    // --- Sakoe-Chiba band width (0 = no constraint) ---
    void set_band_width(int p_width);
    int  get_band_width() const;

    void set_classify_method(int p_method);
    int  get_classify_method() const;

    // --- Core DTW ---
    // Input : two Array<PackedFloat32Array> (MFCC sequences)
    // Output: scalar DTW distance (lower = more similar)
    float compute(const TypedArray<PackedFloat32Array> &p_seq_a,
                  const TypedArray<PackedFloat32Array> &p_seq_b) const;


    // --- Template matching helpers ---
    void   add_template(const String &p_label, const TypedArray<PackedFloat32Array> &p_mfcc);
    void   clear_templates();
    String classify(const TypedArray<PackedFloat32Array> &p_mfcc) const;



    // Returns a Dictionary { label: String, distance: float } for the best match
    Dictionary classify_with_best_score(const TypedArray<PackedFloat32Array> &p_mfcc) const;

    // Returns a Dictionary { label: String, distance: float } for every template
    Dictionary classify_with_every_score(const TypedArray<PackedFloat32Array> &p_mfcc) const;

protected:
    static void _bind_methods();

private:
    float _frame_distance(const PackedFloat32Array &a, const PackedFloat32Array &b) const;
    float _euclidean(const PackedFloat32Array &a, const PackedFloat32Array &b) const;
    float _cosine(const PackedFloat32Array &a, const PackedFloat32Array &b) const;

    // Computes the distance between the input MFCC and all the clips with the given label
    float _computeAverageDistance(const String& p_label, const TypedArray<PackedFloat32Array>& p_mfcc) const;
    float _computeBestDistance(const String& p_label, const TypedArray<PackedFloat32Array>& p_mfcc) const;

    int _distance_metric = EUCLIDEAN;
    int _classify_method = AVERAGE_DISTANCE;
    int _band_width      = 0; // 0 = full matrix (no Sakoe-Chiba constraint)

    std::map<String, std::vector<TypedArray<PackedFloat32Array>>> _templates;
};

VARIANT_ENUM_CAST(DTWMatcher::DistanceMetric);
VARIANT_ENUM_CAST(DTWMatcher::ClassifyMethod);

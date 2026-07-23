#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include "keyword_detection/dtw_matcher_core.h"

using namespace godot;

class DTWMatcher : public RefCounted {
    GDCLASS(DTWMatcher, RefCounted)
public:
    DTWMatcher();
    ~DTWMatcher();

    enum DistanceMetric {
        EUCLIDEAN = DTWMatcherCore::EUCLIDEAN,
        COSINE = DTWMatcherCore::COSINE
    };

    enum ClassifyMethod {
        AVERAGE_DISTANCE = DTWMatcherCore::AVERAGE_DISTANCE,
        BEST_MATCH = DTWMatcherCore::BEST_MATCH
    };

    // --- Distance metric ---
    void set_distance_metric(int p_metric);
    int get_distance_metric() const;

    // --- Sakoe-Chiba band width (1 = no constraint) ---
    void set_band_width(float);
    float get_band_width() const;

    void set_classify_method(int p_method);
    int get_classify_method() const;

    float compute(const TypedArray<PackedFloat32Array>& p_seq_a,
        const TypedArray<PackedFloat32Array>& p_seq_b) const;

    void add_template(const String& p_label, const TypedArray<PackedFloat32Array>& p_mfcc);
    void clear_templates();

    String classify(const TypedArray<PackedFloat32Array>& p_mfcc) const;

    // Returns a Dictionary { label: String, distance: float } for the best match
    Dictionary classify_with_best_score(const TypedArray<PackedFloat32Array>& p_mfcc) const;

    // Returns a Dictionary { label: String, distance: float } for every template
    Dictionary classify_with_every_score(const TypedArray<PackedFloat32Array>& p_mfcc) const;

protected:
    static void _bind_methods();
private:
    DTWMatcherCore core;
};

VARIANT_ENUM_CAST(DTWMatcher::DistanceMetric);
VARIANT_ENUM_CAST(DTWMatcher::ClassifyMethod);

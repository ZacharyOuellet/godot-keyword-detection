#include "dtw_matcher.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static FeatureSequence to_feature_sequence(
    const TypedArray<PackedFloat32Array>& array)
{
    FeatureSequence result;
    result.reserve(array.size());
    for (int i = 0; i < array.size(); i++)
    {
        PackedFloat32Array frame = array[i];

        std::vector<float> values;
        values.resize(frame.size());

        for (int j = 0; j < frame.size(); j++)
        {
            values[j] = frame[j];
        }

        result.push_back(values);
    }
    return result;
}


static Dictionary result_to_dictionary(
    const DTWMatcherCore::ClassifyResult& result)
{
    Dictionary dict;
    dict["label"] = String(result.label.c_str());
    dict["distance"] = result.distance;
    return dict;
}

DTWMatcher::DTWMatcher() {}
DTWMatcher::~DTWMatcher() {}

void DTWMatcher::_bind_methods()
{
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
    ClassDB::bind_method(D_METHOD("add_template", "label", "features"), &DTWMatcher::add_template);
    ClassDB::bind_method(D_METHOD("classify", "features"), &DTWMatcher::classify);
    ClassDB::bind_method(D_METHOD("classify_with_best_score", "features"), &DTWMatcher::classify_with_best_score);
    ClassDB::bind_method(D_METHOD("classify_with_every_score", "features"), &DTWMatcher::classify_with_every_score);
    ClassDB::bind_method(D_METHOD("clear_templates"), &DTWMatcher::clear_templates);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "distance_metric"), "set_distance_metric", "get_distance_metric");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "band_width"), "set_band_width", "get_band_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "classify_method"), "set_classify_method", "get_classify_method");
}

void DTWMatcher::set_distance_metric(int metric) { core.set_distance_metric(metric); }
int DTWMatcher::get_distance_metric() const { return core.get_distance_metric(); }
void DTWMatcher::set_band_width(float width) { core.set_band_width(width); }
float DTWMatcher::get_band_width() const { return core.get_band_width(); }
void DTWMatcher::set_classify_method(int method) { core.set_classify_method(method); }
int DTWMatcher::get_classify_method() const { return core.get_classify_method(); }

float DTWMatcher::compute(const TypedArray<PackedFloat32Array>& seq_a, const TypedArray<PackedFloat32Array>& seq_b) const
{
    FeatureSequence a = to_feature_sequence(seq_a);
    FeatureSequence b = to_feature_sequence(seq_b);

    return core.compute(a, b);
}

void DTWMatcher::add_template(
    const String& label,
    const TypedArray<PackedFloat32Array>& mfcc)
{
    FeatureSequence features = to_feature_sequence(mfcc);
    core.add_template(
        std::string(label.utf8().get_data()),
        features
    );
}

void DTWMatcher::clear_templates()
{
    core.clear_templates();
}

String DTWMatcher::classify(const TypedArray<PackedFloat32Array>& mfcc) const
{
    FeatureSequence features = to_feature_sequence(mfcc);
    std::string result = core.classify(features);
    return String(result.c_str());
}

Dictionary DTWMatcher::classify_with_best_score(const TypedArray<PackedFloat32Array>& mfcc) const
{
    FeatureSequence features = to_feature_sequence(mfcc);
    auto result = core.classify_with_best_score(features);
    return result_to_dictionary(result);
}

Dictionary DTWMatcher::classify_with_every_score(const TypedArray<PackedFloat32Array>& mfcc) const
{
    FeatureSequence features = to_feature_sequence(mfcc);
    auto results = core.classify_with_every_score(features);
    Dictionary dict;
    for (const auto& result : results)
    {
        dict[String(result.label.c_str())] = result.distance;
    }
    return dict;
}

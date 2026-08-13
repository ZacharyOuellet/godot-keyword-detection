#include "dtw_matcher.h"
#include "types_transformation.h"
#include <godot_cpp/core/class_db.hpp>

using namespace godot;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------



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

    ADD_PROPERTY(PropertyInfo(Variant::INT, "distance_metric", PROPERTY_HINT_ENUM), "set_distance_metric", "get_distance_metric");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "band_width"), "set_band_width", "get_band_width");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "classify_method", PROPERTY_HINT_ENUM), "set_classify_method", "get_classify_method");
}

void DTWMatcher::set_distance_metric(DistanceMetric metric) { core.set_distance_metric(metric); }
DTWMatcher::DistanceMetric DTWMatcher::get_distance_metric() const { return (DistanceMetric)core.get_distance_metric(); }
void DTWMatcher::set_band_width(float width) { core.set_band_width(width); }
float DTWMatcher::get_band_width() const { return core.get_band_width(); }
void DTWMatcher::set_classify_method(ClassifyMethod method) { core.set_classify_method(method); }
DTWMatcher::ClassifyMethod DTWMatcher::get_classify_method() const { return (ClassifyMethod)core.get_classify_method(); }

float DTWMatcher::compute(const TypedArray<PackedFloat32Array>& seq_a, const TypedArray<PackedFloat32Array>& seq_b) const
{
    FeatureSequence a = arrayPackedArrayToVectorVector(seq_a);
    FeatureSequence b = arrayPackedArrayToVectorVector(seq_b);

    return core.compute(a, b);
}

void DTWMatcher::add_template(
    const String& label,
    const TypedArray<PackedFloat32Array>& features)
{
    FeatureSequence cpp_features = arrayPackedArrayToVectorVector(features);
    core.add_template(
        std::string(label.utf8().get_data()),
        cpp_features
    );
}

void DTWMatcher::clear_templates()
{
    core.clear_templates();
}

String DTWMatcher::classify(const TypedArray<PackedFloat32Array>& features) const
{
    FeatureSequence cpp_features = arrayPackedArrayToVectorVector(features);
    std::string result = core.classify(cpp_features);
    return String(result.c_str());
}

Dictionary DTWMatcher::classify_with_best_score(const TypedArray<PackedFloat32Array>& features) const
{
    FeatureSequence cpp_features = arrayPackedArrayToVectorVector(features);
    auto result = core.classify_with_best_score(cpp_features);
    return result_to_dictionary(result);
}

Dictionary DTWMatcher::classify_with_every_score(const TypedArray<PackedFloat32Array>& features) const
{
    FeatureSequence cpp_features = arrayPackedArrayToVectorVector(features);
    auto results = core.classify_with_every_score(cpp_features);
    Dictionary dict;
    for (const auto& result : results)
    {
        dict[String(result.label.c_str())] = result.distance;
    }
    return dict;
}

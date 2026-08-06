#include "pncc_processor.h"

#include <godot_cpp/core/class_db.hpp>
#include "types_transformation.h"

using namespace godot;

PNCCProcessor::PNCCProcessor() {}
PNCCProcessor::~PNCCProcessor() {}

void PNCCProcessor::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_sample_rate", "rate"), &PNCCProcessor::set_sample_rate);
    ClassDB::bind_method(D_METHOD("get_sample_rate"), &PNCCProcessor::get_sample_rate);
    ClassDB::bind_method(D_METHOD("set_num_coeffs", "n"), &PNCCProcessor::set_num_coeffs);
    ClassDB::bind_method(D_METHOD("get_num_coeffs"), &PNCCProcessor::get_num_coeffs);
    ClassDB::bind_method(D_METHOD("set_frame_length", "length"), &PNCCProcessor::set_frame_length);
    ClassDB::bind_method(D_METHOD("get_frame_length"), &PNCCProcessor::get_frame_length);
    ClassDB::bind_method(D_METHOD("set_hop_length", "hop"), &PNCCProcessor::set_hop_length);
    ClassDB::bind_method(D_METHOD("get_hop_length"), &PNCCProcessor::get_hop_length);
    ClassDB::bind_method(D_METHOD("set_num_gamma_bands", "bands"), &PNCCProcessor::set_num_gamma_bands);
    ClassDB::bind_method(D_METHOD("get_num_gamma_bands"), &PNCCProcessor::get_num_gamma_bands);
    ClassDB::bind_method(D_METHOD("set_power_law_exponent", "exponent"), &PNCCProcessor::set_power_law_exponent);
    ClassDB::bind_method(D_METHOD("get_power_law_exponent"), &PNCCProcessor::get_power_law_exponent);
    ClassDB::bind_method(D_METHOD("set_medium_time_frames", "frames"), &PNCCProcessor::set_medium_time_frames);
    ClassDB::bind_method(D_METHOD("get_medium_time_frames"), &PNCCProcessor::get_medium_time_frames);
    ClassDB::bind_method(D_METHOD("set_lambda_a", "lambda_a"), &PNCCProcessor::set_lambda_a);
    ClassDB::bind_method(D_METHOD("get_lambda_a"), &PNCCProcessor::get_lambda_a);
    ClassDB::bind_method(D_METHOD("set_lambda_b", "lambda_b"), &PNCCProcessor::set_lambda_b);
    ClassDB::bind_method(D_METHOD("get_lambda_b"), &PNCCProcessor::get_lambda_b);
    ClassDB::bind_method(D_METHOD("set_lambda_t", "lambda_t"), &PNCCProcessor::set_lambda_t);
    ClassDB::bind_method(D_METHOD("get_lambda_t"), &PNCCProcessor::get_lambda_t);
    ClassDB::bind_method(D_METHOD("compute", "samples"), &PNCCProcessor::compute);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_rate"), "set_sample_rate", "get_sample_rate");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_coeffs"), "set_num_coeffs", "get_num_coeffs");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "frame_length"), "set_frame_length", "get_frame_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hop_length"), "set_hop_length", "get_hop_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_gamma_bands"), "set_num_gamma_bands", "get_num_gamma_bands");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "power_law_exponent"), "set_power_law_exponent", "get_power_law_exponent");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "medium_time_frames"), "set_medium_time_frames", "get_medium_time_frames");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lambda_a"), "set_lambda_a", "get_lambda_a");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lambda_b"), "set_lambda_b", "get_lambda_b");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lambda_t"), "set_lambda_t", "get_lambda_t");
}

void PNCCProcessor::set_sample_rate(int rate) { core.set_sample_rate(rate); }
int PNCCProcessor::get_sample_rate() const { return core.get_sample_rate(); }
void PNCCProcessor::set_num_coeffs(int n) { core.set_num_coeffs(n); }
int PNCCProcessor::get_num_coeffs() const { return core.get_num_coeffs(); }
void PNCCProcessor::set_frame_length(int length) { core.set_frame_length(length); }
int PNCCProcessor::get_frame_length() const { return core.get_frame_length(); }
void PNCCProcessor::set_hop_length(int hop) { core.set_hop_length(hop); }
int PNCCProcessor::get_hop_length() const { return core.get_hop_length(); }
void PNCCProcessor::set_num_gamma_bands(int bands) { core.set_num_gamma_bands(bands); }
int PNCCProcessor::get_num_gamma_bands() const { return core.get_num_gamma_bands(); }
void PNCCProcessor::set_power_law_exponent(float exponent) { core.set_power_law_exponent(exponent); }
float PNCCProcessor::get_power_law_exponent() const { return core.get_power_law_exponent(); }
void PNCCProcessor::set_medium_time_frames(int frames) { core.set_medium_time_frames(frames); }
int PNCCProcessor::get_medium_time_frames() const { return core.get_medium_time_frames(); }
void PNCCProcessor::set_lambda_a(float lambda_a) { core.set_lambda_a(lambda_a); }
float PNCCProcessor::get_lambda_a() const { return core.get_lambda_a(); }
void PNCCProcessor::set_lambda_b(float lambda_b) { core.set_lambda_b(lambda_b); }
float PNCCProcessor::get_lambda_b() const { return core.get_lambda_b(); }
void PNCCProcessor::set_lambda_t(float lambda_t) { core.set_lambda_t(lambda_t); }
float PNCCProcessor::get_lambda_t() const { return core.get_lambda_t(); }


TypedArray<PackedFloat32Array> PNCCProcessor::compute(
    const PackedFloat32Array& samples)
{
    std::vector<float> pcm = packedArrayToVector(samples);

    PNCC features = core.compute(pcm);

    return vectorVectorToArrayPackedArray(features);
}

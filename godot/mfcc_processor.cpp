#include "mfcc_processor.h"

using namespace godot;

MFCCProcessor::MFCCProcessor() {}
MFCCProcessor::~MFCCProcessor() {}

// ---- Godot bindings --------------------------------------------------------
void MFCCProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_sample_rate", "rate"), &MFCCProcessor::set_sample_rate);
    ClassDB::bind_method(D_METHOD("get_sample_rate"), &MFCCProcessor::get_sample_rate);
    ClassDB::bind_method(D_METHOD("set_num_coeffs", "n"), &MFCCProcessor::set_num_coeffs);
    ClassDB::bind_method(D_METHOD("get_num_coeffs"), &MFCCProcessor::get_num_coeffs);
    ClassDB::bind_method(D_METHOD("set_frame_length", "len"), &MFCCProcessor::set_frame_length);
    ClassDB::bind_method(D_METHOD("get_frame_length"), &MFCCProcessor::get_frame_length);
    ClassDB::bind_method(D_METHOD("set_hop_length", "hop"), &MFCCProcessor::set_hop_length);
    ClassDB::bind_method(D_METHOD("get_hop_length"), &MFCCProcessor::get_hop_length);
    ClassDB::bind_method(D_METHOD("set_num_mel_bands", "bands"), &MFCCProcessor::set_num_mel_bands);
    ClassDB::bind_method(D_METHOD("get_num_mel_bands"), &MFCCProcessor::get_num_mel_bands);
    ClassDB::bind_method(D_METHOD("compute", "samples"), &MFCCProcessor::compute);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_rate"), "set_sample_rate", "get_sample_rate");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_coeffs"), "set_num_coeffs", "get_num_coeffs");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "frame_length"), "set_frame_length", "get_frame_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hop_length"), "set_hop_length", "get_hop_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_mel_bands"), "set_num_mel_bands", "get_num_mel_bands");
}

// ---- Setters / Getters -----------------------------------------------------
void MFCCProcessor::set_sample_rate(int p_rate) { core.set_sample_rate(p_rate); }
int  MFCCProcessor::get_sample_rate()  const { return core.get_sample_rate(); }
void MFCCProcessor::set_num_coeffs(int p_n) { core.set_num_coeffs(p_n); }
int  MFCCProcessor::get_num_coeffs()   const { return core.get_num_coeffs(); }
void MFCCProcessor::set_frame_length(int p_len) { core.set_frame_length(p_len); }
int  MFCCProcessor::get_frame_length() const { return core.get_frame_length(); }
void MFCCProcessor::set_hop_length(int p_hop) { core.set_hop_length(p_hop); }
int  MFCCProcessor::get_hop_length()   const { return core.get_hop_length(); }
void MFCCProcessor::set_num_mel_bands(int p_bands) { core.set_num_mel_bands(p_bands); }
int  MFCCProcessor::get_num_mel_bands() const { return core.get_num_mel_bands(); }


godot::TypedArray<godot::PackedFloat32Array>
MFCCProcessor::compute(const godot::PackedFloat32Array& samples)
{
    std::vector<float> input;
    input.resize(samples.size());
    for (int i = 0; i < samples.size(); i++)
        input[i] = samples[i];

    auto features = core.compute(input);
    godot::TypedArray<godot::PackedFloat32Array> result;
    for (const auto& frame : features)
    {
        godot::PackedFloat32Array godot_frame;
        godot_frame.resize(frame.size());
        for (int i = 0; i < frame.size(); i++)
            godot_frame[i] = frame[i];
        result.push_back(godot_frame);
    }
    return result;
}

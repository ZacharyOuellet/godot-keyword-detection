#pragma once
#include "keyword_detection/mfcc_processor_core.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

using namespace godot;
class MFCCProcessor : public godot::RefCounted {
    GDCLASS(MFCCProcessor, godot::RefCounted)

public:
    MFCCProcessor();
    ~MFCCProcessor();

    void set_sample_rate(int rate);
    int get_sample_rate() const;

    void set_num_coeffs(int n);
    int get_num_coeffs() const;

    void set_frame_length(int len);
    int get_frame_length() const;

    void set_hop_length(int hop);
    int get_hop_length() const;

    void set_num_mel_bands(int bands);
    int get_num_mel_bands() const;

    TypedArray<PackedFloat32Array> compute(
        const PackedFloat32Array& samples
    );

protected:
    static void _bind_methods();

private:
    MFCCProcessorCore core;
};

#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include "keyword_detection/pncc_processor_core.h"

namespace godot {
    class PNCCProcessor : public RefCounted {
        GDCLASS(PNCCProcessor, RefCounted)
    public:
        PNCCProcessor();
        ~PNCCProcessor();

        void set_sample_rate(int rate);
        int get_sample_rate() const;

        void set_num_coeffs(int n);
        int get_num_coeffs() const;

        void set_frame_length(int length);
        int get_frame_length() const;

        void set_hop_length(int hop);
        int get_hop_length() const;

        void set_num_gamma_bands(int bands);
        int get_num_gamma_bands() const;

        void set_power_law_exponent(float exponent);
        float get_power_law_exponent() const;

        void set_medium_time_frames(int frames);
        int get_medium_time_frames() const;

        TypedArray<PackedFloat32Array> compute(
            const PackedFloat32Array& samples
        );

    protected:
        static void _bind_methods();

    private:
        PNCCProcessorCore core;
    };
}

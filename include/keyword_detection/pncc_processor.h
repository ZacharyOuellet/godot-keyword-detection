#pragma once


#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <vector>

namespace godot {

    // PNCCProcessor -- Power-Normalized Cepstral Coefficients.
    //
    // Structurally this mirrors MFCCProcessor (same property pattern, same
    // framing loop), but PNCC's noise robustness comes from a "medium-time"
    // processing stage (asymmetric noise suppression + temporal masking) that
    // looks across many frames at once, so compute() has to run in multiple
    // passes over the whole utterance instead of finishing one frame at a time.
    class PNCCProcessor : public RefCounted {
        GDCLASS(PNCCProcessor, RefCounted)

    private:
        int _sample_rate = 16000;
        int _num_coeffs = 13;
        int _frame_length = 512;
        int _hop_length = 256;
        int _num_gamma_bands = 40;

        // PNCC-specific tunables (see Kim & Stern, "Power-Normalized Cepstral
        // Coefficients (PNCC) for Robust Speech Recognition", 2016, for the
        // full algorithm this is a simplified/practical version of).
        float _power_law_exponent = 1.0f / 15.0f; // replaces MFCC's log()
        int _medium_time_frames = 2;              // half-width M of the medium-time window (2M+1 frames)
        float _lambda_a = 0.999f;                 // ANS "attack" (rising) forgetting factor
        float _lambda_b = 0.5f;                   // ANS "decay" (falling) forgetting factor
        float _lambda_t = 0.85f;                  // temporal-masking peak decay factor

        bool _filterbank_dirty = true;
        std::vector<std::vector<float>> _gamma_filterbank; // [band][fft_bin]

    protected:
        static void _bind_methods();

    public:
        PNCCProcessor();
        ~PNCCProcessor();

        void set_sample_rate(int p_rate);
        int  get_sample_rate() const;
        void set_num_coeffs(int p_n);
        int  get_num_coeffs() const;
        void set_frame_length(int p_len);
        int  get_frame_length() const;
        void set_hop_length(int p_hop);
        int  get_hop_length() const;
        void set_num_gamma_bands(int p_bands);
        int  get_num_gamma_bands() const;
        void set_power_law_exponent(float p_exp);
        float get_power_law_exponent() const;
        void set_medium_time_frames(int p_m);
        int  get_medium_time_frames() const;

        TypedArray<PackedFloat32Array> compute(const PackedFloat32Array& p_samples);

    private:
        // Filterbank (ERB-spaced triangular filters standing in for gammatone
        // magnitude responses -- cheap, and accurate enough for the band-power
        // estimates the medium-time stage needs).
        void _build_gammatone_filterbank();
        std::vector<float> _apply_gammatone_filterbank(const std::vector<float>& power) const;

        // Medium-time / temporal processing. Each operates on the full
        // [frame][band] matrix built during pass 1 of compute().
        std::vector<std::vector<float>> _medium_time_power(const std::vector<std::vector<float>>& q) const;
        std::vector<std::vector<float>> _asymmetric_noise_suppression(const std::vector<std::vector<float>>& q_mt) const;
        std::vector<std::vector<float>> _temporal_masking(const std::vector<std::vector<float>>& q_mt,
            const std::vector<std::vector<float>>& q0) const;
        std::vector<std::vector<float>> _weight_smoothing(const std::vector<std::vector<float>>& q,
            const std::vector<std::vector<float>>& q_tm,
            const std::vector<std::vector<float>>& q_mt) const;
        void _mean_power_normalize(std::vector<std::vector<float>>& q) const;
    };

}

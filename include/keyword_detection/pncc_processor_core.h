#pragma once
#include <vector>

using PNCC = std::vector<std::vector<float>>;

class PNCCProcessorCore {
public:
    PNCCProcessorCore();
    ~PNCCProcessorCore();

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------

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

    // PNCC parameters

    void set_power_law_exponent(float exponent);
    float get_power_law_exponent() const;

    void set_medium_time_frames(int frames);
    int get_medium_time_frames() const;

    // -----------------------------------------------------------------------
    // Processing
    // -----------------------------------------------------------------------

    PNCC compute(
        const std::vector<float>& samples
    );

private:
    // -----------------------------------------------------------------------
    // Filterbank
    // -----------------------------------------------------------------------
    void _build_gammatone_filterbank();

    std::vector<float> _apply_gammatone_filterbank(
        const std::vector<float>& power
    ) const;

    // -----------------------------------------------------------------------
    // PNCC medium-time processing
    // -----------------------------------------------------------------------

    std::vector<std::vector<float>> _medium_time_power(
        const std::vector<std::vector<float>>& q
    ) const;

    std::vector<std::vector<float>> _asymmetric_noise_suppression(
        const std::vector<std::vector<float>>& q_mt
    ) const;

    std::vector<std::vector<float>> _temporal_masking(
        const std::vector<std::vector<float>>& q_mt,
        const std::vector<std::vector<float>>& q0
    ) const;

    std::vector<std::vector<float>> _weight_smoothing(
        const std::vector<std::vector<float>>& q,
        const std::vector<std::vector<float>>& q_tm,
        const std::vector<std::vector<float>>& q_mt
    ) const;

    void _mean_power_normalize(
        std::vector<std::vector<float>>& q
    ) const;

private:
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    int _sample_rate = 16000;
    int _num_coeffs = 13;
    int _frame_length = 512;
    int _hop_length = 256;
    int _num_gamma_bands = 40;

    // PNCC specific parameters
    float _power_law_exponent = 1.0f / 15.0f;

    // Half-width of medium-time window:
    // total window = 2M + 1 frames
    int _medium_time_frames = 2;

    // Asymmetric noise suppression
    float _lambda_a = 0.999f;
    float _lambda_b = 0.5f;

    // Temporal masking
    float _lambda_t = 0.85f;

    // Cached filterbank
    bool _filterbank_dirty = true;

    // [band][fft_bin]
    std::vector<std::vector<float>> _gamma_filterbank;
};

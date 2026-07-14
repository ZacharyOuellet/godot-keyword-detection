#pragma once

#include <vector>
#include <complex>

using MFCCSequence = std::vector<std::vector<float>>;

class MFCCProcessorCore {
public:
    MFCCProcessorCore();
    ~MFCCProcessorCore();

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

    MFCCSequence compute(const std::vector<float>& samples);

private:
    void _build_mel_filterbank();

    std::vector<float> _apply_mel_filterbank(
        const std::vector<float>& power
    ) const;

    int _sample_rate = 22050;
    int _num_coeffs = 13;
    int _frame_length = 512;
    int _hop_length = 256;
    int _num_mel_bands = 40;

    MFCCSequence _mel_filterbank;
    bool _filterbank_dirty = true;
};

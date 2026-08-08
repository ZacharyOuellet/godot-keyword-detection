#include "keyword_detection/mfcc_processor_core.h"
#include "keyword_detection/utils.h"


#include <cmath>
#include <algorithm>



// ---- Frequency conversions -------------------------------------------------
static float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

// ============================================================================
MFCCProcessorCore::MFCCProcessorCore() {}
MFCCProcessorCore::~MFCCProcessorCore() {}


// ---- Setters / Getters -----------------------------------------------------
void MFCCProcessorCore::set_sample_rate(int p_rate) { _sample_rate = p_rate;  _filterbank_dirty = true; }
int  MFCCProcessorCore::get_sample_rate()  const { return _sample_rate; }
void MFCCProcessorCore::set_num_coeffs(int p_n) { _num_coeffs = p_n; }
int  MFCCProcessorCore::get_num_coeffs()   const { return _num_coeffs; }
void MFCCProcessorCore::set_frame_length(int p_len) { _frame_length = p_len;  _filterbank_dirty = true; }
int  MFCCProcessorCore::get_frame_length() const { return _frame_length; }
void MFCCProcessorCore::set_hop_length(int p_hop) { _hop_length = p_hop; }
int  MFCCProcessorCore::get_hop_length()   const { return _hop_length; }
void MFCCProcessorCore::set_num_mel_bands(int p_bands) { _num_mel_bands = p_bands; _filterbank_dirty = true; }
int  MFCCProcessorCore::get_num_mel_bands() const { return _num_mel_bands; }

// ============================================================================
// Public: compute
// ============================================================================
MFCCSequence MFCCProcessorCore::compute(const std::vector<float>& samples) {
    MFCCSequence result;

    if (_filterbank_dirty) {
        _build_mel_filterbank();
        _filterbank_dirty = false;
    }

    const int n_samples = samples.size();
    if (n_samples < _frame_length) {
        return result;
    }

    const int fft_size = _frame_length;
    const int spec_bins = fft_size / 2 + 1;

    std::vector<float> power(spec_bins);
    std::vector<std::complex<float>> fft_values(fft_size);

    for (int start = 0; start + _frame_length <= n_samples; start += _hop_length) {
        // 1. Copy frame
        fft_values.assign(fft_size, std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < _frame_length; ++i) {
            fft_values[i] = std::complex<float>(samples[start + i], 0.0f);
        }

        Utils::apply_preemphasis(fft_values);

        // 2. Hann window
        Utils::apply_hann_window(fft_values);

        // 3. FFT
        Utils::apply_fft(fft_values);

        // 4. Power spectrum
        Utils::power_spectrum(fft_values, power);

        // 5. Mel filterbank
        std::vector<float> mel_energies = _apply_mel_filterbank(power);

        // 6. Log
        for (float& e : mel_energies) {
            e = std::log(std::max(e, Utils::LOG_FLOOR));
        }

        // 7. DCT → MFCCs
        std::vector<float> coeffs = Utils::dct(mel_energies, _num_coeffs);
        coeffs.resize(_num_coeffs, 0.0f);
        result.push_back(coeffs);
    }

    // 8. Cepstral mean normalization
    Utils::apply_cepstral_mean_normalize(result);

    return result;
}

// ============================================================================
// Private helpers
// ============================================================================

void MFCCProcessorCore::_build_mel_filterbank() {
    const int spec_bins = _frame_length / 2 + 1;
    const float nyquist = _sample_rate / 2.0f;

    const float mel_min = hz_to_mel(0.0f);
    const float mel_max = hz_to_mel(nyquist);

    // Evenly spaced mel points (num_mel_bands + 2 to include edges)
    std::vector<float> mel_points(_num_mel_bands + 2);
    for (int i = 0; i < _num_mel_bands + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (_num_mel_bands + 1);
    }

    // Convert back to Hz, then to FFT bin indices
    std::vector<int> bins(_num_mel_bands + 2);
    for (int i = 0; i < _num_mel_bands + 2; ++i) {
        float hz = mel_to_hz(mel_points[i]);
        bins[i] = static_cast<int>(std::floor((spec_bins)*hz / nyquist));
        bins[i] = std::clamp(bins[i], 0, spec_bins - 1);
    }

    _mel_filterbank.assign(_num_mel_bands, std::vector<float>(spec_bins, 0.0f));

    for (int m = 0; m < _num_mel_bands; ++m) {
        int left = bins[m];
        int center = bins[m + 1];
        int right = bins[m + 2];

        for (int k = left; k <= center; ++k) {
            if (center != left) {
                _mel_filterbank[m][k] = float(k - left) / float(center - left);
            }
        }
        for (int k = center; k <= right; ++k) {
            if (right != center) {
                _mel_filterbank[m][k] = float(right - k) / float(right - center);
            }
        }
    }
}

std::vector<float> MFCCProcessorCore::_apply_mel_filterbank(const std::vector<float>& power) const {
    std::vector<float> energies(_num_mel_bands, 0.0f);
    for (int m = 0; m < _num_mel_bands; ++m) {
        for (int k = 0; k < (int)power.size(); ++k) {
            energies[m] += _mel_filterbank[m][k] * power[k];
        }
    }
    return energies;
}

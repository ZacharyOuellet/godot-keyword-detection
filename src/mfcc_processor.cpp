#include "mfcc_processor.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cmath>
#include <algorithm>

using namespace godot;

// ---- Constants -------------------------------------------------------------
static constexpr float PI  = 3.14159265358979323846f;
static constexpr float LOG_FLOOR = 1e-10f; // avoid log(0)

// ---- Frequency conversions -------------------------------------------------
static float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

// ============================================================================
MFCCProcessor::MFCCProcessor()  {}
MFCCProcessor::~MFCCProcessor() {}

// ---- Godot bindings --------------------------------------------------------
void MFCCProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_sample_rate",   "rate"),  &MFCCProcessor::set_sample_rate);
    ClassDB::bind_method(D_METHOD("get_sample_rate"),            &MFCCProcessor::get_sample_rate);
    ClassDB::bind_method(D_METHOD("set_num_coeffs",    "n"),     &MFCCProcessor::set_num_coeffs);
    ClassDB::bind_method(D_METHOD("get_num_coeffs"),             &MFCCProcessor::get_num_coeffs);
    ClassDB::bind_method(D_METHOD("set_frame_length",  "len"),   &MFCCProcessor::set_frame_length);
    ClassDB::bind_method(D_METHOD("get_frame_length"),           &MFCCProcessor::get_frame_length);
    ClassDB::bind_method(D_METHOD("set_hop_length",    "hop"),   &MFCCProcessor::set_hop_length);
    ClassDB::bind_method(D_METHOD("get_hop_length"),             &MFCCProcessor::get_hop_length);
    ClassDB::bind_method(D_METHOD("set_num_mel_bands", "bands"), &MFCCProcessor::set_num_mel_bands);
    ClassDB::bind_method(D_METHOD("get_num_mel_bands"),          &MFCCProcessor::get_num_mel_bands);
    ClassDB::bind_method(D_METHOD("compute",           "samples"), &MFCCProcessor::compute);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_rate"),   "set_sample_rate",   "get_sample_rate");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_coeffs"),    "set_num_coeffs",    "get_num_coeffs");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "frame_length"),  "set_frame_length",  "get_frame_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hop_length"),    "set_hop_length",    "get_hop_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_mel_bands"), "set_num_mel_bands", "get_num_mel_bands");
}

// ---- Setters / Getters -----------------------------------------------------
void MFCCProcessor::set_sample_rate(int p_rate)   { _sample_rate   = p_rate;  _filterbank_dirty = true; }
int  MFCCProcessor::get_sample_rate()  const       { return _sample_rate; }
void MFCCProcessor::set_num_coeffs(int p_n)        { _num_coeffs    = p_n; }
int  MFCCProcessor::get_num_coeffs()   const       { return _num_coeffs; }
void MFCCProcessor::set_frame_length(int p_len)    { _frame_length  = p_len;  _filterbank_dirty = true; }
int  MFCCProcessor::get_frame_length() const       { return _frame_length; }
void MFCCProcessor::set_hop_length(int p_hop)      { _hop_length    = p_hop; }
int  MFCCProcessor::get_hop_length()   const       { return _hop_length; }
void MFCCProcessor::set_num_mel_bands(int p_bands) { _num_mel_bands = p_bands; _filterbank_dirty = true; }
int  MFCCProcessor::get_num_mel_bands() const      { return _num_mel_bands; }

// ============================================================================
// Public: compute
// ============================================================================
TypedArray<PackedFloat32Array> MFCCProcessor::compute(const PackedFloat32Array &p_samples) {
    TypedArray<PackedFloat32Array> result;

    if (_filterbank_dirty) {
        _build_mel_filterbank();
        _filterbank_dirty = false;
    }

    const int n_samples = p_samples.size();
    if (n_samples < _frame_length) {
        UtilityFunctions::push_warning("MFCCProcessor: not enough samples for even one frame.");
        return result;
    }

    const int fft_size   = _frame_length;
    const int spec_bins  = fft_size / 2 + 1;

    std::vector<float> real(fft_size), imag(fft_size), power(spec_bins);

    for (int start = 0; start + _frame_length <= n_samples; start += _hop_length) {
        // 1. Copy frame
        real.assign(fft_size, 0.0f);
        imag.assign(fft_size, 0.0f);
        for (int i = 0; i < _frame_length; ++i) {
            real[i] = p_samples[start + i];
        }

        // 2. Hann window
        _apply_hann_window(real);

        // 3. FFT
        _fft(real, imag);

        // 4. Power spectrum
        _power_spectrum(real, imag, power);

        // 5. Mel filterbank
        std::vector<float> mel_energies = _apply_mel_filterbank(power);

        // 6. Log
        for (float &e : mel_energies) {
            e = std::log(std::max(e, LOG_FLOOR));
        }

        // 7. DCT → MFCCs
        std::vector<float> coeffs = _dct(mel_energies);

        // 8. Pack into Godot array
        PackedFloat32Array frame_coeffs;
        frame_coeffs.resize(_num_coeffs);
        for (int k = 0; k < _num_coeffs; ++k) {
            frame_coeffs[k] = (k < (int)coeffs.size()) ? coeffs[k] : 0.0f;
        }
        result.push_back(frame_coeffs);
    }

    return result;
}

// ============================================================================
// Private helpers
// ============================================================================

void MFCCProcessor::_build_mel_filterbank() {
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
        float hz  = mel_to_hz(mel_points[i]);
        bins[i]   = static_cast<int>(std::floor((spec_bins) * hz / nyquist));
        bins[i]   = std::clamp(bins[i], 0, spec_bins - 1);
    }

    _mel_filterbank.assign(_num_mel_bands, std::vector<float>(spec_bins, 0.0f));

    for (int m = 0; m < _num_mel_bands; ++m) {
        int left   = bins[m];
        int center = bins[m + 1];
        int right  = bins[m + 2];

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

void MFCCProcessor::_apply_hann_window(std::vector<float> &frame) const {
    const int N = static_cast<int>(frame.size());
    for (int i = 0; i < N; ++i) {
        frame[i] *= 0.5f * (1.0f - std::cos(2.0f * PI * i / (N - 1)));
    }
}

// Cooley-Tukey iterative FFT (radix-2, in-place).
// Requires frame size to be a power of two.
void MFCCProcessor::_fft(std::vector<float> &real, std::vector<float> &imag) const {
    const int N = static_cast<int>(real.size());

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) { j ^= bit; }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Butterfly stages
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * PI / len;
        float wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < N; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                float ur = real[i + j],           ui = imag[i + j];
                float vr = real[i + j + len/2] * cr - imag[i + j + len/2] * ci;
                float vi = real[i + j + len/2] * ci + imag[i + j + len/2] * cr;
                real[i + j]          = ur + vr;
                imag[i + j]          = ui + vi;
                real[i + j + len/2]  = ur - vr;
                imag[i + j + len/2]  = ui - vi;
                float new_cr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = new_cr;
            }
        }
    }
}

void MFCCProcessor::_power_spectrum(const std::vector<float> &real,
                                    const std::vector<float> &imag,
                                    std::vector<float>       &power) const {
    const int bins = static_cast<int>(power.size());
    for (int k = 0; k < bins; ++k) {
        power[k] = real[k] * real[k] + imag[k] * imag[k];
    }
}

std::vector<float> MFCCProcessor::_apply_mel_filterbank(const std::vector<float> &power) const {
    std::vector<float> energies(_num_mel_bands, 0.0f);
    for (int m = 0; m < _num_mel_bands; ++m) {
        for (int k = 0; k < (int)power.size(); ++k) {
            energies[m] += _mel_filterbank[m][k] * power[k];
        }
    }
    return energies;
}

// Type-II DCT (orthonormal)
std::vector<float> MFCCProcessor::_dct(const std::vector<float> &mel_energies) const {
    const int M = static_cast<int>(mel_energies.size());
    std::vector<float> out(_num_coeffs, 0.0f);
    for (int k = 0; k < _num_coeffs; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < M; ++n) {
            sum += mel_energies[n] * std::cos(PI * k * (2 * n + 1) / (2.0f * M));
        }
        out[k] = sum * (k == 0 ? std::sqrt(1.0f / M) : std::sqrt(2.0f / M));
    }
    return out;
}

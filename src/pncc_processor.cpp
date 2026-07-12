#include "pncc_processor.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cmath>
#include <algorithm>
#include "utils.h"

using namespace godot;


// ---- Frequency conversions -------------------------------------------------
// PNCC's filterbank is spaced on the ERB (Equivalent Rectangular Bandwidth)
// scale rather than mel, since it's meant to approximate a gammatone
// (cochlear) filterbank.
static float hz_to_erb(float hz) { return 21.4f * std::log10(1.0f + 0.00437f * hz); }
static float erb_to_hz(float erb) { return (std::pow(10.0f, erb / 21.4f) - 1.0f) / 0.00437f; }

// ============================================================================
PNCCProcessor::PNCCProcessor() {}
PNCCProcessor::~PNCCProcessor() {}

// ---- Godot bindings --------------------------------------------------------
void PNCCProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_sample_rate", "rate"), &PNCCProcessor::set_sample_rate);
    ClassDB::bind_method(D_METHOD("get_sample_rate"), &PNCCProcessor::get_sample_rate);
    ClassDB::bind_method(D_METHOD("set_num_coeffs", "n"), &PNCCProcessor::set_num_coeffs);
    ClassDB::bind_method(D_METHOD("get_num_coeffs"), &PNCCProcessor::get_num_coeffs);
    ClassDB::bind_method(D_METHOD("set_frame_length", "len"), &PNCCProcessor::set_frame_length);
    ClassDB::bind_method(D_METHOD("get_frame_length"), &PNCCProcessor::get_frame_length);
    ClassDB::bind_method(D_METHOD("set_hop_length", "hop"), &PNCCProcessor::set_hop_length);
    ClassDB::bind_method(D_METHOD("get_hop_length"), &PNCCProcessor::get_hop_length);
    ClassDB::bind_method(D_METHOD("set_num_gamma_bands", "bands"), &PNCCProcessor::set_num_gamma_bands);
    ClassDB::bind_method(D_METHOD("get_num_gamma_bands"), &PNCCProcessor::get_num_gamma_bands);
    ClassDB::bind_method(D_METHOD("set_power_law_exponent", "exponent"), &PNCCProcessor::set_power_law_exponent);
    ClassDB::bind_method(D_METHOD("get_power_law_exponent"), &PNCCProcessor::get_power_law_exponent);
    ClassDB::bind_method(D_METHOD("set_medium_time_frames", "m"), &PNCCProcessor::set_medium_time_frames);
    ClassDB::bind_method(D_METHOD("get_medium_time_frames"), &PNCCProcessor::get_medium_time_frames);
    ClassDB::bind_method(D_METHOD("compute", "samples"), &PNCCProcessor::compute);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_rate"), "set_sample_rate", "get_sample_rate");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_coeffs"), "set_num_coeffs", "get_num_coeffs");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "frame_length"), "set_frame_length", "get_frame_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hop_length"), "set_hop_length", "get_hop_length");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "num_gamma_bands"), "set_num_gamma_bands", "get_num_gamma_bands");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "power_law_exponent"), "set_power_law_exponent", "get_power_law_exponent");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "medium_time_frames"), "set_medium_time_frames", "get_medium_time_frames");
}

// ---- Setters / Getters -----------------------------------------------------
void  PNCCProcessor::set_sample_rate(int p_rate) { _sample_rate = p_rate; _filterbank_dirty = true; }
int   PNCCProcessor::get_sample_rate()   const { return _sample_rate; }
void  PNCCProcessor::set_num_coeffs(int p_n) { _num_coeffs = p_n; }
int   PNCCProcessor::get_num_coeffs()    const { return _num_coeffs; }
void  PNCCProcessor::set_frame_length(int p_len) { _frame_length = p_len; _filterbank_dirty = true; }
int   PNCCProcessor::get_frame_length()  const { return _frame_length; }
void  PNCCProcessor::set_hop_length(int p_hop) { _hop_length = p_hop; }
int   PNCCProcessor::get_hop_length()    const { return _hop_length; }
void  PNCCProcessor::set_num_gamma_bands(int p_bands) { _num_gamma_bands = p_bands; _filterbank_dirty = true; }
int   PNCCProcessor::get_num_gamma_bands() const { return _num_gamma_bands; }
void  PNCCProcessor::set_power_law_exponent(float p_exp) { _power_law_exponent = p_exp; }
float PNCCProcessor::get_power_law_exponent() const { return _power_law_exponent; }
void  PNCCProcessor::set_medium_time_frames(int p_m) { _medium_time_frames = p_m; }
int   PNCCProcessor::get_medium_time_frames() const { return _medium_time_frames; }

// ============================================================================
// Public: compute
// ============================================================================
TypedArray<PackedFloat32Array> PNCCProcessor::compute(const PackedFloat32Array& p_samples) {
    TypedArray<PackedFloat32Array> result;

    if (_filterbank_dirty) {
        _build_gammatone_filterbank();
        _filterbank_dirty = false;
    }

    const int n_samples = p_samples.size();
    if (n_samples < _frame_length) {
        UtilityFunctions::push_warning("PNCCProcessor: not enough samples for even one frame.");
        return result;
    }

    const int fft_size = _frame_length;
    const int spec_bins = fft_size / 2 + 1;

    std::vector<float> power(spec_bins);
    std::vector<std::complex<float>> fft_values(fft_size);

    // ---- Pass 1: per-frame gammatone band power -------------------------
    // Unlike MFCC, PNCC's medium-time processing (noise suppression,
    // temporal masking) needs the band-power trajectory of the *whole*
    // utterance before any single frame can be finished, so we can't fold
    // this into one frame-at-a-time loop the way MFCC does.
    std::vector<std::vector<float>> band_power; // [frame][band]

    for (int start = 0; start + _frame_length <= n_samples; start += _hop_length) {
        fft_values.assign(fft_size, std::complex<float>(0.0f, 0.0f));
        for (int i = 0; i < _frame_length; ++i) {
            fft_values[i] = std::complex<float>(p_samples[start + i], 0.0f);
        }

        // Note: PNCC conventionally skips pre-emphasis -- the gammatone
        // filterbank already approximates the ear's frequency response,
        // and the medium-time stage handles the rest of the shaping.
        Utils::apply_hann_window(fft_values);
        Utils::apply_fft(fft_values);
        Utils::power_spectrum(fft_values, power);

        band_power.push_back(_apply_gammatone_filterbank(power));
    }

    if (band_power.empty()) {
        return result;
    }

    // ---- Pass 2: medium-time power normalization -------------------------
    std::vector<std::vector<float>> q_mt = _medium_time_power(band_power);
    std::vector<std::vector<float>> q0 = _asymmetric_noise_suppression(q_mt);
    std::vector<std::vector<float>> q_tm = _temporal_masking(q_mt, q0);
    std::vector<std::vector<float>> s = _weight_smoothing(band_power, q_tm, q_mt);
    _mean_power_normalize(s);

    // ---- Pass 3: power-law nonlinearity + DCT per frame -------------------
    const int n_frames = (int)s.size();
    for (int m = 0; m < n_frames; ++m) {
        // Power-law compression replaces MFCC's log() -- PNCC uses x^(1/15)
        // rather than log because it better matches the intensity-loudness
        // relationship at low SNRs and avoids log's blowup near zero.
        std::vector<float> compressed(s[m].size());
        for (size_t l = 0; l < s[m].size(); ++l) {
            compressed[l] = std::pow(std::max(s[m][l], 0.0f), _power_law_exponent);
        }

        std::vector<float> coeffs = Utils::dct(compressed, _num_coeffs);
        Utils::apply_mean_normalize(coeffs);

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

void PNCCProcessor::_build_gammatone_filterbank() {
    const int spec_bins = _frame_length / 2 + 1;
    const float nyquist = _sample_rate / 2.0f;

    const float erb_min = hz_to_erb(0.0f);
    const float erb_max = hz_to_erb(nyquist);

    // Evenly spaced ERB points (num_gamma_bands + 2 to include edges)
    std::vector<float> erb_points(_num_gamma_bands + 2);
    for (int i = 0; i < _num_gamma_bands + 2; ++i) {
        erb_points[i] = erb_min + (erb_max - erb_min) * i / (_num_gamma_bands + 1);
    }

    // Convert back to Hz, then to FFT bin indices
    std::vector<int> bins(_num_gamma_bands + 2);
    for (int i = 0; i < _num_gamma_bands + 2; ++i) {
        float hz = erb_to_hz(erb_points[i]);
        bins[i] = static_cast<int>(std::floor((spec_bins)*hz / nyquist));
        bins[i] = std::clamp(bins[i], 0, spec_bins - 1);
    }

    _gamma_filterbank.assign(_num_gamma_bands, std::vector<float>(spec_bins, 0.0f));

    // Triangular filters on the ERB scale stand in for true gammatone
    // magnitude responses here -- much cheaper to build and apply, and
    // accurate enough for the band-power estimates the medium-time stage
    // consumes. Swap this out for real gammatone filter shapes if you need
    // closer fidelity to the reference PNCC algorithm.
    for (int m = 0; m < _num_gamma_bands; ++m) {
        int left = bins[m];
        int center = bins[m + 1];
        int right = bins[m + 2];

        for (int k = left; k <= center; ++k) {
            if (center != left) {
                _gamma_filterbank[m][k] = float(k - left) / float(center - left);
            }
        }
        for (int k = center; k <= right; ++k) {
            if (right != center) {
                _gamma_filterbank[m][k] = float(right - k) / float(right - center);
            }
        }
    }
}

std::vector<float> PNCCProcessor::_apply_gammatone_filterbank(const std::vector<float>& power) const {
    std::vector<float> energies(_num_gamma_bands, 0.0f);
    for (int m = 0; m < _num_gamma_bands; ++m) {
        for (int k = 0; k < (int)power.size(); ++k) {
            energies[m] += _gamma_filterbank[m][k] * power[k];
        }
    }
    return energies;
}

// ---- Medium-time power: a simple centered moving average across frames ----
// Smooths each band's power trajectory over a window of (2M+1) frames so
// the noise-floor tracking below isn't chasing frame-to-frame jitter.
std::vector<std::vector<float>> PNCCProcessor::_medium_time_power(const std::vector<std::vector<float>>& q) const {
    const int n_frames = (int)q.size();
    if (n_frames == 0) return q;
    const int n_bands = (int)q[0].size();

    std::vector<std::vector<float>> q_mt(n_frames, std::vector<float>(n_bands, 0.0f));

    const int M = std::max(0, _medium_time_frames);
    for (int m = 0; m < n_frames; ++m) {
        int lo = std::max(0, m - M);
        int hi = std::min(n_frames - 1, m + M);
        float count = float(hi - lo + 1);
        for (int l = 0; l < n_bands; ++l) {
            float sum = 0.0f;
            for (int mm = lo; mm <= hi; ++mm) sum += q[mm][l];
            q_mt[m][l] = sum / count;
        }
    }
    return q_mt;
}

// ---- Asymmetric noise suppression (ANS) ------------------------------------
// Tracks a per-band noise floor with an asymmetric leaky integrator: it
// rises slowly (lambda_a, close to 1) so transient speech energy doesn't
// drag the floor up, but falls faster (lambda_b) so it can follow the
// noise back down between utterances. Subtracting the floor suppresses
// stationary background noise while leaving speech onsets intact.
std::vector<std::vector<float>> PNCCProcessor::_asymmetric_noise_suppression(const std::vector<std::vector<float>>& q_mt) const {
    const int n_frames = (int)q_mt.size();
    if (n_frames == 0) return q_mt;
    const int n_bands = (int)q_mt[0].size();

    const float floor_ratio = 0.01f; // residual floor so bands never hit a hard zero

    std::vector<std::vector<float>> q0(n_frames, std::vector<float>(n_bands, 0.0f));

    for (int l = 0; l < n_bands; ++l) {
        float q_le = q_mt[0][l];
        for (int m = 0; m < n_frames; ++m) {
            float q_val = q_mt[m][l];
            if (q_val >= q_le) {
                q_le = _lambda_a * q_le + (1.0f - _lambda_a) * q_val;
            }
            else {
                q_le = _lambda_b * q_le + (1.0f - _lambda_b) * q_val;
            }
            float suppressed = q_val - q_le;
            q0[m][l] = std::max(suppressed, floor_ratio * q_val);
        }
    }
    return q0;
}

// ---- Temporal masking ------------------------------------------------------
// Models the ear's forward masking: a loud onset in a band suppresses
// perception of quieter energy that follows it shortly after. We track a
// decaying peak per band and clamp anything that falls too far below it,
// which further reduces sensitivity to noise/reverberation tails.
std::vector<std::vector<float>> PNCCProcessor::_temporal_masking(const std::vector<std::vector<float>>& q_mt,
    const std::vector<std::vector<float>>& q0) const {
    const int n_frames = (int)q_mt.size();
    if (n_frames == 0) return q0;
    const int n_bands = (int)q_mt[0].size();

    const float masking_floor = 0.2f; // ratio below the running peak that gets masked

    std::vector<std::vector<float>> q_tm(n_frames, std::vector<float>(n_bands, 0.0f));

    for (int l = 0; l < n_bands; ++l) {
        float peak = q0[0][l];
        for (int m = 0; m < n_frames; ++m) {
            float current = q0[m][l];
            peak = std::max(_lambda_t * peak, current);
            q_tm[m][l] = (current >= masking_floor * peak) ? current : masking_floor * peak;
        }
    }
    return q_tm;
}

// ---- Weight smoothing -------------------------------------------------------
// Turns the noise-suppressed/masked medium-time energy back into a per-bin
// weighting ratio, applies it to the ORIGINAL short-time band power (so we
// recover full time resolution instead of staying stuck at medium-time
// resolution), then smooths the result across neighboring bands to reduce
// the musical-noise-like artifacts that per-band weighting can introduce.
std::vector<std::vector<float>> PNCCProcessor::_weight_smoothing(const std::vector<std::vector<float>>& q,
    const std::vector<std::vector<float>>& q_tm,
    const std::vector<std::vector<float>>& q_mt) const {
    const int n_frames = (int)q.size();
    if (n_frames == 0) return q;
    const int n_bands = (int)q[0].size();
    const float eps = 1e-8f;

    std::vector<std::vector<float>> s(n_frames, std::vector<float>(n_bands, 0.0f));
    for (int m = 0; m < n_frames; ++m) {
        for (int l = 0; l < n_bands; ++l) {
            float w = q_tm[m][l] / (q_mt[m][l] + eps);
            w = std::clamp(w, 0.0f, 5.0f);
            s[m][l] = q[m][l] * w;
        }
    }

    std::vector<std::vector<float>> s_smooth(n_frames, std::vector<float>(n_bands, 0.0f));
    for (int m = 0; m < n_frames; ++m) {
        for (int l = 0; l < n_bands; ++l) {
            int lo = std::max(0, l - 1);
            int hi = std::min(n_bands - 1, l + 1);
            float sum = 0.0f;
            for (int ll = lo; ll <= hi; ++ll) sum += s[m][ll];
            s_smooth[m][l] = sum / float(hi - lo + 1);
        }
    }
    return s_smooth;
}

// ---- Mean power normalization ----------------------------------------------
// Rescales the whole [frame][band] matrix so its mean matches a fixed
// target, playing the role of an automatic gain control step so overall
// recording loudness doesn't shift the final coefficients.
void PNCCProcessor::_mean_power_normalize(std::vector<std::vector<float>>& q) const {
    double sum = 0.0;
    size_t count = 0;
    for (auto& row : q) {
        for (float v : row) { sum += v; ++count; }
    }
    if (count == 0) return;

    float mean = float(sum / double(count));
    const float target_mean = 1e4f;
    float scale = (mean > 1e-8f) ? (target_mean / mean) : 1.0f;

    for (auto& row : q) {
        for (float& v : row) v *= scale;
    }
}

#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <vector>

using namespace godot;

// ---------------------------------------------------------------------------
// MFCCProcessor
//
// Computes Mel-Frequency Cepstral Coefficients from raw PCM audio samples.
//
// Typical usage (GDScript):
//   var mfcc = MFCCProcessor.new()
//   mfcc.sample_rate   = 22050
//   mfcc.num_coeffs    = 13
//   mfcc.frame_length  = 512
//   mfcc.hop_length    = 256
//   mfcc.num_mel_bands = 40
//   var features: Array = mfcc.compute(pcm_samples)  # Array of PackedFloat32Array
// ---------------------------------------------------------------------------
class MFCCProcessor : public RefCounted {
    GDCLASS(MFCCProcessor, RefCounted)

public:
    MFCCProcessor();
    ~MFCCProcessor();

    // --- Parameters (set before calling compute) ---
    void set_sample_rate(int p_rate);
    int  get_sample_rate() const;

    void set_num_coeffs(int p_n);
    int  get_num_coeffs() const;

    void set_frame_length(int p_len);
    int  get_frame_length() const;

    void set_hop_length(int p_hop);
    int  get_hop_length() const;

    void set_num_mel_bands(int p_bands);
    int  get_num_mel_bands() const;

    // --- Main API ---
    // Input : PackedFloat32Array of mono PCM samples in [-1, 1]
    // Output: Array of PackedFloat32Array  (one entry per frame, length = num_coeffs)
    TypedArray<PackedFloat32Array> compute(const PackedFloat32Array& p_samples);

protected:
    static void _bind_methods();

private:
    // DSP helpers
    void  _build_mel_filterbank();
    void  _apply_hann_window(std::vector<float>& frame) const;
    void  _fft(std::vector<float>& real, std::vector<float>& imag) const;
    void  _power_spectrum(const std::vector<float>& real,
        const std::vector<float>& imag,
        std::vector<float>& power) const;
    std::vector<float> _apply_mel_filterbank(const std::vector<float>& power) const;
    std::vector<float> _dct(const std::vector<float>& mel_energies) const;

    // Parameters
    int _sample_rate = 22050;
    int _num_coeffs = 13;
    int _frame_length = 512;
    int _hop_length = 256;
    int _num_mel_bands = 40;

    // Pre-computed filterbank  [_num_mel_bands][_frame_length/2 + 1]
    std::vector<std::vector<float>> _mel_filterbank;
    bool _filterbank_dirty = true;
};

#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include <cassert>

namespace Utils {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float LOG_FLOOR = 1e-10f; // avoid log(0)

    template <typename T> // for float or std::complex<float>
    inline void apply_preemphasis(std::vector<T>& values, const float coeff = 0.97f) {
        if (values.empty()) {
            return;
        }
        // First sample remains unchanged
        for (size_t i = values.size() - 1; i > 0; --i) {
            values[i] = values[i] - coeff * values[i - 1];
        }
    }

    inline std::vector<float> dct(const std::vector<float>& values, const int num_coeffs) {
        const int N = static_cast<int>(values.size());
        std::vector<float> out(num_coeffs, 0.0f);
        for (int k = 0; k < num_coeffs; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += values[n] * std::cos(PI * k * (2 * n + 1) / (2.0f * N));
            }
            out[k] = sum * (k == 0 ? std::sqrt(1.0f / N) : std::sqrt(2.0f / N));
        }
        return out;
    }

    // Real Cepstral Mean Normalization: for each coefficient index k,
    // subtract the mean of c_k across ALL frames of the utterance. This is
    // what actually removes constant channel/gain/mic-distance mismatch
    // between two different recordings of the same word. Call this once
    // per utterance, after every frame's coefficients have been computed -
    // never per-frame.
    inline void apply_cepstral_mean_normalize(std::vector<std::vector<float>>& frames) {
        if (frames.empty()) return;
        const size_t num_coeffs = frames[0].size();

        std::vector<float> mean(num_coeffs, 0.0f);
        for (const auto& frame : frames) {
            for (size_t k = 0; k < num_coeffs; ++k) {
                mean[k] += frame[k];
            }
        }
        for (size_t k = 0; k < num_coeffs; ++k) {
            mean[k] /= static_cast<float>(frames.size());
        }

        for (auto& frame : frames) {
            for (size_t k = 0; k < num_coeffs; ++k) {
                frame[k] -= mean[k];
            }
        }
    }


    // Taken from https://cp-algorithms.com/algebra/fft.html
    inline void apply_fft(std::vector<std::complex<float>>& values, bool invert = false) {
        const int N = static_cast<int>(values.size());
        if (N == 1) return;
        assert((N & (N - 1)) == 0); // N must be a power of 2

        // Bit-reversal permutation
        for (int i = 1, j = 0; i < N; ++i) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1) { j ^= bit; }
            j ^= bit;
            if (i < j) {
                std::swap(values[i], values[j]);
            }
        }

        // Butterfly stages
        for (int len = 2; len <= N; len <<= 1) {
            float ang = 2.0f * PI / len * (invert ? -1.0f : 1.0f);
            std::complex<float> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < N; i += len) {
                std::complex<float> w(1.0f, 0.0f);
                for (int j = 0; j < len / 2; ++j) {
                    std::complex<float> u = values[i + j];
                    std::complex<float> v = values[i + j + len / 2] * w;
                    values[i + j] = u + v;
                    values[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (invert) {
            for (std::complex<float>& x : values) {
                x /= N;
            }
        }
    }

    template <typename T> // for float or std::complex<float>
    inline void apply_hann_window(std::vector<T>& frame) {
        const int N = static_cast<int>(frame.size())>1?static_cast<int>(frame.size()):2;
        for (int i = 0; i < N; ++i) {
            frame[i] *= 0.5f * (1.0f - std::cos(2.0f * PI * i / (N - 1)));
        }
    }

    inline void power_spectrum(const std::vector<std::complex<float>>& fft_values, std::vector<float>& power) {
        const int bins = static_cast<int>(power.size());
        for (int k = 0; k < bins; ++k) {
            const auto& val = fft_values[k];
            power[k] = val.real() * val.real() + val.imag() * val.imag();
        }
    }
}

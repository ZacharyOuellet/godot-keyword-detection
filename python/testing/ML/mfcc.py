"""
Python MFCC extractor that mirrors mfcc_processor_core.cpp / utils.h
step for step, so training-time features match your C++ inference code
for the same configuration (sample_rate, frame_length, hop_length,
num_mel_bands, num_coeffs).

Pipeline (matches the C++ exactly):
    per frame:
      1. copy frame_length samples (no padding -- frame_length == fft_size)
      2. pre-emphasis, RESET PER FRAME (not applied to the whole signal
         before framing -- values[0] of each frame is left unchanged,
         same as apply_preemphasis in utils.h)
      3. symmetric Hann window (N = frame_length)
      4. real FFT -> power spectrum over spec_bins = frame_length//2 + 1
      5. HTK-style triangular mel filterbank (2595*log10(1+hz/700)),
         bin edges computed the same way (floor(spec_bins*hz/nyquist),
         clipped) as _build_mel_filterbank
      6. natural log, floored at 1e-10
      7. orthonormal DCT-II (scipy dct type=2, norm='ortho'), truncated
         to num_coeffs -- matches Utils::dct's scaling exactly
      8. per-frame mean normalization

IMPORTANT: the C++ header defaults sample_rate to 22050. Your audio
(MSWC / this HF dataset) is very likely 16 kHz. Make sure whatever
sample_rate you configure here is the SAME one you set on the C++
side via set_sample_rate(), or the mel filterbank edges (and therefore
the features) won't match between training and deployment.
"""

import numpy as np
from scipy.fft import dct

LOG_FLOOR = 1e-10

try:
    import keyword_detection as _kd  # your pybind11 bindings, if built + on PYTHONPATH
    _HAS_CPP_BACKEND = True
except ImportError:
    _HAS_CPP_BACKEND = False


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate, frame_length, num_mel_bands):
    """Reproduces MFCCProcessorCore::_build_mel_filterbank exactly."""
    spec_bins = frame_length // 2 + 1
    nyquist = sample_rate / 2.0

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(nyquist)
    mel_points = mel_min + (mel_max - mel_min) * np.arange(num_mel_bands + 2) / (num_mel_bands + 1)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor(spec_bins * hz_points / nyquist).astype(int)
    bins = np.clip(bins, 0, spec_bins - 1)

    fb = np.zeros((num_mel_bands, spec_bins), dtype=np.float32)
    for m in range(num_mel_bands):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center != left:
            k = np.arange(left, center + 1)
            fb[m, k] = (k - left) / (center - left)
        if right != center:
            k = np.arange(center, right + 1)
            fb[m, k] = (right - k) / (right - center)
    return fb


class MFCCExtractor:
    """
    Waveform (1D array-like, float, any length) -> MFCC sequence.
    Call returns a numpy array of shape [num_coeffs, n_frames]
    (already transposed to the [freq, time] layout Net expects).

    Frames shorter than frame_length produce zero frames (n_frames may
    legitimately be 0 for very short clips -- caller should handle that,
    e.g. by skipping the sample or padding the waveform up front).
    """

    def __init__(self, sample_rate=16000, frame_length=512, hop_length=256,
                 num_mel_bands=40, num_coeffs=13, preemphasis_coeff=0.97):
        assert (frame_length & (frame_length - 1)) == 0, "frame_length must be a power of 2 (FFT requirement)"
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.num_mel_bands = num_mel_bands
        self.num_coeffs = num_coeffs
        self.preemphasis_coeff = preemphasis_coeff
        self._filterbank = build_mel_filterbank(sample_rate, frame_length, num_mel_bands)
        self._hann = np.hanning(frame_length).astype(np.float32)  # symmetric, matches utils.h

    def __call__(self, waveform):
        x = np.asarray(waveform, dtype=np.float32)
        n = x.shape[-1]
        if n < self.frame_length:
            return np.zeros((self.num_coeffs, 0), dtype=np.float32)

        n_frames = 1 + (n - self.frame_length) // self.hop_length
        frames = np.lib.stride_tricks.sliding_window_view(x, self.frame_length)[::self.hop_length][:n_frames]
        frames = frames.copy()  # writable

        # 1. pre-emphasis, reset per frame
        pre = frames.copy()
        pre[:, 1:] = frames[:, 1:] - self.preemphasis_coeff * frames[:, :-1]

        # 2. Hann window
        windowed = pre * self._hann[None, :]

        # 3. FFT -> power spectrum
        spec = np.fft.rfft(windowed, n=self.frame_length, axis=-1)
        power = spec.real ** 2 + spec.imag ** 2  # [n_frames, spec_bins]

        # 4. mel filterbank
        mel_energies = power @ self._filterbank.T  # [n_frames, num_mel_bands]

        # 5. log
        log_mel = np.log(np.maximum(mel_energies, LOG_FLOOR))

        # 6. DCT -> MFCCs (orthonormal, truncated to num_coeffs)
        mfcc = dct(log_mel, type=2, axis=-1, norm="ortho")[:, :self.num_coeffs]

        # 7. per-frame mean normalize
        mfcc = mfcc - mfcc.mean(axis=-1, keepdims=True)

        return mfcc.T.astype(np.float32)  # [num_coeffs, n_frames]


class CppMFCCExtractor:
    """
    Same interface as MFCCExtractor (waveform -> [num_coeffs, n_frames]),
    but calls straight into your compiled keyword_detection.MFCCProcessor
    (built from bindings.cpp / setup.py) instead of reimplementing the
    algorithm in numpy. This is the recommended extractor for real
    training runs: it's the literal code that will run on-device, so
    there's no reimplementation-drift risk.

    Requires the `keyword_detection` extension to be built and importable
    (e.g. `pip install -e .` from the directory containing setup.py).
    """

    def __init__(self, sample_rate=16000, frame_length=512, hop_length=256,
                 num_mel_bands=40, num_coeffs=13, preemphasis_coeff=0.97):
        if not _HAS_CPP_BACKEND:
            raise ImportError(
                "keyword_detection extension not found. Build it first: "
                "`pip install -e .` from the directory with your setup.py, "
                "or use MFCCExtractor (pure numpy) instead."
            )
        if preemphasis_coeff != 0.97:
            raise ValueError(
                "MFCCProcessorCore hardcodes the pre-emphasis coefficient at 0.97 "
                "(no setter exposed) -- can't override it through the C++ backend."
            )
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.num_coeffs = num_coeffs
        self._proc = None
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.num_mel_bands = num_mel_bands
        self.num_coeffs = num_coeffs

    def _get_proc(self):
        if self._proc is None:
            self._proc = _kd.MFCCProcessor()
            self._proc.sample_rate = self.sample_rate
            self._proc.frame_length = self.frame_length
            self._proc.hop_length = self.hop_length
            self._proc.num_mel_bands = self.num_mel_bands
            self._proc.num_coeffs = self.num_coeffs

        return self._proc

    def __call__(self, waveform):
        x = np.asarray(waveform, dtype=np.float32)
        proc = self._get_proc()
        out = proc.compute(x)  # (n_frames, num_coeffs)
        if out.shape[0] == 0:
            return np.zeros((self.num_coeffs, 0), dtype=np.float32)
        return out.T  # -> [num_coeffs, n_frames], same layout as MFCCExtractor


def make_extractor(prefer_cpp=True, **kwargs):
    """
    Convenience factory: returns CppMFCCExtractor if the compiled backend
    is available (and prefer_cpp=True), otherwise falls back to the pure
    numpy MFCCExtractor with a warning. Both have the same call signature.
    """
    if prefer_cpp and _HAS_CPP_BACKEND:
        return CppMFCCExtractor(**kwargs)
    if prefer_cpp and not _HAS_CPP_BACKEND:
        import warnings
        warnings.warn(
            "keyword_detection C++ extension not found -- falling back to the "
            "pure numpy MFCCExtractor. Build the extension for guaranteed "
            "parity with your deployed C++ inference code."
        )
    return MFCCExtractor(**kwargs)

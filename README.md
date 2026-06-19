# godot-mfcc-dtw

A Godot 4 GDExtension that exposes **MFCC** (Mel-Frequency Cepstral Coefficients)
and **DTW** (Dynamic Time Warping) as first-class Godot classes.
Designed for real-time audio gesture / speech-command recognition.

---

## Classes

### `MFCCProcessor`

Converts raw PCM audio (mono `PackedFloat32Array` in `[-1, 1]`) into a sequence
of MFCC feature vectors.

| Property        | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `sample_rate`   | 22050   | Sample rate of the input audio       |
| `num_coeffs`    | 13      | Number of cepstral coefficients      |
| `frame_length`  | 512     | FFT frame size (must be power of 2)  |
| `hop_length`    | 256     | Step between frames                  |
| `num_mel_bands` | 40      | Mel filterbank bands                 |

```gdscript
var mfcc := MFCCProcessor.new()
mfcc.sample_rate   = 22050
mfcc.num_coeffs    = 13
mfcc.frame_length  = 512
mfcc.hop_length    = 256
mfcc.num_mel_bands = 40

# features: Array[PackedFloat32Array]  — one entry per frame
var features: Array = mfcc.compute(pcm_samples)
```

---

### `DTWMatcher`

Computes the DTW distance between two MFCC sequences, or classifies an unknown
sequence against a set of labelled templates.

| Property          | Default     | Description                              |
|-------------------|-------------|------------------------------------------|
| `distance_metric` | `EUCLIDEAN` | Frame distance: `EUCLIDEAN` or `COSINE`  |
| `band_width`      | 0           | Sakoe-Chiba band (0 = no constraint)     |

```gdscript
var dtw := DTWMatcher.new()
dtw.distance_metric = DTWMatcher.EUCLIDEAN
dtw.band_width      = 20   # optional: constrain warping path

# Raw distance between two sequences
var dist: float = dtw.compute(features_a, features_b)

# --- Template matching ---
dtw.add_template("hello",  mfcc_hello_reference)
dtw.add_template("stop",   mfcc_stop_reference)

var label: String     = dtw.classify(unknown_features)
var result: Dictionary = dtw.classify_with_score(unknown_features)
# result == { "label": "hello", "distance": 0.42 }
```

---

## Building

### Prerequisites

- Python 3 + SCons (`pip install scons`)
- A C++17 compiler (GCC, Clang, or MSVC)
- godot-cpp checked out as a submodule

```bash
git clone --recurse-submodules https://github.com/YOUR_USER/godot-mfcc-dtw.git
cd godot-mfcc-dtw

# Debug build for the current platform
scons target=template_debug

# Release build
scons target=template_release
```

Binaries are written to `addons/mfcc_dtw/bin/`.

---

## Installation

Copy the entire `addons/mfcc_dtw/` folder into your project's `addons/` directory,
then enable **MFCC-DTW** in **Project → Project Settings → Plugins**.

---

## Full example

```gdscript
extends Node

@export var audio_stream: AudioStreamWAV

func _ready() -> void:
    var pcm := _wav_to_pcm(audio_stream)

    # 1. Compute MFCC features
    var mfcc := MFCCProcessor.new()
    var features: Array = mfcc.compute(pcm)

    # 2. Compare two recordings
    var dtw := DTWMatcher.new()
    dtw.band_width = 20

    dtw.add_template("hello", features)   # enroll a template

    # Later, with a new recording:
    var new_features: Array = mfcc.compute(_wav_to_pcm(another_stream))
    var result: Dictionary  = dtw.classify_with_score(new_features)
    print("Best match: ", result["label"], "  distance: ", result["distance"])

# Convert AudioStreamWAV data to a flat PackedFloat32Array of mono samples.
func _wav_to_pcm(stream: AudioStreamWAV) -> PackedFloat32Array:
    var data := stream.data           # PackedByteArray (16-bit LE stereo/mono)
    var out  := PackedFloat32Array()
    var i    := 0
    while i + 1 < data.size():
        var sample := int(data[i]) | (int(data[i + 1]) << 8)
        if sample >= 32768: sample -= 65536
        out.append(sample / 32768.0)
        i += 2 * (2 if stream.stereo else 1)  # skip right channel if stereo
    return out
```

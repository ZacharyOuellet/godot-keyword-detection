# godot-keyword-detection

A Godot 4 plugin that allows keyword spotting using **Dynamic Time Warping (DTW)**. The main computation happens in a GDExtension in C++ but a handy node makes it easy to use.

---

## Installation
See releases for the content to add it to your godot project. It is also available on the Godot Asset Library


## How to use
To be able to recognize keywords with this library you need an instance of the `KeywordMatcher` node. This node handles the underlying C++ API. It is configured with a `KeywordMatcherSetting` resource. In that resource, you will need to provide labels and ***UNCOMPRESSED*** `.wav` audio clips for the labels. The compression setting for the file used must be changed from the default in the import tab of Godot.


### `KeywordMatcher`
This is the main class for the plugin. Just put that in your scene and set the settings to a valid `KeywordMatcherSetting` resource.

It has 4 main methods you will need to use:
|Method|Return Type |Description|
|-----|-----|-----|
|`recognize_wav(stream: AudioStreamWAV) `|`String`|Used to classify a AudioStreamWAV. It will return the best match for the labels in the KeywordMatcherSetting attached to the instance.|
| `recognize_pcm(pcm: PackedFloat32Array)` |`String`|If you are using audio data other than an `AudioStreamWAV`, convert it to a `PackedFloat32Array` containing PCM audio samples. It needs to have the same sample rate as the settings.|
|`recognize_wav_detailed(stream: AudioStreamWAV)`|`Dictionary`| This works similarly to the normal `recognize_wav` method but it will instead return a `Dictionary` with all labels as keys and the distance related to each. A lower score (distance) means a better match. |
|`recognize_pcm_detailed(pcm: PackedFloat32Array)`|`Dictionary`| If you read the other descriptions, this will be pretty straight forward. Transform your audio sample into a PCM, pass it to this method and it will return a `Dictionary` with labels and scores.
> [!WARNING]
> WAVs are expected to be uncompressed 16 bits mono audio samples.

> [!WARNING]
> PCMs are expected to represent a single channel floats, normalized (`[-1.0 , 1.0]`)

### Using the gdextension
If you don't feel like using the provided node, you can directly call the `DTWMatcher` from the GDExtension. The documentation is all provided *Godot style*. Visible on the editor or your favorite VSCode extension.

## Important stuff for best results
Since this library is based on the distance between two clips there are some consideration that can affect the results drastically.
### Silence trimming
The node will try to handle the `AudioStreamWAV` to trim silence. It is a basic implementation and manual trimming could yield better results. Trimming silence allows the algorithm to focus on the actual words instead of comparing silence and background noise from two clips.
### Compression
The node unpacks a `.wav` file into a `PackedFloat32Array` and expects the `.wav` file to be uncompressed. If it uses the default import setting (Quite Ok Audio), the results will be corrupted and it will lead to random results.
### Diversity
The best results happen when there is a similarity between voices. To match for a variety of speakers, you need a variety of speakers for your samples. Try to have different types of voices and accents. Better yet, you can calibrate for a specific user by saving some things he said.
### Less labels => More precision
Since the algorithm compares to each label's audio clip, if a label is close to another, it will lead to imprecision and wrong classification. It can deal with a couple of words, but I don't think it is the best option for 10+ words (see demo). The more distinct the words, the more accurate it gets.
## Demo
In this repository, you can find a godot project that detects a spoken number in english (0-9). It can be used to try out some settings and their effect.
## Supported platforms
|Platform|Architecture|Supported|
|:--:|:----:|:--:|
|linux|x86_64|:white_check_mark:|
|linux|x86_32|:white_check_mark:|
|linux|arm64|:white_check_mark:|
|linux|arm32|:white_check_mark:|
|windows|x86_64|:white_check_mark:|
|windows|x86_32|:white_check_mark:|
|windows|arm64|:white_check_mark:|
|macos|universal|:white_check_mark:|
|android|x86_64|:white_check_mark:|
|android|x86_32|:white_check_mark:|
|android|arm64|:white_check_mark:|
|android|arm32|:white_check_mark:|
|ios|arm64|:white_check_mark:|
|web|wasm32|:white_check_mark:|
> [!NOTE]
> The plugin was developed on `windows x86_64` and it was tested on this platform only. The build for other platforms should work but I did not test them.

## Compiling
You can compile the GDExtension by using `scons`.

Make sure you have the python dependencies installed (in a virtual environment if you want):
```bash
pip install -r requirements.txt
```
Then you can run the build:
```
scons
```
> [!TIP]
> You can run `scons combiledb=yes` to allow extensions and IDEs to recognize symbols

## Repository structures
|Folder|Content|
|------|-------|
|`/project`|Contains the addon folder and the demo project. This is where the release package comes from|
|`/include` and `/src`|The C++ code lives there, it is the code that is built for the GDExtension|
|`/godot`|The Godot wrapper (with `godot-cpp`) to bridge the core C++ implementation|
|`/python`|A python wrapper that uses `pybind11` to create bindings for the core library. It was used for testing and benchmarking the library|
|`/doc_classes`|Used for the Godot documentation bundled with the GDExtension|

## Possible future work
- Implement a way to save the MFCC or PNCC values instead of shipping with the audio wav files.
- Noise filtering
- Calibrate for a user by pitching up and down their voice recordings when classifying

## Contributing
Open to issues and suggestions, PRs are also welcome.

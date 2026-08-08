class_name KeywordMatcher
extends Node


signal keyword_recognized(label: String)
signal no_keyword_recognized

@export var settings: KeywordMatcherSettings:
	set(value):
		settings = value
		if is_inside_tree():
			_rebuild()

var _feature_extractor: Variant
var _dtw: DTWMatcher


func _ready() -> void:
	_rebuild()


## Rebuilds the DTW matcher/extractor and reloads all templates from
## [member settings]. Called automatically on _ready() and whenever [member
## settings] is reassigned at runtime; call it manually if you edit the
## settings resource in place (e.g. change sample_rate) and want that to
## take effect without reassigning the whole resource.
func _rebuild() -> void:
	_dtw = null
	_feature_extractor = null

	if settings == null:
		push_warning("KeywordMatcher has no KeywordMatcherSettings assigned; it will not recognize anything.")
		return

	if settings.keyword_dataset.is_empty():
		push_warning("KeywordMatcherSettings has no baked audio templates yet - add labels and associated .wav files")

	_dtw = settings.create_dtw_matcher()
	_feature_extractor = settings.create_feature_extractor()

	for label in settings.keyword_dataset.keys():
		var feature_list: Array = []
		for audio in settings.keyword_dataset[label]:
			if audio is AudioStreamWAV:
				_dtw.add_template(label, _feature_extractor.compute(_wav_to_pcm(audio)))


## Runs keyword spotting on a full WAV clip. Returns the recognized label, or
## "" if nothing matched. Also emits keyword_recognized / no_keyword_recognized.
func recognize_wav(stream: AudioStreamWAV) -> String:
	if settings == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return ""
	var resampled_stream = linear_resample(stream, settings.sample_rate)
	return recognize_pcm(_wav_to_pcm(resampled_stream))


## Runs keyword spotting on raw, already-decoded PCM samples (mono float
## samples, at the sample rate configured on [member settings]).
## Make sure the sample rate (or mix rate) is correct before calling this function
func recognize_pcm(pcm: PackedFloat32Array) -> String:
	if _dtw == null or _feature_extractor == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return ""
	var feature: PackedFloat32Array = _feature_extractor.compute(pcm)

	var label: String = _dtw.classify(feature)

	if label.is_empty():
		no_keyword_recognized.emit()
	else:
		keyword_recognized.emit(label)
	return label


## Instead of returning only the best label, it returns a dictionary with scores for each label
## A bit less performant but can be used for debugging or specific features
func recognize_wav_detailed(stream: AudioStreamWAV) -> Dictionary:
	if settings == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return {}
	var resampled_stream = linear_resample(stream, settings.sample_rate)
	return recognize_pcm_detailed(_wav_to_pcm(resampled_stream))


func recognize_pcm_detailed(pcm: PackedFloat32Array) -> Dictionary:
	if _dtw == null or _feature_extractor == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return {}
	var feature: Array[PackedFloat32Array] = _feature_extractor.compute(pcm)
	var scores: Dictionary = _dtw.classify_with_every_score(feature)
	return scores

## Resamples [param input]'s PCM data from [param original_rate] to
## [param new_rate] using linear interpolation between neighboring samples,
## so playback duration/pitch is preserved (unlike just changing mix_rate).
## Only supports 16-bit PCM AudioStreamWAV data, matching [method _wav_to_pcm].
## Loop points (if any) are rescaled to match the new frame count.
static func linear_resample(input: AudioStreamWAV, new_rate: int) -> AudioStreamWAV:
	var output: AudioStreamWAV = input.duplicate(true)
	if input.mix_rate == new_rate:
		return output
	output.set_mix_rate(new_rate)
	if input.format != AudioStreamWAV.FORMAT_16_BITS:
		push_error("KeywordMatcher.resample only supports 16-bit PCM AudioStreamWAV data (format=%d given). Convert the source WAV to 16-bit PCM first." % input.format)
		return output

	var channels: int = 2 if input.stereo else 1
	var bytes_per_frame: int = 2 * channels
	var data: PackedByteArray = input.data
	var frame_count: int = data.size() / bytes_per_frame

	if frame_count <= 1 or input.mix_rate <= 0 or new_rate <= 0:
		return output

	# Decode interleaved 16-bit PCM into per-channel float buffers.
	var channel_samples: Array[PackedFloat32Array] = []
	for c in channels:
		var samples := PackedFloat32Array()
		samples.resize(frame_count)
		channel_samples.append(samples)

	for frame in frame_count:
		for c in channels:
			var byte_index: int = (frame * channels + c) * 2
			var sample: int = int(data[byte_index]) | (int(data[byte_index + 1]) << 8)
			if sample >= 32768:
				sample -= 65536
			channel_samples[c][frame] = sample / 32768.0

	# Preserve duration: out_frame_count / new_rate == frame_count / original_rate.
	var ratio: float = float(new_rate) / float(input.mix_rate)
	var out_frame_count: int = max(1, int(round(frame_count * ratio)))
	var step: float = float(input.mix_rate) / float(new_rate) # input frames per output frame

	var out_data := PackedByteArray()
	out_data.resize(out_frame_count * bytes_per_frame)

	for out_frame in out_frame_count:
		var src_pos: float = out_frame * step
		var src_index: int = min(int(floor(src_pos)), frame_count - 1)
		var src_index_next: int = min(src_index + 1, frame_count - 1)
		var frac: float = src_pos - src_index

		for c in channels:
			var s0: float = channel_samples[c][src_index]
			var s1: float = channel_samples[c][src_index_next]
			var interpolated: float = s0 + (s1 - s0) * frac

			var sample_int: int = int(round(clamp(interpolated, -1.0, 1.0) * 32767.0))
			var byte_index: int = (out_frame * channels + c) * 2
			out_data[byte_index] = sample_int & 0xFF
			out_data[byte_index + 1] = (sample_int >> 8) & 0xFF

	output.data = out_data

	# Loop points are stored as frame indices; rescale so loops still land
	# on the same relative position in the resampled audio.
	if input.loop_mode != AudioStreamWAV.LOOP_DISABLED:
		output.loop_begin = int(round(input.loop_begin * ratio))
		output.loop_end = int(round(input.loop_end * ratio))

	return output


func _wav_to_pcm(stream: AudioStreamWAV) -> PackedFloat32Array:
	var data := stream.data # PackedByteArray (16-bit LE stereo/mono)
	var out := PackedFloat32Array()
	var i := 0
	while i + 1 < data.size():
		var sample := int(data[i]) | (int(data[i + 1]) << 8)
		if sample >= 32768: sample -= 65536
		out.append(sample / 32768.0)
		i += 2 * (2 if stream.stereo else 1) # skip right channel if stereo
	return out

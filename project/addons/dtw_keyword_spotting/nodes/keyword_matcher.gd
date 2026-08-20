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

## used to debug, create an audio stream from a pcm
func _make_playable_stream(pcm: PackedFloat32Array, sample_rate: int) -> AudioStreamWAV:
	var stream: AudioStreamWAV = AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.mix_rate = sample_rate
	stream.stereo = false

	var bytes: PackedByteArray = PackedByteArray()
	bytes.resize(pcm.size() * 2)
	for i in pcm.size():
		var clamped: float = clamp(pcm[i], -1.0, 1.0)
		var value: int = int(round(clamped * 32767.0))
		bytes[i * 2] = value & 0xFF
		bytes[i * 2 + 1] = (value >> 8) & 0xFF
	stream.data = bytes
	return stream


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
		var clip_index := 0
		for audio in settings.keyword_dataset[label]:
			if audio is AudioStreamWAV:
				var resampled_audio := linear_resample(audio, settings.sample_rate)
				var pcm := _trim_silence(_wav_to_pcm(resampled_audio), settings.sample_rate)
				clip_index += 1

				_dtw.add_template(label, _feature_extractor.compute(pcm))


## Runs keyword spotting on a full WAV clip. Returns the recognized label, or
## "" if nothing matched. Also emits keyword_recognized / no_keyword_recognized.
func recognize_wav(stream: AudioStreamWAV) -> String:
	if settings == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return ""
	if stream ==null :
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
	var trimmed: PackedFloat32Array = _trim_silence(pcm, settings.sample_rate)
	var feature: Array[PackedFloat32Array] = _feature_extractor.compute(trimmed)

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
	if stream ==null :
		return {}
	var resampled_stream = linear_resample(stream, settings.sample_rate)
	return recognize_pcm_detailed(_wav_to_pcm(resampled_stream))


func recognize_pcm_detailed(pcm: PackedFloat32Array) -> Dictionary:
	if _dtw == null or _feature_extractor == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return {}
	var trimmed: PackedFloat32Array = _trim_silence(pcm, settings.sample_rate)
	var feature: Array[PackedFloat32Array] = _feature_extractor.compute(trimmed)
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


## Strips leading/trailing silence based on short-window RMS energy relative
## to the clip's peak RMS. Without this, both DTW distance and PNCC's global
## power normalization end up dominated by however much silence padding a
## given recording happens to have, which makes matching unstable and biases
## results toward whichever templates happen to have a similar
## silence-to-speech ratio rather than the closest-sounding word.
## window_ms/threshold_ratio are conservative defaults; tune to your mic/noise floor.
static func _trim_silence(pcm: PackedFloat32Array, sample_rate: int, window_ms: float = 20.0, threshold_ratio: float = 0.05) -> PackedFloat32Array:
	if pcm.is_empty():
		return pcm

	var window: int = max(1, int(round(window_ms * sample_rate / 1000.0)))
	var num_windows: int = int(ceil(float(pcm.size()) / window))
	var rms := PackedFloat32Array()
	rms.resize(num_windows)

	var peak_rms := 0.0
	for w in num_windows:
		var start: int = w * window
		var end: int = min(start + window, pcm.size())
		var sum_sq := 0.0
		for i in range(start, end):
			sum_sq += pcm[i] * pcm[i]
		var window_rms: float = sqrt(sum_sq / float(end - start))
		rms[w] = window_rms
		if window_rms > peak_rms:
			peak_rms = window_rms

	if peak_rms <= 0.0:
		return pcm # fully silent clip, nothing sensible to trim

	var threshold: float = peak_rms * threshold_ratio

	var first_window := -1
	for w in num_windows:
		if rms[w] >= threshold:
			first_window = w
			break
	if first_window == -1:
		return pcm # nothing above threshold, leave as-is

	var last_window := 0
	for w in range(num_windows - 1, -1, -1):
		if rms[w] >= threshold:
			last_window = w
			break

	var start_sample: int = first_window * window
	var end_sample: int = min((last_window + 1) * window, pcm.size())
	return pcm.slice(start_sample, end_sample)


func _wav_to_pcm(stream: AudioStreamWAV) -> PackedFloat32Array:
	if stream.format != AudioStreamWAV.FORMAT_16_BITS:
		push_error("KeywordMatcher._wav_to_pcm only supports 16-bit PCM (format=%d given%s). Select the .wav in the FileSystem dock -> Import tab and set it to uncompressed 16-bit PCM, then Reimport." % [
			stream.format,
			(" for " + stream.resource_path) if stream.resource_path != "" else ""
		])
		return PackedFloat32Array()

	var data := stream.data # PackedByteArray (16-bit LE stereo/mono)
	var out := PackedFloat32Array()
	var i := 0
	while i + 1 < data.size():
		var sample := int(data[i]) | (int(data[i + 1]) << 8)
		if sample >= 32768: sample -= 65536
		out.append(sample / 32768.0)
		i += 2 * (2 if stream.stereo else 1) # skip right channel if stereo
	return out

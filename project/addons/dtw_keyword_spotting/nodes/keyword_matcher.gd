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
				_dtw.add_template(label, _feature_extractor.compute(settings.wav_to_pcm(audio)))


## Runs keyword spotting on a full WAV clip. Returns the recognized label, or
## "" if nothing matched. Also emits keyword_recognized / no_keyword_recognized.
func recognize_wav(stream: AudioStreamWAV) -> String:
	if settings == null:
		push_error("KeywordMatcher is not ready: assign a KeywordMatcherSettings resource first.")
		return ""
	return recognize_pcm(settings.wav_to_pcm(stream))


## Runs keyword spotting on raw, already-decoded PCM samples (mono float
## samples, at the sample rate configured on [member settings]).
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

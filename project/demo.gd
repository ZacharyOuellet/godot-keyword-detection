extends Control

@onready var keyword_matcher: KeywordMatcher = $"KeywordMatcher"
@onready var result_label: Label = $CompareContainer/DistanceLabel

func _on_compare_request(audioStream: AudioStreamWAV) -> void:
	_display_result(keyword_matcher.recognize_wav_detailed(audioStream))

func _display_result(result: Dictionary) -> void:
	var text: String = ""

	var max_score: float = 0.0
	var min_score: float = INF
	var score_sum: float = 0.0
	for label: String in result:
		var score: float = result[label]
		score_sum += score
		if score < min_score: min_score = score
		if score > max_score: max_score = score

	text += "AVG : {0}\nMIN : {1}\nMAX : {2}\n".format([score_sum / result.size(), min_score, max_score])

	_sort_dict(result)
	for label: String in result:
		var score: float = result[label]
		text += "{0} : {1}\n".format([label, score])

	result_label.text = text


func _sort_dict(dict: Dictionary) -> void:
	var pairs: Array = dict.keys().map(func(key: String) -> Array: return [key, dict[key]])
	pairs.sort_custom(func(a: Array, b: Array) -> bool:
		return a[1] < b[1]
	)
	dict.clear()
	for p: Array in pairs:
		dict[p[0]] = p[1]

func set_matching_mode(classify_method: DTWMatcher.ClassifyMethod) -> void:
	keyword_matcher.settings.classification_method = classify_method

func set_distance_metric(distance_metric: DTWMatcher.DistanceMetric) -> void:
	keyword_matcher.settings.distance_metric = distance_metric

func set_feature_extractor(feature_extractor: KeywordMatcherSettings.FeatureExtractionMethod) -> void:
	keyword_matcher.settings.feature_extraction_method = feature_extractor

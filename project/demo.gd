extends Control

@onready var keyword_matcher: KeywordMatcher = $"KeywordMatcher"


func _on_compare_request(audioStream: AudioStreamWAV):
	_display_result(keyword_matcher.recognize_wav_detailed(audioStream))

func _display_result(result: Dictionary):
	var text: String = ""

	var max_score := 0.0
	var min_score := INF
	var score_sum := 0.0
	for label in result:
		var score: float = result[label]
		score_sum += score
		if score < min_score: min_score = score
		if score > max_score: max_score = score

	text += "AVG : {0}\nMIN : {1}\nMAX : {2}\n".format([score_sum / result.size(), min_score, max_score])

	_sort_dict(result)
	for label in result:
		var score: float = result[label]
		text += "{0} : {1}\n".format([label, score])

	$CompareContainer/DistanceLabel.text = text


func _sort_dict(dict: Dictionary) -> void:
	var pairs = dict.keys().map(func(key): return [key, dict[key]])
	pairs.sort_custom(func(a, b):
		return a[1] < b[1]
	)
	dict.clear()
	for p in pairs:
		dict[p[0]] = p[1]

func set_matching_mode(classify_method: DTWMatcher.ClassifyMethod):
	keyword_matcher.settings.classification_method = classify_method

func set_distance_metric(distance_metric: DTWMatcher.DistanceMetric):
	keyword_matcher.settings.distance_metric = distance_metric

func set_feature_extractor(feature_extractor: KeywordMatcherSettings.FeatureExtractionMethod):
	keyword_matcher.settings.feature_extraction_method = feature_extractor

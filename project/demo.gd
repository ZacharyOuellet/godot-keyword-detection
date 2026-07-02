extends Control

func _on_compare_request(audioStream: AudioStreamWAV):
	var mfcc := MFCCProcessor.new()
	var matcher := DTWMatcher.new()
	for templatePath: String in $SamplesPanel.get_all_file_paths():
		matcher.add_template(templatePath.split("/")[-1], mfcc.compute(_wav_to_pcm(AudioStreamWAV.load_from_file(templatePath))))

	var arr = mfcc.compute(_wav_to_pcm(audioStream))
	_display_result(matcher.classify_with_every_score(arr))


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

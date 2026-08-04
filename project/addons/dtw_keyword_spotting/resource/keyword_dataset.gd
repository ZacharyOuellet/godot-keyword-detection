@tool
class_name KeywordDataset
extends Resource

# No more WavList wrapper needed — the custom inspector plugin (see
# addons/keyword_dataset_inspector) renders this dictionary directly as
# label -> [AudioStreamWAV, AudioStreamWAV, ...] rows.
@export var source_files: Dictionary = {}

@export_storage var baked_features: Dictionary = {}


func _init() -> void:
	if Engine.is_editor_hint():
		if self not in DtwKeywordSpotting.open_datasets:
			DtwKeywordSpotting.open_datasets.append(self)


func bake_features() -> void:
	var extractor = PNCCProcessor.new()
	baked_features.clear()
	for label in source_files.keys():
		var feature_list: Array[PackedFloat32Array] = []
		for audio in source_files[label]:
			if audio is AudioStreamWAV:
				feature_list.append(extractor.compute(_wav_to_pcm(audio)))
		baked_features[label] = feature_list
	var err := ResourceSaver.save(self, resource_path)
	if err != OK:
		push_error("Failed to save baked dataset: %s" % err)


func _get_content_hash() -> int:
	var hash_input := {}
	for label in source_files.keys():
		var entries := []
		for stream in source_files[label]:
			var path: String = stream.resource_path if stream is Resource else str(stream)
			var mtime := FileAccess.get_modified_time(path)
			entries.append([path, mtime])
		hash_input[label] = entries
	return hash(hash_input)


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

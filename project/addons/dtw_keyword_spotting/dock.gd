@tool
extends Control

@export var sample_path_scene: PackedScene = preload("res://addons/dtw_keyword_spotting/editor/components/sample_path.tscn")

var _stream_being_saved: AudioStreamWAV = null

func get_all_file_paths_and_labels() -> Array[Dictionary]:
	var paths: Array[Dictionary] = []
	for node in %SampleContainer.get_children():
		paths.append({"label": node.label, "path": node.path})
	return paths

func _on_load_from_file() -> void:
	$FileDialog.popup_file_dialog()

func _on_file_selected(path: String) -> void:
	add_to_list(path)

func _on_files_selected(paths: PackedStringArray) -> void:
	for path in paths:
		add_to_list(path)

func _on_dir_selected(dir_path: String) -> void:
	var dir = DirAccess.open(dir_path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()

		while file_name != "":
			# Ignore hidden system navigation items
			if file_name == "." or file_name == "..":
				file_name = dir.get_next()
				continue
			if !dir.current_is_dir() and file_name.ends_with(".wav"):
				var file_path = dir_path.path_join(file_name)
				add_to_list(file_path)

			file_name = dir.get_next()
		dir.list_dir_end()
	else:
		push_error("Failed to access path: ", dir_path)


func _on_recording_confirmed(audioStream: AudioStreamWAV):
	$SaveFileDialog.popup()
	_stream_being_saved = audioStream


func _on_save_location_chosen(path: String):
	_stream_being_saved.save_to_wav(path)
	add_to_list(path)

func _on_cancel_save():
	_stream_being_saved = null

func add_to_list(filepath: String):
	for node in %LabelContainer.get_children():
		if "text" in node and node.path == filepath: return # No duplicates
	var node := sample_path_scene.instantiate()
	%LabelContainer.add_child(node)

func _await_one_of_signals(signalsArray: Array[Signal]):
	var wrapper: RefCounted = RefCounted.new()
	wrapper.add_user_signal("one_triggered")

	var handler = func():
		if wrapper.has_user_signal("triggered"):
			wrapper.emit_signal("triggered")
	for awaitedSignal in signalsArray:
		awaitedSignal.connect(handler)

	await wrapper.triggered

	for awaitedSignal in signalsArray:
		awaitedSignal.disconnect(handler)


func on_export():
	print("a")

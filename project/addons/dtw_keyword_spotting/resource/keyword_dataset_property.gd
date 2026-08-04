@tool
extends EditorProperty
## Custom inspector widget for KeywordDataset.source_files.
## Renders: [label name] [wav picker] [wav picker] [+] ... as compact rows,
## with no need to create a separate WavList resource by hand.

var _rows_container: VBoxContainer
var _updating := false # guards against re-entrant rebuilds while we're the one writing


func _init() -> void:
	_rows_container = VBoxContainer.new()
	add_child(_rows_container)
	add_focusable(_rows_container)
	set_bottom_editor(_rows_container)


func _update_property() -> void:
	if _updating:
		return
	_rebuild()


func _rebuild() -> void:
	for c in _rows_container.get_children():
		c.queue_free()

	var dict: Dictionary = _get_dict()

	for label in dict.keys():
		_rows_container.add_child(_build_label_section(label, dict[label]))

	var add_label_btn := Button.new()
	add_label_btn.text = "+ Add Keyword Label"
	add_label_btn.pressed.connect(_on_add_label_pressed)
	_rows_container.add_child(add_label_btn)


func _get_dict() -> Dictionary:
	var v = get_edited_object().get(get_edited_property())
	return v if v is Dictionary else {}


# ---------------------------------------------------------------------------
# Section: one label + its list of wav files
# ---------------------------------------------------------------------------

func _build_label_section(label: String, wavs: Array) -> Control:
	var panel := PanelContainer.new()
	var vbox := VBoxContainer.new()
	panel.add_child(vbox)

	var header := HBoxContainer.new()
	header.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var label_edit := LineEdit.new()
	label_edit.text = label
	label_edit.custom_minimum_size.x = 140
	label_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	label_edit.text_submitted.connect(func(new_text): _on_label_renamed(label, new_text, label_edit))
	header.add_child(label_edit)

	var count_label := Label.new()
	count_label.text = "%d wav(s)" % wavs.size()
	count_label.modulate.a = 0.6
	header.add_child(count_label)

	var remove_label_btn := Button.new()
	remove_label_btn.text = "Remove Label"
	remove_label_btn.pressed.connect(_on_remove_label.bind(label))
	header.add_child(remove_label_btn)

	vbox.add_child(header)

	for i in wavs.size():
		vbox.add_child(_build_wav_row(label, i, wavs[i]))

	var add_wav_btn := Button.new()
	add_wav_btn.text = "+ Add WAV slot"
	add_wav_btn.pressed.connect(_on_add_wav.bind(label))
	vbox.add_child(add_wav_btn)

	# Accept drag-and-drop of .wav resources straight from the FileSystem dock
	# onto this section to append them.
	var drop_target := _WavDropTarget.new()
	drop_target.on_files_dropped = func(paths: Array): _on_wavs_dropped(label, paths)
	drop_target.custom_minimum_size.y = 18
	var drop_hint := Label.new()
	drop_hint.text = "(drop .wav files here)"
	drop_hint.modulate.a = 0.4
	drop_target.add_child(drop_hint)
	vbox.add_child(drop_target)

	return panel


func _build_wav_row(label: String, index: int, stream: AudioStreamWAV) -> Control:
	var row := HBoxContainer.new()

	var picker := EditorResourcePicker.new()
	picker.base_type = "AudioStreamWAV"
	picker.edited_resource = stream
	picker.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	picker.resource_changed.connect(_on_wav_changed.bind(label, index))
	row.add_child(picker)

	var remove_btn := Button.new()
	remove_btn.text = "x"
	remove_btn.pressed.connect(_on_remove_wav.bind(label, index))
	row.add_child(remove_btn)

	return row


# ---------------------------------------------------------------------------
# Mutation handlers — each clones the dict, edits the clone, then commits it
# ---------------------------------------------------------------------------

func _commit(dict: Dictionary) -> void:
	_updating = true
	emit_changed(get_edited_property(), dict)
	_updating = false
	_rebuild()


func _on_add_label_pressed() -> void:
	var dict := _get_dict().duplicate(true)
	var base_name := "new_label"
	var name := base_name
	var i := 1
	while dict.has(name):
		name = "%s_%d" % [base_name, i]
		i += 1
	dict[name] = []
	_commit(dict)


func _on_remove_label(label: String) -> void:
	var dict := _get_dict().duplicate(true)
	dict.erase(label)
	_commit(dict)


func _on_label_renamed(old_name: String, new_name: String, edit: LineEdit) -> void:
	if new_name == old_name or new_name.is_empty():
		edit.text = old_name
		return
	var dict := _get_dict().duplicate(true)
	if dict.has(new_name):
		push_warning("Label '%s' already exists." % new_name)
		edit.text = old_name
		return
	dict[new_name] = dict[old_name]
	dict.erase(old_name)
	_commit(dict)


func _on_add_wav(label: String) -> void:
	var dict := _get_dict().duplicate(true)
	var arr: Array = (dict[label] as Array).duplicate()
	arr.append(null)
	dict[label] = arr
	_commit(dict)


func _on_remove_wav(label: String, index: int) -> void:
	var dict := _get_dict().duplicate(true)
	var arr: Array = (dict[label] as Array).duplicate()
	arr.remove_at(index)
	dict[label] = arr
	_commit(dict)


func _on_wav_changed(stream: Resource, label: String, index: int) -> void:
	var dict := _get_dict().duplicate(true)
	var arr: Array = (dict[label] as Array).duplicate()
	arr[index] = stream
	dict[label] = arr
	_commit(dict)


func _on_wavs_dropped(label: String, paths: Array) -> void:
	var dict := _get_dict().duplicate(true)
	var arr: Array = (dict[label] as Array).duplicate()
	for path in paths:
		if String(path).get_extension().to_lower() == "wav":
			var res := load(path)
			if res is AudioStreamWAV:
				arr.append(res)
	dict[label] = arr
	_commit(dict)


# ---------------------------------------------------------------------------
# Small helper control: a drop zone that accepts FileSystem dock drags
# ---------------------------------------------------------------------------

class _WavDropTarget extends PanelContainer:
	var on_files_dropped: Callable

	func _can_drop_data(_pos: Vector2, data: Variant) -> bool:
		return typeof(data) == TYPE_DICTIONARY and data.get("type") == "files"

	func _drop_data(_pos: Vector2, data: Variant) -> void:
		if on_files_dropped.is_valid():
			on_files_dropped.call(data.get("files", []))

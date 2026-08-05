@tool
extends EditorProperty
## Custom inspector widget for KeywordDataset.source_files.
## Renders: [label name] [wav picker] [wav picker] [+] ... as compact rows,
## with no need to create a separate WavList resource by hand.

var _rows_container: VBoxContainer
var _updating := false # guards against re-entrant rebuilds while we're the one writing


func _init() -> void:
	_rows_container = VBoxContainer.new()
	_rows_container.add_theme_constant_override("separation", 0)
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
	var labels := dict.keys()

	for idx in labels.size():
		var label = labels[idx]
		_rows_container.add_child(_build_label_section(label, dict[label]))
		_rows_container.add_child(_build_separator())

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
	var row := HBoxContainer.new()

	var remove_label_btn := Button.new()
	remove_label_btn.icon = get_theme_icon("Remove", "EditorIcons")
	remove_label_btn.size_flags_horizontal = Control.SIZE_FILL
	remove_label_btn.tooltip_text = "Remove this label and all its wavs"
	remove_label_btn.pressed.connect(_on_remove_label.bind(label))
	row.add_child(remove_label_btn)


	var left := VBoxContainer.new()
	left.custom_minimum_size.x = 250
	left.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(left)

	var label_edit := LineEdit.new()
	label_edit.text = label
	label_edit.text_submitted.connect(func(new_text): _on_label_renamed(label, new_text, label_edit))
	left.add_child(label_edit)

	var count_label := Label.new()
	count_label.text = "%d wav(s)" % wavs.size()
	count_label.modulate.a = 0.6
	count_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left.add_child(count_label)


	row.add_child(_build_separator(true))

	# --- Right: one row per wav, stacked vertically ---
	var wav_column := _WavColumnTarget.new()
	wav_column.on_files_dropped = func(paths: Array): _on_wavs_dropped(label, paths)
	wav_column.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	wav_column.add_theme_constant_override("separation", 2)
	row.add_child(wav_column)

	for i in wavs.size():
		wav_column.add_child(_build_wav_row(label, i, wavs[i]))

	var add_wav_btn := Button.new()
	add_wav_btn.text = "+ Add WAV slot"
	add_wav_btn.tooltip_text = "Or drop .wav files anywhere in this column"
	add_wav_btn.pressed.connect(_on_add_wav.bind(label))
	wav_column.add_child(add_wav_btn)

	return row

func _build_wav_row(label: String, index: int, stream: AudioStreamWAV) -> Control:
	var chip := HBoxContainer.new()
	chip.add_theme_constant_override("separation", 4)

	var name_label := Label.new()
	name_label.text = stream.resource_path.get_file() if stream else "(none)"
	name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	chip.add_child(name_label) # alongside the picker, before the remove button

	var picker := EditorResourcePicker.new()
	picker.base_type = "AudioStreamWAV"
	picker.edited_resource = stream
	picker.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	picker.resource_changed.connect(_on_wav_changed.bind(label, index))
	chip.add_child(picker)

	var remove_btn := Button.new()
	remove_btn.icon = get_theme_icon("Remove", "EditorIcons")
	remove_btn.pressed.connect(_on_remove_wav.bind(label, index))
	chip.add_child(remove_btn)

	return chip


func _build_separator(is_vertical:bool = false):
	var sep := VSeparator.new() if is_vertical else HSeparator.new()
	var sb := StyleBoxLine.new()
	sb.color = get_theme_color("font_color", "Label")
	sb.color.a = 0.3
	sb.thickness = 3
	sb.vertical = is_vertical
	sep.add_theme_stylebox_override("separator", sb)
	return sep

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
# Small helper control: a VBoxContainer that stacks wav rows one per line,
# and accepts FileSystem dock drags anywhere in the column.
# ---------------------------------------------------------------------------

class _WavColumnTarget extends VBoxContainer:
	var on_files_dropped: Callable

	func _can_drop_data(_pos: Vector2, data: Variant) -> bool:
		return typeof(data) == TYPE_DICTIONARY and data.get("type") == "files"

	func _drop_data(_pos: Vector2, data: Variant) -> void:
		if on_files_dropped.is_valid():
			on_files_dropped.call(data.get("files", []))

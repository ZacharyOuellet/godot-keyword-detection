@tool
extends EditorInspectorPlugin

const KeywordDatasetProperty := preload("res://addons/dtw_keyword_spotting/resource/keyword_dataset_property.gd")


func _can_handle(object: Object) -> bool:
	return object is KeywordDataset


func _parse_property(
	object: Object,
	type: Variant.Type,
	name: String,
	hint_type: PropertyHint,
	hint_string: String,
	usage_flags: int,
	wide: bool
) -> bool:
	if name != "source_files":
		return false

	var prop := KeywordDatasetProperty.new()
	add_property_editor(name, prop)
	return true # tells the inspector: don't draw the default editor for this property

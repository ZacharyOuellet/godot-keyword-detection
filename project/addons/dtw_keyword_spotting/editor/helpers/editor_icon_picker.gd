@tool
extends Button

@export var editor_icon_name: StringName:
	set(value):
		editor_icon_name = value
		_ready()

func _ready() -> void:
	icon = get_theme_icon(editor_icon_name, "EditorIcons")

@tool
class_name DtwKeywordSpotting
extends EditorPlugin


const InspectorPluginScript := preload("res://addons/dtw_keyword_spotting/keyword_dataset_inspector_plugin.gd")
const KeywordMatcherScript := preload("res://addons/dtw_keyword_spotting/nodes/keyword_matcher.gd")
var inspector_plugin: EditorInspectorPlugin

func _enable_plugin() -> void:
	# Add autoloads here.
	pass

func _disable_plugin() -> void:
	# Remove autoloads here.
	pass

func _enter_tree() -> void:
	inspector_plugin = InspectorPluginScript.new()
	add_inspector_plugin(inspector_plugin)

	var icon := EditorInterface.get_base_control().get_theme_icon("AudioStreamPlayer", "EditorIcons")
	add_custom_type("KeywordMatcher", "Node", KeywordMatcherScript, preload("res://addons/dtw_keyword_spotting/assets/KeywordMatcher.svg"))


func _exit_tree() -> void:
	remove_inspector_plugin(inspector_plugin)
	inspector_plugin = null
	remove_custom_type("KeywordMatcher")

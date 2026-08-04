@tool
class_name DtwKeywordSpotting
extends EditorPlugin

static var open_datasets: Array[KeywordDataset] = []

const InspectorPluginScript := preload("res://addons/dtw_keyword_spotting/keyword_dataset_inspector_plugin.gd")
var inspector_plugin: EditorInspectorPlugin

func _save_external_data() -> void:
	for dataset in open_datasets:
		if is_instance_valid(dataset): # TODO
			print("Baking dataset : ", dataset)
			dataset.bake_features()

func _enable_plugin() -> void:
	# Add autoloads here.
	pass


func _disable_plugin() -> void:
	# Remove autoloads here.
	pass


func _enter_tree() -> void:
	inspector_plugin = InspectorPluginScript.new()
	add_inspector_plugin(inspector_plugin)

func _exit_tree() -> void:
	remove_inspector_plugin(inspector_plugin)
	inspector_plugin = null

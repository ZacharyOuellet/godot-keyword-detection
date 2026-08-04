@tool
class_name DtwKeywordSpotting
extends EditorPlugin

var dock
static var open_datasets: Array[KeywordDataset] = []

const InspectorPluginScript := preload("res://addons/dtw_keyword_spotting/keyword_dataset_inspector_plugin.gd")
var inspector_plugin: EditorInspectorPlugin

func _save_external_data() -> void:
	for dataset in open_datasets:
		if is_instance_valid(dataset) and dataset._dirty:
			print("Baking dataset : ", dataset)
			dataset.bake_features()

func _enable_plugin() -> void:
	# Add autoloads here.
	pass


func _disable_plugin() -> void:
	# Remove autoloads here.
	pass


func _enter_tree() -> void:
	dock = EditorDock.new()
	dock.title = "Keyword library maker"
	dock.default_slot = EditorDock.DOCK_SLOT_LEFT_UR
	var dock_content = preload("res://addons/dtw_keyword_spotting/dock.tscn").instantiate()
	dock.add_child(dock_content)
	add_dock(dock)

	inspector_plugin = InspectorPluginScript.new()
	add_inspector_plugin(inspector_plugin)

func _exit_tree() -> void:
	remove_dock(dock)
	dock.queue_free()
	dock = null

	remove_inspector_plugin(inspector_plugin)
	inspector_plugin = null



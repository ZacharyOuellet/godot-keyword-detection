@tool
extends Control

signal play_requested(wav: AudioStreamWAV)

var path: String:
	get():
		return $Path.text
	set(value):
		$Path.text = value
		audio = AudioStreamWAV.load_from_file(value)

var label: String:
	get():
		if ($Label.text):
			return $Label.text
		else:
			return $Path.text.split("/")[-1]
	set(value):
		$Label.text = value

var audio: AudioStreamWAV

func _ready() -> void:
	label = label

func _on_delete_pressed():
	queue_free()

func _on_play_pressed():
	play_requested.emit(audio)

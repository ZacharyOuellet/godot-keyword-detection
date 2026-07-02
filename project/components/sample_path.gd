@tool
extends Control

signal play_requested(wav: AudioStreamWAV)

var text: String:
	get():
		return $Label.text
	set(value):
		$Label.text = value
		audio = AudioStreamWAV.load_from_file(value)

var audio: AudioStreamWAV

func _on_delete_pressed():
	queue_free()

func _on_play_pressed():
	play_requested.emit(audio)

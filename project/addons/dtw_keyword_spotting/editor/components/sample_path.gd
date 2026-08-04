@tool
extends Control

var audio: AudioStreamWAV

var path: String:
	get():
		return $Path.text
	set(value):
		$Path.text = value
		audio = AudioStreamWAV.load_from_file(value)


@onready var audioStreamPlayer: AudioStreamPlayer = $"AudioStreamPlayer"

func _on_delete_pressed():
	queue_free()

func _on_play_pressed():
	audioStreamPlayer.stream = audio
	audioStreamPlayer.play()

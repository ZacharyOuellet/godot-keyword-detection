@tool
extends Control

signal recording_confirmed(audioStream: AudioStreamWAV)
signal recording_started()
signal recording_stopped()
signal recording_cancelled()

@export var default_label_text = "Record":
	get:
		return default_label_text
	set(value):
		$Label.text = value
		default_label_text = value

@export var confirm_button_texture: Texture2D:
	get:
		return confirm_button_texture
	set(value):
		confirm_button_texture = value
		$Buttons/Save.icon = confirm_button_texture

var recordEffect: AudioEffectRecord
var recording: AudioStreamWAV:
	get:
		return recording
	set(value):
		var exists := value != null
		$Buttons/Record.disabled = exists
		$Buttons/Play.disabled = !exists
		$Buttons/Save.disabled = !exists
		$Buttons/Cancel.disabled = !exists
		recording = value

var stereo: bool = true
var mix_rate := 44100 # This is the default mix rate on recordings\
var format := AudioStreamWAV.FORMAT_16_BITS # This is the default format on recordings.

func _ready() -> void:
	var idx := AudioServer.get_bus_index("Record")
	recordEffect = AudioServer.get_bus_effect(idx, 0)
	recording = null
	$Label.text = default_label_text

func _on_record_button_pressed() -> void:
	if recordEffect.is_recording_active():
		recordEffect.set_recording_active(false)
		recording = recordEffect.get_recording()
		recording.set_mix_rate(mix_rate)
		recording.set_format(format)
		recording.set_stereo(stereo)
		$Label.text = default_label_text
		recording_stopped.emit()
	else:
		recordEffect.set_recording_active(true)
		$Label.text = "Recording..."
		recording_started.emit()

func _on_play_button_pressed() -> void:
	$AudioStreamOutput.stream = recording
	$AudioStreamOutput.play()

func _on_save_button_pressed() -> void:
	recording_confirmed.emit(recording)
	recording = null

func _on_cancel_button_pressed() -> void:
	recording_cancelled.emit()
	recording = null

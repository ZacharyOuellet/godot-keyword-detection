extends Control

signal recording_confirmed(audioStream: AudioStreamWAV)
signal recording_started()
signal recording_stopped()

var recordEffect: AudioEffectRecord
var recording: AudioStreamWAV:
	get:
		return recording
	set(value):
		var exists: bool = value != null
		record_button.disabled = exists
		play_button.disabled = !exists
		confirm_button.disabled = !exists
		recording = value

@onready var audio_output_player: AudioStreamPlayer = $AudioStreamOutput
@onready var record_button: Button = $Buttons/Record
@onready var play_button: Button = $Buttons/Play
@onready var confirm_button: Button = $Buttons/Confirm
@onready var label: Label = $Label

func _ready() -> void:
	var idx: int = AudioServer.get_bus_index("Record")
	recordEffect = AudioServer.get_bus_effect(idx, 0)
	recording = null

func _on_record_button_pressed() -> void:
	if recordEffect.is_recording_active():
		recordEffect.set_recording_active(false)
		recording = recordEffect.get_recording()
		label.text = "Record a wav"
		recording_stopped.emit()
	else:
		recordEffect.set_recording_active(true)
		label.text = "Recording..."
		recording_started.emit()

func _on_play_button_pressed() -> void:
	audio_output_player.stream = recording
	audio_output_player.play()

func _on_confirm_button_pressed() -> void:
	recording_confirmed.emit(recording)
	recording = null

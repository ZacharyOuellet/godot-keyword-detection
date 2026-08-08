extends Control

signal recording_confirmed(audioStream: AudioStreamWAV)
signal recording_started()
signal recording_stopped()

var recordEffect: AudioEffectRecord
var recording: AudioStreamWAV:
	get:
		return recording
	set(value):
		var exists := value != null
		$Buttons/Record.disabled = exists
		$Buttons/Play.disabled = !exists
		$Buttons/Confirm.disabled = !exists
		recording = value

# var stereo: bool = false
# @export var format := AudioStreamWAV.FORMAT_16_BITS

func _ready() -> void:
	var idx := AudioServer.get_bus_index("Record")
	recordEffect = AudioServer.get_bus_effect(idx, 0)
	recording = null

func _on_record_button_pressed() -> void:
	if recordEffect.is_recording_active():
		recordEffect.set_recording_active(false)
		recording = recordEffect.get_recording()
		# recording.set_format(format)
		# recording.set_stereo(stereo)
		recording_stopped.emit()
	else:
		recordEffect.set_recording_active(true)
		$Label.text = "Recording..."
		recording_started.emit()

func _on_play_button_pressed() -> void:
	$AudioStreamOutput.stream = recording
	$AudioStreamOutput.play()

func _on_confirm_button_pressed() -> void:
	recording_confirmed.emit(recording)
	recording = null

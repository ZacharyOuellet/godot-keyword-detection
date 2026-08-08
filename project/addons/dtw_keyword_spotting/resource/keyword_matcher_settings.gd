@tool
class_name KeywordMatcherSettings
extends Resource

enum FeatureExtractionMethod
{
	MFCC,
	PNCC
}

@export_group("DTW settings")
## When measuring distances between two feature vectors, it can be linear (euclidean) or the cosine
## My tests showed that the euclidean distance was better for most languages
@export var distance_metric: DTWMatcher.DistanceMetric = DTWMatcher.DistanceMetric.EUCLIDEAN
## When you have multiple templates for the same label.
## You can match by averaging distances or only base your decision on the best score for each label.
@export var classification_method: DTWMatcher.ClassifyMethod = DTWMatcher.ClassifyMethod.BEST_MATCH
## The relative width of the Sakoe-Chiba band.
## 1 means the full width
## 0 means we compare only the diagonals
@export_range(0, 1, 0.01) var sakoe_chiba_band_width: float = 0.2

@export_group("Feature extraction")
@export var feature_extraction_method: FeatureExtractionMethod = FeatureExtractionMethod.PNCC
## [b][color=orange]WARNING:[/color][/b] The sample rate setting here must be the same has the one in the .wav templates.
## Without it, unexpected inaccuracies could happen
@export_custom(PROPERTY_HINT_NONE, "suffix:Hz") var sample_rate: int = 44100
## More coefficients make DTW more sensitive to fine spectral differences, while fewer coefficients result in a coarser and potentially more robust comparison.
@export var num_coeffs: int = 13
## The number of audio samples contained in each feature extraction frame.
## Larger frames provide better frequency resolution but lower temporal resolution.
@export var frame_length: int = 512
## The number of samples between consecutive feature extraction frames.
## Smaller values produce more overlapping frames and provide finer temporal resolution.
@export var hop_length: int = 256

@export_subgroup("MFCC specific")
## The number of Mel filter banks used to represent the frequency spectrum before computing the MFCCs.
## More Mel bands provide a finer representation of the spectral envelope.
@export var num_mel_bands: int = 40

@export_subgroup("PNCC specific")
## The number of frequency bands used to represent the power spectrum in the PNCC extraction.
## More bands provide a finer spectral representation.
@export var num_gamma_bands: int = 40
## Controls the compression applied to the power spectrum.
## Lower values reduce the influence of large amplitude variations.
@export var power_law_exponent: float = 1.0 / 15.0
## Controls the length of the medium-time integration window used by PNCC.
@export var medium_time_frame: int = 2
@export_subgroup("PNCC specific/Asymmetric noise suppression")
## Controls the adaptation rate for the asymmetric noise suppression.
## Higher values make the noise estimate adapt more slowly.
@export var lambda_a: float = 0.999
## Controls the strength of the asymmetric noise suppression.
@export var lambda_b: float = 0.5
@export_subgroup("PNCC specific/Temporal masking")
## Controls the temporal masking applied during PNCC extraction.
## Higher values give more weight to the temporal masking effect.
@export var lambda_t: float = 0.85


@export_group("Words and labels")
@export var keyword_dataset: Dictionary

var dirty := false


func create_dtw_matcher() -> DTWMatcher:
	var dtw = DTWMatcher.new()
	dtw.distance_metric = distance_metric
	dtw.band_width = sakoe_chiba_band_width
	dtw.classify_method = classification_method
	return dtw


func create_feature_extractor() -> Variant:
	match feature_extraction_method:
		FeatureExtractionMethod.MFCC:
			return _create_mfcc_processor()
		FeatureExtractionMethod.PNCC:
			return _create_pncc_processor()
		_:
			push_error("feature extraction invalid, defaulting to mfcc")
			return _create_mfcc_processor()


func _create_pncc_processor() -> PNCCProcessor:
	var pncc = PNCCProcessor.new()
	pncc.num_gamma_bands = num_gamma_bands
	pncc.power_law_exponent = power_law_exponent
	pncc.medium_time_frames = medium_time_frame
	pncc.sample_rate = sample_rate
	pncc.num_coeffs = num_coeffs
	pncc.frame_length = frame_length
	pncc.hop_length = hop_length
	return pncc


func _create_mfcc_processor() -> MFCCProcessor:
	var mfcc = MFCCProcessor.new()
	mfcc.num_mel_bands = num_mel_bands
	mfcc.sample_rate = sample_rate
	mfcc.num_coeffs = num_coeffs
	mfcc.frame_length = frame_length
	mfcc.hop_length = hop_length
	return mfcc

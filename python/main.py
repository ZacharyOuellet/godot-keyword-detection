import numpy as np
import keyword_detection as kd

mfcc = kd.MFCCProcessor()
mfcc.sample_rate = 16000
mfcc.num_coeffs = 13

samples = np.random.uniform(-1, 1, 16000).astype(np.float32)
features = mfcc.compute(samples)          # (n_frames, 13) float32 array

matcher = kd.DTWMatcher()
matcher.distance_metric = kd.DTWMatcher.DistanceMetric.COSINE
matcher.classify_method = kd.DTWMatcher.ClassifyMethod.BEST_MATCH
matcher.add_template("hello", features)

result = matcher.classify_with_best_score(features)
print(result)   # {'label': 'hello', 'distance': 0.0}

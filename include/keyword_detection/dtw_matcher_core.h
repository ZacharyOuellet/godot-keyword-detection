#pragma once

#include <vector>
#include <map>
#include <string>

using FeatureSequence = std::vector<std::vector<float>>;

class DTWMatcherCore {
public:
    enum DistanceMetric {
        EUCLIDEAN = 0,
        COSINE = 1,
    };

    enum ClassifyMethod {
        AVERAGE_DISTANCE = 0,
        BEST_MATCH = 1,
    };

    struct ClassifyResult {
        std::string label;
        float distance;
    };

    DTWMatcherCore();
    ~DTWMatcherCore();

    // --- Distance metric ---
    void set_distance_metric(int metric);
    int  get_distance_metric() const;

    // --- Sakoe-Chiba band width (0 = no constraint) ---
    void set_band_width(int width);
    int  get_band_width() const;

    void set_classify_method(int method);
    int  get_classify_method() const;

    // --- Core DTW ---
    // Input : two Array<PackedFloat32Array> (MFCC sequences)
    // Output: scalar DTW distance (lower = more similar)
    float compute(const std::vector<std::vector<float>>& seq_a,
        const std::vector<std::vector<float>>& seq_b) const;


    // --- Template matching helpers ---
    void   add_template(const std::string& label, const FeatureSequence& features);
    void   clear_templates();
    std::string classify(const FeatureSequence& features) const;

    // Returns a Dictionary { label: String, distance: float } for the best match
    ClassifyResult classify_with_best_score(const FeatureSequence& features) const;

    // Returns a Dictionary { label: String, distance: float } for every template
    std::vector<ClassifyResult> classify_with_every_score(const FeatureSequence& features) const;

private:
    float _frame_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float _euclidean(const std::vector<float>& a, const std::vector<float>& b) const;
    float _cosine(const std::vector<float>& a, const std::vector<float>& b) const;

    // Computes the distance between the input MFCC and all the clips with the given label
    float _computeAverageDistance(const std::string& label, const FeatureSequence& features) const;
    float _computeBestDistance(const std::string& label, const FeatureSequence& features) const;

    int _distance_metric = EUCLIDEAN;
    int _classify_method = AVERAGE_DISTANCE;
    int _band_width = 0; // 0 = full matrix (no Sakoe-Chiba constraint)


    std::map<std::string, std::vector<std::vector<std::vector<float>>>> _templates;
};

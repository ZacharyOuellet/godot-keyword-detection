// Python bindings for the keyword-detection core library.
// Mirrors the Godot GDExtension layer (mfcc_processor.cpp, pncc_processor.cpp,
// dtw_matcher.cpp, register_types.cpp) but targets pybind11 + NumPy instead
// of godot_cpp. The underlying core classes are untouched.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "keyword_detection/mfcc_processor_core.h"
#include "keyword_detection/pncc_processor_core.h"
#include "keyword_detection/dtw_matcher_core.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Conversion helpers (equivalent to to_feature_sequence / result_to_dictionary
// in the Godot dtw_matcher.cpp, but numpy <-> std::vector instead of
// TypedArray<PackedFloat32Array> <-> std::vector)
// ---------------------------------------------------------------------------

// 1D numpy float32 array -> std::vector<float>
static std::vector<float> array_to_vector(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& arr)
{
    if (arr.ndim() != 1)
        throw std::runtime_error("expected a 1D array of samples");

    const float* data = arr.data();
    return std::vector<float>(data, data + arr.shape(0));
}

// 2D numpy float32 array (n_frames, n_coeffs) -> vector<vector<float>>
static std::vector<std::vector<float>> array_to_feature_sequence(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& arr)
{
    if (arr.ndim() != 2)
        throw std::runtime_error("expected a 2D array shaped (n_frames, n_coeffs)");

    const auto n_frames = arr.shape(0);
    const auto n_coeffs = arr.shape(1);
    const float* data = arr.data();

    std::vector<std::vector<float>> result;
    result.reserve(n_frames);
    for (py::ssize_t i = 0; i < n_frames; ++i)
    {
        result.emplace_back(data + i * n_coeffs, data + (i + 1) * n_coeffs);
    }
    return result;
}

// vector<vector<float>> -> 2D numpy float32 array (n_frames, n_coeffs)
// (the reverse direction of the helper above; used for MFCC/PNCC::compute output)
static py::array_t<float> feature_sequence_to_array(
    const std::vector<std::vector<float>>& seq)
{
    const py::ssize_t n_frames = static_cast<py::ssize_t>(seq.size());
    const py::ssize_t n_coeffs = n_frames > 0 ? static_cast<py::ssize_t>(seq[0].size()) : 0;

    py::array_t<float> result({ n_frames, n_coeffs });
    float* out = result.mutable_data();
    for (py::ssize_t i = 0; i < n_frames; ++i)
    {
        std::copy(seq[i].begin(), seq[i].end(), out + i * n_coeffs);
    }
    return result;
}

// DTWMatcherCore::ClassifyResult -> {"label": str, "distance": float}
static py::dict classify_result_to_dict(const DTWMatcherCore::ClassifyResult& result)
{
    py::dict d;
    d["label"] = result.label;
    d["distance"] = result.distance;
    return d;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(keyword_detection, m)
{
    m.doc() = "MFCC / PNCC feature extraction and DTW template matching "
        "(pybind11 bindings around the shared C++ core)";

    // ---- MFCCProcessor -----------------------------------------------------
    py::class_<MFCCProcessorCore>(m, "MFCCProcessor")
        .def(py::init<>())
        .def_property("sample_rate", &MFCCProcessorCore::get_sample_rate, &MFCCProcessorCore::set_sample_rate)
        .def_property("num_coeffs", &MFCCProcessorCore::get_num_coeffs, &MFCCProcessorCore::set_num_coeffs)
        .def_property("frame_length", &MFCCProcessorCore::get_frame_length, &MFCCProcessorCore::set_frame_length)
        .def_property("hop_length", &MFCCProcessorCore::get_hop_length, &MFCCProcessorCore::set_hop_length)
        .def_property("num_mel_bands", &MFCCProcessorCore::get_num_mel_bands, &MFCCProcessorCore::set_num_mel_bands)
        .def(
            "compute",
            [](MFCCProcessorCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& samples) {
                    return feature_sequence_to_array(self.compute(array_to_vector(samples)));
            },
            py::arg("samples"),
            "Compute MFCCs for a 1D float32 array of PCM samples.\n"
            "Returns a (n_frames, num_coeffs) float32 array.");

    // ---- PNCCProcessor -------------------------------------------------------
    py::class_<PNCCProcessorCore>(m, "PNCCProcessor")
        .def(py::init<>())
        .def_property("sample_rate", &PNCCProcessorCore::get_sample_rate, &PNCCProcessorCore::set_sample_rate)
        .def_property("num_coeffs", &PNCCProcessorCore::get_num_coeffs, &PNCCProcessorCore::set_num_coeffs)
        .def_property("frame_length", &PNCCProcessorCore::get_frame_length, &PNCCProcessorCore::set_frame_length)
        .def_property("hop_length", &PNCCProcessorCore::get_hop_length, &PNCCProcessorCore::set_hop_length)
        .def_property("num_gamma_bands", &PNCCProcessorCore::get_num_gamma_bands, &PNCCProcessorCore::set_num_gamma_bands)
        .def_property("power_law_exponent", &PNCCProcessorCore::get_power_law_exponent, &PNCCProcessorCore::set_power_law_exponent)
        .def_property("medium_time_frames", &PNCCProcessorCore::get_medium_time_frames, &PNCCProcessorCore::set_medium_time_frames)
        .def(
            "compute",
            [](PNCCProcessorCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& samples) {
                    return feature_sequence_to_array(self.compute(array_to_vector(samples)));
            },
            py::arg("samples"),
            "Compute PNCCs for a 1D float32 array of PCM samples.\n"
            "Returns a (n_frames, num_coeffs) float32 array.");

    // ---- DTWMatcher ----------------------------------------------------------
    py::class_<DTWMatcherCore> dtw(m, "DTWMatcher");

    py::enum_<DTWMatcherCore::DistanceMetric>(dtw, "DistanceMetric")
        .value("EUCLIDEAN", DTWMatcherCore::EUCLIDEAN)
        .value("COSINE", DTWMatcherCore::COSINE)
        .export_values();

    py::enum_<DTWMatcherCore::ClassifyMethod>(dtw, "ClassifyMethod")
        .value("AVERAGE_DISTANCE", DTWMatcherCore::AVERAGE_DISTANCE)
        .value("BEST_MATCH", DTWMatcherCore::BEST_MATCH)
        .export_values();

    dtw.def(py::init<>())
        .def_property("distance_metric", &DTWMatcherCore::get_distance_metric, &DTWMatcherCore::set_distance_metric)
        .def_property("band_width", &DTWMatcherCore::get_band_width, &DTWMatcherCore::set_band_width)
        .def_property("classify_method", &DTWMatcherCore::get_classify_method, &DTWMatcherCore::set_classify_method)
        .def(
            "compute",
            [](const DTWMatcherCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& seq_a,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& seq_b) {
                    return self.compute(array_to_feature_sequence(seq_a), array_to_feature_sequence(seq_b));
            },
            py::arg("seq_a"), py::arg("seq_b"),
            "DTW distance between two (n_frames, n_coeffs) feature sequences.")
        .def(
            "add_template",
            [](DTWMatcherCore& self, const std::string& label,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& features) {
                    self.add_template(label, array_to_feature_sequence(features));
            },
            py::arg("label"), py::arg("features"))
        .def("clear_templates", &DTWMatcherCore::clear_templates)
        .def(
            "classify",
            [](const DTWMatcherCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& features) {
                    return self.classify(array_to_feature_sequence(features));
            },
            py::arg("features"))
        .def(
            "classify_with_best_score",
            [](const DTWMatcherCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& features) {
                    return classify_result_to_dict(self.classify_with_best_score(array_to_feature_sequence(features)));
            },
            py::arg("features"),
            "Returns {'label': str, 'distance': float} for the best-matching template.")
        .def(
            "classify_with_every_score",
            [](const DTWMatcherCore& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& features) {
                    py::dict result;
                    for (const auto& r : self.classify_with_every_score(array_to_feature_sequence(features)))
                        result[py::str(r.label)] = r.distance;
                    return result;
            },
            py::arg("features"),
            "Returns {label: distance} for every registered template.");
}

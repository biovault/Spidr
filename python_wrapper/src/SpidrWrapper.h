#pragma once

#include <SpidrAnalysis.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <optional>

namespace py = pybind11;

// This class introduces a unified&simplified constructor for SpidrAnalysis
class SpidrWrapper {
public:
	SpidrWrapper(
		feat_dist featDist = feat_dist::PC_CHA,
		loc_Neigh_Weighting kernelType = loc_Neigh_Weighting::WEIGHT_UNIF,
		size_t numLocNeighbors = 0,
		size_t numHistBins = 0,
		float pixelWeight = 0.5,
		knn_library aknnAlgType = knn_library::KNN_HNSW,
		size_t numIterations = 1000,
		size_t perplexity = 30,
		size_t exaggeration = 250,
		size_t expDecay = 70,
		bool forceCalcBackgroundFeatures = false,
		py::array_t<float, py::array::c_style | py::array::forcecast> initial_embedding = py::array_t<float, py::array::c_style | py::array::forcecast>());

	// compute knn dists and ids (and as part of that also the features), does not return knn
	void compute_fit(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		size_t imgWidth, size_t imgHeight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);

	// compute knn dists and ids (and as part of that also the features), returns knn
	std::tuple<std::vector<int>, std::vector<float>> fit(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		size_t imgWidth, size_t imgHeight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);

	// compute embedding based on knn, does not return knn
	void compute_transform();

	// compute embedding based on knn, returns knn
	py::array_t<float, py::array::c_style> transform();

	// computes knn and embedding, returns knn
	py::array_t<float, py::array::c_style> fit_transform(
		py::array_t<float, py::array::c_style | py::array::forcecast> X,
		py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
		size_t imgWidth, size_t imgHeight,
		std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal);

	// returns the computed embedding
	py::array_t<float, py::array::c_style> get_embedding();

	size_t get_nn() { return _nn; }
	size_t get_perplexity() { return _perplexity; }
	size_t get_numIterations() { return _numIterations; }

	// sets externally computed knn, no need to call any *fit* function afterwards, just go with *transform
	void set_kNN(py::array_t<int, py::array::c_style | py::array::forcecast> knn_indices, py::array_t<float, py::array::c_style | py::array::forcecast> knn_distances);

	// returns the knn distances and indices computed in one of the *fit functions
	std::tuple<std::vector<int>, std::vector<float>> get_kNN();

private:

	std::unique_ptr<SpidrAnalysis> _SpidrAnalysis;

	feat_dist _featDist;
	feature_type _featType;
	distance_metric _distMetric;
	loc_Neigh_Weighting _kernelType;
	size_t _numLocNeighbors;
	size_t _numHistBins;
	float _pixelWeight;
	knn_library _aknnAlgType;
	size_t _numIterations;
	size_t _perplexity;
	size_t _exaggeration;
	size_t _expDecay;
	bool _forceCalcBackgroundFeatures;

	std::vector<float> _initial_embedding;
	bool _has_preset_embedding;
	size_t _num_data_points_initial_embedding;

	size_t _numDims;
	py::ssize_t _numPoints;
	ImgSize _imgSize;

	bool _fitted;		// whether transform can be called (can be called when knn are set)
	bool _transformed;	// whether an embedding was calculated and can be returned
	size_t _nn;

};


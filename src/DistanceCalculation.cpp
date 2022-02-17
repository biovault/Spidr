#include "DistanceCalculation.h"

#include "SpidrAnalysisParameters.h"
#include "KNNUtils.h"
#include "EvalUtils.h"

#include "hnswlib/hnswlib.h"
#include "spdlog/spdlog-inl.h"

#include <chrono>
#include <algorithm>            // std::none_of
#include <iterator>             // std::make_move_iterator, find
#include <utility>              // std::move


//DistanceCalculation::DistanceCalculation()
//{
//}


void DistanceCalculation::setup(const Feature dataFeatures, const std::vector<unsigned int>& foregroundIDsGlobal, SpidrParameters& params) {
    spdlog::info("Distance calculation: Setup");
    _featureType = params._featureType;
    _numFeatureValsPerPoint = params._numFeatureValsPerPoint;

    // SpidrParameters
    _knn_lib = params._aknn_algorithm;
    _knn_metric = params._aknn_metric;
    _nn = params.get_nn();                           // see SpidrAnalysis::update_nn ->  (size_t)(params.get_perplexity() * params.get_perplexity_multiplier() + 1)
    _neighborhoodSize = params.get_neighborhoodSize();    // square neighborhood with _numNeighborsInEachDirection to each side from the center
    _neighborhoodWeighting = params._neighWeighting;
    _pixelWeight = params._pixelWeight;

    // Data
    // Input
    _numPoints = params._numPoints;
    _numForegroundPoints = params._numForegroundPoints;
    _numDims = params._numDims;
    _numHistBins = params._numHistBins;
    _embeddingName = params._embeddingName;
    _imgWidth = params._imgSize.width;

    _dataFeatures = dataFeatures;
    _foregroundIDsGlobal = foregroundIDsGlobal;

    // Output
    //_knn_indices.resize(_numForegroundPoints*_nn, -1);              // unnecessary, done in ComputeHNSWkNN
    //_knn_distances.resize(_numForegroundPoints*_nn, -1);    // unnecessary, done in ComputeHNSWkNN

    spdlog::info("Distance calculation: Feature values per point: {0}, Number of NN to calculate {1}. Metric: {2}", _numFeatureValsPerPoint, _nn, logging::distance_metric_name(_knn_metric));

    if (_numPoints != _numForegroundPoints)
        spdlog::info("Distance calculation: Do not consider {} background points", _numPoints - _numForegroundPoints);
}

void DistanceCalculation::compute() {
	spdlog::info("Distance calculation: Started");

    computekNN();

	spdlog::info("Distance calculation: Finished");

}

void DistanceCalculation::computekNN() {
    
	spdlog::info("Distance calculation: Setting up metric space");
    auto t_start_CreateHNSWSpace = std::chrono::steady_clock::now();

    // setup hsnw index
    hnswlib::SpaceInterface<float> *space = CreateHNSWSpace(_knn_metric, _featureType, _numDims, _neighborhoodSize, _neighborhoodWeighting, _numHistBins, _pixelWeight);
    assert(space != NULL);

    auto t_end_CreateHNSWSpace = std::chrono::steady_clock::now();
	spdlog::info("Distance calculation: Build time metric space (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_CreateHNSWSpace - t_start_CreateHNSWSpace).count()) / 1000);
    spdlog::info("Distance calculation: Compute kNN");

    auto t_start_ComputeDist = std::chrono::steady_clock::now();

    if (_knn_lib == knn_library::KNN_HNSW) {
		spdlog::info("Distance calculation: HNSWLib for knn computation");

        std::tie(_knn_indices, _knn_distances) = ComputeHNSWkNN(_dataFeatures, space, _foregroundIDsGlobal, _nn);

    }
    else if (_knn_lib == knn_library::KKN_EXACT) {
		spdlog::info("Distance calculation: Exact kNN computation");

        std::tie(_knn_indices, _knn_distances) = ComputeExactKNN(_dataFeatures, space, _foregroundIDsGlobal, _nn);

    }
	else if (_knn_lib == knn_library::FULL_DIST_BRUTE_FORCE) {
		// Use entire distance matrix 
		spdlog::info("Distance calculation: Calc full distance matrix brute force");
		std::tie(_knn_indices, _knn_distances) = ComputeFullDistMat(_dataFeatures, space, _foregroundIDsGlobal);

	}
	else
		throw std::runtime_error("Distance calculation: Unknown knn_library");

    auto t_end_ComputeDist = std::chrono::steady_clock::now();
	spdlog::info("Distance calculation: Computation duration (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_ComputeDist - t_start_ComputeDist).count()) / 1000);

    // -1 would mark unset values
    assert(_knn_indices.size() == _numForegroundPoints * _nn);
    assert(_knn_distances.size() == _numForegroundPoints * _nn);
    assert(std::none_of(_knn_indices.begin(), _knn_indices.end(), [](int i) {return i == -1; }));
    assert(std::none_of(_knn_distances.begin(), _knn_distances.end(), [](float i) {return i == -1.0f; }));

}

std::tuple< std::vector<int>, std::vector<float>> DistanceCalculation::output() const {
    return { _knn_indices, _knn_distances };
}

std::vector<int> DistanceCalculation::get_knn_indices() const {
    return _knn_indices;
}

std::vector<float> DistanceCalculation::get_knn_distances() const {
    return _knn_distances;
}


void DistanceCalculation::setKnnAlgorithm(knn_library knn)
{
    _knn_lib = knn;
}

void DistanceCalculation::setDistanceMetric(distance_metric metric)
{
    _knn_metric = metric;
}
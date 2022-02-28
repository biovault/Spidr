#include "SpidrAnalysis.h"
#include "spdlog/spdlog-inl.h"

#include <cmath>
#include <algorithm>
#include <chrono>       // std::chrono

SpidrAnalysis::SpidrAnalysis() :
    _featExtraction(),
    _distCalc(),
    _tsne()
{

}


SpidrAnalysis::SpidrAnalysis(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const size_t numDimensions, const ImgSize imgSize, const feature_type featType, \
    const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, const float pixelDistanceWeight, \
    const knn_library aknnAlgType, const distance_metric aknnMetric, \
    const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, \
    const std::vector<float>& initial_embedding , const std::string embeddingName /* = "emd" */, \
    bool forceCalcBackgroundFeatures /* = false */, const std::vector<unsigned int>& backgroundIDsGlobal /* = std::vector<unsigned int>() */) :
    _featExtraction(),
    _distCalc(),
    _tsne()
{
    // set data
    setupData(attribute_data, pointIDsGlobal, numDimensions, imgSize, embeddingName, initial_embedding, backgroundIDsGlobal);

    // computation settings
    initializeAnalysisSettings(featType, kernelType, numLocNeighbors, numHistBins, pixelDistanceWeight, aknnAlgType, aknnMetric, numIterations, perplexity, exaggeration, expDecay, forceCalcBackgroundFeatures);
}

SpidrAnalysis::SpidrAnalysis(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const size_t numDimensions, const ImgSize imgSize, const feature_type featType, \
    const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, const float pixelDistanceWeight, \
    const knn_library aknnAlgType, const distance_metric aknnMetric, \
    const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, \
    const std::string embeddingName /* = "emd" */, \
    bool forceCalcBackgroundFeatures /* = false */, const std::vector<unsigned int>& backgroundIDsGlobal /* = std::vector<unsigned int>() */) :
    SpidrAnalysis(attribute_data, pointIDsGlobal, numDimensions, imgSize, featType, kernelType, numLocNeighbors, numHistBins, pixelDistanceWeight, \
        aknnAlgType, aknnMetric, numIterations, perplexity, exaggeration, expDecay, std::vector<float>(), embeddingName, forceCalcBackgroundFeatures, backgroundIDsGlobal)
{
    // no inital embedding set like in the other constructor
}


void SpidrAnalysis::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, \
    const std::vector<unsigned int>& backgroundIDsGlobal/* = std::vector<unsigned int>() */) {

	// Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;

    _backgroundIDsGlobal = backgroundIDsGlobal;
    std::sort(_backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end());
    // IDs that are not background are in the foreground
    std::set_difference(_pointIDsGlobal.begin(), _pointIDsGlobal.end(),
                        _backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end(), 
                        std::inserter(_foregroundIDsGlobal, _foregroundIDsGlobal.begin()));

    // Set parameters
    _params._numPoints = _pointIDsGlobal.size();
    _params._numDims = numDimensions;
	_params._imgSize = imgSize;
    _params._embeddingName = embeddingName;
    _params._numForegroundPoints = _foregroundIDsGlobal.size();

	spdlog::info("SpidrAnalysis: Setup data with number of points: {0}, num dims: {1}, image size (width, height): ({2}, {3})", _params._numPoints, _params._numDims, _params._imgSize.width, _params._imgSize.height);
    if(!_backgroundIDsGlobal.empty())
        spdlog::info("SpidrAnalysis: Excluding {} background points and respective features", _backgroundIDsGlobal.size());

    assert(_params._numForegroundPoints + _backgroundIDsGlobal.size() == _params._numPoints);
}

void SpidrAnalysis::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, const std::vector<float>& initial_embedding, \
    const std::vector<unsigned int>& backgroundIDsGlobal/* = std::vector<unsigned int>() */) 
{
    // Set data
    setupData(attribute_data, pointIDsGlobal, numDimensions, imgSize, embeddingName, backgroundIDsGlobal);

    // Set initial embedding
    if (!initial_embedding.empty())
    {
        if (initial_embedding.size() != _params._numForegroundPoints * 2)
        {
            spdlog::warn("SpidrWrapper::compute_fit: User-defined initial embedding does not have the same number of (foreground) points. Initial embedding is ignored.");
            return;
        }

        _params._has_preset_embedding = true;
        _initial_embedding = initial_embedding;
    }
}

void SpidrAnalysis::initializeAnalysisSettings(const feature_type featType, const loc_Neigh_Weighting kernelWeightType, const size_t numLocNeighbors, const size_t numHistBins, float pixelDistanceWeight, \
                                               const knn_library aknnAlgType, const distance_metric aknnMetric,\
                                               const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, bool forceCalcBackgroundFeatures) {
	if (_params._numDims < 0 || _params._numHistBins < 0)
		spdlog::error("SpidrWrapper: first call SpidrAnalysis::setupData() before initializing the settings with SpidrAnalysis::initializeAnalysisSettings since some might depend on the data dimensions.");

	// the following set* functions set values in _params
	// initialize Feature Extraction Settings
    setFeatureType(featType);
    setKernelWeight(kernelWeightType);
    setNumLocNeighbors(numLocNeighbors);    // Sets both _params._kernelWidth and _params._neighborhoodSize
    setNumHistBins(numHistBins);
    setPixelDistanceWeight(pixelDistanceWeight);

    // initialize Distance Calculation Settings
    // number of nn is dertermined by perplexity, set in setPerplexity
    setKnnAlgorithm(aknnAlgType);
    setDistanceMetric(aknnMetric);

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);
    setExpDecay(expDecay);

    // Derived parameters
	setNumFeatureValsPerPoint(featType, _params._numDims, _params._numHistBins, _params.get_neighborhoodSize());			// sets _params._numFeatureValsPerPoint
    setForceCalcBackgroundFeatures(forceCalcBackgroundFeatures);													        // sets _params._forceCalcBackgroundFeatures

    if (check_feat_dist(featType, aknnMetric) == false)
        spdlog::warn("SpidrAnalysis: Feature type and distance metric do not work well together. Results might not make sense or computation might fail.");

	spdlog::info("SpidrAnalysis: Initialized all settings");
}


void SpidrAnalysis::computeFeatures() {
	_featExtraction.setup(_pointIDsGlobal, _attribute_data, _params, _backgroundIDsGlobal, _foregroundIDsGlobal);
	_featExtraction.compute();
	spdlog::info("SpidrAnalysis: Get computed feature values");
	_dataFeats = _featExtraction.output();
}

void SpidrAnalysis::computekNN() {
	_distCalc.setup(_dataFeats, _foregroundIDsGlobal, _params);
	_distCalc.compute();
	_knn_indices = _distCalc.get_knn_indices();
	_knn_distances = _distCalc.get_knn_distances();
}

void SpidrAnalysis::computeEmbedding() {
    if(_knn_indices.empty() | _knn_distances.empty())
        spdlog::error("SpidrAnalysis: knn must not be empty");

    if (_knn_indices.size() != _knn_distances.size())
        spdlog::error("SpidrAnalysis: knn indices and knn distance do not align");

    if (_params._has_preset_embedding)
    {
        spdlog::info("computeEmbedding: _params._has_preset_embedding");
        _tsne.setup(_knn_indices, _knn_distances, _params, _initial_embedding);
    }
    else
    {
        spdlog::info("computeEmbedding: _params.NO");
        _tsne.setup(_knn_indices, _knn_distances, _params);
    }
	
    _tsne.compute();
}

void SpidrAnalysis::compute() {
	// Extract features
	computeFeatures();

    // Caclculate distances and kNN
	computekNN();

    // Compute t-SNE with the given data
	computeEmbedding();

	spdlog::info("SpidrAnalysis: Finished");
}


void SpidrAnalysis::setFeatureType(const feature_type feature_type) {
	_params._featureType = feature_type;
}

void SpidrAnalysis::setKernelWeight(const loc_Neigh_Weighting loc_Neigh_Weighting) {
    _params._neighWeighting = loc_Neigh_Weighting;
}

void SpidrAnalysis::setNumLocNeighbors(const size_t num) {
    _params.set_numNeighborsInEachDirection(num);
    //_params._kernelWidth = (2 * _params.get_numNeighborsInEachDirection()) + 1;   // set in set_numNeighborsInEachDirection
    //_params._neighborhoodSize = _params._kernelWidth * _params._kernelWidth;;
}

void SpidrAnalysis::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysis::setKnnAlgorithm(const knn_library knn_library) {
    _params._aknn_algorithm = knn_library;
}

void SpidrAnalysis::setDistanceMetric(const distance_metric distance_metric) {
    _params._aknn_metric = distance_metric;
}

void SpidrAnalysis::setPerplexity(const size_t perplexity) {
    _params.set_perplexity(perplexity);
}

void SpidrAnalysis::setNumIterations(const size_t numIt) {
    _params._numIterations = numIt;
}

void SpidrAnalysis::setExaggeration(const size_t exag) {
    _params._exaggeration = exag;
}

void SpidrAnalysis::setExpDecay(const size_t expDecay) {
    _params._expDecay = expDecay;
}

void SpidrAnalysis::setPixelDistanceWeight(const float pixelWeight) {
    _params._pixelWeight = pixelWeight;
}

void SpidrAnalysis::setNumFeatureValsPerPoint(feature_type featType, size_t numDims, size_t numHistBins, size_t neighborhoodSize) {
	_params._numFeatureValsPerPoint = NumFeatureValsPerPoint(featType, numDims, numHistBins, neighborhoodSize);
}

void SpidrAnalysis::setForceCalcBackgroundFeatures(const bool CalcBackgroundFeatures) {
    _params._forceCalcBackgroundFeatures = CalcBackgroundFeatures;
}

const size_t SpidrAnalysis::getNumForegroundPoints() {
    return _params._numForegroundPoints;
}

const size_t SpidrAnalysis::getNumImagePoints() {
    assert(_pointIDsGlobal.size() == _params._numForegroundPoints + _backgroundIDsGlobal.size());
    return _params._numPoints;
}

bool SpidrAnalysis::embeddingIsRunning() {
    return _tsne.isTsneRunning();
}

const std::vector<float>& SpidrAnalysis::output() const {
    return _tsne.output();
}

const std::vector<float> SpidrAnalysis::output_copy() const {
    std::vector<float> emb;
    const std::vector<float>& emb_ref = _tsne.output();
    emb.assign(emb_ref.begin(), emb_ref.end());  // copy vector 
    return emb;
}

const std::vector<float>& SpidrAnalysis::outputWithBackground() {
    const std::vector<float>& emb = _tsne.output();

    if (_backgroundIDsGlobal.empty())
    {
        return emb;
    }
    else
    {
        addBackgroundToEmbedding(_emd_with_backgound, emb);
        return _emd_with_backgound;
    }
}

const std::vector<float> SpidrAnalysis::outputWithBackground(std::vector<float> emd_without_background)
{

    if (_backgroundIDsGlobal.empty())
    {
        spdlog::info("No background points specified, returning input embedding.");
        return emd_without_background;
    }
    else
    {
        std::vector<float> emb;
        addBackgroundToEmbedding(emb, emd_without_background);
        return emb;
    }

}

const std::vector<float> SpidrAnalysis::outputWithBackground_copy() const {
    const std::vector<float>& emb = _tsne.output();
    std::vector<float> emd_with_backgound;

    if (_backgroundIDsGlobal.empty())
    {
        return emb;
    }
    else
    {
        addBackgroundToEmbedding(emd_with_backgound, emb);
        return emd_with_backgound;
    }
}

void SpidrAnalysis::addBackgroundToEmbedding(std::vector<float>& emb, const std::vector<float>& emb_wo_bg) const {
    spdlog::info("SpidrAnalysis: Add background back to embedding");
    auto start = std::chrono::steady_clock::now();

    emb.resize(_pointIDsGlobal.size() * 2); // _params._numPoints = _pointIDsGlobal.size()

    // find min x and min y embedding positions
    float minx = emb_wo_bg[0];
    float miny = emb_wo_bg[1];

    for (size_t i = 0; i < emb_wo_bg.size(); i += 2) {
        if (emb_wo_bg[i] < minx)
            minx = emb_wo_bg[i];

        if (emb_wo_bg[i + 1] < miny)
            miny = emb_wo_bg[i + 1];
    }

    minx -= std::abs(minx) * 0.05f;
    miny -= std::abs(miny) * 0.05f;

    // Place all background pixel in the lower left corner of the embedding
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < _backgroundIDsGlobal.size(); i++) {
        emb[2 * _backgroundIDsGlobal[i]] = minx;
        emb[2 * _backgroundIDsGlobal[i] + 1] = miny;

    }

    // Copy the foreground embedding positions
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < _foregroundIDsGlobal.size(); i++) {
        emb[2 * _foregroundIDsGlobal[i]] = emb_wo_bg[2 * i];
        emb[2 * _foregroundIDsGlobal[i] + 1] = emb_wo_bg[2 * i + 1];
    }

    auto end = std::chrono::steady_clock::now();
    spdlog::info("SpidrAnalysis: Add backgorund (sec): {}", ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000);


}

void SpidrAnalysis::stopComputation() {
    _featExtraction.stopFeatureCopmutation();
    _tsne.stopGradientDescent();
}

const SpidrParameters SpidrAnalysis::getParameters() const {
    return _params;
}

const Feature SpidrAnalysis::getDataFeatures() const {
	return _dataFeats;
}

const std::tuple<std::vector<int>, std::vector<float>> SpidrAnalysis::getKnn() const {
	return std::make_tuple(_knn_indices, _knn_distances);
}

const std::vector<int> SpidrAnalysis::getKnnIndices() const {
    return _knn_indices;
}

const std::vector<float> SpidrAnalysis::getKnnDistances() const {
    return _knn_distances;
}

void SpidrAnalysis::setKnn(std::vector<int>& indices, std::vector<float>& distances) {
    _knn_indices = indices;
    _knn_distances = distances;

    // set meta data	
    assert(_knn_indices.size() % _params.get_nn() == 0);        // knn size must align with perplexity
    size_t num_points = _knn_indices.size() / _params.get_nn();

    _params._numPoints = num_points;
    _params._numForegroundPoints = num_points;
}

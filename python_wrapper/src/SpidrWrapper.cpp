#include "SpidrWrapper.h"
#include "spdlog/spdlog-inl.h"

#include <algorithm>

SpidrWrapper::SpidrWrapper(feat_dist featDist,
	loc_Neigh_Weighting kernelType,
	size_t numLocNeighbors,
	size_t numHistBins,
	float pixelWeight,
	knn_library aknnAlgType,
	size_t numIterations,
	size_t perplexity,
	size_t exaggeration,
	size_t expDecay,
	bool forceCalcBackgroundFeatures,
	py::array_t<float, py::array::c_style | py::array::forcecast> initial_embedding
) : _kernelType(kernelType), _numHistBins(numHistBins), _pixelWeight(pixelWeight), _aknnAlgType(aknnAlgType), _featDist(featDist), _numIterations(numIterations), _numLocNeighbors(numLocNeighbors),
    _perplexity(perplexity), _exaggeration(exaggeration), _expDecay(expDecay), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures), _fitted(false), _transformed(false), _has_preset_embedding(false), 
	_numDims(0), _numPoints(0) /* _numDims and _numPoints are set later and derived from the data*/
{
	
	if (_numLocNeighbors <= 0)
	{
		spdlog::error("SpidrWrapper::Constructor: Spatial Neighbors must be larger 0");
		return;
	}

	// set _featType and _distMetric based on _feat_dist (not all feat_dist are exposed in the python wrapper, see SpiderBind.cpp)
	std::tie(_featType, _distMetric) = get_feat_and_dist(_featDist);

	// check number of histogram bins
	if (_featType == feature_type::TEXTURE_HIST_1D && _numHistBins <= 0)
	{
		auto kernelWidth = (2 * _numLocNeighbors) + 1;
		auto neighborhoodSize = kernelWidth * kernelWidth;

		_numHistBins = RiceBinSize(neighborhoodSize);
		spdlog::warn("SpidrWrapper:: Number of histogram bins must be larger than 0, automatically set it to {}", _numHistBins);
	}

	// Check pixel weight
	if (_pixelWeight > 1.0f || _pixelWeight < 0.0f)
	{
		spdlog::warn("SpidrWrapper::pixelWeight {} outside range [0, 1], automatically clamp it", _pixelWeight);
		_pixelWeight = std::clamp(_pixelWeight, 0.0f, 1.0f);
	}

	_SpidrAnalysis = std::make_unique<SpidrAnalysis>();
	_nn = _perplexity * 3 + 1;  // _perplexity_multiplier = 3
	
	// initial embedding given?
	if (initial_embedding.size() > 1)
	{
		_has_preset_embedding = true;
		_num_data_points_initial_embedding = initial_embedding.size() / 2;

		// copy initial embedding
		_initial_embedding.resize(initial_embedding.size());
		std::memcpy(_initial_embedding.data(), initial_embedding.data(), initial_embedding.size() * sizeof(float));
	}
}


void SpidrWrapper::compute_fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	size_t imgWidth, size_t imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {
	// check input dimensions
	if (X.ndim() != 2)
		throw std::runtime_error("SpidrWrapper::compute_fit: Input should be 2-D NumPy array");

	// copy data from py::array to std::vector
	std::vector<float> dat(X.size());
	std::memcpy(dat.data(), X.data(), X.size() * sizeof(float));

	std::vector<unsigned int> IDs(pointIDsGlobal.size());
	std::memcpy(IDs.data(), pointIDsGlobal.data(), pointIDsGlobal.size() * sizeof(unsigned int));

	// Get other data info
	_numDims = X.shape()[1];
	_numPoints = X.shape()[0];
	_imgSize = ImgSize(imgWidth, imgHight);

	// Background data?
	std::vector<unsigned int> IDsBack;
	if (!backgroundIDsGlobal.has_value())
	{
		// no background
		IDsBack = std::vector<unsigned int>();
	}
	else
	{
		IDsBack.reserve(backgroundIDsGlobal->size());
		std::memcpy(IDsBack.data(), backgroundIDsGlobal->data(), backgroundIDsGlobal->size() * sizeof(unsigned int));
	}

	// Pass data to SpidrLib
	if (_has_preset_embedding)
	{
		_SpidrAnalysis->setupData(dat, IDs, _numDims, _imgSize, "SpidrWrapper", _initial_embedding, IDsBack);
	}
	else
	{
		_SpidrAnalysis->setupData(dat, IDs, _numDims, _imgSize, "SpidrWrapper", IDsBack);
	}
	
	// Init all settings (setupData must have been called before initing the settings.)
	_SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numLocNeighbors, _numHistBins, _pixelWeight, _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceCalcBackgroundFeatures);

	// Compute knn dists and inds
	_SpidrAnalysis->computeFeatures();
	_SpidrAnalysis->computekNN();

	_fitted = true;
	_transformed = false;
}


std::tuple<std::vector<int>, std::vector<float>> SpidrWrapper::fit(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	size_t imgWidth, size_t imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	// Init settings, (Extract features), compute similarities, embed data
	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	return _SpidrAnalysis->getKnn();
}


void SpidrWrapper::compute_transform() {
	if (_fitted == false) {
		spdlog::error("SpidrWrapper::compute_transform: Call fit(...) before transform() or go with fit_transform() or set knn manually with set_kNN(...)");
		return;
	}

	// computes t-SNE based on previously computed high-dimensional distances
	_SpidrAnalysis->computeEmbedding();

	_transformed = true;
}


py::array_t<float, py::array::c_style> SpidrWrapper::transform() {

	// compute embedding
	compute_transform();

	// return embedding
	return get_embedding();
}


py::array_t<float, py::array::c_style> SpidrWrapper::fit_transform(
	py::array_t<float, py::array::c_style | py::array::forcecast> X,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> pointIDsGlobal,
	size_t imgWidth, size_t imgHight,
	std::optional<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>> backgroundIDsGlobal) {

	// Init settings, (Extract features), compute similarities
	compute_fit(X, pointIDsGlobal, imgWidth, imgHight, backgroundIDsGlobal);

	// embed data
	compute_transform();

	// return embedding
	return get_embedding();
}

void SpidrWrapper::set_kNN(py::array_t<int, py::array::c_style | py::array::forcecast> knn_indices, py::array_t<float, py::array::c_style | py::array::forcecast> knn_distances) {
	// copy data from py::array to std::vector
	std::vector<int> indices(knn_indices.size());
	std::memcpy(indices.data(), knn_indices.data(), knn_indices.size() * sizeof(float));
	
	std::vector<float> distances(knn_distances.size());
	std::memcpy(distances.data(), knn_distances.data(), knn_distances.size() * sizeof(float));

	// check values
	if (indices.size() != distances.size())
	{
		spdlog::error("SpidrWrapper::setKNN: knn indices and distances do not align.");
		return;
	}

	if (indices.size() % _nn != 0)
	{
		spdlog::error("SpidrWrapper::setKNN: size of indices vector must be multiple of number of neighbors.");
		return;
	}

	// set knn values
	_SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numLocNeighbors, _numHistBins, 0.5, _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceCalcBackgroundFeatures);
	_SpidrAnalysis->setKnn(indices, distances);

	// set number of points as it is used in transform()
	_numPoints = indices.size() / _nn;

	_fitted = true;
	_transformed = false;
}

py::array_t<float, py::array::c_style> SpidrWrapper::get_embedding() {
	if (_transformed == false) {
		spdlog::error("SpidrWrapper::get_embedding: Call compute_transform() or fit_transform() first");
		return py::array_t<float>();
	}

	// get embedding
	std::vector<float> emb = _SpidrAnalysis->outputWithBackground();

	return py::array(py::buffer_info(
		emb.data(),													/* data as contiguous array  */
		sizeof(float),												/* size of one scalar        */
		py::format_descriptor<float>::format(),						/* data type                 */
		2,															/* number of dimensions      */
		std::vector<py::ssize_t>{_numPoints, 2},					/* shape of the matrix       */
		std::vector<py::ssize_t>{sizeof(float) * 2, sizeof(float)}	/* strides for each axis     */
	));

}

std::tuple<std::vector<int>, std::vector<float>> SpidrWrapper::get_kNN()
{
	if (_fitted == false) {
		spdlog::error("SpidrWrapper::get_kNN: Call fit(...) before transform() or go with fit_transform() or set knn manually with set_kNN(...)");
		return std::make_tuple(std::vector<int>{ -1 }, std::vector<float>{ 0 });	// return dummy values
	}

	return _SpidrAnalysis->getKnn();
}
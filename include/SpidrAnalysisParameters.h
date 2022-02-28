#pragma once

#include <string>
#include <stdexcept>
#include <tuple>

#include "spdlog/spdlog-inl.h"

/*! Image width and height container */
typedef struct ImgSize {
    size_t width;
    size_t height;

	ImgSize() : width(0), height(0) {};
	ImgSize(size_t width, size_t height) : width(width), height(height) {};

} ImgSize;


/*! kNN algorithm that is used for kNN computations
 * The librarires are extended in order to work with different feature types
 */
enum class knn_library : size_t
{
	KKN_EXACT,              /*!< No aknn library in use, no approximation i.e. exact kNN computation */
	KNN_HNSW,			    /*!< approximated kNN with HNSWLib */
	FULL_DIST_BRUTE_FORCE,	/*!< straight forward brute force computation of the entire distance matric - not intended to be used with t-SNE but rather if you are interested in the distance matrix and not the knn like for MDS */
};

/*! Defines the distance metric
* add respective entries to logging::distance_metric_name
 */
enum class distance_metric : size_t
{
	METRIC_QF,       /*!< Quadratic form distance */
	METRIC_HEL,      /*!< Hellinger distance */
	METRIC_EUC,      /*!< Euclidean distance */
	METRIC_CHA,      /*!< Chamfer distance (point cloud)*/
	METRIC_SSD,      /*!< Sum of squared distances (point cloud)*/
	METRIC_HAU,      /*!< Hausdorff distance (point cloud)*/
	METRIC_HAU_med,      /*!< Hausdorff distance (point cloud) but with median instead of max*/
	METRIC_BHATTACHARYYA,      /*!< Bhattacharyya distance between two multivariate normal distributions, https://en.wikipedia.org/wiki/Bhattacharyya_distance */
	METRIC_BHATTACHARYYATESTONLYMEANS,      /*!< TEST Bhattacharyya distance only means */
	METRIC_DETMATRATIO,      /*!< Deteterminant Ratio part of Bhattacharyya distance, i.e. Bat distance between two distributions with the same mean */
	METRIC_CMD_covmat,      /*!< Correlation Matrix distance http://dx.doi.org/10.1109/VETECS.2005.1543265 */
	METRIC_FRECHET_Gen,      /*!< The Fréchet distance between multivariate normal distributions, https://doi.org/10.1016/0047-259X(82)90077-X */
    METRIC_FRECHET_CovMat,      /*!< The Fréchet distance between multivariate normal distributions but ignoring the means, https://doi.org/10.1016/0047-259X(82)90077-X */
	METRIC_FROBENIUS_CovMat,      /*!< Frobenius norm of element-wise differences between covmatrices */
    METRIC_EUC_sep,     /*!< Add pixel location (x,y) as feature, compute euc norm separately and combine with weight */
};


/*! Types of neighborhood features
 *
 * Adding a new feature type? Make sure to adjust NumFeatureValsPerPoint()
 */
enum class feature_type : size_t
{
	TEXTURE_HIST_1D,    /*!< Histograms of data point neighborhood, vector feature */
	LOCALMORANSI,       /*!< Local Moran's I (Local Indicator of Spatial Associations), scalar feaure */
	LOCALGEARYC,        /*!< Local Geary's C (Local Indicator of Spatial Associations), scalar feature */
	PCLOUD,             /*!< Point cloud, i.e. just the neighborhood, no transformations*/
	MULTIVAR_NORM,      /*!< Mean and covariance matrix  */
	CHANNEL_HIST,       /*!< Histogram with one bis per channel that counts active (>1) values */
    PIXEL_LOCATION,     /*!< Add pixel location (x,y) as feature */
	PIXEL_LOCATION_RANGENORM,/*!< Add pixel location (x,y) as feature, norm the x and y range to the attribute range: [0, largestPixelIndex] -> [_minAttriVal, _maxAttriVal]  */
};


/*! Main combination of distance and feature
 * get_feat_and_dist(feat_dist) will return the corresponding feature_type and distance_metric
 * when creating a new feat_dist, make sure to update get_feat_and_dist
 */
enum class feat_dist : size_t
{
	HIST_QF,        /*!< Channel histogram and QF distance */
	HIST_HEL,       /*!< Channel histogram and Hellinger */
	LMI_EUC,        /*!< Local Moran's I and euclidean distance */
	LGC_EUC,        /*!< Local Geary's C and euclidean distance */
	PC_CHA,         /*!< Point cloud distance: Chamfer */
	PC_HAU,         /*!< Point cloud distance: Hausdorff */
	PC_HAU_MED,     /*!< Point cloud distance: Hausdorff median*/
	PC_SSD,         /*!< Point cloud distance: SSD */
	MVN_BHAT,       /*!< Mean and covariance matrix feaure and Bhattacharyya distance */
	MVN_FRO,        /*!< Mean and covariance matrix feaure and Frobenius norm of element-wise differences between covmatrices*/
	CHIST_EUC,      /*!< Histogram with one bis per channel that counts active (>1) values and euclidean distance */
	XY_EUC,         /*!< Add pixel location (x,y) as features, euclidean norm */
	XY_EUCW,        /*!< Add pixel location (x,y) as features, sum weighted euclidean norms between attributes and between xy coords */
    XYRNORM_EUC,    /*!< Add pixel location (x,y) as feature, norm the x and y range to the attribute range: [0, largestPixelIndex] -> [_minAttriVal, _maxAttriVal], euclidean norm */	
};

/*! All feature and distance combinations that make most sense
 */
constexpr feat_dist all_feat_dists[] = { feat_dist::HIST_QF, feat_dist::HIST_HEL, feat_dist::LMI_EUC, feat_dist::LGC_EUC, feat_dist::PC_CHA, feat_dist::PC_HAU, feat_dist::PC_HAU_MED,
										 feat_dist::PC_SSD , feat_dist::MVN_BHAT , feat_dist::MVN_FRO , feat_dist::CHIST_EUC , feat_dist::XY_EUC , feat_dist::XY_EUCW , feat_dist::XYRNORM_EUC };


/*! Checks wheter a feat and dist combination are in all_feat_dists (whether they make sense)
 */
bool check_feat_dist(feature_type feat, distance_metric dist);

/*! Get the feature and distance metric from a feat_dist
 *  use:  std::tie(feat, dist) = get_feat_and_dist(feat_dist)
 */
std::tuple< feature_type, distance_metric> get_feat_and_dist(feat_dist feat_dist);

/*! Types of ground distance calculation that are used as the basis for bin similarities
 */
enum class bin_sim : size_t
{
    SIM_EUC,    /*!< 1 - sqrt(Euclidean distance between bins)/(Max dist) */
    SIM_EXP,    /*!< exp(-(Euclidean distance between bins)^2/(Max dist)) */
    SIM_UNI,    /*!< 1 (uniform) */
};

// Heuristic for setting the histogram bin size
enum class histBinSizeHeuristic : size_t
{
	MANUAL,    /*!< Manually  adjust histogram bin size */
	SQRT,      /*!< ceil(sqrt(n)), n = neighborhood size */
	STURGES,   /*!< ceil(log_2(n))+1, n = neighborhood size */
	RICE,      /*!< ceil(2*pow(n, 1/3)), n = neighborhood size */
};


/*! Weighting of local neighborhoods
 * Used e.g. in histogram creation, spatial weighting in LOCALMORANSI and Point cloud distance
 */
enum class loc_Neigh_Weighting : size_t
{
	WEIGHT_UNIF,    /*!< Uniform weighting (all 1) */
	WEIGHT_BINO,    /*!< Weighting binomial approximation of 2D gaussian */
	WEIGHT_GAUS,    /*!< Weighting given by 2D gaussian */
};

/*! Normalize to max, sum=1 or none
 */
enum class norm_vec : size_t
{
	NORM_NONE,  /*!< No normalization */
	NORM_MAX,   /*!< Normalization such that max = 1 (usually center value) */
	NORM_SUM,   /*!< Normalization such that sum = 1 */
};

/*! Heuristics for determining the Number of histogram bins based on the number of data points
* https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
 */
enum class bin_size : size_t
{
	MANUAL,     /*!< define manually> */
	SQRT,       /*!< sqrt(n) > */
	STURGES,    /*!< ceil(log2(n)) + 1> */
	RICE,       /*!< 2 * sqrt3(n) > */
};


/*! Calculates the size of an feature wrt to the feature type
 * Used as a step size for adding points to an HNSWlib index
 *
 * \param featureType type of feature (e.g. scalar LOCALMORANSI or vector Texture Histogram)
 * \param numDims Number of data channels
 * \param numHistBins Number of histogram bins of feature type is a vector i.e. histogram
 * \param neighborhoodSize Size of neighborhood, must be a perfect square
 * \return
 */
static const size_t NumFeatureValsPerPoint(const feature_type featureType, const size_t numDims, const size_t numHistBins, const size_t neighborhoodSize) {
	size_t featureSize = 0;
	switch (featureType) {
	case feature_type::TEXTURE_HIST_1D: featureSize = numDims * numHistBins; break;
	case feature_type::CHANNEL_HIST:    // same as Geary's C, one bin per channel in this type of histogram
	case feature_type::LOCALMORANSI:    // same as Geary's C, one scalar value per channel
	case feature_type::LOCALGEARYC:     featureSize = numDims; break;
	case feature_type::PCLOUD:          featureSize = neighborhoodSize; break; // numDims * neighborhoodSize for copying data instead of IDs
	case feature_type::MULTIVAR_NORM: featureSize = numDims + numDims * numDims + 2; break; // channel-wise means + covaraince matrix
	case feature_type::PIXEL_LOCATION:  // same as PIXEL_LOCATION_RANGENORM, attribute feature + (x, y) pixel location
	case feature_type::PIXEL_LOCATION_RANGENORM:   featureSize = numDims + 2; break;
    default: throw std::runtime_error("No feature size defined for this feature");
	}

	return featureSize;
}

namespace logging {
    std::string distance_metric_name(const distance_metric& metric);
    std::string feature_type_name(const feature_type& feature);
    std::string feat_dist_name(const feat_dist& feat_dist);
    std::string neighborhood_weighting_name(const loc_Neigh_Weighting& weighting);
}


/*! Stores all parameters used in the Spatial Analysis.
 *
 * Used to set parameters for FeatureExtraction, DistanceCalculation and TsneComputatio
 */
class SpidrParameters {
public:

	/*! The default constructor sets un-useable values - the user has to set them
	 */
	SpidrParameters() :
        _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1), _embeddingName(""),
        _featureType(feature_type::TEXTURE_HIST_1D), _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF), _numNeighborsInEachDirection(0), _numHistBins(0),
        _kernelWidth(0), _neighborhoodSize(0), _numFeatureValsPerPoint(0), _forceCalcBackgroundFeatures(false), _pixelWeight(0.5),
        _aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(distance_metric::METRIC_QF), _numForegroundPoints(-1),
		_perplexity(30), _perplexity_multiplier(3), _numIterations(1000), _exaggeration(250), _expDecay(250), _has_preset_embedding(false)
	{
        // The default constructor sets un-useable values - the user has to set them
    }

	/*! Set all parameters used during feature extract and distance computation for spatially aware embeddings
	 * 
	 * All points in the data are used, no background is defined
	 * 
	 * \param numPoints Number of data points
	 * \param numDims Number of data channels
	 * \param imgSize Width and height of image,. width*height = numPoints
	 * \param embeddingName name of embedding
	 * \param dataVecBegin points to first entry in attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointMdimN]
	 * \param featureType Feature (e.g. texture) to be extracted. Check feat_dist in for valid combinations with a supported distance metric
	 * \param neighWeighting Local neighborhood weighting
	 * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus numLocNeighbors=1 -> 3x3 neighborhood
	 * \param numHistBins Number of histogram bins for histogram features. You might want to use a heuristic for this, see FeatureUtils.h
	 * \param aknn_algorithm exact or approximated knn or full distance matrix
	 * \param aknn_metric Distance metric. Check feat_dist for valid combinations with a supported feature type
	 * \param pixelWeight weight between eucl. feature and pixel distance in distance_metric::METRIC_EUC_sep, i.e. feat_dist::XY_EUCW
	 * \param perplexity t-sne perplexity, defines number of nearest neighbors: nn = perplexity * 3 +1
	 * \param numIterations number of t-sne gradient descent iterations
	 * \param exaggeration iterations with complete exaggeration of the attractive forces
	 * \param expDecay iterations required to remove the exaggeration using an exponential decay
	 */
	SpidrParameters(size_t numPoints, size_t numDims, ImgSize imgSize, std::string embeddingName, const float* dataVecBegin,
		feature_type featureType, loc_Neigh_Weighting neighWeighting, size_t numLocNeighbors, size_t numHistBins,
		knn_library aknn_algorithm, distance_metric aknn_metric, float pixelWeight,
        size_t perplexity, size_t numIterations, size_t exaggeration, size_t expDecay=250) :
		_numPoints(numPoints), _numDims(numDims), _imgSize(imgSize), _embeddingName(embeddingName),
		_featureType(featureType), _neighWeighting(neighWeighting), _numHistBins(numHistBins), _pixelWeight(pixelWeight),
		_aknn_algorithm(aknn_algorithm), _aknn_metric(aknn_metric), _forceCalcBackgroundFeatures(false),
		_perplexity_multiplier(3), _numIterations(numIterations), _exaggeration(exaggeration), _expDecay(expDecay), _has_preset_embedding(false)
	{
        set_perplexity(perplexity);  // sets nn based on perplexity
		_numForegroundPoints = numPoints; // No background, default that all points are in the foreground
        set_numNeighborsInEachDirection(numLocNeighbors);  // sets _kernelWidth and _neighborhoodSize
        _numFeatureValsPerPoint = NumFeatureValsPerPoint(_featureType, _numDims, _numHistBins, _neighborhoodSize);
	}

	/*! Set all parameters used during feature extract and distance computation for spatially aware embeddings
	 *
	 * Background IDs are given.
	 *
	 * \param numPoints Number of data points
	 * \param numDims Number of data channels
	 * \param imgSize Width and height of image,. width*height = numPoints
	 * \param embeddingName name of embedding
	 * \param dataVecBegin points to first entry in attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointMdimN]
	 * \param numForegroundPoints Number of points in the foreground. numPoints - numForegroundPoints = numBackgroundPoints
	 * \param featureType Feature (e.g. texture) to be extracted. Check feat_dist in for valid combinations with a supported distance metric
	 * \param neighWeighting Local neighborhood weighting
	 * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus numLocNeighbors=1 -> 3x3 neighborhood
	 * \param numHistBins Number of histogram bins for histogram features. You might want to use a heuristic for this, see FeatureUtils.h
	 * \param aknn_algorithm exact or approximated knn or full distance matrix
	 * \param aknn_metric Distance metric. Check feat_dist for valid combinations with a supported feature type
	 * \param pixelWeight weight between eucl. feature and pixel distance in distance_metric::METRIC_EUC_sep, i.e. feat_dist::XY_EUCW
	 * \param perplexity t-sne perplexity, defines number of nearest neighbors: nn = perplexity * 3 +1
	 * \param numIterations number of t-sne gradient descent iterations
	 * \param exaggeration iterations with complete exaggeration of the attractive forces
	 * \param expDecay iterations required to remove the exaggeration using an exponential decay
	 * \param forceCalcBackgroundFeatures force the computation of features for background points, they are still not included in the embedding computation
	 */
	SpidrParameters(size_t numPoints, size_t numDims, ImgSize imgSize, std::string embeddingName, const float* dataVecBegin, size_t numForegroundPoints,
        feature_type featureType, loc_Neigh_Weighting neighWeighting, size_t numLocNeighbors, size_t numHistBins, float pixelWeight,
        knn_library aknn_algorithm, distance_metric aknn_metric,
        size_t perplexity, size_t numIterations, size_t exaggeration, size_t expDecay = 250, bool forceCalcBackgroundFeatures = false) :
        _numPoints(numPoints), _numDims(numDims), _imgSize(imgSize), _embeddingName(embeddingName), _numForegroundPoints(numForegroundPoints),
        _featureType(featureType), _neighWeighting(neighWeighting), _numHistBins(numHistBins), _pixelWeight(pixelWeight),
        _aknn_algorithm(aknn_algorithm), _aknn_metric(aknn_metric), _forceCalcBackgroundFeatures(forceCalcBackgroundFeatures),
        _perplexity_multiplier(3), _numIterations(numIterations), _exaggeration(exaggeration), _expDecay(expDecay), _has_preset_embedding(false)
    {
        set_perplexity(perplexity);  // sets nn based on perplexity
        set_numNeighborsInEachDirection(numLocNeighbors);  // sets _kernelWidth and _neighborhoodSize
        _numFeatureValsPerPoint = NumFeatureValsPerPoint(_featureType, _numDims, _numHistBins, _neighborhoodSize);
    }

	// Getter
	
	size_t get_nn() const { return _nn; }; 	// number of nn depends on perplexity, thus the user should not be able to set it
    size_t get_perplexity() const { return _perplexity; };
    size_t get_perplexity_multiplier() const { return _perplexity_multiplier; };

    size_t get_kernelWidth() const { return _kernelWidth; };
    size_t get_neighborhoodSize() const { return _neighborhoodSize; };
    size_t get_numNeighborsInEachDirection() const { return _numNeighborsInEachDirection; };

	// Setter

	/*! Setting the perplexity also changes the number of knn
	 * 
	 * \param perp t-sne perplexity, defines number of nearest neighbors: nn = perplexity * 3 +1
	*/
	void set_perplexity(size_t perp) {
        if (perp <= 0) 
		{
            spdlog::warn("SpidrParameters: perplexity must be positive - set to 5 instead");
            _perplexity = 5;
        }
		else
		{
			_perplexity = perp;
		}

		update_nn();	// sets nn based on perplexity
	}

	/*! Sets _numNeighborsInEachDirection, _kernelWidth, _neighborhoodSize
	 *
	 * kernelWidth		= (2 * _numNeighborsInEachDirection) + 1
	 * neighborhoodSize = kernelWidth^2
	 * 
	 * \param numNeighborsInEachDirection
	*/
	void set_numNeighborsInEachDirection(size_t numNeighborsInEachDirection) {
        _numNeighborsInEachDirection = numNeighborsInEachDirection;
        _kernelWidth = (2 * _numNeighborsInEachDirection) + 1;
        _neighborhoodSize = _kernelWidth * _kernelWidth;
    }

private:

	/*! nn = perplexity * 3 + 1
	*/
	void update_nn() {
		// see Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.
		_nn = _perplexity * _perplexity_multiplier + (size_t)1;

		// For small data sets, cap the kNN at the number of points
		if (_nn > _numPoints)
		{
			spdlog::warn("SpidrParameters: Few data points - reduce number nn to number of points");
			_nn = _numPoints;
		}
	}

public:
	// data
	size_t              _numPoints;             /*!< Number of points in the data set> */
	size_t              _numDims;               /*!< Number of dimensions/channels> */
	ImgSize             _imgSize;               /*!< Image height and width> */
	std::string         _embeddingName;         /*!< Name of the embedding > */
    size_t              _numForegroundPoints;   /*!< number of foreground points (background points are not included in the embedding > */

	// features
	feature_type        _featureType;           /*!< Type of data feature to be extracted > */
	size_t              _numFeatureValsPerPoint;/*!< Depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)> */
	loc_Neigh_Weighting _neighWeighting;        /*!< Weighting type of the neighborhood > */
	size_t              _numHistBins;           /*!< Number of bins in a histogram feature > */
    float               _pixelWeight;           /*!< For METRIC_EUC_sep: 0 is only attribute dist, 1 is only pixel dist > */
    bool                _forceCalcBackgroundFeatures; /*!< Usually features are not computed for the background, but you can force it anyway > */

	// distance
	knn_library         _aknn_algorithm;        /*!< kNN algo type, e.g. exact kNN vs approximated kNN > */
	distance_metric     _aknn_metric;           /*!< Distance between features/attributes > */

	// embedding
    size_t              _numIterations;         /*!< Number of gradient descent iterations> */
    size_t              _exaggeration;          /*!< Number of iterations with complete exaggeration of the attractive forces> */
    size_t              _expDecay;              /*!< Number of iterations required to remove the exaggeration using an exponential decay> */
	bool				_has_preset_embedding;	/*!< Whether an initial embedding is given, default: random>*/

private:
	size_t              _nn;                    /*!< Number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1> */
	const size_t        _perplexity_multiplier; /*!< Multiplied by the perplexity gives the number of nearest neighbors used> */
    size_t              _perplexity;            /*!< Perplexity value in evert distribution.> */
    size_t              _kernelWidth;           /*!< (2 * _numNeighborsInEachDirection) + 1;> */
    size_t              _neighborhoodSize;      /*!< _kernelWidth * _kernelWidth> */
    size_t              _numNeighborsInEachDirection;       /*!< Number of neighbors in each direction, i.e. 1 yields a 3x3 neighborhood> */

};
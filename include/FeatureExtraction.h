#pragma once

#include "FeatureUtils.h"	// struct ImgSize;

#include <vector>
#include <Eigen/Dense>

class SpidrParameters;
enum class loc_Neigh_Weighting : uint32_t;
enum class feature_type : uint32_t;

/*! Support class for SpidrAnalysis, used to extract data features
 * 
 * See FeatureUtils.h for info on the Feature class (output of the computation).
 * 
 * Use as:
 * FeatureExtraction featExtraction();
 * 	featExtraction.setup(_pointIDsGlobal, _attribute_data, _params, _backgroundIDsGlobal, _foregroundIDsGlobal);
 *	featExtraction.compute();
 *	const Feature dataFeats = featExtraction.output();
 * 
 * All feature extraction functions must have the signature (size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs)
 * 
 */
class FeatureExtraction
{
public:
    FeatureExtraction();
    ~FeatureExtraction();

    /*! Setup this helper class
     *
     * \param pointIDsGlobal reference to global IDs
     * \param attribute_data reference to data points
     * \param params Spidr parameters
     * \param backgroundIDsGlobal Ids for which NO features are computed
     * \param foregroundIDsGlobal Ids for which features are computed
     */
    void setup(const std::vector<unsigned int>& pointIDsGlobal, const std::vector<float>& attribute_data, const SpidrParameters& params, const std::vector<unsigned int>& backgroundIDsGlobal, const std::vector<unsigned int>& foregroundIDsGlobal);

    /*! Calculates features, basically calls initExtraction and extractFeatures
    */
    void compute();

    /*! Get feature data
     * 
     * See FeatureUtils.h for info on the Feature class
     * 
     * \return Feature: 
     */
    Feature output();

    /*! Sets _numNeighborsInEachDirection, _kernelWidth and _neighborhoodSize based
     *
     * \param size _numNeighborsInEachDirection
    */
    void setNumLocNeighbors(size_t size);

    /*! Set and _neighborhoodWeighting and computes _neighborhoodWeights
     *
     * \param weighting weighting type
    */
    void setNeighborhoodWeighting(loc_Neigh_Weighting weighting);

    /*! Sets number of histogram bins.
     * Best use some heuristic for this
     *
     * \param size number of bins
    */
    void setNumHistBins(size_t size);

    /*! called in SpidrAnalysis::stopComputation() but currently does nothing
     *
    */
    void stopFeatureCopmutation();	// 

    /*! Currently unused
     * \returns whether features are currently computed
    */
    bool requestedStop();

    loc_Neigh_Weighting getNeighborhoodWeighting();

private:

    /*! Inits some summary values of the data depending on the feature type and resizes the output
     * The summary values are min, max, mean and var per dimension. Not all
     * summary values are computed for each feature type
    */
    void initExtraction();

    /*! Compute spatial features of the data
     * Depending on _featType, these can be classic texture features or other indicator of spatial association. 
     * Sets the output variables.
     */
    void extractFeatures();

    /*! Calculate Texture histograms
     * For each dimension compute a 1D histogram of the neighborhood values for pointID.
     * Sets _outFeatures.
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void calculateHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Calculate Channel histograms
     * Compute one 1D histogram where each bin corresponds to one channel and counts active (above a threshold) values.
     * Note: a bin does therefor not correspond with a value range.
     * Currently only binary thresholding is implemented. That is, values >= 1 are counted. Data has to be thresholded by the user in advance.
     *
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void calculateChannelHistogram(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Calculate Local Indicator of Spatial Association features for each item
     * Compute Local Moran's I of the neighborhood values for pointID. 
     * Sets _outFeatures.
     * See doi:10.1111/j.1538-4632.1995.tb00338.x
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
    */
    void calculateLISA(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Calculate Geary's C features for each item
     * Compute Geary's C of the neighborhood values for pointID.
     * Sets _outFeatures.
     * See doi:10.1111/j.1538-4632.1995.tb00338.x
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
    */
    void calculateGearysC(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Multivariate normal distributions feature: covariance matrix and channel-wise mean
     * 
     * \param pointInd
     * \param neighborValues
    */
    void multivarNormDistDescriptor(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Adds two features: x and y location of data point
     *
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void addPixelLocationToAttributes(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Adds two features: normed x and y location of data point
     *  norm pixel range to attribute range: [0, largestPixelIndex] -> [_minAttriVal, _maxAttriVal]
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void addPixelLocationNormedToAttributes(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Sets the Feature per element to all it's neighbors attributes
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void allNeighborhoodVals(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Sets the Feature per element to all its neighbor's IDs
     * 
     * \param pointInd global ID of point for which to compute the feature
     * \param neighborValues data values of all neighborhood points, serialized like the attribute_data
     * \param neighborIDs global IDs of all neighborhood points
     */
    void allNeighborhoodIDs(size_t pointInd, std::vector<float> neighborValues, std::vector<int> neighborIDs);

    /*! Inits the neighborhood weighting
     * 
     * \param weighting weighting type
     */
    void weightNeighborhood(loc_Neigh_Weighting weighting);

    /*! Pointer to function that computer features
     * E.g. calculateHistogram or calculateLISA. 
     */
    void(FeatureExtraction::*featFunct)(size_t, std::vector<float>, std::vector<int>);

private:
    // Options 
    feature_type _featType;                         /*!< Type of feature to extract */
    distance_metric _distType;                      /*!< Distance between features */
    size_t       _numFeatureValsPerPoint;           /*!< depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)> */
    size_t       _numNeighborsInEachDirection;      /*!< Number of neighbors in each direction */
    size_t       _kernelWidth;                      /*!< Width of the kernel (2* _numNeighborsInEachDirection +1) */
    size_t       _neighborhoodSize;                 /*!< Square neighborhood centered around an item with _neighborhoodSize neighbors to the left, right, top and buttom */
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!< Weighting type of neighborhood kernel */
    std::vector<float> _neighborhoodWeights;        /*!< Weightings of neighborhood kernel */
    Eigen::VectorXf _neighborhoodWeights_eig;       /*!< Weightings of neighborhood kernel */
    float _neighborhoodWeightsSum;                  /*!< Sum of weightings in neighborhood kernel */
    size_t       _numHistBins;                      /*!< Number of bins in each histogram */
    bool _forceCalcBackgroundFeatures;              /*!< Force calculation of features for background data */
    bool _stopFeatureComputation;                   /*!< Stops the computation (TODO: break the openmp parallel loop) */

    // Data
    // Input
	ImgSize      _imgSize;                          /*!< image width and height > */
    size_t       _numDims;                          /*!< number of image channels > */
    size_t       _numPoints;                        /*!< number of data points, image width*height > */
    std::vector<unsigned int> _pointIDsGlobal;      /*!< global point IDs > */
    std::vector<float> _attribute_data;             /*!< actual data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointMdimN] > */
    std::vector<unsigned int> _backgroundIDsGlobal;  /*!< background IDs, per default, no features are computed for these > */
    std::vector<unsigned int> _foregroundIDsGlobal;  /*!< foreground IDs, per default all global IDs are foreground > */
    std::vector<float> _minMaxVals;                 /*!< Extrema for each dimension/channel, i.e. [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...] */
    float _maxAttriVal;                             /*!< Max attribute value overall */
    float _minAttriVal;                             /*!< Min attribute value overall */
    std::vector<float> _meanVals;                   /*!< Avg for each dimension/channel, i.e. [mean_Ch0, meam_Ch1, ...] */
    std::vector<float> _varVals;                    /*!< Variance estimate for each dimension/channel, i.e. [mean_Ch0, meam_Ch1, ...] */

    Eigen::MatrixXui _indices_mat_padded;            /*!< Eigen matrix of _pointIDsGlobal for easier neighborhood extraction, padded with pad size _numNeighborsInEachDirection and edge values> */

    // Output
    Feature _outFeatures;                            /*!< Features for each item. > */
};
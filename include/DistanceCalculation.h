#pragma once

#include <tuple>
#include <vector>
#include <string>   
#include "FeatureUtils.h"

class SpidrParameters;
enum class knn_library : uint32_t;
enum class distance_metric : uint32_t;
enum class feature_type : uint32_t;
enum class loc_Neigh_Weighting : uint32_t;

/*! Support class for SpidrAnalysis, used to compute distances between features
 * 
 * Use as:
 * DistanceCalculation distCalc();
 *  distCalc.setup(_dataFeats, _foregroundIDsGlobal, _params);
 *	distCalc.compute();
 *	_knn_indices = distCalc.get_knn_indices();
 *	_knn_distances = _distCalc.get_knn_distances();
 * 
 */
class DistanceCalculation 
{
public:
//    DistanceCalculation();


     /*! Set up the helper class
      *
      * \param dataFeatures features as computed with FeatureExtraction
      * \param foregroundIDsGlobal global IDs of all points which are used in the nearest neighbor computation
      * \param params spidr parameters
     */
    void setup(const Feature dataFeatures, const std::vector<unsigned int>& foregroundIDsGlobal, SpidrParameters& params);

    /*! Get knn indices and distances 
     * Use with std::tie(_knn_indices, _knn_distances) = output()
     * \returns knn indices and distances as a tuple
     */
    std::tuple< std::vector<int>, std::vector<float>> output() const; // tuple of indices and dists
    
    /*! Get knn indices
     * \returns knn indices 
     */
    std::vector<int> get_knn_indices() const;

    /*! Get knn distances
     * \returns knn distances
     */
    std::vector<float> get_knn_distances() const;

    /*! set knn algorithm
     * \param knn knn algorithm, e.g. exact or approx knn
     */
    void setKnnAlgorithm(knn_library knn);

    /*! set distance metric
     * \param metric distances metric
     */
    void setDistanceMetric(distance_metric metric);

    /*! Computes knn and prints some log info
     */
    void compute();

private:

    /*! Computes knn
     */
    void computekNN();

private:
    // Options
    feature_type _featureType;                      /*!< Type of feature > */
    knn_library _knn_lib;                           /*!< knn algorithm, e.g. exact or approx > */
    distance_metric _knn_metric;                    /*!< Distance metric between feature type > */
    size_t _nn;                                     /*!< number of nearest neighbors to be computed> */
    size_t _neighborhoodSize;                       /*!< number of neighbors = kernelWidth * kernelWidth > */
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!< neighborhood weights, used when calculating distance directly from high-dim points (_featureType is no feature/PCLOUD) > */
    float _pixelWeight;                             /*!< For METRIC_EUC_sep: 0 is only attribute dist, 1 is only pixel dist > */

    // Data
    // Input
    Feature _dataFeatures;                          /*!< Computed features to be compared> */
    size_t _numFeatureValsPerPoint;                 /*!< Feature Values per Point> */
    size_t _numDims;                                /*!< number of image channels> */
    size_t _numPoints;                              /*!< total number of points> */
    size_t _numForegroundPoints;                    /*!< number of foreground points> */
    size_t _numHistBins;                            /*!< number of histogram bins, if that was the feature> */
    std::string _embeddingName;                     /*!< Name of the embedding */
    const float* _dataVecBegin;                     /*!< Points to the first element in the data vector, might be used in some distance computatinos> */
    std::vector<unsigned int> _foregroundIDsGlobal; /*!< global foreground point IDs> */
    size_t _imgWidth;                               /*!< image width> */

    // Output
    std::vector<int> _knn_indices;                  /*!< knn indices, serialized > */
    std::vector<float> _knn_distances;              /*!< knn distances, serialized > */
};
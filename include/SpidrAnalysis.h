#pragma once

#include "SpidrAnalysisParameters.h"
#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "FeatureUtils.h"
#include "KNNUtils.h"

#include <vector>
#include <string>
#include <tuple>

/*! Main class for spatially aware embedding computation
 * 
 * Computes spatially aware t-SNE embeddings. 
 * Spatial information is incorporated by using distance functions between high-dimensional data points that take the spatial neighborhood of data points into account.
 * 
 * Use as:
 * SpidrAnalysis spidr(...);
 * spidr.compute();
 * const std::vector<float> embedding = spidr.output();
 * 
 * OR
 * SpidrAnalysis spidr();
 * spidr.setupData(...);
 * spidr.initializeAnalysisSettings(...);
 * spidr.compute();
 * const std::vector<float> embedding = spidr.output();
 *
 * You might also only compute nearest neighbors and use them in other embedding techniques than t-SNE (see python wrapper fit() function and examples)
 * 
 * Specific distance functions are written for hnswlib, see KNNDists.h
 * 
 */
class SpidrAnalysis
{
public:

    /*! Setup data and setting for spatially-aware embedding
     *
     * Use as:
     * SpidrAnalysis spidr();
     * spidr.setupData(...);
     * spidr.initializeAnalysisSettings(...);
     * spidr.compute();
     * const std::vector<float> embedding = spidr.output();
    */
    SpidrAnalysis();

    /*! Setup data and setting for spatially-aware embedding
     * 
     * For details see SpidrAnalysisParameters.h
     * 
     * Use as:
     * SpidrAnalysis spidr(...);
     * spidr.compute();
     * const std::vector<float> embedding = spidr.output();
     * 
     * \param attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointMdimN]
     * \param pointIDsGlobal number of data points
     * \param numDimensions number of dimensions
     * \param imgSize Width and height of image,. width*height = pointIDsGlobal
     * \param featType Feature (e.g. texture) to be extracted. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported distance metric
     * \param kernelType Local neighborhood weighting
     * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus numLocNeighbors=1 -> 3x3 neighborhood
     * \param numHistBins Number of histogram bins for histogram features. You might want to use a heuristic for this, see FeatureUtils.h
     * \param pixelDistanceWeight weight between eucl. feature and pixel distance in distance_metric::METRIC_EUC_sep, i.e. feat_dist::XY_EUCW
     * \param aknnAlgType exact or approximated knn or full distance matrix
     * \param aknnMetric Distance metric. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported feature type
     * \param numIterations number of t-sne gradient descent iterations
     * \param perplexity t-sne perplexity, defines number of nearest neighbors: nn = perplexity * 3 +1
     * \param exaggeration iterations with complete exaggeration of the attractive forces
     * \param expDecay iterations required to remove the exaggeration using an exponential decay
     * \param initial_embedding vector with initial embedding, serialized [point0X, point0Y, point1X, point1Y, ..., pointMX, pointMY]
     * \param embeddingName name of embedding
     * \param forceCalcBackgroundFeatures If background IDs are given you can force the computation of features for these data points
     * \param backgroundIDsGlobal IDs of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation. Backround points are added to the embedding in the lower left corner to keep the embedding length consistent with the number of points in the data
    */
    SpidrAnalysis(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const feature_type featType, \
        const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, const float pixelDistanceWeight, \
        const knn_library aknnAlgType, const distance_metric aknnMetric, \
        const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, \
        const std::vector<float>& initial_embedding, \
        const std::string embeddingName = "emd", bool forceCalcBackgroundFeatures = false, const std::vector<unsigned int>& backgroundIDsGlobal = std::vector<unsigned int>());

    /*! Setup data and setting for spatially-aware embedding
     *
     * For details see SpidrAnalysisParameters.h
     *
     * Use as:
     * SpidrAnalysis spidr(...);
     * spidr.compute();
     * const std::vector<float> embedding = spidr.output();
     *
     * \param attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointMdimN]
     * \param pointIDsGlobal number of data points
     * \param numDimensions number of dimensions
     * \param imgSize Width and height of image,. width*height = pointIDsGlobal
     * \param featType Feature (e.g. texture) to be extracted. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported distance metric
     * \param kernelType Local neighborhood weighting
     * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus numLocNeighbors=1 -> 3x3 neighborhood
     * \param numHistBins Number of histogram bins for histogram features. You might want to use a heuristic for this, see FeatureUtils.h
     * \param pixelDistanceWeight weight between eucl. feature and pixel distance in distance_metric::METRIC_EUC_sep, i.e. feat_dist::XY_EUCW
     * \param aknnAlgType exact or approximated knn or full distance matrix
     * \param aknnMetric Distance metric. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported feature type
     * \param numIterations number of t-sne gradient descent iterations
     * \param perplexity t-sne perplexity, defines number of nearest neighbors: nn = perplexity * 3 +1
     * \param exaggeration iterations with complete exaggeration of the attractive forces
     * \param expDecay iterations required to remove the exaggeration using an exponential decay
     * \param embeddingName name of embedding
     * \param forceCalcBackgroundFeatures If background IDs are given you can force the computation of features for these data points
     * \param backgroundIDsGlobal IDs of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation. Backround points are added to the embedding in the lower left corner to keep the embedding length consistent with the number of points in the data
    */
    SpidrAnalysis(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const feature_type featType, \
        const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, const float pixelDistanceWeight, \
        const knn_library aknnAlgType, const distance_metric aknnMetric, \
        const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, \
        const std::string embeddingName = "emd", bool forceCalcBackgroundFeatures = false, const std::vector<unsigned int>& backgroundIDsGlobal = std::vector<unsigned int>());

    SpidrAnalysis(const SpidrAnalysis&) = delete;
    SpidrAnalysis& operator= (const SpidrAnalysis&) = delete;

    /*! Set the data
     *
     * \param attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointndimn]
     * \param pointIDsGlobal number of data points
     * \param numDimensions number of dimensions
     * \param imgSize Width and height of image,. width*height = pointIDsGlobal
     * \param embeddingName name of embedding
     * \param backgroundIDsGlobal ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
    */
    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, \
        const std::vector<unsigned int>& backgroundIDsGlobal = std::vector<unsigned int>());

    /*! Set the data
     *
     * \param attribute_data high-dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointndimn]
     * \param pointIDsGlobal number of data points
     * \param numDimensions number of dimensions
     * \param imgSize Width and height of image,. width*height = pointIDsGlobal
     * \param embeddingName name of embedding
     * \param initial_embedding vector with initial embedding, serialized [point0X, point0Y, point1X, point1Y, ..., pointMX, pointMY]
     * \param backgroundIDsGlobal ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
    */
    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const std::string embeddingName, \
        const std::vector<float>& initial_embedding, const std::vector<unsigned int>& backgroundIDsGlobal = std::vector<unsigned int>());

    /*! Set the parameters of the entire Analysis
     * Use the input from e.g a GUI
     *
     * \param featType Feature (e.g. texture) to be extracted. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported distance metric
     * \param kernelType Local neighborhood weighting
     * \param numLocNeighbors Number of spatial Neighbors In Each Direction, thus 1 -> 3x3 neighborhood
     * \param numHistBins Number of histogram bins for histogram features. You might want to use a heuristic for this, see FeatureUtils.h
     * \param pixelDistanceWeight weight between eucl. feature and pixel distance in distance_metric::METRIC_EUC_sep, i.e. feat_dist::XY_EUCW
     * \param aknnAlgType exact or approximated knn or full distance matrix
     * \param aknnMetric Distance metric. Check feat_dist in SpidrAnalysisParameters.h for valid combinations with a supported feature type
     * \param numIterations number of t-sne gradient descent iterations
     * \param perplexity t-sne perplexity, defines number of nearest neighbors
     * \param exaggeration early exageration interations in t-sne gradient descent
     * \param expDecay exponential decay during t-sne gradient descent
     * \param forceCalcBackgroundFeatures If background IDs are given you can force the computation of features for these data points
     */
    void initializeAnalysisSettings(const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, const float pixelDistanceWeight, \
        const knn_library aknnAlgType, const distance_metric aknnMetric, \
        const size_t numIterations, const size_t perplexity, const size_t exaggeration, const size_t expDecay, bool forceCalcBackgroundFeatures = false);


	/*! Compute feature extraction and embedding
	 * Calls computeFeatures, computekNN and computeEmbedding, in that order
	 */
	void compute();

	/*! Compute Features from raw data
	 * sets _dataFeats, used for kNN computation in computekNN()
	 */
	void computeFeatures();

	/*! Compute kNN based on features
	 * sets _knn_indices and _knn_distances based on _dataFeats, used for embedding computation in computeEmbedding()
	 */
	void computekNN();

	/*! Compute t-SNE embedding
	 * based on _knn_indices and _knn_distances
     * Either call computeFeature() and computekNN() before of use setKnn()
	 */
	void computeEmbedding();

    /*! Release openGL context of the t-SNE computation
     */
    void stopComputation();

    // Getter

    /*! Return number of foreground points
    * Total number of points minus background points
     */
    const size_t getNumForegroundPoints();

    /*! Return total number of data points
     */
    const size_t getNumImagePoints();

    /*! Returns true of gradient descent is currently iterating
     */
    bool embeddingIsRunning();

    /*! Return reference to embdding
     */
     const std::vector<float> &output() const;

     /*! Return copy of embdding
      */
     const std::vector<float> output_copy() const;

	/*! Return reference to embdding with background
	 * Checks if during setupData() any background points were specified and, if so, adds them into a corner in the embedding
	 */
    const std::vector<float> &outputWithBackground();

    /*! Return embdding with added background points
     *
     * This function is useful in cases where you used this class for knn and distance computation and computed an embedding externally.
     *
     * Checks if during setupData() any background points were specified and, if so, adds them into a corner in the embedding
     *
     * \param emd_without_background embedding without background points
     * \returns embedding with background points
    */
    const std::vector<float> outputWithBackground(std::vector<float> emd_without_background);

    /*! Return copy of embdding with background
     * Checks if during setupData() any background points were specified and, if so, adds them into a corner in the embedding
     */
    const std::vector<float> outputWithBackground_copy() const;

    /*! Return current spidr parameters
     */
    const SpidrParameters getParameters() const;

    /*! Return computed data features
     */
    const Feature getDataFeatures() const;

    /* Returns _knn_indices, _knn_distances, use with std::tie(_knnIds, _knnDists) = getKnn(); */
    const std::tuple<std::vector<int>, std::vector<float>> getKnn() const;

    /*! Return copy knn indices, also see getKnn()
     */
    const std::vector<int> getKnnIndices() const;

    /*! Return knn distance, also see getKnn()
     */
    const std::vector<float> getKnnDistances() const;

    // Setter

    /* Set externally computed knn
     * Automatically sets number of points  
    */
    void setKnn(std::vector<int>& indices, std::vector<float>& distances);

private:
    
    // Setter

    /*! Sets feature type as in enum class feature_type in FeatureUtils.h
    *
    * \param feature_type_index, see enum class feature_type in FeatureUtils.h
    */
    void setFeatureType(const feature_type feature_type_index);

    /*! Sets feature type as in enum class loc_Neigh_Weighting in FeatureUtils.h
    *
    * \param loc_Neigh_Weighting_index, see enum class loc_Neigh_Weighting in FeatureUtils.h
    */
    void setKernelWeight(const loc_Neigh_Weighting loc_Neigh_Weighting_index);

    /*! Sets the number of spatially local pixel neighbors in each direction. Sets _params._kernelWidth and _params._neighborhoodSize as well*/
    void setNumLocNeighbors(const size_t num);

    /*! Sets the number of histogram bins */
    void setNumHistBins(const size_t num);

    /*! Sets knn algorithm type as in enum class feature_type in KNNUtils.h
    *
    * \param knn_library_index, see enum class feature_type in KNNUtils.h
    */
    void setKnnAlgorithm(const knn_library knn_library_index);

    /*! Sets knn algorithm type as in enum class distance_metric in KNNUtils.h
    *
    * \param distance_metric_index, see enum class distance_metric in KNNUtils.h
    */
    void setDistanceMetric(const distance_metric distance_metric_index);

    /*! Sets the perplexity and automatically determines the number of approximated kNN
    * nn = 3 * perplexity + 1 
    *
    * \param perplexity t-SNE perlexity
    */
    void setPerplexity(const size_t perplexity);
    /*! Sets the number of histogram bins */

    /*! Sets the number of gradient descent iteration */
    void setNumIterations(const size_t numIt);

    /*! Sets the exageration during gradient descent */
    void setExaggeration(const size_t exag);

    /*! Sets the exponential decay during gradient descent */
    void setExpDecay(const size_t expDacay);

    /*! Sets the pixelWeight for weighting attribute and pos distance with distance_metric::METRIC_EUC_sep */
    void setPixelDistanceWeight(const float pixelWeight);

    /*! Sets the size of a feature, derived from other parameters */
    void setNumFeatureValsPerPoint(feature_type featType, size_t numDims, size_t numHistBins, size_t neighborhoodSize);

    void setForceCalcBackgroundFeatures(const bool forceCalcBackgroundFeatures);


    // Utility
    
    /* Add bg points to emb, uses the ID info set for an instance of this class */
    void addBackgroundToEmbedding(std::vector<float>& emb, const std::vector<float>& emb_wo_bg) const;


private:
    // worker classes
    FeatureExtraction _featExtraction;					/*!< Class that computes features based on feature_type > */
    DistanceCalculation _distCalc;						/*!< Class that computes distances between features based on distance_metric > */
    TsneComputation _tsne;								/*!< Class that computes t-SNE embedding based on knn distances > */
    
    // data and settings
    std::vector<float> _attribute_data;					/*!< High dimensional data, serialized [point0dim0, point0dim1, ..., point1dim0, point1dim1, ... pointndimn] > */
    std::vector<unsigned int> _pointIDsGlobal;			/*!< Global data point IDs > */
    std::vector<unsigned int> _backgroundIDsGlobal;		/*!< ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation > */
    std::vector<unsigned int> _foregroundIDsGlobal;		/*!< ID of points which are used during the t-SNE embedding > */
    SpidrParameters _params;							/*!< Container for Spidr parameters like feature_type and distance_metric> */
    std::vector<float> _emd_with_backgound;             /*!< Used in SpidrAnalysis::outputWithBackground to return embedding with background > */
    std::vector<float> _initial_embedding;               /*!< Initial embedding, default: random > */

	// features and knn
	Feature _dataFeats;						            /*!< Computed features > */
	std::vector<int> _knn_indices ;						/*!< Computed knn indices> */
	std::vector<float> _knn_distances;			        /*!< Computed knn distances> */

};



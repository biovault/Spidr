#pragma once
#include "SpidrAnalysisParameters.h"

#ifdef __APPLE__
#include "glad/glad_3_3.h"
#define __gl3_h_
#endif
#include <GLFW/glfw3.h>

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"

#include <vector>
#include <string>

class SpidrParameters;

/*! Support class for SpidrAnalysis, used to compute a t-SNE embedding
 *
 * Builds on https://github.com/biovault/HDILib
 * 
 * Use as:
 *  TsneComputation tsne();
 *  tsne.setup(_knn_indices, _knn_distances, _params);
 *  tsne.compute();
 *	const std::vector<float>& emb = _tsne.output();
 *
 */
class TsneComputation
{
public:
    TsneComputation();

    /*! Set up the helper class
     *
     * \param knn_indices knn indices, global IDs
     * \param knn_distances knn distances
     * \param params spidr parameters
     * \param initial_embedding vector with initial embedding, serialized [point0X, point0Y, point1X, point1Y, ..., pointMX, pointMY]
     */
    void setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params, std::vector<float> initial_embedding = std::vector<float>());

    /*! Computes t-SNE embedding based on knn indices and distances
     */
    void compute();

    /*! Get t-SNE embedding after current iteration
     * \return t-SNE embedding
     */
    const std::vector<float>& output() const;

    // Setter

    void setVerbose(bool verbose);
    void setIterations(int iterations);
    void setExaggerationIter(int exaggerationIter);
    void setExponentialDecay(int exponentialDecay);
    void setPerplexity(float perplexity);
    void setNumDimensionsOutput(int numDimensionsOutput);

    /*! Stops embedding process at current gradient descent iteration
     */
    void stopGradientDescent();

    /*! Stops embedding process at current gradient descent iteration and sets internal _isMarkedForDeletion true
     */
    void markForDeletion();

    // Getter

    inline bool verbose() { return _verbose; }
    inline size_t iterations() { return _iterations; }
    inline int exaggerationIter() { return _exaggerationIter; }
    inline float perplexity() { return _perplexity; }
    inline int numDimensionsOutput() { return _numDimensionsOutput; }

    inline bool isTsneRunning() const { return _isTsneRunning; }
    inline bool isGradientDescentRunning() const { return _isGradientDescentRunning; }
    inline bool isMarkedForDeletion() const { return _isMarkedForDeletion; }

private:

    /*! Computes A-tSNE probability distribution
     */
    void initTSNE();

    /*! Calls initGradientDescent() and embed()
     */
    void computeGradientDescent();

    /*! Initializes GPU based gradient descent 
     */
    void initGradientDescent();

    /*! Performs gradient descent iterations
     */
    void embed();

    /*! Copies current embedding to output variable every 10 iterations
     */
    void copyFloatOutput();

private:
    // TSNE structures
    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type _probabilityDistribution;    /*!< Generator for a joint probability distribution that describes similarities in the high dimensional data > */
    hdi::dr::GradientDescentTSNETexture _GPGPU_tSNE;                                                    /*!< Main gpu based t-sne computation class> */
    hdi::data::Embedding<float> _embedding;                                                             /*!< Container for the embedding > */

    // Data
    std::vector<int> _knn_indices;              /*!< knn indices, global IDs, serialized> */
    std::vector<float> _knn_distances;          /*!< knn distances, serialized> */
    size_t _numForegroundPoints;                /*!< number of points to be embedded> */
    std::vector<float> _outputData;             /*!< output embedding> */

    // Options
    size_t _iterations;                         /*!< number of gradient descent iterations> */
    int _numTrees;                              /*!< unused, for compatibility with HDIlib: Number of trees used int the AKNN> */
    int _numChecks;                             /*!< unused, for compatibility with HDIlib: Number of checks used int the AKNN> */
    int _exaggerationIter;                      /*!< iterations with complete exaggeration of the attractive forces> */
    int _exponentialDecay;                      /*!< iterations required to remove the exaggeration using an exponential decay> */
    float _perplexity;                          /*!< Perplexity value in distribution> */
    int _perplexity_multiplier;                 /*!< 3. Multiplied by the perplexity gives the number of nearest neighbors used> */
    int _numDimensionsOutput;                   /*!< 2. > */
    int _nn;                                    /*!< number of nearest neighbors> */
    bool _has_inital_emb;                       /*!< Whether the user set an initial embedding> */

    // Flags
    bool _verbose;                              /*!< Controls number of print statements of HDIlib> */
    bool _isGradientDescentRunning;             /*!< Returns whether the gradient descent is currently ongoing> */
    bool _isTsneRunning;                        /*!< Returns whether the embedding process is currently ongoing> */
    bool _isMarkedForDeletion;                  /*!< Returns whether an instance of this class might be savely deleted> */

    size_t _continueFromIteration;              /*!< Currently unsed. Stores the iter number when stopGradientDescent() is called > */
	GLFWwindow* _offscreen_context;             /*!< GLFWwindow used for gpu-based gradient descent > */
};

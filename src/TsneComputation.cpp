#include "TsneComputation.h"

#include "EvalUtils.h"

#include <algorithm>            // std::min, max
#include <vector>
#include <assert.h>

#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/utils/scoped_timers.h"

// not present in glfw 3.1.2
#ifndef GLFW_FALSE
#define GLFW_FALSE 0
#endif

TsneComputation::TsneComputation() :
    _iterations(1000),
    _numTrees(4),
    _numChecks(1024),
    _exaggerationIter(250),
    _exponentialDecay(250),
    _perplexity(30),
    _perplexity_multiplier(3),
    _numDimensionsOutput(2),
    _verbose(false),
    _isGradientDescentRunning(false),
    _isTsneRunning(false),
    _isMarkedForDeletion(false),
    _continueFromIteration(0),
    _numForegroundPoints(0),
    _has_inital_emb(false),
_offscreen_context(nullptr)
{
    _nn = static_cast<int> (_perplexity * _perplexity_multiplier + 1);
}


void TsneComputation::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneComputation::setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params, std::vector<float> initial_embedding /* = std::vector<float>() */) {
    // SpidrParameters
    _iterations = params._numIterations;
    _perplexity = static_cast<float> (params.get_perplexity());
    _exaggerationIter = static_cast<unsigned int> (params._exaggeration);
    _exponentialDecay = static_cast<unsigned int> (params._expDecay);
    _nn = static_cast<int> (params.get_nn());                       // same as in constructor = _perplexity * 3 + 1;
    _numForegroundPoints = params._numForegroundPoints;         // if no background IDs are given, _numForegroundPoints = _numPoints
    _perplexity_multiplier = static_cast<int> (params.get_perplexity_multiplier());

    // Data
    _knn_indices = knn_indices;
    _knn_distances = knn_distances;

	spdlog::info("t-SNE computation: Num data points: {0} with {1} precalculated nearest neighbors. Perplexity: {2}, Iterations: {3}", _numForegroundPoints, params.get_nn(), _perplexity, _iterations);

    assert(_knn_indices.size() == _numForegroundPoints * _nn);

    // Set user-given initial embedding
    if (!initial_embedding.empty())
    {
        assert(params._has_preset_embedding);
        spdlog::info("TsneComputation::setup: Use user-provided initial embedding");

        _embedding = hdi::data::Embedding<float>(2, _numForegroundPoints);
        _embedding.getContainer() = initial_embedding;
        _has_inital_emb = params._has_preset_embedding;
    }
}


void TsneComputation::initTSNE()
{
        
    // Computation of the high dimensional similarities
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = _perplexity_multiplier;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;

		spdlog::info("tSNE initialized.");

        _probabilityDistribution.clear();
        _probabilityDistribution.resize(_numForegroundPoints);
		spdlog::info("Sparse matrix allocated.");

        hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
        double t = 0.0;
        {
            hdi::utils::ScopedTimer<double> timer(t);
            probabilityGenerator.computeGaussianDistributions(_knn_distances, _knn_indices, _nn, _probabilityDistribution, probGenParams);
        }
		spdlog::info("Probability distributions calculated.");
		spdlog::info("================================================================================");
		spdlog::info("A-tSNE: Compute probability distribution: {} seconds", t / 1000);
		spdlog::info("--------------------------------------------------------------------------------");
    }
}

void TsneComputation::initGradientDescent()
{
    _continueFromIteration = 0;

    _isTsneRunning = true;

    hdi::dr::TsneParameters tsneParams;

    tsneParams._embedding_dimensionality = _numDimensionsOutput;
    tsneParams._mom_switching_iter = _exaggerationIter;
    tsneParams._remove_exaggeration_iter = _exaggerationIter;
    tsneParams._exponential_decay_iter = _exponentialDecay;
    tsneParams._exaggeration_factor = 4 + _numForegroundPoints / 60000.0;
    tsneParams._presetEmbedding = _has_inital_emb;

    // Create a offscreen window
	if (!glfwInit()) {
		throw std::runtime_error("Unable to initialize GLFW.");
	}
#ifdef __APPLE__
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // invisible - ie offscreen, window
	_offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
	if (_offscreen_context == NULL) {
		glfwTerminate();
		throw std::runtime_error("Failed to create GLFW window");
	}
	glfwMakeContextCurrent(_offscreen_context);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		glfwTerminate();
		throw std::runtime_error("Failed to initialize OpenGL context");
	}
    // Initialize GPGPU-SNE
    _GPGPU_tSNE.initialize(_probabilityDistribution, &_embedding, tsneParams);
    
    copyFloatOutput();
}

// Computing gradient descent
void TsneComputation::embed()
{
    double elapsed = 0;
    double t = 0;
    {
		spdlog::info("A-tSNE: Computing gradient descent..\n");
        _isGradientDescentRunning = true;

        // Performs gradient descent for every iteration
        for (size_t iter = 0; iter < _iterations; ++iter)
        {
            hdi::utils::ScopedTimer<double> timer(t);
            if (!_isGradientDescentRunning)
            {
                _continueFromIteration = iter;
                break;
            }

            // Perform a GPGPU-SNE iteration
            _GPGPU_tSNE.doAnIteration();

            if (iter > 0 && iter % 10 == 0)
            {
                copyFloatOutput();
            }

            if (t > 1000)
				spdlog::info("Time: {}", t);

            elapsed += t;
        }
		glfwDestroyWindow(_offscreen_context);
		glfwTerminate();

        copyFloatOutput();
        
        _isGradientDescentRunning = false;
        _isTsneRunning = false;

    }

	spdlog::info("--------------------------------------------------------------------------------");
	spdlog::info("A-tSNE: Finished embedding of tSNE Analysis in: {} seconds", elapsed / 1000);
	spdlog::info("================================================================================");

}

void TsneComputation::compute() {
    initTSNE();
    computeGradientDescent();
}

// Copy tSNE output to our output
void TsneComputation::copyFloatOutput()
{
    _outputData = _embedding.getContainer();
}

const std::vector<float>& TsneComputation::output() const
{
    return _outputData;
}

void TsneComputation::setVerbose(bool verbose)
{
    _verbose = verbose;
}

void TsneComputation::setIterations(int iterations)
{
    _iterations = iterations;
}

void TsneComputation::setExaggerationIter(int exaggerationIter)
{
    _exaggerationIter = exaggerationIter;
}

void TsneComputation::setExponentialDecay(int exponentialDecay)
{
    _exponentialDecay = exponentialDecay;
}

void TsneComputation::setPerplexity(float perplexity)
{
    _perplexity = perplexity;
}

void TsneComputation::setNumDimensionsOutput(int numDimensionsOutput)
{
    _numDimensionsOutput = numDimensionsOutput;
}

void TsneComputation::stopGradientDescent()
{
    _isGradientDescentRunning = false;
}

void TsneComputation::markForDeletion()
{
    _isMarkedForDeletion = true;

    stopGradientDescent();
}

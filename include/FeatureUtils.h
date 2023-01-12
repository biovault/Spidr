#pragma once

#include <cmath>
#include <vector>
#include <utility>   // std::pair
#include <tuple>     // std::tuple
#include <memory>    // std::unique_ptr
#include <algorithm> // std::for_each
#include <execution> // std::par_unseq

#include "SpidrAnalysisParameters.h"
#include <Eigen/Dense>

/*! Normalizes all values in vec wrt to normVal
 * Basically normedVec[i] = vec[i] / normVal
 *
 * \param vec data vector
 * \param normVal nomalization constant
 */
template<typename T>
void NormVector(std::vector<T>& vec, T normVal){

    std::for_each(std::execution::par_unseq, std::begin(vec), std::end(vec), [normVal](auto& val) {
        val /= normVal;
    });

}

/*! compute a row of Pascal's triangle  https://en.wikipedia.org/wiki/Pascal%27s_triangle
 *
 * \param n row number
 * \return triangle values
 */
std::vector<unsigned int> PascalsTriangleRow(const size_t n);

/*! Compute 2d binomial kernel
 *
 * \param width kernel will be of size width*width
 * \param norm whether to normalize all values
 * \return serialized 2d kernel [row0col0, row0col1, ... row1col0, row1col1, ... rowWidthcolWidth]
 */
std::vector<float> BinomialKernel2D(const size_t width, norm_vec norm = norm_vec::NORM_NONE);

/*! Compute 1d gaussian kernel
 *
 * \param width width of kernen
 * \param sd standard deviation
 * \return kernel values
 */
std::vector<float> GaussianKernel1D(const size_t width, const float sd = 1);

/*! Compute 2d gaussian kernel
 *
 * \param width kernel will be of size width*width
 * \param sd standard deviation
 * \param norm whether to normalize all values
 * \return serialized 2d kernel [row0col0, row0col1, ... row1col0, row1col1, ... rowWidthcolWidth]
 */
std::vector<float> GaussianKernel2D(const size_t width, const float sd = 1, norm_vec norm = norm_vec::NORM_NONE);

/*! heuristic to determine number of histogram bins given number of items
* 
* ceil(sqrt(numItems))
 *
 * \param numItems number of values
 * \return number of histogram bins
 */
unsigned int SqrtBinSize(unsigned int numItems);

/*! heuristic to determine number of histogram bins given number of items
*
* ceil(log2(numItems) + 1))
 *
 * \param numItems number of values
 * \return number of histogram bins
 */
unsigned int SturgesBinSize(unsigned int numItems);

/*! heuristic to determine number of histogram bins given number of items
*
* ceil((2 * pow(numItems, 1.0/3))))
 *
 * \param numItems number of values
 * \return number of histogram bins
 */
unsigned int RiceBinSize(unsigned int numItems);


/*! Get data for all neighborhood point ids
 * Padding: if neighbor is outside selection (outside the image), assign 0 to all dimension values
 * 
 * \param neighborIDs vector of global neighborhood point IDs
 * \param _attribute_data reference to the data
 * \param _neighborhoodSize number of neighbors
 * \param _numDims number of dimensions
 * \return data layout with dimension d and neighbor n: [n0d0, n0d1, n0d2, ..., n1d0, n1d1, ..., n2d0, n2d1, ...], size = neighborhoodSize * numDims

 */
std::vector<float> getNeighborhoodValues(const std::vector<int>& neighborIDs, const std::vector<float>& attribute_data, const size_t neighborhoodSize, const size_t numDims);

/*! Calculate the minimum and maximum value for each channel
 *
 * \param numPoints number of data points
 * \param numDims number of dimensions
 * \param attribute_data data
 * \return vector with [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMinMaxPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data, const std::vector<unsigned int>& globalPointIDs) {
    std::vector<float> minMaxVals(2 * numDims, 0);

    // for each dimension iterate over all values
    // remember data stucture (point1 d0, point1 d1,... point1 dn, point2 d0, point2 d1, ...)
    for (unsigned int dimCount = 0; dimCount < numDims; dimCount++) {
        // init min and max
        float currentVal = attribute_data[dimCount];
        minMaxVals[2 * dimCount] = currentVal;
        minMaxVals[2 * dimCount + 1] = currentVal;

        for(auto& pointID: globalPointIDs) {
            currentVal = attribute_data[pointID * numDims + dimCount];
            // min
            if (currentVal < minMaxVals[2 * dimCount])
                minMaxVals[2 * dimCount] = currentVal;
            // max
            else if (currentVal > minMaxVals[2 * dimCount + 1])
                minMaxVals[2 * dimCount + 1] = currentVal;
        }
    }

    return minMaxVals;
}

/*! Calculate the mean value for each channel
 *
 * \param numPoints number of data points
 * \param numDims number of dimensions
 * \param attribute_data data
 * \return vector with [mean_Ch0, mean_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMeanPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data, const std::vector<unsigned int>& globalPointIDs) {
    std::vector<float> meanVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        for (auto& pointID : globalPointIDs) {
            sum += attribute_data[pointID * numDims + dimCount];
        }

        meanVals[dimCount] = sum / numPoints;
    }

    return meanVals;
}

/*! Calculate estimate of the variance
 *  Assuming equally likely values, a (biased) estimated of the variance is computed for each dimension
 *
 * \param numPoints number of data points
 * \param numDims number of dimensions
 * \param attribute_data data
 * \return vector with [var_Ch0, var_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcVarEstimate(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data, const std::vector<float> &meanVals, const std::vector<unsigned int>& globalPointIDs) {
    std::vector<float> varVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        float temp_diff = 0;
        for (auto& pointID : globalPointIDs) {
            temp_diff = attribute_data[pointID * numDims + dimCount] - meanVals[dimCount];
            sum += (temp_diff * temp_diff);
        }

        varVals[dimCount] = (sum > 0) ? sum / numPoints : 0.00000001f;   // make sure that variance is not zero for noise-free data

    }

    return varVals;
}


namespace Eigen {
    // add short matrix version for unsigned int, works just as MatrixXi
	typedef Matrix<unsigned int, -1, -1> MatrixXui;
}


/*! Helper struct for padding, example from Eigen
 * From https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
 * This example was used as the basis for padAllDirections
 * 
 * What does this do? 
 * pad{3, 5} creates a sequence of indices [0 0 0 1 2]
 * Now a slicing operation A(seqN(i,m), seqN(j,n) selects a block starting at i,j having m rows, and n columns (equivalent to A.block(i,j,m,n)).
 * Slicing like A(pad{3,N}, pad{3,N}) will thus return a matrix that was padded left and top with 2 rows
 *
*/
struct padUpperLeft {
	Eigen::Index size() const { return out_size; }
	Eigen::Index operator[] (Eigen::Index i) const { return std::max<Eigen::Index>(0, i - (out_size - in_size)); }
	Eigen::Index in_size, out_size;
};

/*! Helper struct for constant padding, see padEdge
 *  Creates a sequence of indices: padAllDirections{3, 1} -> [0 0 1 2 2]
 *  Thus padding const values like [0 1 2] -> [(0) 0 1 2 (2)]
 *
 * See padEdge() for usage
 * 
 * \param in_size length of sequence to pad
 * \param pad_size number of pad values on both sites
 */
struct padAllDirections {
	padAllDirections(Eigen::Index in_size, Eigen::Index pad_size) : in_size(in_size), pad_size(pad_size) {}
	Eigen::Index size() const { return in_size + 2 * pad_size; }
	Eigen::Index operator[] (Eigen::Index i) const { return std::min<Eigen::Index>(std::max<Eigen::Index>(0, i - pad_size), in_size - 1); }
	Eigen::Index in_size, pad_size;
};


/*! Pads a matrix (2d) in all directions with the border values
 * Similar to numpy's np.pad(..., mode='edge')
 *
 * \param mat input matrix
 * \param pad_size number of pad values on each site
 */
Eigen::MatrixXui padEdge(Eigen::MatrixXui mat, Eigen::Index pad_size);

/*! Get rectangle neighborhood point ids for one data item
 *  
 * Padding: constant border value
 * See https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
 * 
 * \param coord_row point row ID (of upper left corner)
 * \param coord_col point col ID (of upper left corner)
 * \param kernelWidth extract Ids from kernelWidth*kernelWidth block
 * \param padded_ids
 * \return
 */
std::vector<int> getNeighborhoodInds(const unsigned int coord_row, const unsigned int coord_col, const size_t kernelWidth, Eigen::MatrixXui* padded_ids);


/*! Returns a random Eigen vector sampled from a uniform distribution
 * \param len length of the vector
 * \param lo lower bound of uniform distribution
 * \param high higher bound of uniform distribution
*/
Eigen::VectorXf randomVector(unsigned int len, float lo, float hi);

/*! Base class for histograms
*/
template <class scalar_type>
class Histogram_Base
{
public:
    Histogram_Base() = delete;
    Histogram_Base(float min, float max, unsigned int numberOfBins);
    // The last bin might be smaller than the rest of (max-min)/binWidth does not yield an integer
    Histogram_Base(float min, float max, float binWidth);

    void fill(const float value);
    void fill(const std::vector<float> values);

    // Getter

    unsigned int getNumBins() const { return _counts.size(); };
    unsigned int getCount(unsigned int bin) const { return _counts[bin]; };

    unsigned int getCount() const { return _countBinValid; };
    unsigned int getCountAll() const { return _countBinTotal; };
    unsigned int getCountUnderflow() const { return _countBinUnderflow; };
    unsigned int getCountOverflow() const { return _countBinOverflow; };

    float getMin() const { return _minVal; };
    float getMax() const { return _maxVal; };
    float getBinLower(unsigned int bin) const { return _minVal + bin * _binWidth; };
    float getBinUpper(unsigned int bin) const { return _minVal + (bin + 1) * _binWidth; };

    auto cbegin() const { return _counts.cbegin(); };
    auto cend() const { return _counts.cend(); };

    scalar_type operator[](int index) const;

    Eigen::Vector<scalar_type, -1> counts() const { return _counts; };
    Eigen::VectorXf normalizedCounts() const { return _counts.template cast<float>() / _counts.sum(); };

protected:
    Eigen::Vector<scalar_type, -1> _counts;
    unsigned int _countBinOverflow;
    unsigned int _countBinUnderflow;
    unsigned int _countBinTotal;
    unsigned int _countBinValid;
    float _binWidth;
    float _binNormed;
    float _minVal;
    float _maxVal;
    unsigned int _numBins;

    void commonInit();
};


/*! Histogram class
 *
 * If newVal == binMax then it will not count as overflow but is counted in the largest bin
 */
class Histogram : public Histogram_Base<unsigned int>
{
public:
    Histogram() = delete;
    Histogram(float min, float max, unsigned int numberOfBins) : Histogram_Base(min, max, numberOfBins) { };
    Histogram(float min, float max, float binWidth) : Histogram_Base(min, max, binWidth) { };

};

/*! Weighted Histogram class
 *
 * Introduces the fill_weighted function to the histogram base class
 * 
 * If newVal == binMax then it will not count as overflow but is counted in the largest bin
 */
class Histogram_Weighted : public Histogram_Base<float>
{
public:
    Histogram_Weighted() = delete;
    Histogram_Weighted(float min, float max, unsigned int numberOfBins) : Histogram_Base(min, max, numberOfBins) { };
    Histogram_Weighted(float min, float max, float binWidth) : Histogram_Base(min, max, binWidth) { };

    void fill_weighted(const float value, const float weight);
    void fill_weighted(const std::vector<float> values, const std::vector<float> weights);

};


/*! Base class for channel histograms
 *
 * This histogram class counts active channel values, i.e. one bin is one channel and not a value range
 */
template <class scalar_type>
class Channel_Histogram_Base
{
public:
    Channel_Histogram_Base() = delete;
    Channel_Histogram_Base(size_t numDims, float threshold = 1);
    Channel_Histogram_Base(std::vector<float> tresholds);

    void fill_ch(const size_t ch, const float value);
    void fill_ch(const size_t ch, const std::vector<float> values);

    // Getter

    unsigned int getNumBins() const { return _counts.size(); };
    unsigned int getCount(unsigned int bin) const { return _counts[bin]; };

    unsigned int getCount() const { return _totalBinCounts; };

    auto cbegin() const { return _counts.cbegin(); };
    auto cend() const { return _counts.cend(); };

    scalar_type operator[](size_t index) const;

    Eigen::Vector<scalar_type, -1> counts() const { return _counts; };
    const Eigen::Vector<scalar_type, -1>* countsp() const { return &_counts; };
    Eigen::VectorXf normalizedCounts() const { return _counts.template cast<float>() / _counts.sum(); };

    std::vector<scalar_type> counts_std() const { return std::vector<scalar_type>(_counts.data(), _counts.data() + _counts.size()); };
    std::vector<float> normalizedCounts_std() const { auto eigen_counts_norm = normalizedCounts(); return std::vector<scalar_type>(eigen_counts_norm.data(), eigen_counts_norm.data() + eigen_counts_norm.size()); };


protected:
    Eigen::Vector<scalar_type, -1> _counts;

    std::vector<float> _tresholds;

    size_t _totalBinCounts;
    size_t _numBins;

};


/*! Channel Histogram class
 * *
 * If newVal == binMax then it will not count as overflow but is counted in the largest bin
 */
class Channel_Histogram : public Channel_Histogram_Base<unsigned int>
{
public:
    Channel_Histogram() = delete;
    Channel_Histogram(unsigned int numDims, float threshold = 1) : Channel_Histogram_Base(numDims, threshold) { };
    Channel_Histogram(std::vector<float> tresholds) : Channel_Histogram_Base(tresholds) { };

};

/*! Weighted Channel Histogram class
 *
 * Introduces the fill_weighted function to the histogram base class
 *
 * If newVal == binMax then it will not count as overflow but is counted in the largest bin
 */
class Channel_Histogram_Weighted : public Channel_Histogram_Base<float>
{
public:
    Channel_Histogram_Weighted() = delete;
    Channel_Histogram_Weighted(size_t numDims, float threshold = 1) : Channel_Histogram_Base(numDims, threshold) { };
    Channel_Histogram_Weighted(std::vector<float> tresholds) : Channel_Histogram_Base(tresholds) { };

    void fill_ch_weighted(const size_t ch, const float value, const float weight);
    void fill_ch_weighted(const size_t ch, const std::vector<float> values, const std::vector<float> weights);

};

/*! Compute the variance of given data
 *
 * \params vec data values
 * \return variance
 */
float variance(Eigen::VectorXf vec);

/*! Compute the covariance of two given data
 *
 * \params vec data values
 * \return variance
 */
float covariance(Eigen::VectorXf vec1, Eigen::VectorXf vec2);

/*! Compute the covariance matrix of given data
 *  Assumes uniformly distributed data
 * 
 * see https://stackoverflow.com/a/15142446 and https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance
 * also https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Unbiasedness for discussion over (1 / (neighborhood.cols() - 1)) and (1 / neighborhood.cols())

 * \params vec data values
 * \return variance
 */
Eigen::MatrixXf covmat(Eigen::MatrixXf data);

/*! Compute the covariance matrix of given data
 *
 * see https://stackoverflow.com/a/15142446 and https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Definition_of_sample_covariance
 * also https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Unbiasedness
 * see https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Weighted_samples
 * 
 * \params vec data values
 * \params probs probablities
 * \return variance
 */
Eigen::MatrixXf covmat(Eigen::MatrixXf data, Eigen::VectorXf probs);

/*! Feature struct: mean vector, covariance matrix and it's determinant
*/
typedef struct Multivar_normal{
    Eigen::VectorXf mean_vec;
    Eigen::MatrixXf cov_mat;
    float cov_mat_det;

    Multivar_normal() = delete;
    Multivar_normal(Eigen::VectorXf m, Eigen::MatrixXf c) : mean_vec(m), cov_mat(c), cov_mat_det(c.determinant()) {};
    Multivar_normal(Eigen::VectorXf m, Eigen::MatrixXf c, float d) : mean_vec(m), cov_mat(c), cov_mat_det(d) {};
} Multivar_normal;

/*! Feature struct: mean vector, covariance matrix and square root of it's determinant
*/
typedef struct MeanCov_feat {
    Eigen::VectorXf mean_vec;
    Eigen::MatrixXf cov_mat;
    float cov_mat_det_sqrt;

    MeanCov_feat() = delete;
    MeanCov_feat(Eigen::VectorXf m, Eigen::MatrixXf c, float d) : mean_vec(m), cov_mat(c), cov_mat_det_sqrt(d) {};
} MeanCov_feat;

/*! Compute Multivar_normal feature (mean vector, covariance matrix and square root of it's determinant)
 *
 * \params vec data values
 * \return Multivar_normal struct
 */
Multivar_normal compMultiVarFeatures(Eigen::MatrixXf data);

/*! Compute Multivar_normal feature (mean vector, covariance matrix and square root of it's determinant)
 *
 * \params vec data values
 * \params probs probablities
 * \return Multivar_normal struct
 */
Multivar_normal compMultiVarFeatures(Eigen::MatrixXf data, Eigen::VectorXf probs);

/*! Interface class for Feature data
 * 
 * Empty base class to allow FeatureData<T> store arbitrary data while keeping a vector of IFeatureData* in the Feature class
 */
class IFeatureData
{
};

/*! Stores feature data for one data point
*/
template<class T>
class FeatureData : public IFeatureData
{
public:
    FeatureData(T d) : data(d) {};
    T data;
};


/*! Features for all data
 *
 * Use as:
 * Feature features();
 * features.resize(_numPoints);
 * 
 * // example feature: channel-wise histogram of spatially neighboring data points
 * std::vector<Eigen::VectorXf> feat(_numDims);             // vector of histograms per channel
 * for (size_t dim = 0; dim < _numDims; dim++)
 *      feat(dim) = computeHistogram();
 * 
 * features.at(dataPointID) = new FeatureData<std::vector<Eigen::VectorXf>>(feat);
 * 
 */
class Feature
{
public:
    Feature() { };
    ~Feature() { }

    /*! For setting feature data
     * \param ID point ID
     * \return reference of pointer to Feature data of point ID
    */
    IFeatureData*& at(size_t ID) { return featdata.at(ID); };

    /*! For getting feature data when you work with const Feature and want to make sure to not modify anything
     * \param ID point ID
     * \return pointer to Feature data of point ID
    */
    IFeatureData* get(size_t ID) const { return featdata.at(ID); };

    /*! Resize the feature data vector
    * \param newSize number of data points
     */
    void resize(size_t newSize) { featdata.resize(newSize); };

private:
    std::vector<IFeatureData*> featdata;    /* < Vector of feature data > */
};

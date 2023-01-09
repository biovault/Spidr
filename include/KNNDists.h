#pragma once
#include "hnswlib/hnswlib.h"    // defines USE_SSE and USE_AVX and includes intrinsics

#if defined(__GNUC__)
#define PORTABLE_ALIGN32hnsw __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32hnsw __declspec(align(32))
#endif

#include <omp.h>

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product, std:accumulate 
#include <algorithm> // std::find, fill, sort
#include <vector>
#include <iterator>
#include <thread>
#include <atomic>

#include "hdi/data/map_mem_eff.h" // hdi::data::MapMemEff

#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>    //  MatrixBase::sqrt()

#include "SpidrAnalysisParameters.h"
#include "FeatureUtils.h"
#include "KNNUtils.h"

/* ! Add new distance metrics to the hnswlib
 * 
 * To add a new metric MYMETRIC, define: 
 *   - A class MYMETRICSpace that inherits from hnswlib::SpaceInterface
 *   - A static float MYMETRIC_dist function with the signature (const void *pVect1v, const void *pVect2v, const void *qty_ptr)
 *   - A struct MYMETRICSpace_params
 * 
 * pVect1v and pVect2v will be pointers to the features of two points that are to be compared, qty_ptr points to the parameter struct
 * 
*/
namespace hnswlib {

    /* ! Replacement for the openmp '#pragma omp parallel for' directive
     * The method is borrowed from nmslib, https://github.com/nmslib/nmslib/blob/master/similarity_search/include/thread_pool.h
     */
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        }
        else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        }
                        catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }

    }


    // ---------------
    // Quadratic form for 1D Histograms
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_QF {
        size_t dim;
        size_t bin;
        ::std::vector<float> A;     // bin similarity matrix for 1D histograms: entry A_ij refers to the sim between bin i and bin j 
        Eigen::MatrixXf weights;    // same as A
    };

    static float
        QFSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        FeatureData<std::vector<Eigen::VectorXf>>* histos1 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v);
        FeatureData<std::vector<Eigen::VectorXf>>* histos2 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v);

        const space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;
        const float* pWeight = sparam->A.data();

        float res = 0;
        float t1 = 0;
        float t2 = 0;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )
            for (size_t i = 0; i < nbin; i++) {
                t1 = histos1->data[d][i] - histos2->data[d][i];
                for (size_t j = 0; j < nbin; j++) {
                    t2 = histos1->data[d][j] - histos2->data[d][j];
                    res += *(pWeight + i * nbin + j) * t1 * t2;
                }
            }
        }

        return res;
    }

    // This one is much slower
    //static float
    //    QFEigenSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    //    Eigen::VectorXf* histos1 = (static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v)->data).data();
    //    Eigen::VectorXf* histos2 = (static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v)->data).data();

    //    const space_params_QF* sparam = (space_params_QF*)qty_ptr;
    //    const size_t ndim = sparam->dim;
    //    const size_t nbin = sparam->bin;
    //    const Eigen::MatrixXf weights = sparam->weights;

    //    float res = 0;
    //    float t1 = 0;
    //    float t2 = 0;

    //    Eigen::VectorXf diff;

    //    // add the histogram distance for each dimension
    //    for (size_t d = 0; d < ndim; d++) {
    //        // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )

    //        diff = *histos1 - *histos2;
    //        res += diff.transpose() * weights * diff;
    //        
    //        // point to histograms of next dimension
    //        histos1++;
    //        histos2++;
    //    }

    //    return res;
    //}

    static float
        QFSqrSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        std::vector<Eigen::VectorXf>* histos1 = &(static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v)->data);
        std::vector<Eigen::VectorXf>* histos2 = &(static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v)->data);

        float* pVect1 = nullptr;
        float* pVect2 = nullptr;

        space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        size_t nbin4 = nbin >> 2 << 2;		// right shift by 2, left-shift by 2: create a multiple of 4

        float res = 0;
        float PORTABLE_ALIGN32hnsw TmpRes[8];			// memory aligned float array
        __m128 v1, v2, TmpSum, wRow, diff;			// write in registers of 128 bit size
        float *pA, *pEnd1, *pW, *pWend, *pwR;
        unsigned int wloc;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            pA = sparam->A.data();					// reset to first weight for every dimension

           // calculate the QF distance for each dimension
            pVect1 = (histos1->at(d)).data();
            pVect2 = (histos2->at(d)).data();

           // 1. calculate w = (pVect1-pVect2)
            std::vector<float> w(nbin);
            wloc = 0;
            pEnd1 = pVect1 + nbin4;			// point to the first dimension not to be vectorized
            while (pVect1 < pEnd1) {
                v1 = _mm_loadu_ps(pVect1);					// Load the next four float values
                v2 = _mm_loadu_ps(pVect2);
                diff = _mm_sub_ps(v1, v2);					// substract all float values
                _mm_store_ps(&w[wloc], diff);				// store diff values in memory
                pVect1 += 4;								// advance pointer to position after loaded values
                pVect2 += 4;
                wloc += 4;
            }

            // manually calc the rest dims
            for (wloc; wloc < nbin; wloc++) {
                w[wloc] = *pVect1 - *pVect2;
                pVect1++;
                pVect2++;
            }

            // 2. calculate d = w'Aw
            for (unsigned int row = 0; row < nbin; row++) {
                TmpSum = _mm_set1_ps(0);
                pW = w.data();					// pointer to first float in w
                pWend = pW + nbin4;			// point to the first dimension not to be vectorized
                pwR = pW + row;
                wRow = _mm_load1_ps(pwR);					// load one float into all elements fo wRow

                while (pW < pWend) {
                    v1 = _mm_loadu_ps(pW);
                    v2 = _mm_loadu_ps(pA);
                    TmpSum = _mm_add_ps(TmpSum, _mm_mul_ps(wRow, _mm_mul_ps(v1, v2)));	// multiply all values and add them to temp sum values
                    pW += 4;
                    pA += 4;
                }
                _mm_store_ps(TmpRes, TmpSum);
                res += TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

                // manually calc the rest dims
                for (unsigned int uloc = static_cast<unsigned int>(nbin4); uloc < nbin; uloc++) {
                    res += *pwR * *pW * *pA;
                    pW++;
                    pA++;
                }
            }

            // point to next dimension is done in the last iteration
            // of the for loop in the rest calc under point 1. (no pVect1++ necessary here)
        }

        return res;
    }


    class QFSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_QF params_;

    public:
        QFSpace(size_t dim, size_t bin, bin_sim ground_type = bin_sim::SIM_EUC) {
            spdlog::info("KNNDist: create QFSpace");

            fstdistfunc_ = QFSqr;
            // Not entirely sure why this only shows positive effects for high bin counts...
            if (bin >= 12)
            {
                fstdistfunc_ = QFSqrSSE;
            }

            data_size_ = sizeof(std::vector<Eigen::VectorXf>);

            ::std::vector<float> A = BinSimilarities(bin, ground_type);
            
            Eigen::MatrixXf weights = Eigen::Map<Eigen::MatrixXf>(&A[0], bin, bin);;

            params_ = { dim, bin, A, weights };
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~QFSpace() {}
    };
       
    // ---------------
    //    Hellinger
    // ---------------

    // data struct for distance calculation in HellingerSpace
    struct space_params_Hel {
        size_t dim;
        size_t bin;
    };

    static float
        HelSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        FeatureData<std::vector<Eigen::VectorXf>>* histos1 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect1v);
        FeatureData<std::vector<Eigen::VectorXf>>* histos2 = static_cast<FeatureData<std::vector<Eigen::VectorXf>>*>((IFeatureData*)pVect2v);

        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
       
        const space_params_Hel* sparam = (space_params_Hel*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        float res = 0;

        // Calculate Hellinger distance based on Bhattacharyya coefficient 
        float binSim = 0;
        float histDiff = 1;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            histDiff = 1;
            for (size_t b = 0; b < nbin; b++) {
                binSim = histos1->data[d][b] * histos2->data[d][b];
                histDiff -= ::std::sqrt(binSim);
            }
            res += (histDiff>=0) ? ::std::sqrt(histDiff) : 0; // sometimes histDiff is slightly below 0 due to rounding errors
        }

        return (res);
    }


    class HellingerSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Hel params_;

    public:
        HellingerSpace(size_t dim, size_t bin) {
            spdlog::info("KNNDist: create HellingerSpace");
            fstdistfunc_ = HelSqr;
            params_ = { dim, bin };
            data_size_ = sizeof(std::vector<Eigen::VectorXf>);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~HellingerSpace() {}
    };


    // ---------------
    //    Adapt L2 space
    // ---------------

    struct space_params_L2Feat {
        size_t dim;
        DISTFUNC<float> L2distfunc_;
    };


    static float
        L2FeatSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        //FeatureData<std::vector<float>>* histos1 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v);
        //FeatureData<std::vector<float>>* histos2 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v);
        float *pVect1 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v)->data).data();
        float *pVect2 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v)->data).data();

        const space_params_L2Feat* sparam = (space_params_L2Feat*)qty_ptr;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        return L2distfunc_(pVect1, pVect2, &(sparam->dim));
    }


    class L2FeatSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

        space_params_L2Feat params_;

    public:
        L2FeatSpace(size_t dim) {
            spdlog::info("KNNDist: create L2FeatSpace");
            fstdistfunc_ = L2FeatSqr;

            dim_ = dim;
            data_size_ = sizeof(std::vector<float>);

            // The actual hnswlib L2 norm function is part of the space_params_L2Feat
            // since L2FeatSqr has to access the feature vector correctly before calling it
            params_ = { dim_, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif

        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~L2FeatSpace() {}
    };


    // ---------------
    //    Adapt L2 space with weights like the InnerProduct above
    //    For use with attribtues + pos features
    // ---------------

    struct space_params_L2sepFeat {
        size_t dim;
        float weight;
        DISTFUNC<float> L2distfunc_;
    };


    static float
        L2sepFeatSqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        //FeatureData<std::vector<float>>* histos1 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v);
        //FeatureData<std::vector<float>>* histos2 = static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v);
        float* pVect1 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect1v)->data).data();
        float* pVect2 = (static_cast<FeatureData<std::vector<float>>*>((IFeatureData*)pVect2v)->data).data();

        const space_params_L2sepFeat* sparam = (space_params_L2sepFeat*)qty_ptr;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;
        float weight = sparam->weight;

        size_t pixelpos_dims = 2;
        size_t attribute_dims = sparam->dim - pixelpos_dims;

        float attribute_dist = L2distfunc_(pVect1, pVect2, &attribute_dims);
        float pixelpos_dist = L2Sqr(pVect1 + attribute_dims, pVect2 + attribute_dims, &pixelpos_dims);

        float dist = (1.0f - weight) * attribute_dist + weight * pixelpos_dist;

        return dist;
    }


    class L2sepFeatSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

        space_params_L2sepFeat params_;

    public:
        L2sepFeatSpace(size_t dim, float weight = 0.5f) {
            spdlog::info("KNNDist: create L2sepFeatSpace with weight {}", weight);
            fstdistfunc_ = L2sepFeatSqr;

            dim_ = dim;
            data_size_ = sizeof(std::vector<float>);

            // The actual hnswlib L2 norm function is part of the space_params_L2Feat
            // since L2FeatSqr has to access the feature vector correctly before calling it
            params_ = { dim_, weight, L2Sqr };

            size_t dim_attributes = dim_ - 2;       // (x,y) coordinates were added as features
#if defined(USE_SSE) || defined(USE_AVX)
            if (dim_attributes % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim_attributes % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim_attributes > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim_attributes > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif

        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void* get_dist_func_param() {
            return &params_;
        }

        ~L2sepFeatSpace() {}
    };


    // ---------------
    //    Point cloud distance (Chamfer)
    // ---------------

    // data struct for distance calculation in ChamferSpace
    struct space_params_Chamf {
        size_t dim;
        Eigen::VectorXf weights;         // neighborhood similarity matrix
        size_t neighborhoodSize;
        DISTFUNC<float> L2distfunc_;
    };

    static float
        ChamferDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const float* valsN1Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect1v)->data.data(); // pointer to vector with values in neighborhood 1
        const float* valsN2Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect2v)->data.data();

        // parameters
        const space_params_Chamf* sparam = (space_params_Chamf*)qty_ptr;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const size_t ndim = sparam->dim;
        const Eigen::VectorXf weights = sparam->weights;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        //Eigen::VectorXf colDistMins(valsN1.cols()); // (2 * (params._numNeighborsInEachDirection) + 1) * (2 * (params._numNeighborsInEachDirection) + 1)
        //Eigen::VectorXf rowDistMins(valsN1.cols());

        Eigen::MatrixXf distMat(neighborhoodSize, neighborhoodSize);

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (int n1 = 0; n1 < neighborhoodSize; n1++) {
            for (int n2 = 0; n2 < neighborhoodSize; n2++) {
                distMat(n1, n2) = L2distfunc_(valsN1Begin + (n1*ndim), valsN2Begin + (n2*ndim), &ndim);
            }
        }
        // Using the SSE function from HSNW is faster than then the matrix version from Eigen
        // There is probably a smart formulation with the Eigen matrices that would be better though...
        // not faster:
        //for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
        //    distMat.col(n1) = (valsN1.colwise() - valsN2.col(n1)).colwise().squaredNorm();
        //}

        // weight min of each col and row, and sum over them
        //colDistMins = distMat.rowwise().minCoeff();
        //rowDistMins = distMat.colwise().minCoeff();

        //return colSum / numNeighbors1 + rowSum / numNeighbors2;
        return (distMat.rowwise().minCoeff().dot(weights) + distMat.colwise().minCoeff().dot(weights)) / neighborhoodSize;
    }


    class ChamferSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Chamf params_;

    public:
        ChamferSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting) {
            spdlog::info("KNNDist: create ChamferSpace");
            fstdistfunc_ = ChamferDist;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<Eigen::MatrixXf>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1.0f); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1.0f);  break;  // no implemented weighting type given. 
            }

            Eigen::VectorXf weights = Eigen::Map<Eigen::VectorXf>(&A[0], neighborhoodSize);;

            params_ = { dim, weights, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~ChamferSpace() {}
    };


    // ---------------
    //    Point cloud distance (Sum of squared distances)
    // ---------------

// data struct for distance calculation in SSDSpace
    struct space_params_SSD {
        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numNeighborsInEachDirection) + 1) * (2 * (params._numNeighborsInEachDirection) + 1)
        DISTFUNC<float> L2distfunc_;
    };

    static float
        SumSquaredDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const float* valsN1Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect1v)->data.data(); // pointer to vector with values in neighborhood 1
        const float* valsN2Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect2v)->data.data();

        // parameters
        space_params_SSD* sparam = (space_params_SSD*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const std::vector<float> weights = sparam->A;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        float res = 0;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                res += (weights[n1] + weights[n2]) * L2distfunc_(valsN1Begin + (n1*ndim), valsN2Begin + (n2*ndim), &ndim);

            }
        }

        return res / (neighborhoodSize * neighborhoodSize);
    }


    class SSDSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_SSD params_;

    public:
        SSDSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting) {
            spdlog::info("KNNDist: create SSDSpace");
            fstdistfunc_ = SumSquaredDist;
            data_size_ = sizeof(FeatureData<Eigen::MatrixXf>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1.0f); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1.0f);  break;  // no implemented weighting type given. 
            }

            params_ = { dim, A, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~SSDSpace() {}
    };


    // ---------------
    //    Point cloud distance (Hausdorff distances)
    // ---------------

    struct space_params_Haus {
        size_t dim;
        Eigen::VectorXf weights;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numNeighborsInEachDirection) + 1) * (2 * (params._numNeighborsInEachDirection) + 1)
        DISTFUNC<float> L2distfunc_;
    };


    static float
        HausdorffDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const float* valsN1Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect1v)->data.data(); // pointer to vector with values in neighborhood 1
        const float* valsN2Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect2v)->data.data();

        // parameters
        space_params_Haus* sparam = (space_params_Haus*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const Eigen::VectorXf weights = sparam->weights;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        //std::vector<float> colDistMins(neighborhoodSize, FLT_MAX);
        //std::vector<float> rowDistMins(neighborhoodSize, FLT_MAX);
        
        Eigen::MatrixXf distMat(neighborhoodSize, neighborhoodSize);

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                distMat(n1, n2) = L2distfunc_(valsN1Begin + (n1*ndim), valsN2Begin + (n2*ndim), &ndim);
            }
        }

        // weight min of each col and row
        //Eigen::VectorXf weightedMinsR = distMat.rowwise().minCoeff().cwiseProduct(weights);
        //Eigen::VectorXf weightedMinsC = distMat.colwise().minCoeff().transpose().cwiseProduct(weights);

        //// max of weighted mins
        //float maxR = weightedMinsR.maxCoeff();
        //float maxC = weightedMinsC.maxCoeff();

        return std::max(distMat.rowwise().minCoeff().cwiseProduct(weights).maxCoeff(), distMat.colwise().minCoeff().transpose().cwiseProduct(weights).maxCoeff());
        //return std::max(maxR, maxC);
    }


    class HausdorffSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Haus params_;

    public:
        HausdorffSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting) {
            spdlog::info("KNNDist: create HausdorffSpace");
            fstdistfunc_ = HausdorffDist;
            //data_size_ = featureValsPerPoint * sizeof(float);
            data_size_ = sizeof(FeatureData<Eigen::MatrixXf>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1.0f); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1.0f);  break;  // no implemented weighting type given. 
            }
            Eigen::VectorXf weights = Eigen::Map<Eigen::VectorXf>(&A[0], neighborhoodSize);;

            params_ = { dim, weights, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~HausdorffSpace() {}
    };

    // ---------------
    //    Point cloud distance (Hausdorff _median distances)
    // ---------------


    static float
        HausdorffDist_median(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const float* valsN1Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect1v)->data.data(); // pointer to vector with values in neighborhood 1
        const float* valsN2Begin = static_cast<FeatureData<Eigen::MatrixXf>*>((IFeatureData*)pVect2v)->data.data();

        // parameters
        space_params_Haus* sparam = (space_params_Haus*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const Eigen::VectorXf weights = sparam->weights;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        Eigen::MatrixXf distMat(neighborhoodSize, neighborhoodSize);

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2 (colDist) and vice versa (rowDist)
        // Weight the colDist and rowDist with the inverse of the number of items in the neighborhood
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                distMat(n1, n2) = L2distfunc_(valsN1Begin + (n1*ndim), valsN2Begin + (n2*ndim), &ndim);
            }
        }

        // weight min of each col and row
        Eigen::VectorXf weightedMinsR = distMat.rowwise().minCoeff().cwiseProduct(weights);
        Eigen::VectorXf weightedMinsC = distMat.colwise().minCoeff().transpose().cwiseProduct(weights);

		// find median of mins
        float colMedian = CalcMedian(weightedMinsR.data(), weightedMinsR.data() + neighborhoodSize, neighborhoodSize);
        float rowMedian = CalcMedian(weightedMinsC.data(), weightedMinsC.data() + neighborhoodSize, neighborhoodSize);

        assert(colMedian < FLT_MAX);
        assert(rowMedian < FLT_MAX);

        return (colMedian + colMedian) / 2;
    }

    class HausdorffSpace_median : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Haus params_;

    public:
        HausdorffSpace_median(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting) {
            spdlog::info("KNNDist: create HausdorffSpace_median");
            fstdistfunc_ = HausdorffDist_median;
            data_size_ = sizeof(FeatureData<Eigen::MatrixXf>);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1.0f); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1.0f);  break;  // no implemented weighting type given. 
            }
            Eigen::VectorXf weights = Eigen::Map<Eigen::VectorXf>(&A[0], neighborhoodSize);;

            params_ = { dim, weights, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~HausdorffSpace_median() {}
    };


    // ---------------
    //    Multivariate Covariant Matrix Space
    // ---------------

    // function takes: mean vec 1, covmat 1, sqrt of covmat det 1, mean vec 2, covmat 2, sqrt of covmat det 2
    template<typename MTYPE>
    using COVMATDIST = MTYPE(*)(const Eigen::VectorXf&, const Eigen::MatrixXf&, const float, const Eigen::VectorXf&, const Eigen::MatrixXf&, const float);

    struct space_params_Bhattacharyya {
        COVMATDIST<float> fstdistfunc_;
    };

    // Bhattacharyya distance between two multivariate normal distributions 
    // 
    // Computing the distance like this might fails if the covariance matrices are not well behaved, see below for a more robust implementation of this distance
    // 
    // https://en.wikipedia.org/wiki/Bhattacharyya_distance
    // https://doi.org/10.1016/S0031-3203(03)00035-9
    //float distBhattacharyya(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
    //	Eigen::MatrixXf covmat_comb = (covmat1 + covmat2) / 2.0f;
    //	Eigen::VectorXf mean_diff = mean1 - mean2;
    //    const float det_comb = covmat_comb.determinant();
    //    
    //    assert(det_comb > 0);
    //    assert(det_sqrt_1 > 0);
    //    assert(det_sqrt_2 > 0);

    //    if (det_comb > 1e-5f)
    //        return 0.125f * mean_diff.transpose() * covmat_comb.inverse() * mean_diff + 0.5f * std::logf(det_comb / (det_sqrt_1 * det_sqrt_2));
    //    else
    //        return 0.0f;
    //}

    typedef struct SVD {
        Eigen::MatrixXf u;
        Eigen::VectorXf d;
        Eigen::MatrixXf v;
        SVD() = delete;
        SVD(Eigen::MatrixXf u, Eigen::VectorXf d, Eigen::MatrixXf v) : u(u), d(d), v(v) {};
    } SVD;

    typedef Eigen::Matrix<bool, -1, 1> VectorXb;


    SVD svd(Eigen::MatrixXf mat) {
        Eigen::BDCSVD<Eigen::MatrixXf> bdcsvd;
        bdcsvd.compute(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        return SVD{ bdcsvd.matrixU(), bdcsvd.singularValues(), bdcsvd.matrixV() };
    }

    inline float LogDetReduced(const SVD& svd, const Eigen::MatrixXf& mat, std::vector<int> activeD) {

        Eigen::MatrixXf Ut = svd.u(Eigen::all, activeD).transpose();
        Eigen::MatrixXf V = svd.v(Eigen::all, activeD);

        Eigen::MatrixXf reduced = Ut * mat * V;
        float det = reduced.determinant();

        return std::log(std::abs(det));
    }

    inline std::vector<int> GetActiveDIds(Eigen::VectorXf vec) {
        std::vector<int> ids_ActiveD;

        VectorXb activeD = vec.array() > 1e-5f;

        for (int i = 0; i < vec.size(); i++) {
            if (activeD[i] == true)
                ids_ActiveD.push_back(i);
        }

        return ids_ActiveD;
    }

    /*! Bhattacharyya distance between two multivariate normal distributions 
     * See https://stats.stackexchange.com/a/429723 and https://en.wikipedia.org/wiki/Bhattacharyya_distance
    */
    float distBhattacharyya(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        Eigen::MatrixXf covmat_comb = (covmat1 + covmat2) / 2.0f;
        Eigen::VectorXf mean_diff = mean1 - mean2;

        // log determinant terms

        SVD c_svd = svd(covmat_comb);

        std::vector<int> activeDIds = GetActiveDIds(c_svd.d);

        float a = c_svd.d(activeDIds).array().log().sum();
        float b = LogDetReduced(c_svd, covmat1, activeDIds) / (-2.0f);
        float c = LogDetReduced(c_svd, covmat2, activeDIds) / (-2.0f);

        float batlogdet = a + b + c;

        // Mahalanobis term

        // check validity
        if (activeDIds.size() < static_cast<size_t>(c_svd.d.size()) )
        {
            std::vector<int> allIds{ static_cast<int>(c_svd.d.size()) };
            std::iota(allIds.begin(), allIds.end(), 0);
            std::vector<int> inactiveDIds;

            std::set_difference(allIds.begin(), allIds.end(), activeDIds.begin(), activeDIds.end(),
                std::inserter(inactiveDIds, activeDIds.begin()));

            Eigen::VectorXf checkSpan = c_svd.u(Eigen::all, inactiveDIds).transpose() * mean_diff;

            if (std::any_of(checkSpan.begin(), checkSpan.end(), [](auto& val) { return val != 0; }))
            {
                spdlog::warn("KNNDists: distBhattacharyya: FLT_MAX.");
                return FLT_MAX;
            }
        }


        Eigen::VectorXf inverse_activeDs = c_svd.d(activeDIds).array().inverse();
        Eigen::MatrixXf inverse_activeDs_diag = inverse_activeDs.asDiagonal();
        Eigen::MatrixXf inverse_activeDs_diag2 = inverse_activeDs.asDiagonal().toDenseMatrix();

        Eigen::MatrixXf activeV = c_svd.v(Eigen::all, activeDIds);
        Eigen::MatrixXf activeU = c_svd.u(Eigen::all, activeDIds);

        Eigen::VectorXf maha = activeV * inverse_activeDs_diag * activeU.transpose() * mean_diff;

        return (mean_diff.cwiseProduct(maha)).sum() / 8 + batlogdet / 2;
    }


    // Bhattacharyya distance, only determinant ratio
    // i.e. for two distributions with the same means
    float distDetRatio(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        const float det_comb = ((covmat1 + covmat2) / 2.0f).determinant();
        
        assert(det_comb > 0);
        assert(det_sqrt_1 > 0);
        assert(det_sqrt_2 > 0);

        return std::logf(det_comb / (det_sqrt_1 * det_sqrt_2));
    }

    // Bhattacharyya distance, only Mahalanobis part
    float distBhattacharyyaMean(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        Eigen::MatrixXf covmat_comb = (covmat1 + covmat2) / 2.0f;
        Eigen::VectorXf mean_diff = mean1 - mean2;
        const float det_comb = covmat_comb.determinant();

        assert(det_comb > 0);

        return 0.125f * mean_diff.transpose() * covmat_comb.inverse() * mean_diff;
    }


    // Correlation Matrix distance
    // http://dx.doi.org/10.1109/VETECS.2005.1543265
    float CMD(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        return 1 - (covmat1*covmat2).trace() / (covmat1.norm() * covmat2.norm());
    }

    // The Fréchet distance between multivariate normal distributions
    // https://doi.org/10.1016/0047-259X(82)90077-X
    float FrechetDistGeneral(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        return (mean1- mean2).squaredNorm() + (covmat1 + covmat2 - 2 * (covmat2*covmat1).sqrt()).trace();
    }
    // Fréchet distance without the means comparison
    float FrechetDistCovMat(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        return (covmat1 + covmat2 - 2 * (covmat2*covmat1).sqrt()).trace();
    }

    // Frobenius between covmatrices
    float FrobeniusCovMat(const Eigen::VectorXf& mean1, const Eigen::MatrixXf& covmat1, const float det_sqrt_1, const Eigen::VectorXf& mean2, const Eigen::MatrixXf& covmat2, const float det_sqrt_2) {
        return (covmat1 - covmat2).squaredNorm();
    }


    /*! Calls one of many distance functions between distriptors for multivariate normal distributions (mean vector,  covariance matrix, covariance matrix determinant) as specified by the user
    */
    static float
        MultiVarCovMat(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        MeanCov_feat pVect1 = static_cast<FeatureData<MeanCov_feat>*>((IFeatureData*)pVect1v)->data;
        MeanCov_feat pVect2 = static_cast<FeatureData<MeanCov_feat>*>((IFeatureData*)pVect2v)->data;

        const space_params_Bhattacharyya* sparam = (space_params_Bhattacharyya*)qty_ptr;
        COVMATDIST<float> covmatdist = sparam->fstdistfunc_;

        // Bhattacharyya distance
        return covmatdist(pVect1.mean_vec, pVect1.cov_mat, pVect1.cov_mat_det_sqrt, pVect2.mean_vec, pVect2.cov_mat, pVect2.cov_mat_det_sqrt);
    }

    class MultiVarCovMat_Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_Bhattacharyya params_;

    public:
        MultiVarCovMat_Space(distance_metric distanceMetric) {
            spdlog::info("KNNDist: create MultiVarCovMat_Space");

            data_size_ = sizeof(FeatureData<MeanCov_feat>);
            fstdistfunc_ = MultiVarCovMat;

            if (distanceMetric == distance_metric::METRIC_BHATTACHARYYA)
                params_ = { distBhattacharyya };
            else if (distanceMetric == distance_metric::METRIC_DETMATRATIO)
                params_ = { distDetRatio };
            else if (distanceMetric == distance_metric::METRIC_BHATTACHARYYATESTONLYMEANS)
                params_ = { distBhattacharyyaMean };
            else if (distanceMetric == distance_metric::METRIC_CMD_covmat)
                params_ = { CMD };
            else if (distanceMetric == distance_metric::METRIC_FRECHET_Gen)
                params_ = { FrechetDistGeneral };
            else if (distanceMetric == distance_metric::METRIC_FRECHET_CovMat)
                params_ = { FrechetDistCovMat };
            else if (distanceMetric == distance_metric::METRIC_FROBENIUS_CovMat)
                params_ = { FrobeniusCovMat };
            else
            {
                params_ = { distBhattacharyya };
                spdlog::error("KNNDists: ERROR: No suitable distance for multivar covmat space provided. Defaulting to Bhattacharyya distance.");
            }

        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *)&params_;
        }

        ~MultiVarCovMat_Space() {}
    };

}

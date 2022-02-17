#pragma once
#include "KNNUtils.h"
#include "KNNDists.h"
#include "FeatureUtils.h"

#include "spdlog/spdlog-inl.h"
#include "hnswlib/hnswlib.h"

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product, std:accumulate, std::iota
#include <algorithm> // std::find, fill, sort
#include <assert.h>


metricPair MakeMetricPair(feature_type ft, distance_metric dm) {
    return std::make_tuple(ft, dm);
}

distance_metric GetDistMetricFromMetricPair(const metricPair metricPair) {
    return std::get<1>(metricPair);
}

feature_type GetFeatureTypeFromMetricPair(const metricPair metricPair) {
    return std::get<0>(metricPair);
}

template<typename T>
T CalcMedian(std::vector<T>& vec, size_t vecSize) {
	T median;
    const T half = static_cast<const T> (0.5);

	size_t n = vecSize / 2;
	std::nth_element(vec.begin(), vec.begin() + n, vec.end());
	T vn = vec[n];
	if (vecSize % 2 == 1)	// uneven length
	{
		median = vn;
	}
	else					// even length, median is average of the central two items
	{
		std::nth_element(vec.begin(), vec.begin() + n - 1, vec.end());
		median = half *(vn + vec[n - 1]);
	}

	return median;
}
template float CalcMedian<float>(std::vector<float>& vec, size_t vecSize);

template<typename T>
T CalcMedian(T* first, T* last, size_t vecSize) {
    T median;
    const T half = static_cast<const T>( 0.5);

    size_t n = vecSize / 2;
    std::nth_element(first, first + n, last);
    T vn = *(first + n);
    if (vecSize % 2 == 1)	// uneven length
    {
        median = vn;
    }
    else					// even length, median is average of the central two items
    {
        std::nth_element(first, first + n - 1, last);
        median = half * (vn + *(first + n - 1));
    }

    return median;
}
template float CalcMedian<float>(float* first, float* last, size_t vecSize);

std::vector<float> BinSimilarities(size_t num_bins, bin_sim sim_type, float sim_weight) {
	std::vector<float> A(num_bins*num_bins, -1);
	size_t ground_dist_max = num_bins - 1;

	if (sim_type == bin_sim::SIM_EUC) {
		for (int i = 0; i < (int)num_bins; i++) {
			for (int j = 0; j < (int)num_bins; j++) {
				A[i * num_bins + j] = 1 - (float(std::abs(i - j)) / float(ground_dist_max));
			}
		}
	}
	else if (sim_type == bin_sim::SIM_EXP) {
		for (int i = 0; i < (int)num_bins; i++) {
			for (int j = 0; j < (int)num_bins; j++) {
				A[i * num_bins + j] = ::std::exp(-1 * sim_weight * float(std::abs(i - j)));
			}
		}
	}
	else if (sim_type == bin_sim::SIM_UNI) {
		std::fill(A.begin(), A.end(), 1.0f);
	}

	// if there is a -1 in A, this value was not set (invalid ground_type option selected)
	assert(std::find(A.begin(), A.end(), -1.0f) == A.end());

	return A;
}

std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const Feature& dataFeatures, hnswlib::SpaceInterface<float> *space, const std::vector<unsigned int>& foregroundIDsGlobal, const size_t nn) {
    auto numForegroundPoints = foregroundIDsGlobal.size();

    std::vector<int> indices(numForegroundPoints * nn, -1);
    std::vector<float> distances_squared(numForegroundPoints * nn, -1);

	spdlog::info("Distance calculation: Build akNN Index");

    hnswlib::HierarchicalNSW<float> appr_alg(space, numForegroundPoints, 16, 200, 0);   // use default HNSW values for M, ef_construction random_seed

    // add data points: each data point holds indMultiplier values (number of feature values)
    // add the first data point outside the parallel loop
    appr_alg.addPoint((const void*)(dataFeatures.get(foregroundIDsGlobal[0])), (std::size_t)0);

#ifdef NDEBUG
    // This loop is for release mode, it's parallel loop implementation from hnswlib
    int num_threads = std::thread::hardware_concurrency();
    hnswlib::ParallelFor(1, numForegroundPoints, num_threads, [&](size_t i, size_t threadId) {
        appr_alg.addPoint((const void*)(dataFeatures.get(foregroundIDsGlobal[i])), (hnswlib::labeltype)i);
        });
#else
    // This loop is for debugging, when you want to sequentially add points
    for (int i = 1; i < numForegroundPoints; ++i)
    {
        appr_alg.addPoint((const void*)(dataFeatures.get(foregroundIDsGlobal[i])), (hnswlib::labeltype)i);
    }
#endif
    
	spdlog::info("Distance calculation: Search akNN Index");

    // query dataset
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < numForegroundPoints; ++i)
    {
        // find nearest neighbors
        auto top_candidates = appr_alg.searchKnn((void*)(dataFeatures.get(foregroundIDsGlobal[i])), (hnswlib::labeltype)nn);
        while (top_candidates.size() > nn) {
            top_candidates.pop();
        }

        assert(top_candidates.size() == nn);

        // save nn in _knn_indices and _knn_distances 
        auto* distances_offset = distances_squared.data() + (i * nn);
        auto indices_offset = indices.data() + (i * nn);
        int j = 0;
        while (top_candidates.size() > 0) {
            auto rez = top_candidates.top();
            distances_offset[nn - j - 1] = rez.first;
            indices_offset[nn - j - 1] = appr_alg.getExternalLabel(rez.second);
            top_candidates.pop();
            ++j;
        }
    }
    
    return std::make_tuple(indices, distances_squared);
}

std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const Feature& dataFeatures, hnswlib::SpaceInterface<float> *space, const std::vector<unsigned int>& foregroundIDsGlobal, const size_t nn, bool fullDistMat) {
    auto numForegroundPoints = foregroundIDsGlobal.size();
    
    std::vector<std::pair<int, float>> indices_distances(numForegroundPoints);
	std::vector<int> knn_indices(numForegroundPoints*nn, -1);
	std::vector<float> knn_distances_squared(numForegroundPoints*nn, -1.0f);

	hnswlib::DISTFUNC<float> distfunc = space->get_dist_func();
	void* params = space->get_dist_func_param();

	// only used if fullDistMat == true 
	std::vector<int> idx_row(nn);
	std::iota(idx_row.begin(), idx_row.end(), 0);

	// For each point, calc distances to all other
	// and take the nn smallest as kNN
	for (int i = 0; i < (int)numForegroundPoints; i++) {
		// Calculate distance to all points  using the respective metric
#ifdef NDEBUG
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)numForegroundPoints; j++) {
            indices_distances[j] = std::make_pair(j, distfunc(dataFeatures.get(foregroundIDsGlobal[i]), dataFeatures.get(foregroundIDsGlobal[j]), params));
        }

		if (!fullDistMat)
		{
			// compute knn, not full distance matrix
			assert(nn < numForegroundPoints);
			// sort all distances to point i
			std::sort(indices_distances.begin(), indices_distances.end(), [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second; });
	
			// Take the first nn indices 
			std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_indices.begin() + i * nn, [](const std::pair<int, float>& p) { return p.first; });
		}
		else
		{
			assert(nn == numForegroundPoints);
			// for full distance matrix, sort the indices depending on the distances
			std::sort(idx_row.begin(), idx_row.end(), [&indices_distances](int i1, int i2) {return indices_distances[i1].second < indices_distances[i2].second; });

			// Take the first nn indices (just copy them from the sorted indices)
			std::copy(idx_row.begin(), idx_row.begin() + nn, knn_indices.begin() + i * nn);
		}

		// Take the first nn distances 
		std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_distances_squared.begin() + i * nn, [](const std::pair<int, float>& p) { return p.second; });
	}

	return std::make_tuple(knn_indices, knn_distances_squared);
}

std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat(const Feature& dataFeatures, hnswlib::SpaceInterface<float> *space, const std::vector<unsigned int>& foregroundIDsGlobal) {
	// set nn = numForegroundPoints and don't sort the nn
	return ComputeExactKNN(dataFeatures, space, foregroundIDsGlobal, foregroundIDsGlobal.size(), true);
}


hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const feature_type feature_type, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t numHistBins, const float pixelWeight) {
    // chose distance metric
    hnswlib::SpaceInterface<float> *space = NULL;
    spdlog::info("Distance calculation: Metric {}", logging::distance_metric_name(knn_metric));

    switch (knn_metric) {
    case distance_metric::METRIC_QF:
        assert(numHistBins > 0);
        space = new hnswlib::QFSpace(numDims, numHistBins);
        break;

    case distance_metric::METRIC_HEL:
        assert(numHistBins > 0);
        space = new hnswlib::HellingerSpace(numDims, numHistBins);
        break;

    case distance_metric::METRIC_EUC:
        if ((feature_type == feature_type::PIXEL_LOCATION) || (feature_type == feature_type::PIXEL_LOCATION_RANGENORM))
            space = new hnswlib::L2FeatSpace(numDims+2);    // (x,y) coordinates are added as features
        else
            space = new hnswlib::L2FeatSpace(numDims);
        break;

    case distance_metric::METRIC_CHA:
        space = new hnswlib::ChamferSpace(numDims, neighborhoodSize, neighborhoodWeighting);
        break;

    case distance_metric::METRIC_SSD:
        space = new hnswlib::SSDSpace(numDims, neighborhoodSize, neighborhoodWeighting);
        break;

    case distance_metric::METRIC_HAU:
        space = new hnswlib::HausdorffSpace(numDims, neighborhoodSize, neighborhoodWeighting);
        break;

    case distance_metric::METRIC_HAU_med:
        space = new hnswlib::HausdorffSpace_median(numDims, neighborhoodSize, neighborhoodWeighting);
        break;

    case distance_metric::METRIC_BHATTACHARYYA:
    case distance_metric::METRIC_DETMATRATIO:
    case distance_metric::METRIC_BHATTACHARYYATESTONLYMEANS:
    case distance_metric::METRIC_CMD_covmat:
    case distance_metric::METRIC_FRECHET_Gen:
    case distance_metric::METRIC_FRECHET_CovMat:
    case distance_metric::METRIC_FROBENIUS_CovMat:
        space = new hnswlib::MultiVarCovMat_Space(knn_metric);
        break;

    case distance_metric::METRIC_EUC_sep:
        space = new hnswlib::L2sepFeatSpace(numDims+2, pixelWeight);        // (x,y) coordinates are added as features
        break;

    default:
        spdlog::error("Distance calculation: ERROR: Distance metric unknown.");
    }

    return space;
}


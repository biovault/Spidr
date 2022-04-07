#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SpidrWrapper.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(spidr, m) {
	m.doc() = "SpidrWrapper";

	// ENUMS
	py::enum_<feat_dist>(m, "DistMetric", "Distance metric, the choice of distance will set the feature type")
		.value("QF_hist", feat_dist::HIST_QF)
		.value("Hel_hist", feat_dist::HIST_HEL)
		.value("Chamfer_pc", feat_dist::PC_CHA)
		.value("Hausdorff_med_pc", feat_dist::PC_HAU_MED)
		.value("SSD_pc", feat_dist::PC_SSD)
		.value("Hausdorff_pc", feat_dist::PC_HAU)
		.value("Morans_I", feat_dist::LMI_EUC)
		.value("Bhattacharyya", feat_dist::MVN_BHAT)
		.value("Pixel loc (weighted)", feat_dist::XY_EUCW)
		.value("Pixel loc (normed)", feat_dist::XYRNORM_EUC);

	py::enum_<loc_Neigh_Weighting>(m, "WeightLoc", "Local neighborhood weighting")
		.value("uniform", loc_Neigh_Weighting::WEIGHT_UNIF)
		.value("bino", loc_Neigh_Weighting::WEIGHT_BINO)
		.value("gauss", loc_Neigh_Weighting::WEIGHT_GAUS);

	py::enum_<knn_library>(m, "KnnAlgorithm", "kNN computation: exact, approximated or full distance matrix")
		.value("hnsw", knn_library::KNN_HNSW)
		.value("exact_knn", knn_library::KKN_EXACT)
		.value("full_dist_matrix", knn_library::FULL_DIST_BRUTE_FORCE);

	// MAIN WRAPPER: here SpidrWrapper, on python side SpidrAnalysis
	py::class_<SpidrWrapper> spidrAnalysis(m, "SpidrAnalysis");

	spidrAnalysis.def(py::init<feat_dist, loc_Neigh_Weighting, size_t, size_t, float, knn_library, size_t, size_t, size_t, size_t, bool, py::array_t<float, py::array::c_style | py::array::forcecast>>(), "Init SpidrLib",
		py::arg("distMetric") = feat_dist::PC_CHA,
		py::arg("kernelType") = loc_Neigh_Weighting::WEIGHT_UNIF,
		py::arg("numLocNeighbors") = 0,
		py::arg("numHistBins") = 0,
		py::arg("pixelWeight") = 0.5,
		py::arg("aknnAlgType") = knn_library::KNN_HNSW,
		py::arg("numIterations") = 1000,
		py::arg("perplexity") = 30,
		py::arg("exaggeration") = 250,
		py::arg("expDecay") = 70,
		py::arg("forceCalcBackgroundFeatures") = false, 
		py::arg("initial_embedding") = py::none());


	spidrAnalysis.def("fit", &SpidrWrapper::fit, "Compute kNN dists and indices and return them",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("compute_fit", &SpidrWrapper::compute_fit, "Compute kNN dists and indices, do not return anything",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("transform", &SpidrWrapper::transform, "Compute embedding and return it, fit() must have been called previously");

	spidrAnalysis.def("compute_transform", &SpidrWrapper::compute_transform, "Compute embedding but do not return anything, fit() must have been called previously");

	spidrAnalysis.def("fit_transform", &SpidrWrapper::fit_transform, "Compute embedding and return it",
		py::arg("X"),
		py::arg("pointIDsGlobal"),
		py::arg("imgWidth"),
		py::arg("imgHeight"),
		py::arg("backgroundIDsGlobal") = py::none());

	spidrAnalysis.def("set_kNN", &SpidrWrapper::set_kNN, "Compute embedding, calls fit()",
		py::arg("knn_indices"),
		py::arg("knn_distances"));

	spidrAnalysis.def("get_kNN", &SpidrWrapper::get_kNN, "Returns the kNN dists and indices");

	spidrAnalysis.def_property_readonly("perplexity", &SpidrWrapper::get_perplexity, "t-SNE perplexity");
	spidrAnalysis.def_property_readonly("iterations", &SpidrWrapper::get_numIterations, "t-SNE iterations");
	spidrAnalysis.def_property_readonly("nn", &SpidrWrapper::get_nn, "Number of nearest neighbors");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif


}

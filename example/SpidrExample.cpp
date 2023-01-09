#include <string>
#include <filesystem>
#include <vector>
#include <numeric>
#include <fstream>
#include <stdexcept>

#include <SpidrAnalysis.h>

std::vector<float> readData(const std::string fileName);
void writeData(const std::vector<float> data, const std::string fileName);

/*! Example of using the spidt library for computing a spatially aware t-SNE embedding
 * 
 * A synthetic data set is used. This image only has two channels. There are several distinct regions in the image
 * That cannot be distinguished based on the values alone but that differ in texture.
 * 
 * This example uses the chamfer point cloud distance to incorporate spatial neighborhood information into the embedding.
 * 
 * The image data set is loaded, an embedding computed and saved to disk.
 * 
*/
int main() {
	// set data info
	const std::filesystem::path projectDir = std::filesystem::current_path().parent_path();
	const std::string fileName = "CheckeredBoxes_2Ch_32.bin";
	const std::string emebddingName = "CheckeredBoxes_2Ch_32_sp-tSNE_Chamfer.bin";

	const std::string loadPath = projectDir.string() + "/example/data/" + fileName;
	const std::string savePath = projectDir.string() + "/example/data/" + emebddingName;

	// load data
	const std::vector<float> data = readData(loadPath);

	// image data
	const size_t numPoints = 32 * 32;
	const size_t numDims = 2;
	const ImgSize imgSize(32, 32);
	std::vector<unsigned int> pointIDsGlobal(numPoints);
	std::iota(pointIDsGlobal.begin(), pointIDsGlobal.end(), 0);

	// spidr settings
	const feature_type featureType = feature_type::PCLOUD;
	const knn_library knnLibrary = knn_library::KNN_HNSW;
	const distance_metric distanceMetric = distance_metric::METRIC_CHA;
	const loc_Neigh_Weighting neighborhoodWeighting = loc_Neigh_Weighting::WEIGHT_UNIF;
	const size_t spatialNeighborsInEachDirection = 1;

	// t-SNE settings
	const int numIterations = 1000;
	const int perplexity = 20;
	const int exaggeration = 250;
	const int expDecay = 70;

	// compute spatially informed embedding
	SpidrAnalysis spidr(data, pointIDsGlobal, numDims, imgSize, featureType, neighborhoodWeighting, spatialNeighborsInEachDirection, 0, 0, /* unused settings for the selected distance */ 
						knnLibrary, distanceMetric, numIterations, perplexity, exaggeration, expDecay, emebddingName);

	spidr.compute();
	const std::vector<float> embedding = spidr.output();

	// save embedding
	writeData(embedding, savePath);

    return 0;
}

std::vector<float> readData(const std::string fileName)
{
	std::vector<float> fileContents;

	// open file 
	std::ifstream fin(fileName, std::ios::in | std::ios::binary);
	if (!fin.is_open()) {
		throw std::invalid_argument("Unable to load file: " + fileName);
	}
	else {
		// number of data points
		fin.seekg(0, std::ios::end);
		auto fileSize = fin.tellg();
		auto numDataPoints = fileSize / sizeof(float);
		fin.seekg(0, std::ios::beg);

		// read data
		fileContents.resize(numDataPoints);
		fin.read(reinterpret_cast<char*>(fileContents.data()), fileSize);
		fin.close();
	}

	return fileContents;
}

void writeData(const std::vector<float> data, const std::string fileName) {
	std::ofstream fout(fileName, std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
	fout.close();
}

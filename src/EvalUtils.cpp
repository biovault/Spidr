#pragma once
#include"EvalUtils.h"

#include <fstream>

template<typename T>
void writeVecToBinary(std::vector<T> vec, std::string writePath) {
    std::ofstream fout(writePath, std::ofstream::out | std::ofstream::binary);
    fout.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    fout.close();
}

// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template void writeVecToBinary<float>(std::vector<float> vec, std::string writePath);
template void writeVecToBinary<double>(std::vector<double> vec, std::string writePath);
template void writeVecToBinary<int>(std::vector<int> vec, std::string writePath);
template void writeVecToBinary<unsigned int>(std::vector<unsigned int> vec, std::string writePath);
template void writeVecToBinary<size_t>(std::vector<size_t> vec, std::string writePath);

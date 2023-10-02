#include"EvalUtils.h"

#include <fstream>

template<typename T>
void writeVecToBinary(const std::vector<T>& vec, const std::string& writePath) {
    std::ofstream fout(writePath, std::ofstream::out | std::ofstream::binary);
    fout.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    fout.close();
}

// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template void writeVecToBinary<float>(const std::vector<float>& vec, const std::string& writePath);
template void writeVecToBinary<double>(const std::vector<double>& vec, const std::string& writePath);
template void writeVecToBinary<int>(const std::vector<int>& vec, const std::string& writePath);
template void writeVecToBinary<unsigned int>(const std::vector<unsigned int>& vec, const std::string& writePath);
template void writeVecToBinary<size_t>(const std::vector<size_t>& vec, const std::string& writePath);

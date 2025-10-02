#ifndef readinput_h
#define readinput_h
#include "allFiles.h"
#include <string>
void read_wbsparse_cpp_to_matrix(const std::string& file_path, SparseMatrix64& mat, double threshold);
#endif
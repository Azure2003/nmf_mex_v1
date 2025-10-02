#include <fstream>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
#include "readInput.h"
#include "mex.h"
void read_wbsparse_cpp_to_matrix(const std::string& file_path, SparseMatrix64& mat, double threshold) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    // Read magic
    char magic[8];
    in.read(magic, 8);
    const char expected[] = {0, 0, 0, 0, 'c', 's', 't', 0};
    for (int i = 0; i < 8; ++i) {
        if (magic[i] != expected[i]) {
            throw std::runtime_error("Invalid magic header");
        }
    }

    // Read dimensions
    int64_t nrows = 0, ncols = 0;
    in.read(reinterpret_cast<char*>(&nrows), 8);
    in.read(reinterpret_cast<char*>(&ncols), 8);
    if (nrows <= 0 || ncols <= 0) {
        throw std::runtime_error("Invalid matrix dimensions");
    }

    // This may be dynamically allocated for earlier clear. For the future.
    std::vector<int64_t> length_array(ncols);
    in.read(reinterpret_cast<char*>(length_array.data()), 8 * ncols);
    if (!in) {
        throw std::runtime_error("Failed to read length array");
    }
    

    // Validate total non-zero count
    int64_t total_nonzeros = 0;
    for (int64_t len : length_array) {
        if (len < 0 || len > nrows) {
            throw std::runtime_error("Invalid length value: " + std::to_string(len));
        }
        total_nonzeros += len;
    }

    const uint32_t MASK = (1 << 10) - 1;  // 0x3FF
    const int64_t total_rows = nrows * 3;

mexPrintf("Reserving space....\n");
mexEvalString("drawnow; disp(' ');");
    mat.resize(total_rows, ncols);
    mat.reserve(total_nonzeros*2);
    mexPrintf("Reading starting....\n");
mexEvalString("drawnow; disp(' ');");
    for (int64_t col = 0; col < ncols; ++col) {

        int64_t len = length_array[col];
        std::vector<std::pair<int64_t, double>> col_entries;

        for (int64_t i = 0; i < len; ++i) {
            int64_t index;
            uint64_t coded;
            in.read(reinterpret_cast<char*>(&index), 8);
            in.read(reinterpret_cast<char*>(&coded), 8);

            if (!in) {
                std::cerr << "\nFile read error at col " << col << ", entry " << i << std::endl;
                break;
            }

            if (index < 0 || index >= nrows) {
                std::cerr << "\nInvalid index " << index << " at col " << col << std::endl;
                continue;
            }

            uint32_t totalCount = coded >> 32;

            uint32_t temp = static_cast<uint32_t>(coded & 0xFFFFFFFF);
            double f2 = ((temp >> 10) & MASK) / 1000.0;
            double f1 = ((temp >> 20) & MASK) / 1000.0;
            double f3 = 1.0 - f1 - f2;
            if (f3 < -0.002 || (temp & (3U << 30))) continue;
            if (f3 < 0.0) f3 = 0.0;

            double fracs[3] = {f1, f2, f3};

            for (int j = 0; j < 3; ++j) {
                int64_t row_idx = index + j*nrows;
                if (row_idx < 0 || row_idx >= total_rows) {
                    std::cerr << "\nOut of bounds: row " << row_idx << ", col " << col << std::endl;
                    continue;
                }
                if (col < 0 || col >= ncols) {
                    std::cerr << "\nOut of bounds: row " << row_idx << ", col " << col << std::endl;
                    continue;
                }
                double val = totalCount * fracs[j];
                if (val >=threshold) {
                    col_entries.emplace_back(row_idx, val);
                }
            }
        }
        std::sort(col_entries.begin(), col_entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Insert sorted entries into sparse matrix
        for (const auto& [row_idx, val] : col_entries) {
            mat.insert(row_idx, col) = val;
        }
    }
in.close(); 
mat.makeCompressed(); 
mexPrintf("Reading complete.\n");
mexPrintf("%lld x %lld matrix\n", total_rows, ncols);
mexPrintf("Triplet count: %zu\n", mat.nonZeros());
mexPrintf("Allocation completed, size of matrix is estimated to be: %.2f GB\n",
          (12.0 * mat.nonZeros() + ncols * 16.0) / 1073741824.0);
mexEvalString("drawnow; disp(' ');");
}
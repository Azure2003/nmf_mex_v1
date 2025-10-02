#include "mex.h"
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "allFiles.h"
// Your other headers
#include "readInput.h"
#include <unordered_map>

// Util: Convert mxArray to std::vector<int>
std::vector<int> mxArrayToIntVector(const mxArray* arr) {
    size_t n = mxGetNumberOfElements(arr);
    double* data = mxGetPr(arr);
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<int>(data[i]);
    return out;
}

// Util: Convert mxArray to std::vector<double>
std::vector<double> mxArrayToDoubleVector(const mxArray* arr) {
    size_t n = mxGetNumberOfElements(arr);
    double* data = mxGetPr(arr);
    return std::vector<double>(data, data + n);
}

Eigen::MatrixXd matlabToEigen(const mxArray* array) {
    if (!mxIsDouble(array) || mxIsComplex(array)) {
        mexErrMsgTxt("Input must be a real double matrix.");
    }

    mwSize rows = mxGetM(array); // MATLAB rows
    mwSize cols = mxGetN(array); // MATLAB cols
    double* data = mxGetPr(array); // Column-major data

    // Map MATLAB's data directly to an Eigen matrix (no copy)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigenMat(data, rows, cols);

    return eigenMat; 
}

Eigen::VectorXd matlabToEigenVector(const mxArray *matlabVec) {
    // Check input is double
    if (!mxIsDouble(matlabVec) || mxIsComplex(matlabVec)) {
        mexErrMsgIdAndTxt("MyToolbox:matlabToEigenVector:invalidType",
                          "Input must be a real double vector.");
    }

    // Get dimensions
    mwSize rows = mxGetM(matlabVec);
    mwSize cols = mxGetN(matlabVec);

    // Ensure it's a vector (1 column or 1 row)
    if (!(rows == 1 || cols == 1)) {
        mexErrMsgIdAndTxt("MyToolbox:matlabToEigenVector:notVector",
                          "Input must be a vector.");
    }

    // Total elements
    mwSize len = rows * cols;

    // Get pointer to MATLAB data
    double *dataPtr = mxGetPr(matlabVec);

    // Copy into Eigen vector
    Eigen::VectorXd eigenVec(len);
    for (mwSize i = 0; i < len; ++i) {
        eigenVec[i] = dataPtr[i];
    }

    return eigenVec;
}

// Core wrapper logic for calling from mexFunction
mxArray* sumRowColumn(std::string file_path, double threshold) {
    SparseMatrix64 mat;
    read_wbsparse_cpp_to_matrix(file_path, mat, threshold);
    const mwSize nRows = mat.rows();
    const mwSize nCols = mat.cols();

    Eigen::VectorXd rowSums = Eigen::VectorXd::Zero(nRows);
    Eigen::VectorXd colSums = Eigen::VectorXd::Zero(nCols);

    // Efficiently iterate over non-zeros (column-major)
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SparseMatrix64::InnerIterator it(mat,k); it; ++it) {
            const int r = it.row();
            const int c = it.col();
            const double v = it.value();
            rowSums[r] += v;
            colSums[c] += v;
        }
    }
    int size=1;
    const mwSize dims[1] = { static_cast<mwSize>(size) };
    const char* field_names[] = {"row", "column"};
    mxArray* resultStruct = mxCreateStructArray(1, dims, 2, field_names);
    
    mwSize rowsRow = rowSums.size();
    mwSize colsRow = 1;  // Column vector in MATLAB

    mxArray* D = mxCreateDoubleMatrix(rowsRow, colsRow, mxREAL); // MATLAB column vector

    Eigen::Map<Eigen::VectorXd> map1(mxGetPr(D), rowsRow);
    map1 = rowSums;

    mwSize rowsCol = 1;
    mwSize colsCol = colSums.size();  // Column vector in MATLAB

    mxArray* D1 = mxCreateDoubleMatrix(colsCol, rowsCol, mxREAL); // MATLAB column vector

    Eigen::Map<Eigen::VectorXd> map2(mxGetPr(D1), colsCol);
    map2 = colSums;
    int i=0;
    mxSetField(resultStruct, i, "row", D);
    mxSetField(resultStruct, i, "column", D1);
    return resultStruct;
}

// MATLAB entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

    // === Required arguments ===
    char* file_path_cstr = mxArrayToString(prhs[0]);
    if (!file_path_cstr) {
        mexErrMsgIdAndTxt("runNMF_mex:invalidInput", "First argument must be a valid string.");
    }
    std::string file_path(file_path_cstr);
    mxFree(file_path_cstr);
    double threshold=1.0;
    for (int i = 1; i + 1 < nrhs; i += 2) {
        char* key_cstr = mxArrayToString(prhs[i]);
        if (!key_cstr) {
            mexErrMsgIdAndTxt("runNMF_mex:invalidArg", "Argument name must be a valid string.");
        }
        std::string key(key_cstr);
        mxFree(key_cstr);

        const mxArray* val = prhs[i + 1];
        if(key=="threshold"){
            threshold = mxGetScalar(val);
        }
    }
    //std::vector<int> k = mxArrayToIntVector(prhs[1]);

    mxArray* result=sumRowColumn(file_path, threshold);
    
    plhs[0] = result;
}
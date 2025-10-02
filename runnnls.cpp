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
#include "prepnnls.h"
#include "runNMF.h"
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
mxArray* runNNLsWrapper(std::string file_path,
                       std::vector<double> L1,
                       std::vector<double> L2,
                       unsigned int threads,
                       double threshold,
                       Eigen::MatrixXd w_init,
                       Eigen::MatrixXd h_init,
                       Eigen::VectorXd d,
                    bool dual) {
    SparseMatrix64 mat;
    read_wbsparse_cpp_to_matrix(file_path, mat, threshold);
    int size=1;
    const mwSize dims[1] = { static_cast<mwSize>(size) };
    const char* field_names[] = { "W", "H", "d", "d1"};
    mxArray* resultStruct = mxCreateStructArray(1, dims, 4, field_names);
    
    Eigen::SparseMatrix<double> link_matrix_h_temp;
    Eigen::MatrixXd w_init_format;
    if(w_init.rows()>0&&w_init.cols()>0){
        w_init_format=w_init.transpose();
    }

    NNLResult result=run_nnl(mat,L1,L2, threads, w_init_format, h_init, d, dual);
    
    
    mwSize rows = result.W.rows();
    mwSize cols = result.W.cols();
    mxArray* W = mxCreateDoubleMatrix(rows, cols, mxREAL);

// Copy using Eigen::Map (clean, readable)
    Eigen::Map<Eigen::MatrixXd> map(mxGetPr(W), rows, cols);
    map = result.W;

        // H
    mwSize rowsH = result.H.rows();
    mwSize colsH = result.H.cols();
    mxArray* H = mxCreateDoubleMatrix(rowsH, colsH, mxREAL);

// Copy using Eigen::Map (clean, readable)
    Eigen::Map<Eigen::MatrixXd> map1(mxGetPr(H), rowsH, colsH);
    map1 = result.H;

        // D
       // Assume result.D is an Eigen::VectorXd
    mwSize rowsD = result.D.size();
    mwSize colsD = 1;  // Column vector in MATLAB

    mxArray* D = mxCreateDoubleMatrix(rowsD, colsD, mxREAL); // MATLAB column vector

// Map MATLAB memory to an Eigen vector and copy
    Eigen::Map<Eigen::VectorXd> map2(mxGetPr(D), rowsD);
    map2 = result.D;

    mwSize rowsD1 = result.D1.size();
    mwSize colsD1 = 1;  // Column vector in MATLAB

    mxArray* D1 = mxCreateDoubleMatrix(rowsD1, colsD1, mxREAL); // MATLAB column vector

// Map MATLAB memory to an Eigen vector and copy
    Eigen::Map<Eigen::VectorXd> map3(mxGetPr(D1), rowsD1);
    map3 = result.D1;

    int i=0;

        mxSetField(resultStruct, i, "W", W);
        mxSetField(resultStruct, i, "H", H);
        mxSetField(resultStruct, i, "d", D);
        mxSetField(resultStruct, i, "d1", D1);

    return resultStruct;
}

// MATLAB entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2) {
        mexErrMsgTxt("Usage: out = nnl(filepath, ...)");
    }

    // === Required arguments ===
    char* file_path_cstr = mxArrayToString(prhs[0]);
    if (!file_path_cstr) {
        mexErrMsgIdAndTxt("runNMF_mex:invalidInput", "First argument must be a valid string.");
    }
    std::string file_path(file_path_cstr);
    mxFree(file_path_cstr);

    //std::vector<int> k = mxArrayToIntVector(prhs[1]);

    // === Optional args with defaults ===
    unsigned int maxit = 100;
    std::vector<double> zeros={0,0};
    std::vector<double> L1=zeros;
    std::vector<double> L2=zeros;
    unsigned int threads = 0;
    double threshold=1.0;
    Eigen::MatrixXd w_init;
    Eigen::MatrixXd h_init;
    Eigen::VectorXd d;
    bool dual=false;

    // === Name-value pair parsing ===
    std::unordered_map<std::string, int> arg_map = { {"maxit", 3}, {"verbose", 4}, {"L1", 5}, {"L2", 6},
        {"seed", 7},  {"upper_bound", 10}, {"threads", 11}, {"mask_zeros", 12}, {"threshold", 13}, {"w_init", 14}
    };

    for (int i = 1; i + 1 < nrhs; i += 2) {
        char* key_cstr = mxArrayToString(prhs[i]);
        if (!key_cstr) {
            mexErrMsgIdAndTxt("runNMF_mex:invalidArg", "Argument name must be a valid string.");
        }
        std::string key(key_cstr);
        mxFree(key_cstr);

        const mxArray* val = prhs[i + 1];
        if(key=="maxit"){
            maxit=static_cast<unsigned int>(mxGetScalar(val));
        } else if (key == "L1") {
            L1 = mxArrayToDoubleVector(val);
        } else if (key == "L2") {
            L2 = mxArrayToDoubleVector(val);
        }  else if (key == "threads") {
            threads = static_cast<unsigned int>(mxGetScalar(val));
        } else if(key=="threshold"){
            threshold = mxGetScalar(val);
        } else if (key=="w_init"){
            w_init=matlabToEigen(val);
        } else if (key=="h_init"){
            h_init=matlabToEigen(val);
        }else if (key=="d"){
            d=matlabToEigenVector(val);
        } else if (key=="dual"){
            dual=mxGetScalar(val);
        }else{
            mexErrMsgIdAndTxt("runNMF_mex:unknownOption", ("Unknown option: " + key).c_str());
        }
    }
    mxArray* result=runNNLsWrapper(file_path, L1, L2, threads, threshold, w_init, h_init, d, dual);
    
    plhs[0] = result;
}
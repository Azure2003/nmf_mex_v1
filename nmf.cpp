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

// Core wrapper logic for calling from mexFunction
mxArray* runNMFWrapper(std::string file_path,
                       std::vector<int> k,
                       double tol,
                       unsigned int maxit,
                       bool verbose,
                       std::vector<double> L1,
                       std::vector<double> L2,
                       std::vector<int> seed,
                       bool link_h,
                       bool sort_model,
                       double upper_bound,
                       unsigned int threads,
                       bool mask_zeros,
                    double threshold,
                    Eigen::MatrixXd w_init) {
    SparseMatrix64 mat;
    read_wbsparse_cpp_to_matrix(file_path, mat, threshold);
    int size=0;
    size=k.size();
    if(size==0){
        size=1;
    }
    const mwSize dims[1] = { static_cast<mwSize>(size) };
    const char* field_names[] = { "W", "H", "d", "tol", "iter", "MSE" };
    mxArray* resultStruct = mxCreateStructArray(1, dims, 6, field_names);
    if(w_init.rows()>0&&w_init.cols()>0){
        int i=0;
        std::vector<double> zeros={0,0};
        Eigen::SparseMatrix<double> link_matrix_h_temp;
        std::vector <Eigen::MatrixXd> w_init_format;
        w_init_format.push_back(w_init.transpose());
        mexPrintf("Progress: %d / %d\n", 1, 1);
        mexPrintf("Running w_init version");
        mexEvalString("drawnow; disp(' ');");
        NMFResult result = run_nmf(mat, tol, maxit, verbose, L1, L2, threads, w_init_format, link_matrix_h_temp, mask_zeros, link_h, sort_model, upper_bound );
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

        // Scalars
        mxArray* tol_out = mxCreateDoubleScalar(result.tol);
        mxArray* iter_out = mxCreateDoubleScalar(result.iter);
        mxArray* mse_out  = mxCreateDoubleScalar(result.mse);

        mxSetField(resultStruct, i, "W", W);
        mxSetField(resultStruct, i, "H", H);
        mxSetField(resultStruct, i, "d", D);
        mxSetField(resultStruct, i, "tol", tol_out);
        mxSetField(resultStruct, i, "iter", iter_out);
        mxSetField(resultStruct, i, "MSE", mse_out);
    }else{
    for (int i = 0; i < k.size(); ++i) {
        mexPrintf("Progress: %d / %d\n", i+1, k.size());
        mexEvalString("drawnow; disp(' ');");

        int ktemp = k[i];
        NMFResult result = run_nmf(mat, tol, maxit, verbose, threads, seed,
                                   ktemp, mask_zeros, link_h, sort_model, upper_bound);

        // W
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

        // Scalars
        mxArray* tol_out = mxCreateDoubleScalar(result.tol);
        mxArray* iter_out = mxCreateDoubleScalar(result.iter);
        mxArray* mse_out  = mxCreateDoubleScalar(result.mse);

        mxSetField(resultStruct, i, "W", W);
        mxSetField(resultStruct, i, "H", H);
        mxSetField(resultStruct, i, "d", D);
        mxSetField(resultStruct, i, "tol", tol_out);
        mxSetField(resultStruct, i, "iter", iter_out);
        mxSetField(resultStruct, i, "MSE", mse_out);
    }
}

    return resultStruct;
}

// MATLAB entry point
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    if (nrhs < 2) {
        mexErrMsgTxt("Usage: out = runNMF_mex(filepath, k_array, 'tol', 1e-4, ...)");
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
    std::vector<int> k;
    double tol = 1e-4;
    unsigned int maxit = 100;
    bool verbose = false;
    std::vector<double> L1{0, 0};
    std::vector<double> L2{0, 0};
    std::vector<int> seed;
    bool link_h = false;
    bool sort_model = true;
    double upper_bound = 0.0;
    unsigned int threads = 0;
    bool mask_zeros = false;
    double threshold=1.0;
    Eigen::MatrixXd w_init;

    // === Name-value pair parsing ===
    std::unordered_map<std::string, int> arg_map = { {"ks", 1},
        {"tol", 2}, {"maxit", 3}, {"verbose", 4}, {"L1", 5}, {"L2", 6},
        {"seed", 7}, {"link_h", 8}, {"sort_model", 9}, {"upper_bound", 10},
        {"threads", 11}, {"mask_zeros", 12}, {"threshold", 13}, {"w_init", 14}
    };

    for (int i = 1; i + 1 < nrhs; i += 2) {
        char* key_cstr = mxArrayToString(prhs[i]);
        if (!key_cstr) {
            mexErrMsgIdAndTxt("runNMF_mex:invalidArg", "Argument name must be a valid string.");
        }
        std::string key(key_cstr);
        mxFree(key_cstr);

        const mxArray* val = prhs[i + 1];
        if (key=="ks"){
            k=mxArrayToIntVector(val);
        }
        else if(key == "tol") {
            tol = mxGetScalar(val);
        } else if (key == "maxit") {
            maxit = static_cast<unsigned int>(mxGetScalar(val));
        } else if (key == "verbose") {
            verbose = mxGetScalar(val);
        } else if (key == "L1") {
            L1 = mxArrayToDoubleVector(val);
        } else if (key == "L2") {
            L2 = mxArrayToDoubleVector(val);
        } else if (key == "seed") {
            seed = mxArrayToIntVector(val);
        } else if (key == "link_h") {
            link_h = mxGetScalar(val);
        } else if (key == "sort_model") {
            sort_model = mxGetScalar(val);
        } else if (key == "upper_bound") {
            upper_bound = mxGetScalar(val);
        } else if (key == "threads") {
            threads = static_cast<unsigned int>(mxGetScalar(val));
        } else if (key == "mask_zeros") {
            mask_zeros = mxGetScalar(val);
        } else if(key=="threshold"){
            threshold = mxGetScalar(val);
        } else if (key=="w_init"){
             w_init=matlabToEigen(val);
        } else {
            mexErrMsgIdAndTxt("runNMF_mex:unknownOption", ("Unknown option: " + key).c_str());
        }
    }
    mxArray* result = runNMFWrapper(file_path, k, tol, maxit, verbose, L1, L2, seed,
                                    link_h, sort_model, upper_bound, threads, mask_zeros, threshold, w_init);

    plhs[0] = result;
}
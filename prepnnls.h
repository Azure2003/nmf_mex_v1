#include "nmf_sparse.h"
#include <random>
#include <iostream>
#include "mex.h"
struct NNLResult {
    Eigen::MatrixXd W;
    Eigen::VectorXd D;
    Eigen::MatrixXd H;
    Eigen::VectorXd D1;
};

NNLResult run_nnl(
    SparseMatrix64& A,
    const std::vector<double>& L1,
    const std::vector<double>& L2,
    unsigned int threads,
    Eigen::MatrixXd& w_init,
    Eigen::MatrixXd& h_init,
    Eigen::VectorXd d,
    bool dual
) {

    nmf<SparseMatrix64> model(A, w_init, h_init, d);
    model.L1 = L1;
    model.L2 = L2;
    model.threads = threads;
    model.dual=dual;


    model.calculate_nnls();

    NNLResult result;
    result.W = model.matrixW().transpose();
    result.D = model.vectorD();
    result.H = model.matrixH();
    result.D1 = model.vectorD1();

    return result;
}
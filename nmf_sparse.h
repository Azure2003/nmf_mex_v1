#ifndef nmf_sparse_H
#define nmf_sparse_H

#include <Eigen/Dense>
#include "allFiles.h"
#include <vector>

template <class T>
class nmf {
   private:
    T& A;
    T t_A;
    Eigen::SparseMatrix<double> mask_matrix, t_mask_matrix;
    Eigen::SparseMatrix<double> link_matrix_w, link_matrix_h;
    Eigen::MatrixXd w;
    Eigen::VectorXd d;
    Eigen::VectorXd d1;
    Eigen::MatrixXd h;
    double tol_ = -1, mse_ = 0;
    unsigned int iter_ = 0, best_model_ = 0;
    bool mask = false, mask_zeros = false, symmetric = false, transposed = false;

   public:
    bool verbose = true;
    bool dual=false;
    unsigned int maxit = 100, threads = 0;
    std::vector<double> L1 = std::vector<double>(2), L2 = std::vector<double>(2);
    std::vector<bool> link = {false, false};
    bool sort_model = true;
    double upper_bound = 0;
    double tol = 1e-4;

    // Constructors
    nmf(T& A, const unsigned int k, const unsigned int seed = 0);
    nmf(T& A, Eigen::MatrixXd w);
    nmf(T& A, Eigen::MatrixXd w, Eigen::MatrixXd h, Eigen::VectorXd d);

    // Setters
    void isSymmetric();
    void maskZeros();
    void maskMatrix(Eigen::SparseMatrix<double>& m);
    void linkH(Eigen::SparseMatrix<double>& l);
    void linkW(Eigen::SparseMatrix<double>& l);
    void upperBound(double upperbound);

    // Getters
    Eigen::MatrixXd matrixW();
    Eigen::VectorXd vectorD();
    Eigen::VectorXd vectorD1();
    Eigen::MatrixXd matrixH();
    double fit_tol();
    unsigned int fit_iter();
    double fit_mse();
    unsigned int best_model();

    // Core methods
    void sortByDiagonal();
    void scaleW();
    void scaleH();
    void predictH();
    void predictW();
    double mse();
    double mse_masked();
    void fit();
    void fit_restarts(std::vector<Eigen::MatrixXd>& w_init);
    void calculate_nnls();
};

#endif 

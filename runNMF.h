#include "nmf_sparse.h"
#include <random>
#include <iostream>
struct NMFResult {
    Eigen::MatrixXd W;
    Eigen::VectorXd D;
    Eigen::MatrixXd H;
    double tol;
    unsigned int iter;
    double mse;
    unsigned int best_model;
};
static const std::vector<std::pair<double, double>> bounds_list = {
    {0.0, 1.0},
    {0.0, 2.0},
    {1.0, 2.0},
    {1.0, 10.0}
};
std::vector<Eigen::MatrixXd> generateWInitFromSeed(
    int k,
    int data_rows,
    const std::vector<int>& seed_ints)
{

    std::vector<Eigen::MatrixXd> w_init;

    for (size_t i = 0; i < seed_ints.size(); ++i) {
        int s = seed_ints[i];
        std::mt19937 rng(s); // seeded RNG

        bool use_runif = false;
        if (i == 0) {
            use_runif = true;
        } else {
            std::uniform_int_distribution<int> coin_flip(0, 1);
            use_runif = (coin_flip(rng) == 1);
        }

        if (use_runif) {
            // select bounds randomly, seeded again for reproducibility
            std::mt19937 rng_bounds(s);
            std::uniform_int_distribution<size_t> bounds_dist(0, bounds_list.size() - 1);
            auto b = bounds_list[bounds_dist(rng_bounds)];

            std::uniform_real_distribution<double> dist(b.first, b.second);

            Eigen::MatrixXd mat(k, data_rows);
            for (int row = 0; row < k; ++row) {
                for (int col = 0; col < data_rows; ++col) {
                    mat(row, col) = dist(rng);
                }
            }
            w_init.push_back(mat);
        } else {
            std::normal_distribution<double> dist(2.0, 1.0);

            Eigen::MatrixXd mat(k, data_rows);
            for (int row = 0; row < k; ++row) {
                for (int col = 0; col < data_rows; ++col) {
                    mat(row, col) = dist(rng);
                }
            }
            w_init.push_back(mat);
        }
    }
    return w_init;
}
std::vector<Eigen::MatrixXd> generateWInitFromSeed(const Eigen::MatrixXd& data, int k, const std::vector<Eigen::MatrixXd>& seed) {
    std::vector<Eigen::MatrixXd> w_init;
    for (const auto& s : seed) {
        if ((s.cols() == data.rows() && s.rows() == k)) {
            w_init.push_back(s);
        } else if ((s.rows() == data.rows() && s.cols() == k)) {
            w_init.push_back(s.transpose());
        } else {
            throw std::invalid_argument("Seed matrix dimensions incompatible");
        }
    }
    return w_init;
}

std::vector<Eigen::MatrixXd> randomInitializeW(int k, int nrows) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::MatrixXd w = Eigen::MatrixXd::NullaryExpr(k, nrows, [&]() { return dis(gen); });
    return {w};
}
NMFResult run_nmf(
    SparseMatrix64& A,
    double tol,
    unsigned int maxit,
    bool verbose,
    const std::vector<double>& L1,
    const std::vector<double>& L2,
    unsigned int threads,
    std::vector<Eigen::MatrixXd>& w_init,
    Eigen::SparseMatrix<double>& link_matrix_h,
    bool mask_zeros,
    bool link_h,
    bool sort_model,
    double upper_bound 
) {
    unsigned int k = w_init.empty() ? 10 : w_init[0].rows();
   Eigen::MatrixXd  w=w_init[0];
    nmf<SparseMatrix64> model(A, w);

    model.tol = tol;
    model.maxit = maxit;
    model.verbose = verbose;
    model.L1 = L1;
    model.L2 = L2;
    model.threads = threads;
    model.sort_model = sort_model;
    model.upper_bound = upper_bound;

    if (link_h)
        model.linkH(link_matrix_h);

    if (mask_zeros)
        model.maskZeros();

    if (w_init.empty())
        model.fit();
    else
        model.fit_restarts(w_init);

    NMFResult result;
    result.W = model.matrixW().transpose();
    result.D = model.vectorD();
    result.H = model.matrixH();
    result.tol = model.fit_tol();
    result.iter = model.fit_iter();
    result.mse = model.fit_mse();
    result.best_model = model.best_model();

    return result;
}
//Will need to edit when we need L1.
NMFResult run_nmf(
    SparseMatrix64& A,
    double tol,
    unsigned int maxit,
    bool verbose,
    unsigned int threads,
    std::vector<int> seed,
    int k,
    bool mask_zeros,
    bool link_h,
    bool sort_model,
    double upper_bound
) {
    std::vector<double> zeros={0,0};
    Eigen::SparseMatrix<double> link_matrix_h;
    std::vector<Eigen::MatrixXd> w_init;
    if(seed.size()==0){
        w_init=randomInitializeW(k, static_cast<int> (A.rows()));
    }else{
        w_init=generateWInitFromSeed(k,static_cast<int> (A.rows()),seed);
    }
    return run_nmf(A, tol, maxit, verbose, zeros, zeros, threads, w_init, link_matrix_h, mask_zeros, link_h, sort_model, upper_bound);
}



#include "bits.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>

Eigen::MatrixXd submat(const Eigen::MatrixXd& x, const Eigen::VectorXi& col_indices) {
    Eigen::MatrixXd x_(x.rows(), col_indices.size());
    for (int i = 0; i < col_indices.size(); ++i)
        x_.col(i) = x.col(col_indices(i));
    return x_;
}

Eigen::VectorXd subvec(const Eigen::MatrixXd& b, const Eigen::VectorXi& ind, unsigned int col) {
    Eigen::VectorXd bsub(ind.size());
    for (int i = 0; i < ind.size(); ++i) bsub(i) = b(ind(i), col);
    return bsub;
}

Eigen::VectorXi find_gtz(const Eigen::MatrixXd& x, unsigned int col) {
    std::vector<int> indices;
    for (int i = 0; i < x.rows(); ++i)
        if (x(i, col) > 0) indices.push_back(i);
    Eigen::VectorXi gtz(indices.size());
    for (int i = 0; i < indices.size(); ++i)
        gtz(i) = indices[i];
    return gtz;
}

double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
    double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    const int n = x.size();
    for (int i = 0; i < n; ++i) {
        x_i = (*(x.data() + i));
        y_i = (*(y.data() + i));
        sum_x += x_i;
        sum_y += y_i;
        sum_xy += x_i * y_i;
        sum_x2 += x_i * x_i;
        sum_y2 += y_i * y_i;
    }
    return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

std::vector<int> sort_index(const Eigen::VectorXd& d) {
    std::vector<int> idx(d.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) { return d[i1] > d[i2]; });
    return idx;
}

Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind) {
    Eigen::MatrixXd x_reordered(ind.size(), x.cols());
    for (int i = 0; i < ind.size(); ++i)
        x_reordered.row(i) = x.row(ind[i]);
    return x_reordered;
}

Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind) {
    Eigen::VectorXd x_reordered(ind.size());
    for (int i = 0; i < ind.size(); ++i)
        x_reordered(i) = x(ind[i]);
    return x_reordered;
}

std::vector<double> getRandomValues(unsigned int len, unsigned int seed) {
    std::vector<double> random_values(len);
    std::mt19937 rng;

    if (seed > 0) {
        rng.seed(seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (unsigned int i = 0; i < len; ++i) {
        random_values[i] = dist(rng);
    }
    return random_values;
}

Eigen::MatrixXd randomMatrix(unsigned int nrow, unsigned int ncol, unsigned int seed) {
    std::vector<double> random_values = getRandomValues(nrow * ncol, seed);
    Eigen::MatrixXd x(nrow, ncol);
    unsigned int indx = 0;
    for (unsigned int r = 0; r < nrow; ++r)
        for (unsigned int c = 0; c < ncol; ++c, ++indx)
            x(r, c) = random_values[indx];
    return x;
}

bool isAppxSymmetric(Eigen::MatrixXd& A, double tol) {
    if (A.rows() != A.cols()) return false;
    for (int i = 0; i < A.cols(); ++i)
        if (std::abs(A(i, 0) - A(0, i)) > tol)
            return false;
    return true;
}

std::vector<unsigned int> nonzeroRowsInCol(const Eigen::MatrixXd& x, unsigned int i) {
    std::vector<unsigned int> nonzeros;
    for (int it = 0; it < x.rows(); ++it)
        if (x(it, i) != 0)
            nonzeros.push_back(it);
    return nonzeros;
}

unsigned int n_nonzeros(const Eigen::MatrixXd& x) {
    unsigned int nz = 0;
    for (int i = 0, size = x.size(); i < size; ++i)
        if (*(x.data() + i) != 0) ++nz;
    return nz;
}

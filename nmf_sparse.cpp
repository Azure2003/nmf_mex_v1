
#include <iostream>
#include <stdexcept>
#include <limits>
#include <numeric>
#include "nmf_sparse.h"
#include "predict.h"
#include "bits.h"
#include "mex.h"
#ifdef _OPENMP
#include <omp.h>

inline unsigned int get_max_threads(unsigned int requested_threads) {
    return requested_threads == 0 ? omp_get_max_threads() : requested_threads;
}
#else
inline unsigned int get_max_threads(unsigned int requested_threads) {
    return 1; // no OpenMP, use serial execution
}
#endif
// Constructor implementations
template <class T>
nmf<T>::nmf(T& A, const unsigned int k, const unsigned int seed) : A(A) {
    w = randomMatrix(k, A.rows(), seed);
    h = Eigen::MatrixXd(k, A.cols());
    d = Eigen::VectorXd::Ones(k);
    isSymmetric();
}

template <class T>
nmf<T>::nmf(T& A, Eigen::MatrixXd w) : A(A), w(w) {
    if (A.rows() != w.cols())
        throw std::runtime_error("number of rows in 'A' and columns in 'w' are not equal!");
    d = Eigen::VectorXd::Ones(w.rows());
    h = Eigen::MatrixXd(w.rows(), A.cols());
    //isSymmetric();
}

template <class T>
nmf<T>::nmf(T& A, Eigen::MatrixXd w, Eigen::MatrixXd h, Eigen::VectorXd d)
    : A(A), w(w), h(h), d(d) {
    if(w.rows()==0&&w.cols()==0&&h.rows()==0&&h.cols()==0){
        throw std::runtime_error("Need one of them provided for this to work");
    }
    if(w.rows()==0&&w.cols()==0){
        mexPrintf("resetting w");
        mexEvalString("drawnow; disp(' ');");
        w.resize(h.rows(), A.rows());
        w.setZero();
        this->w=w;
    }
    if(h.rows()==0&&h.cols()==0){
        mexPrintf("resetting h");
        mexEvalString("drawnow; disp(' ');");
        h.resize(w.rows(), A.cols());
        h.setZero();
        this->h=h;
    }
    if(d.size()==0){
        d = Eigen::VectorXd::Ones(w.rows());
        this->d=d;
    }

    if (A.rows() != w.cols())
        throw std::runtime_error("dimensions of 'w' and 'A' are not compatible");
    if (A.cols() != h.cols())
        throw std::runtime_error("dimensions of 'h' and 'A' are not compatible");
    if (w.rows() != h.rows())
        throw std::runtime_error("rank of 'w' and 'h' are not equal!");
    if (d.size() != w.rows())
        throw std::runtime_error("length of 'd' is not equal to rank of 'w' and 'h'");
    isSymmetric();
}

// Setter implementations
template <class T>
void nmf<T>::isSymmetric() {
    symmetric = (A.rows() == A.cols());
}

template <class T>
void nmf<T>::maskZeros() {
    if (mask)
        throw std::runtime_error("a masking function has already been specified");
    mask_zeros = true;
}

template <class T>
void nmf<T>::maskMatrix(Eigen::SparseMatrix<double>& m) {
    if (mask)
        throw std::runtime_error("a masking function has already been specified");
    if (m.rows() != A.rows() || m.cols() != A.cols())
        throw std::runtime_error("dimensions of masking matrix and 'A' are not equivalent");
    if (mask_zeros)
        throw std::runtime_error("cannot supply both zero-masking and a mask matrix");
    mask = true;
    mask_matrix = m;
    symmetric = (symmetric && (mask_matrix.rows() == mask_matrix.cols()));
}

template <class T>
void nmf<T>::linkH(Eigen::SparseMatrix<double>& l) {
    if (l.cols() == A.cols())
        link_matrix_h = l;
    else
        throw std::runtime_error("dimensions of linking matrix and 'A' are not equivalent");
    link[1] = true;
}

template <class T>
void nmf<T>::linkW(Eigen::SparseMatrix<double>& l) {
    if (l.cols() == A.rows())
        link_matrix_w = l;
    else
        throw std::runtime_error("dimensions of linking matrix and 'A' are not equivalent");
    link[0] = true;
}

template <class T>
void nmf<T>::upperBound(double upperbound) {
    upper_bound = upperbound;
}

// Getter implementations
template <class T>
Eigen::MatrixXd nmf<T>::matrixW() { return w; }

template <class T>
Eigen::VectorXd nmf<T>::vectorD() { return d; }

template <class T>
Eigen::VectorXd nmf<T>::vectorD1() { return d1; }

template <class T>
Eigen::MatrixXd nmf<T>::matrixH() { return h; }

template <class T>
double nmf<T>::fit_tol() { return tol_; }

template <class T>
unsigned int nmf<T>::fit_iter() { return iter_; }

template <class T>
double nmf<T>::fit_mse() { return mse_; }

template <class T>
unsigned int nmf<T>::best_model() { return best_model_; }

// Utility implementations
template <class T>
void nmf<T>::sortByDiagonal() {
    if (w.rows() == 2 && d(0) < d(1)) {
        w.row(1).swap(w.row(0));
        h.row(1).swap(h.row(0));
        std::swap(d(0), d(1));
    } else if (w.rows() > 2) {
        std::vector<int> indx = sort_index(d);
        w = reorder_rows(w, indx);
        d = reorder(d, indx);
        h = reorder_rows(h, indx);
    }
}

template <class T>
void nmf<T>::scaleW() {
    d = w.rowwise().sum();
    d.array() += std::numeric_limits<double>::epsilon();
    for (int i = 0; i < w.rows(); ++i)
        for (int j = 0; j < w.cols(); ++j)
            w(i, j) /= d(i);
}


template <class T>
void nmf<T>::scaleH() {
    d = h.rowwise().sum();
    d.array() += std::numeric_limits<double>::epsilon();
    for (int i = 0; i < h.rows(); ++i)
        for (int j = 0; j < h.cols(); ++j)
            h(i, j) /= d(i);
}

template <class T>
void nmf<T>::predictH() {
    predict(A, mask_matrix, link_matrix_h, w, h, L1[1], L2[1], threads,
            mask_zeros, mask, link[1], upper_bound);
}
//Predict_w is currently a hacked version that does not support any masks. If you have masks, it will be wrong.
template <class T>
void nmf<T>::predictW() {
   predict_W(A, mask_matrix, link_matrix_w, h, w, L1[0], L2[0], threads, mask_zeros, mask, link[0], upper_bound);
}

template <class T>
void nmf<T>::fit() {
    if (verbose){
    mexPrintf("\nIter |     tol\n-----------------\n");
    mexEvalString("drawnow; disp(' ');");
    }
    for (; iter_ < maxit; ++iter_) {
        Eigen::MatrixXd w_it = w;
        predictH();
        scaleH();
        predictW();
        scaleW();

        tol_ = cor(w, w_it);
        if (verbose) {
            if (verbose) {
            mexPrintf(" %d   | %.6g\n", static_cast<int>(iter_ + 1), tol_);
            mexEvalString("drawnow; disp(' ');");
            }
        }
        if (tol_ < tol) break;
    }

    if (tol_ > tol && iter_ == maxit && verbose){
        mexPrintf("Convergence not reached in %d iterations\n", static_cast<int>(iter_));
        mexPrintf("  (actual tol = %.6g, target tol = %.6g)\n", tol_, tol);
        mexEvalString("drawnow; disp(' ');");
    }
    if (sort_model)
        sortByDiagonal();
}

template <class T>
void nmf<T>::calculate_nnls() {
    if(w.isZero()){
        mexPrintf("running w");
        mexEvalString("drawnow; disp(' ');");
        w=Eigen::MatrixXd(h.rows(), A.rows());
        predictW();
        scaleW();
        if(dual==true){
            mexPrintf("running H based on W");
            mexEvalString("drawnow; disp(' ');");
            predictH();
            scaleH();
        }
    }else if(h.isZero()){
        mexPrintf("running h");
        mexEvalString("drawnow; disp(' ');");
        h=Eigen::MatrixXd(w.rows(), A.cols());
        predictH();
        scaleH();
        if(dual==true){
            mexPrintf("running W based on H");
            mexEvalString("drawnow; disp(' ');");
            predictW();
            scaleW();
        }
    }
}

template <class T>
void nmf<T>::fit_restarts(std::vector<Eigen::MatrixXd>& w_init) {
    Eigen::MatrixXd w_best = w;
    Eigen::MatrixXd h_best = h;
    Eigen::VectorXd d_best = d;
    double tol_best = tol_;
    double mse_best = 0;
    double iter_best=0;
    for (unsigned int i = 0; i < w_init.size(); ++i) {
        w = w_init[i];
        tol_ = 1;
        iter_ = 0;
        if (w.rows() != h.rows())
            throw std::runtime_error("rank of 'w' is not equal to rank of 'h'");
        if (w.cols() != A.rows())
            throw std::runtime_error("dimensions of 'w' and 'A' are not compatible");
        fit();
        mse_ = mse();
        if (verbose){
           mexPrintf("MSE: %.6g\n\n", mse_);
           mexEvalString("drawnow; disp(' ');");
        }
        if (i == 0 || mse_ < mse_best) {
            best_model_ = i;
            w_best = w;
            h_best = h;
            d_best = d;
            tol_best = tol_;
            mse_best = mse_;
            iter_best=iter_;
        }
    }
    if (best_model_ != (w_init.size() - 1)) {
        w = w_best;
        h = h_best;
        d = d_best;
        tol_ = tol_best;
        mse_ = mse_best;
        iter_=iter_best;
    }
}

template <>
double nmf<SparseMatrix64>::mse() {
    Eigen::MatrixXd w0 = w.transpose();
    // multiply w0 columns by d(i)
    for (unsigned int i = 0; i < w0.cols(); ++i)
        for (unsigned int j = 0; j < w0.rows(); ++j)
            w0(j, i) *= d(i);

    // compute losses across all samples in parallel
    Eigen::ArrayXd losses(h.cols());
    losses.setZero();
    int threads_=get_max_threads(threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(dynamic)
#endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
        Eigen::VectorXd wh_i = w0 * h.col(i);

        if (mask_zeros) {
            // Iterate over nonzeros of column i in A
            for (SparseMatrix64::InnerIterator iter(A, i); iter; ++iter)
                losses(i) += std::pow(wh_i(iter.row()) - static_cast<double>(iter.value()), 2);
        } else {
            // Subtract known values in sparse matrix from wh_i
            for (SparseMatrix64::InnerIterator iter(A, i); iter; ++iter)
                wh_i(iter.row()) -= static_cast<double>(iter.value());

            if (mask) {
                // mask_matrix assumed Eigen::SparseMatrix<double>
                // Extract inner indices (row indices) of mask_matrix column i
                std::vector<unsigned int> m;
                for (Eigen::SparseMatrix<double>::InnerIterator mit(mask_matrix, i); mit; ++mit)
                    m.push_back(mit.row());

                for (unsigned int idx : m)
                    wh_i(idx) = 0;
            }

            losses(i) += wh_i.array().square().sum();
        }
    }

    // divide total loss by number of applicable measurements
    if (mask)
        return losses.sum() / static_cast<double>((h.cols() * w.cols()) - mask_matrix.nonZeros());
    else if (mask_zeros)
        return losses.sum() / double(A.nonZeros());
    else
        return losses.sum() / static_cast<double>(h.cols() * w.cols());
}
//We just assume this is false as the likelihood of a matrix 4 being symetric is not possible.
/*bool isAppxSymmetricFirstRowCol(SparseMatrix64& mat, double tol) {
    if (mat.rows() != mat.cols())
        return false;

    // Iterator over column 0 (default is column-major)
    SparseMatrix64::InnerIterator col_it(mat, 0);

    // To iterate over row 0, create a row-major view of mat
    Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> mat_row (mat);
    Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t>::InnerIterator row_it(mat_row, 0);

    // Advance iterators and compare corresponding values
    while (col_it && row_it) {
        if (col_it.row() != row_it.col()) {
            // Positions don't match — advance the iterator with smaller index
            if (col_it.row() < row_it.col())
                ++col_it;
            else
                ++row_it;
        } else {
            // Same position: compare values
            if (std::abs(col_it.value() - row_it.value()) > tol)
                return false;
            ++col_it;
            ++row_it;
        }
    }

    // If one iterator still has elements, they differ in sparsity structure
    if (col_it || row_it)
        return false;

    return true;
}


template <>
void nmf<SparseMatrix64>::isSymmetric() {
    symmetric = isAppxSymmetricFirstRowCol(A,1e-12);
}*/

template class nmf<SparseMatrix64>;

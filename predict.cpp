#include "predict.h"
#include <iostream>
#include <limits>
#include "bits.h"
#include "nnls.h"

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




// Main predict for sparse Eigen matrix A
void predict(const SparseMatrix64& A,
             const Eigen::SparseMatrix<double>& mask_A,
             const Eigen::SparseMatrix<double>& mask_h,
             const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h,
             double L1,
             double L2,
             unsigned int threads_,
             bool mask_zeros,
             bool masking_A,
             bool masking_h,
             double upper_bound) {
    unsigned int threads=get_max_threads(threads_);
    if (!mask_zeros) {
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            if (A.outerIndexPtr()[i] == A.outerIndexPtr()[i + 1]) continue; // no nonzeros

            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.outerIndexPtr()[i + 1] - mask_A.outerIndexPtr()[i];

            Eigen::MatrixXd a_i;
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

            if (num_masked == 0) {
                for (SparseMatrix64::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                // Traverse A.col(i) and mask_A.col(i) in parallel to handle masking
                SparseMatrix64::InnerIterator it_A(A, i);
                Eigen::SparseMatrix<double>::InnerIterator it_mask(mask_A, i);

                while (it_A) {
                    if (!it_mask || it_A.row() < it_mask.row()) {
                        b += it_A.value() * w.col(it_A.row());
                        ++it_A;
                    } else if (it_mask && it_A.row() == it_mask.row()) {
                        if (it_mask.value() < 1)
                            b += (it_A.value() * (1 - it_mask.value())) * w.col(it_A.row());
                        ++it_mask;
                        ++it_A;
                    } else if(it_mask) {
                        ++it_mask;
                    }
                }

                // Calculate weighted correction matrix
                Eigen::MatrixXd w_(w.rows(), num_masked);
                int j = 0;
                for (Eigen::SparseMatrix<double>::InnerIterator it(mask_A, i); it; ++it, ++j)
                    w_.col(j) = w.col(it.row()) * it.value();

                Eigen::MatrixXd a_ = w_ * w_.transpose();
                a_i = a - a_;
            }

            if (L1 != 0) b.array() -= L1;

            if (masking_h) {
                // Multiply b by mask_h.col(i)
                for (Eigen::SparseMatrix<double>::InnerIterator it(mask_h, i); it; ++it)
                    b(it.row()) *= it.value();
            }

            if (upper_bound > 0) {
                if (num_masked == 0)
                    c_bnnls(a, b, h, i, (int)upper_bound);
                else
                    c_bnnls(a_i, b, h, i, (int)upper_bound);
            } else {
                if (num_masked == 0)
                    c_nnls(a, b, h, i);
                else
                    c_nnls(a_i, b, h, i);
            }
        }
    } else {  // mask_zeros == true
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            if (A.outerIndexPtr()[i] == A.outerIndexPtr()[i + 1]) continue;

            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.outerIndexPtr()[i + 1] - mask_A.outerIndexPtr()[i];

            ////??????///////    
            Eigen::VectorXi nnz(A.outerIndexPtr()[i + 1] - A.outerIndexPtr()[i]);
            for (int ind = A.outerIndexPtr()[i], j = 0; j < nnz.size(); ++ind, ++j)
                nnz(j) = A.innerIndexPtr()[ind];

            Eigen::MatrixXd w_ = submat(w, nnz);

            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

            if (num_masked == 0) {
                for (SparseMatrix64::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                Eigen::SparseMatrix<double>::InnerIterator it_mask(mask_A, i);
                SparseMatrix64::InnerIterator it_A(A, i);

                int j = 0;
                while (it_mask && it_A) {
                    if (it_mask.row() == it_A.row()) {
                        w_.col(j) *= (1 - it_mask.value());
                        ++it_mask;
                        ++it_A;
                        ++j;
                    } else if (it_mask.row() < it_A.row()) {
                        ++it_mask;
                    } else {
                        ++it_A;
                    }
                }

                Eigen::SparseMatrix<double>::InnerIterator it_mask2(mask_A, i);
                SparseMatrix64::InnerIterator it_A2(A, i);
                while (it_A2) {
                    if (!it_mask2 || it_A2.row() < it_mask2.row()) {
                        b += it_A2.value() * w.col(it_A2.row());
                        ++it_A2;
                    } else if (it_mask2 && it_A2.row() == it_mask2.row()) {
                        if (it_mask2.value() < 1)
                            b += (it_A2.value() * (1 - it_mask2.value())) * w.col(it_A2.row());
                        ++it_mask2;
                        ++it_A2;
                    } else {
                        ++it_mask2;
                    }
                }
            }

            Eigen::MatrixXd a = w_ * w_.transpose();

            if (L1 != 0) b.array() -= L1;
            a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;

            if (masking_h) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(mask_h, i); it; ++it)
                    b(it.row()) *= it.value();
            }

            if (upper_bound > 0) {
                c_bnnls(a, b, h, i, upper_bound);
            } else {
                c_nnls(a, b, h, i);
            }
        }
    }
}

void predict_W(const SparseMatrix64& A,
             const Eigen::SparseMatrix<double>& mask_A,
             const Eigen::SparseMatrix<double>& mask_h,
             const Eigen::MatrixXd& h,
             Eigen::MatrixXd& w,
             double L1,
             double L2,
             unsigned int threads_,
             bool mask_zeros,
             bool masking_A,
             bool masking_h,
             double upper_bound) {
    unsigned int threads=get_max_threads(threads_);
    if (!mask_zeros) {
//Precalculate bs to get O1 access to elements because of iterators, uses local to avoid racing, k by A.rows (Trajectory in this case)
        Eigen::MatrixXd a = h * h.transpose();
        a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;
std::vector<Eigen::MatrixXd> b_locals(threads, Eigen::MatrixXd::Zero(w.rows(), w.cols()));

#pragma omp parallel num_threads(threads)
{
    int tid = omp_get_thread_num();

    #pragma omp for schedule(dynamic)
    for (int col = 0; col < A.outerSize(); ++col) {
        Eigen::VectorXd h_col = h.col(col);
        for (SparseMatrix64::InnerIterator it(A, col); it; ++it) {
            int row = it.row();
            double val = it.value();

            b_locals[tid].col(row) += val * h_col;
        }
    }
}

// Combine b_locals into b
Eigen::MatrixXd b = Eigen::MatrixXd::Zero(w.rows(), w.cols());
for (int tid = 0; tid < threads; ++tid) {
    b += b_locals[tid];
}

#pragma omp parallel for num_threads(threads) schedule(dynamic)
for (int i = 0; i < w.cols(); ++i) {
    Eigen::VectorXd b_col = b.col(i);
    if (b_col.isZero()) continue;

    if (L1 != 0) b_col.array() -= L1;

    if (upper_bound > 0) {
        c_bnnls(a, b_col, w, i, (int)upper_bound);
    } else {
        c_nnls(a, b_col, w, i);
    }
}
    } 
}
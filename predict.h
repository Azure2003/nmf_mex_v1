#ifndef PREDICT_H
#define PREDICT_H

#include "allFiles.h"
#include <Eigen/Dense>
#include <vector>

#ifndef TINY_NUM
#define TINY_NUM 1e-15  // epsilon for numerical stability
#define TINY_NUM_FOR_STABILITY 1e-15
#endif
// Forward declarations for helper functions (implementations expected in .cpp)


// Main predict function for sparse matrix A
void predict(const SparseMatrix64& A,
             const Eigen::SparseMatrix<double>& mask_A,
             const Eigen::SparseMatrix<double>& mask_h,
             const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h,
             double L1,
             double L2,
             unsigned int threads,
             bool mask_zeros,
             bool masking_A,
             bool masking_h,
             double upper_bound = 0);

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
             double upper_bound);
#endif // PREDICT_HPP

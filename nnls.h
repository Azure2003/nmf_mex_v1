#ifndef nnls_h
#define nnls_h

#include <Eigen/Dense>
#ifndef CD_PARAMS
#define CD_PARAMS
#define CD_TOL 1e-8
#define CD_MAXIT 100
#endif
#ifndef TINY_NUM
#define TINY_NUM 1e-15  // epsilon for numerical stability
#define TINY_NUM_FOR_STABILITY 1e-15
#endif
void c_nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const unsigned int sample);
void nnls2(const Eigen::Matrix2d& a, const Eigen::Vector2d& b, const double denom, Eigen::MatrixXd& x, const unsigned int i, const bool nonneg) ;
void c_bnnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const unsigned int sample, const double upper_bound=1);
void nnls2InPlace(const Eigen::Matrix2d& a, const double denom, Eigen::MatrixXd& w, const bool nonneg);
#endif

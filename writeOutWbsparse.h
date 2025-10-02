#ifndef writeoutwbsparse_h
#define writeoutwbsparse_h

#include <Eigen/Dense>
void write_wbsparse_cpp_from_matrix(Eigen::MatrixXd& w, Eigen::VectorXd& D, Eigen::MatrixXd& H, std::string filepath);
#endif
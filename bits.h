#ifndef RCPPML_BITS_HPP
#define RCPPML_BITS_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>

template <class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
    const ArgType& m_arg;
    const RowIndexType& m_rowIndices;
    const ColIndexType& m_colIndices;

public:
    typedef Eigen::Matrix<typename ArgType::Scalar,
                          RowIndexType::SizeAtCompileTime,
                          ColIndexType::SizeAtCompileTime,
                          ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
                          RowIndexType::MaxSizeAtCompileTime,
                          ColIndexType::MaxSizeAtCompileTime>
        MatrixType;

    indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices);
    const typename ArgType::Scalar& operator()(Eigen::Index row, Eigen::Index col) const;
};

template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>,
                      typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
submat(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices);

// Non-template declarations
Eigen::MatrixXd submat(const Eigen::MatrixXd& x, const Eigen::VectorXi& col_indices);
Eigen::VectorXd subvec(const Eigen::MatrixXd& b, const Eigen::VectorXi& ind, unsigned int col);
Eigen::VectorXi find_gtz(const Eigen::MatrixXd& x, unsigned int col);
double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y);
std::vector<int> sort_index(const Eigen::VectorXd& d);
Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind);
Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind);
std::vector<double> getRandomValues(unsigned int len, unsigned int seed);
Eigen::MatrixXd randomMatrix(unsigned int nrow, unsigned int ncol, unsigned int seed);
bool isAppxSymmetric(Eigen::MatrixXd& A, double tol = 1e-12);
std::vector<unsigned int> nonzeroRowsInCol(const Eigen::MatrixXd& x, unsigned int i);
unsigned int n_nonzeros(const Eigen::MatrixXd& x);

// Template definitions (must remain in header)
template <class ArgType, class RowIndexType, class ColIndexType>
inline indexing_functor<ArgType, RowIndexType, ColIndexType>::indexing_functor(
    const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices) {}

template <class ArgType, class RowIndexType, class ColIndexType>
inline const typename ArgType::Scalar& indexing_functor<ArgType, RowIndexType, ColIndexType>::operator()(
    Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
}

template <class ArgType, class RowIndexType, class ColIndexType>
inline Eigen::CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>,
                      typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
submat(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices) {
    typedef indexing_functor<ArgType, RowIndexType, ColIndexType> Func;
    typedef typename Func::MatrixType MatrixType;
    return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

#endif // RCPPML_BITS_HPP

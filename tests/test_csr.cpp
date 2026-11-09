#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "CSR_and_Vectors.h"

// Dense matrix-vector product, written in the most obvious way possible.
// It is the "judge": if CSR disagrees with it, the bug is in CSR.
static DVector dense_times(const std::vector<std::vector<double>>& M, const DVector& x) {
    DVector y(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j)
            y[i] += M[i][j] * x[j];
    return y;
}

// Non-symmetric on purpose (a symmetric matrix would hide a row/column swap)
// and with an empty row (it catches mistakes in row_ptr).
TEST(Csr, SpmvMatchesDenseOnNonSymmetricMatrixWithEmptyRow) {
    std::vector<std::vector<double>> M = {{1, 2, 0}, {0, 0, 0}, {3, 0, 4}};
    CSR A(M);
    DVector x = {1.0, -2.0, 0.5};

    DVector y = A * x;
    DVector expected = dense_times(M, x);

    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < y.size(); ++i)
        EXPECT_NEAR(y[i], expected[i], 1e-14) << "row " << i;
}

// The 4x4 tridiagonal matrix written directly in CSR form.
TEST(Csr, ArrayConstructorHasTheRightDimension) {
    CSR A({2, -1, -1, 2, -1, -1, 2, -1, -1, 2},
          {0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
          {0, 2, 5, 8, 10});

    EXPECT_EQ(A.size(), 4u);          // number of rows, NOT of non-zeros
    EXPECT_EQ(A.get_val_size(), 10u); // number of non-zeros

    DVector y = A * DVector(4, 1.0);
    EXPECT_EQ(y, (DVector{1, 0, 0, 1}));
}

TEST(Csr, ArrayConstructorRejectsABadRowPtr) {
    // row_ptr must start at 0
    EXPECT_THROW(CSR({1.0, 2.0}, {0, 1}, {1, 1, 2}), std::invalid_argument);
}

TEST(Csr, IsSymmetricUsesATolerance) {
    std::vector<std::vector<double>> S = {{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
    std::vector<std::vector<double>> N = {{2, -1, 0}, {-3, 2, 0}, {0, 0, 1}};
    CSR symmetric(S), non_symmetric(N);

    EXPECT_TRUE(symmetric.is_symmetric(1e-12));
    EXPECT_FALSE(non_symmetric.is_symmetric(1e-12));
    EXPECT_TRUE(non_symmetric.is_symmetric(10.0));  // |a_ij - a_ji| = 2 <= 10
}

TEST(Vectors, DotAndNorm) {
    EXPECT_DOUBLE_EQ(dot({1, 2, 3}, {4, 5, 6}), 32.0);
    EXPECT_DOUBLE_EQ(norm({3, 4}), 5.0);
    EXPECT_THROW(dot({1, 2}, {1, 2, 3}), std::invalid_argument);
}

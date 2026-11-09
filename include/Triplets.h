#ifndef TRIPLETS_H
#define TRIPLETS_H

#include <cstddef>
#include <vector>
#include "CSR_and_Vectors.h"

struct Triplet{
    size_t row, col;
    double val;
};

//Sums duplicates, retains zeros, and throws `std::invalid_argument` if `n == 0` or if any triplet has a row or column index ≥ `n`
CSR csr_from_triplets(size_t n, const std::vector<Triplet> &v);

#endif
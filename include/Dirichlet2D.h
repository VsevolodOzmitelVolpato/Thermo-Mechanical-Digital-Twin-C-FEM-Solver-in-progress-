#ifndef DIRICHLET2D_H
#define DIRICHLET2D_H

#include <cstddef>
#include <vector>
#include "Triplets.h"
#include "CSR_and_Vectors.h"

//throws out triplets with fixed row or column, adds (d, d, 1)), and it doesn't add, doesn't remove zeros, doesn't change the received list),
// and when it throws std::invalid_argument (fixed node >= n, repeated fixed node, triplet with row or column >= n).
std::vector<Triplet> dirichlet_matrix_triplets_2d(std::size_t n, const std::vector<Triplet> &T, const std::vector<std::size_t> &fixed_nodes);

//It performs the calculation (using the imposed values ​​for fixed nodes; instead for the for free nodes, it uses f[i] minus the contribution 
//from the fixed nodes; K and f remain the original ones) and throws `std::invalid_argument` in four cases: K and f have different dimensions; 
//the number of provided values ​​does not match the number of nodes; or a fixed node is specified more than once (even if the assigned value is the same)
DVector dirichlet_rhs_2d(const CSR &csr, const DVector &f, const std::vector<std::size_t> &fixed_nodes, const std::vector<double> &fixed_values);

#endif
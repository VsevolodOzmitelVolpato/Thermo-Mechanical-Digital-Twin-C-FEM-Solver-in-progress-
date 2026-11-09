#include "Solver2D.h"
#include <cstddef>
#include <vector>
#include "Mesh2D.h"
#include "CSR_and_Vectors.h"
#include "CGSolver.h"
#include "Dirichlet2D.h"
#include "Assembly2D.h"
#include <stdexcept>

DVector solve_steady_2d(const Mesh2D &M, double k,const DVector &f, const std::vector<std::size_t> &fixed_nodes,const std::vector<double> &fixed_values){
    std::size_t n = M.num_nodes();

    if(!(k > 0.0)){
        throw std::invalid_argument("K must be positive");}
    if(f.size() != n){
        throw std::invalid_argument("the size of the vector doesnt match the size of the mesh");}
    if(fixed_nodes.empty()){
        throw std::invalid_argument("the fixed nodes are empty");}

    std::vector<Triplet> triplets = stiffness_triplets_2d(M, k);

    CSR K = csr_from_triplets(n, triplets);
    CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, triplets, fixed_nodes));

    DVector b = dirichlet_rhs_2d(K, f, fixed_nodes, fixed_values);
    DVector x0(n, 0.0);

    for(std::size_t q {0}; q < fixed_nodes.size(); ++q){
        x0[fixed_nodes[q]] = fixed_values[q];}

    CGSolver CGS(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    CGS.solve(true);

    if(!CGS.has_converged()){
        throw std::runtime_error("The matrix has not converged");}

    return CGS.get_solution();
}

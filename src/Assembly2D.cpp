#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include "Fem2D.h"
#include "Triplets.h"
#include <vector>

CSR assemble_stiffness_2d(const Mesh2D &M, double k){
    std::vector<Triplet> tvec = stiffness_triplets_2d(M, k);
    return csr_from_triplets(M.num_nodes(), tvec);
}

CSR assemble_mass_2d(const Mesh2D &M, double rho_c){
    std::vector<Triplet> tvec = mass_triplets_2d(M, rho_c);
    return csr_from_triplets(M.num_nodes(), tvec);
}

std::vector<double> assemble_load_2d(const Mesh2D &M, double q){
    std::vector<double> vec(M.num_nodes(), 0.0);

    for(std::size_t e {0}; e < M.num_triangles(); ++e){
        Triangle tr = M.triangle(e);
        Vector3 v3 = element_load(M, e, q);

        for(std::size_t i {0}; i < 3; ++i){
            std::size_t global_node = tr[i];
            vec[global_node] += v3[i];
        }
    }
   
    return vec;
}

namespace{

    std::vector<Triplet> matrix_triplets(const Mesh2D &M, Matrix3 (*element_matrix)(const Mesh2D&, std::size_t, double), double coefficient){
    std::vector<Triplet> tvec;
    tvec.reserve(9 * M.num_triangles());

    for(std::size_t e {0}; e < M.num_triangles(); ++e){
        const Triangle& triangle = M.triangle(e);
        const Matrix3 mat = element_matrix(M, e, coefficient);

        for(std::size_t i {0}; i < 3; ++i){
            for(std::size_t j {0}; j < 3; ++j){
                Triplet t {triangle[i], triangle[j], mat[i][j]};
                tvec.push_back(t);}}
    }
    
    return tvec;}
    
}

std::vector<Triplet> stiffness_triplets_2d(const Mesh2D &M, double k){
    return matrix_triplets(M, element_stiffness, k);
}

std::vector<Triplet> mass_triplets_2d(const Mesh2D &M, double rho_c){
    return matrix_triplets(M, element_mass, rho_c);
}
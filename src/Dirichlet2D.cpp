#include "Dirichlet2D.h"
#include <cstddef>
#include <vector>
#include "Triplets.h"
#include <stdexcept>

std::vector<Triplet> dirichlet_matrix_triplets_2d(std::size_t n, const std::vector<Triplet> &T, const std::vector<std::size_t> &fixed_nodes){
    std::vector<bool> bl_vec(n, false);

    for( const auto &d : fixed_nodes){
        if( d >= n){
            throw std::invalid_argument("The fixed node do not exist");}
        if(bl_vec[d]){
            throw std::invalid_argument("repeated node: the two triplets (d, d, 1) would add up to 2");}
        
        bl_vec[d] = true;
    }

    std::vector<Triplet> tvect;
    tvect.reserve(T.size() + fixed_nodes.size());

    for( const auto &t : T){
        if( t.row >= n or t.col >= n){
            throw std::invalid_argument("one of the triplet exceeds in size");}
        
        if(!bl_vec[t.row] && !bl_vec[t.col]){
            tvect.push_back(t);}
    }

    for( const auto &d : fixed_nodes){
        tvect.push_back({d,d,1.0});
    }

    return tvect;
}

DVector dirichlet_rhs_2d(const CSR &csr, const DVector &f, const std::vector<std::size_t> &fixed_nodes, const std::vector<double> &values){
    if(f.size() != csr.size()){
        throw std::invalid_argument("the vector and the CSR matrix have different size");}

    
    if(fixed_nodes.size() != values.size()){
        throw std::invalid_argument("the fixed nodes and the values have different size");}

    size_t n = f.size();
    DVector g_hat(n, 0.0);
    std::vector<bool> is_fixed(n, false);

    for(size_t q{0}; q < fixed_nodes.size(); ++q){
        std::size_t d = fixed_nodes[q];
        if(d >= n){
            throw std::invalid_argument("one of the fixed nodes is invalid");}
        
        if(is_fixed[d]){
            throw std::invalid_argument("one or more nodes are repeated");}

        is_fixed[d] = true;
        g_hat[d] = values[q];
    }

    DVector b = f - csr * g_hat;

    for(size_t q{0}; q < fixed_nodes.size(); ++q){
        b[fixed_nodes[q]] = values[q];}

    return b;
    
}
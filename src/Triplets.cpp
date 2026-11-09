#include "Triplets.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace {
    bool triplets_comparison(const Triplet &t1, const Triplet &t2){
    if (t1.row != t2.row) {
        return t1.row < t2.row;}
    
    return t1.col < t2.col;}
}

CSR csr_from_triplets(size_t n, const std::vector<Triplet> &v){
    if( n == 0){
        throw std::invalid_argument("the n parameter cannot be equal to 0 ");}

    for(auto &t : v){
        if( (t.col >= n) or (t.row >= n)){
            throw std::invalid_argument("either the row or the column have an invalid size");}}
    
    std::vector<Triplet> trips = v;
    std::stable_sort(trips.begin(), trips.end(), triplets_comparison);

    std::vector<double> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr(n + 1, 0);

    for(size_t k {0}; k < trips.size(); ++k){
        if(k > 0 and (trips[k-1].col == trips[k].col and trips[k-1].row == trips[k].row)){
            values.back() += trips[k].val;}
        else{
            values.push_back(trips[k].val);
            col_idx.push_back(static_cast<int>(trips[k].col));
            row_ptr[trips[k].row + 1] +=1;
        }}

    std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

    return CSR(values, col_idx, row_ptr);
}
#include "CSR_and_Vectors.h"
#include <stdexcept>
#include <cmath>

CSR::CSR( const std::vector<std::vector<double>> &Mat){
    size_t s = Mat.size();
    n = s;
    bool is_quad = true;

    for (const auto& line: Mat){
        if(line.size() != n){
            is_quad = false;}}

    if(!is_quad || Mat.empty()){
        throw std::invalid_argument("The matrix is either empty or not square");}
    
    row_ptr.push_back(0);
    for (size_t i = 0; i < n; ++i) {
       for (size_t j = 0; j < n; ++j)
           if (Mat[i][j] != 0) { 
              values.push_back(Mat[i][j]); col_indx.push_back(j);}
       row_ptr.push_back(values.size());
    };
};

std::vector<double> CSR::operator*(const std::vector<double> &v) const{
    std::vector<double> y (n,0.0);
    
    if(n != v.size()){
        throw std::invalid_argument("the dimension of the vector and matrix are different!!");
        return y;
    }
    for(size_t i {0}; i < v.size(); ++i){
        for(size_t j  = row_ptr[i]; j < row_ptr[i+1]; ++j){
            y[i] += values[j]*v[col_indx[j]];
        }}

    return y;
};


double CSR::get(size_t i, size_t j) const {
    for(size_t k = row_ptr[i]; k < row_ptr[i+1]; ++k){
        if(j == col_indx[k]){
            return values[k];}
    }

    return 0.0;
}

std::vector<double> CSR::diag() const{
    std::vector<double> d (n,0);
    for(size_t i {0}; i < n; ++i){
        d[i] = get(i,i);
    }
    return d;
}

size_t CSR::size() const{ return n;};

CSR::CSR( const std::vector<double> &a, const std::vector<int> &b, const std::vector<int> &c) : values(a), col_indx(b), row_ptr(c),
          n(c.empty() ? 0 : c.size() - 1) {
    if (c.empty()    ||    (c[0] != 0)   ||  (a.size() != b.size())   || (static_cast<size_t>(c.back()) != a.size())) {
        throw std::invalid_argument("Invalid CSR matrix, try again!");}

    for (size_t i = 1; i < c.size(); ++i) {
        if (c[i] < c[i-1]) {
            throw std::invalid_argument("The row vector is wrong");}}
}
    

bool CSR::is_symmetric(const double &toll) const{
    
    for( std::size_t i = 0; i < n; i++ ){
        for( std::size_t k = row_ptr[i]; k < row_ptr[i + 1]; k++ ){
            if( std::fabs(values[k] - get(col_indx[k], i)) > toll){
                return false;}}}
    return true;
}

size_t CSR::get_val_size() const{
    return values.size();
}

DVector operator+(const DVector &v1, const DVector &v2){
    DVector z( v1.size() );
    if( v1.size() != v2.size() )
    {
        throw std::invalid_argument("Error: vectors must have the same dimension!");
        return z;
    }
    for( std::size_t i = 0; i < v1.size(); i++ )
        z[i] = v1[i] + v2[i];
    return z;
    
}

DVector operator-(const DVector &v1, const DVector &v2){
    DVector z( v1.size() );
    if( v1.size() != v2.size() )
    {
        throw std::invalid_argument("Error: vectors must have the same dimension!");
        return z;
    }
    for( std::size_t i = 0; i < v1.size(); i++ )
        z[i] = v1[i] - v2[i];
    return z;
    
}

DVector operator*(const double &d, const DVector &v4){
    DVector vv (v4.size());
    for(size_t i {0}; i < v4.size(); ++i){
        vv[i] = d * v4[i];
    }
    return vv;
}

double  dot(const DVector &v6, const DVector &v7){
    double z = 0.0;
    if( v6.size() != v7.size() )
    {
        throw std::invalid_argument("Error: vectors must have the same dimension!");
        return z;
    }

    for(size_t i {0}; i < v6.size(); ++i){
        z += v6[i]*v7[i];
    }
    return z;
}

double norm(const DVector &a){
   return std::sqrt(dot(a,a));
}
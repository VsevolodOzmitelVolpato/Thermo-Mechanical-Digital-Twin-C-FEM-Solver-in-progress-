#include "JacobiPreconditioner.h"
#include <stdexcept>
#include "CSR_and_Vectors.h"

JacobiPreconditioner::JacobiPreconditioner( const CSR &A )
    : inv_diag( A.diag() )
{
    for( std::size_t i = 0; i < inv_diag.size(); i++ )
    {
        if( inv_diag[i] <= 0.0 )
            throw std::invalid_argument("The diagonal entries of A must be positive!");
        inv_diag[i] = 1.0 / inv_diag[i];
    }
}

DVector JacobiPreconditioner::apply( const DVector &r ) const
{
    if( r.size() != inv_diag.size() )
        throw std::invalid_argument("Dimensions don't agree!");
    DVector z( r.size() );
    for( std::size_t i = 0; i < r.size(); i++ )
        z[i] = inv_diag[i] * r[i];
    return z;
}

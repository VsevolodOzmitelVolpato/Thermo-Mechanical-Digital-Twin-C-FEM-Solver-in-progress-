#ifndef JACOBIPRECONDITIONER_H
#define JACOBIPRECONDITIONER_H

#include "CSR_and_Vectors.h"
#include <vector>

class JacobiPreconditioner
{
    private:
        DVector inv_diag;

    public:
        JacobiPreconditioner( const CSR &A );

        DVector apply( const DVector &r ) const;
};


#endif

#ifndef CGSOLVER_H
#define CGSOLVER_H

#include <vector>
#include "CSR_and_Vectors.h"
#include "JacobiPreconditioner.h"

class CGSolver
{
    private:
        CSR A;

        DVector b;
        DVector x0;

        JacobiPreconditioner preconditioner;

        double tol;
        unsigned max_it;

        DVector solution;
        DVector residuals;

        unsigned iterations;
        bool preconditioned;
        bool converged = false;

    public:
        CGSolver( const CSR &matrix, const DVector &rhs,const DVector &initial_guess, double tolerance, unsigned max_iterations);

        void solve(bool use_preconditioner);

        // Getters
        DVector get_solution() const { return solution; }
        DVector get_residuals() const { return residuals; }
        unsigned get_iterations() const { return iterations; }
        bool has_converged() const {return converged;}

        void print() const;

};


#endif // CGSOLVER_HPP

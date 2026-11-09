#include <iostream>
#include <stdexcept>
#include "CGSolver.h"
#include "CSR_and_Vectors.h"

CGSolver::CGSolver( const CSR &matrix, const DVector &rhs,const DVector &initial_guess,
           double tolerance, unsigned max_iterations)
    : A( matrix ), b( rhs ), x0( initial_guess ), preconditioner( matrix ),
      tol( tolerance ), max_it( max_iterations ), iterations( 0 ),
      preconditioned( false )
{
    if( A.size() != b.size() || b.size() != x0.size() )
        throw std::invalid_argument("Dimensions don't agree!");
    if( !A.is_symmetric(1e-12) )
        throw std::invalid_argument("The matrix A must be symmetric!");
}
void CGSolver::print() const
{
    std::cout << "Solved using: "
              << ( preconditioned ? "PCG (Jacobi)" : "CG" ) << std::endl;
    std::cout << "Size n              = " << A.size() << std::endl;
    std::cout << "Non-zero entries    = " << A.get_val_size() << std::endl;
    std::cout << "Tolerance           = " << tol << std::endl;
    std::cout << "Max iterations      = " << max_it << std::endl;
    std::cout << "Iterations          = " << iterations << std::endl;
    if( !residuals.empty() )
        std::cout << "Relative residual   = "
                  << residuals.back() / norm(b) << std::endl;
}


void CGSolver::solve( bool use_preconditioner )
{
    // N.B.: the output members are reset here, so that solve() can be
    // called more than once on the same object
    preconditioned = use_preconditioner;
    solution = x0;
    residuals.clear();
    iterations = 0;

    DVector r = b - A * x0;
    DVector z;
    if (use_preconditioner) {
        z = preconditioner.apply((r)) ;}
    else {
        z = r;}

    DVector p = z;
    double rho = dot(r, z);
    unsigned int k = 0;
    residuals.push_back((norm(r)));
    double a = 0.0, beta= 0.0;

    while ((norm(r) > tol * norm(b)) && k < max_it) {
        DVector q = A*p;
        a = rho/(dot(p, q));
        solution = solution + a * p;
        r = r - a * q;
        residuals.push_back((norm(r)));
        if (use_preconditioner) {
            z = preconditioner.apply((r)) ;}
        else {
            z = r;}
        double rho2 = dot(r, z);
        beta = rho2/ rho;
        rho = rho2;
        p = z + beta * p;
        ++k;}

    iterations = k;

    converged= (norm(r) <= tol * norm(b));
}
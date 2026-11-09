#include <stdexcept>
#include "Thermoelastic1D.h"
#include "CGSolver.h"
#include "Fem1D.h"

DVector assemble_thermal_load(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref) {
    if ( T.size() != x.size()){
        throw std::invalid_argument("The two vectors have different sizes");}

    DVector f (x.size(), 0.0);

    for( size_t i {1}; i < x.size(); ++i){
        double dT_mean = ((T[i-1] - T_ref) + (T[i] - T_ref)) / 2;
        double force = E * alpha * dT_mean;
        f[i-1] -= force;
        f[i] += force;}

    return f;  
}

DVector solve_thermal_expansion(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, RightEnd right_end) {
    if( x.size() < 2){
        throw std::invalid_argument("Vector has one argument");
    }
    if( !(E > 0)){
        throw std::invalid_argument("K argument is not valid");
    }

    size_t n = x.size();
    DMatrix K = assemble_stiffness(x, E);
    DVector f = assemble_thermal_load(x, E, alpha, T, T_ref);

    apply_dirichlet(K, f, 0, 0.0);

    if(right_end == RightEnd::Fixed){
        apply_dirichlet(K, f, n - 1, 0.0);
    }
    
    CSR A(K);
    DVector u0(n, 0.0);
    CGSolver CG(A, f, u0, 1e-12, 10 * n);
    CG.solve(true);

    if(!CG.has_converged()){
        throw std::runtime_error("The solution has not converged");
    }

    return CG.get_solution() ;
}

DVector element_stress(const std::vector<double>& x, double E, double alpha, const DVector& T, double T_ref, const DVector& u) {
    if((u.size() != x.size()) || (T.size() != x.size())){
        throw std::invalid_argument("The vectors have different sizes");
    }
    
    DVector sigma;

    for(size_t i {1}; i < x.size(); ++i){
        double h = x[i] - x[i - 1];
        double tot_deformation = (u[i]- u[i - 1])/h;
        double thermal_deform = alpha * ((T[i - 1] - T_ref) + (T[i] - T_ref)) / 2;
        sigma.push_back(E * (tot_deformation - thermal_deform));
    }

    return sigma;
}

ThermoMechanicalResult solve_thermomechanical(const ThermoMechanicalProblem &problem) {
    ThermoMechanicalResult results;
    
    results.T = solve_transient(problem.x, problem.rho_c, problem.k, problem.q, problem.T_left, 
                               problem.T_right, problem.T0, problem.dt, problem.n_steps, problem.theta);
    
    results.u = solve_thermal_expansion(problem.x, problem.E, problem.alpha, results.T, problem.T_ref, problem.right_end);

    results.sigma = element_stress(problem.x, problem.E, problem.alpha, results.T, problem.T_ref, results.u);

    return results;
}
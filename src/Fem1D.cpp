#include "Fem1D.h"
#include "CSR_and_Vectors.h"
#include "CGSolver.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

const double pi = std::acos(-1.0);

std::vector<double> uniform_mesh(double L, std::size_t n_elements) {
    std::vector<double> mesh (n_elements + 1);

    for(size_t i {0}; i <= n_elements; ++i){
        mesh[i] = L*i/n_elements;}
    
    return mesh; 
}

DMatrix assemble_stiffness(const std::vector<double>& x, double k) {
    size_t n = x.size();
    DMatrix AS (n, std::vector<double> (n, 0.0));
    for(size_t i {1}; i < n; ++i){
        double h = x[i] - x[i-1];
        AS[i-1][i-1] += k/h;
        AS[i][i] += k/h;
        AS[i-1][i] = -k/h;
        AS[i][i-1] = -k/h;}

    return AS;
}

DVector assemble_load(const std::vector<double>& x, double q) {
    size_t n = x.size();
    DVector v(n, 0.0);

    for(size_t i {1}; i < n; ++i){
        double h = x[i] - x[i-1];
        v[i-1] += h*q/2;
        v[i] += h*q/2;
    }

    return v;
}

void apply_dirichlet(DMatrix &K, DVector &F, size_t node, double g){
    size_t size = F.size();
    if (size != K.size()){
        throw std::invalid_argument("the matrix and the vector have different sizes");}

    if (size <= node){
        throw std::invalid_argument("The node is bigger than the size");}
    
    for(size_t i {0}; i < size; ++i){
        if (i != node){
            F[i] -= K[i][node]*g;
        }
        K[node][i] = 0;
        K[i][node] = 0;
    }

    K[node][node] = 1;
    F[node] = g;
}

DVector solve_steady(const std::vector<double>& x, double k, double q, double T_left, double T_right) {
    DMatrix Mat = assemble_stiffness(x, k);
    DVector F = assemble_load(x, q);

    apply_dirichlet(Mat, F, 0, T_left);
    apply_dirichlet(Mat, F, F.size() - 1, T_right);

    CSR K(Mat);
    size_t n = x.size();
    DVector init_guess(n, 0.0);
    CGSolver CG(K, F, init_guess, 1e-12, static_cast<unsigned>(10 * n));

    CG.solve(true);

    if(!CG.has_converged()){
        throw std::runtime_error("CG has not converged");
    }

    return CG.get_solution();
}

DMatrix assemble_mass(const std::vector<double>& x, double rho_c){
    size_t n = x.size();
    DMatrix AS (n, std::vector<double> (n, 0.0));
    for(size_t i {1}; i < n; ++i){
        double h = x[i] - x[i-1];
        double val = rho_c * h / 6;
        AS[i-1][i-1] += 2*val;
        AS[i][i] += 2*val;
        AS[i-1][i] = val;
        AS[i][i-1] = val;}

    return AS;
}

DVector solve_transient(const std::vector<double>& x, double rho_c, double k, double q, double T_left, double T_right, const DVector& T0,
                        double dt, std::size_t n_steps, double theta){
    if(T0.size() != x.size()){
        throw std::invalid_argument("The two vectors dont match the size");}

    if( dt <= 0){
        throw std::invalid_argument("the delta t is lower orequal to 0");}

    if(theta < 0 || theta > 1){
        throw std::invalid_argument("theta has an invalid value");}

    DMatrix K = assemble_stiffness(x, k);
    DVector F = assemble_load(x, q);
    DMatrix M = assemble_mass(x, rho_c);

    size_t n = x.size();

    DMatrix A (n, std::vector<double> (n, 0.0));
    DMatrix B (n, std::vector<double> (n, 0.0));

    for(size_t i {0}; i < n; ++i){
        for(size_t j {0}; j <n; ++j){
            A[i][j] = M[i][j] + theta * dt * K[i][j];
            B[i][j] = M[i][j] - (1 - theta) * dt * K[i][j];
        }}
    
    CSR B_CSR (B);
    DVector T = T0;

    for( size_t i {0}; i < n_steps; ++i){
        DVector rhs = B_CSR * T + dt*F;
        DMatrix A_2 = A;
        apply_dirichlet(A_2, rhs, 0, T_left);
        apply_dirichlet(A_2, rhs, n - 1, T_right);

        CSR A_CSR (A_2);
        CGSolver  CG (A_CSR, rhs, T,1e-12, static_cast<unsigned>(10 * n) );
        CG.solve(true);
        if(!CG.has_converged()){
            throw std::runtime_error("CG has not converged at step " + std::to_string(i));
        }
        T = CG.get_solution();
    }

    return T;
}

double l2_error(const std::vector<double>& x, const DVector& T_num, const DVector& T_ref){
    if(T_num.size() != x.size() || T_ref.size() != x.size()){
        throw std::invalid_argument("One of the Tvectors does not match the vector size");}
    
    DVector e = T_num - T_ref;
    DMatrix M = assemble_mass(x, 1.0);
    CSR M_CSR (M);

    double L2 = std::sqrt(dot(e, M_CSR * e));
    return L2;
}

double observed_order(double e_coarse, double e_fine, double ratio){
    if(!(ratio > 1) || !(e_coarse > 0) || !(e_fine > 0)){
        throw std::invalid_argument("Data are not valid");}
    
    double o = std::log(e_coarse / e_fine) / std::log(ratio);
    return o;
} 

double decaying_sine_error(std::size_t n_elements, double dt, std::size_t n_steps, double theta){
    if ( n_elements < 1){
        throw std::invalid_argument("The number of elements must be greater than 0");}

    DVector mesh = uniform_mesh(1.0, n_elements);
    DVector T(mesh.size());
   
    for( size_t i {0}; i < mesh.size(); ++i){
        T[i] =  std::sin(pi * mesh[i] );}

    DVector solved = solve_transient(mesh, 1.0, 1.0, 0.0, 0.0, 0.0, T, dt, n_steps, theta);

    double t = n_steps * dt;

    DVector T_exact(mesh.size());

    for (size_t i {0}; i < mesh.size(); ++i){
        T_exact[i] = std::exp(pi * pi * (-t)) * std::sin(pi * mesh[i]);}
    
    double d = l2_error(mesh, solved, T_exact);

    return d;
}


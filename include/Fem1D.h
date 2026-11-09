#ifndef FEM1D_H
#define FEM1D_H

#include <cstddef>
#include <vector>
#include "CSR_and_Vectors.h"

typedef std::vector<std::vector<double>> DMatrix;

std::vector<double> uniform_mesh(double L, std::size_t n_elements);

DMatrix assemble_stiffness(const std::vector<double>& x, double k);

DVector assemble_load(const std::vector<double>& x, double q);

void apply_dirichlet(DMatrix &K, DVector &F, size_t node, double g);

DVector solve_steady(const std::vector<double>& x, double k, double q, double T_left, double T_right);
// throw std::runtime_error if CG does not converge (ex. a NaN in the data).

DMatrix assemble_mass(const std::vector<double>& x, double rho_c);

DVector solve_transient(const std::vector<double>& x, double rho_c, double k, double q, double T_left, double T_right, const DVector& T0,
                        double dt, std::size_t n_steps, double theta);
// throw std::runtime_error if CG does not converge in a step (ex. a NaN in the data).
// or with an instable scheme where the numbers are not a double

double l2_error(const std::vector<double>& x, const DVector& T_num, const DVector& T_ref);

double observed_order(double e_coarse, double e_fine, double ratio);

double decaying_sine_error(std::size_t n_elements, double dt, std::size_t n_steps, double theta);
// correct order of the arguments: x, rho_c, k, q, T_left, T_right, T0, dt, n_steps, theta


#endif
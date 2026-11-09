#ifndef ASSEMBLY2D_H
#define ASSEMBLY2D_H

#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include <vector>
#include "Triplets.h"

CSR assemble_stiffness_2d(const Mesh2D &M, double k);

CSR assemble_mass_2d(const Mesh2D &M, double rho_c);

std::vector<double> assemble_load_2d(const Mesh2D &M, double q);

std::vector<Triplet> stiffness_triplets_2d(const Mesh2D &M, double k);

std::vector<Triplet> mass_triplets_2d(const Mesh2D &M, double rho_c);

#endif
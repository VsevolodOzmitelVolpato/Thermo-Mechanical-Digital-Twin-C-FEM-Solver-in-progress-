#ifndef FEM2D_H
#define FEM2D_H

#include <array>
#include <cstddef>
#include "Mesh2D.h"

using Matrix3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

Matrix3 element_stiffness(const Mesh2D &, size_t, double);

Matrix3 element_mass(const Mesh2D&, size_t, double);

Vector3 element_load(const Mesh2D &M, size_t e, double q);

#endif
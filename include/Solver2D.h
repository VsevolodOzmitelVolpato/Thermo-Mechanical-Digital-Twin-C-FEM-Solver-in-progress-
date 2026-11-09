#ifndef SOLVER2D_H
#define SOLVER2D_H

#include <cstddef>
#include <vector>
#include "Mesh2D.h"
#include "CSR_and_Vectors.h"

//it returns the temperature of each node—n values ​​in total; for the fixed nodes, the values ​​are exactly those that have been imposed.
//it needs at leats 1 fixed node because K is singular. it launchs invalid argument if: k is not positive or is NaN; f has a number of elements 
//other than n; no nodes are fixed; the number of values ​​differs from the number of fixed nodes; a fixed node is >= n; a fixed node is repeated
//or an unfixed node does not belong to any triangle. If CG doesnt converge it launchs a runtime error
DVector solve_steady_2d(const Mesh2D &M, double k,const DVector &f, const std::vector<std::size_t> &fixed_nodes,const std::vector<double> &values);



#endif
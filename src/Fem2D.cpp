#include "Fem2D.h"

Matrix3 element_stiffness(const Mesh2D &M, size_t e, double k){
    Triangle t = M.triangle(e);
    double area = M.area(e);

    std::array<double, 3> x{}, y{}, b{}, c{};

    for( size_t i {0}; i < 3; ++i){
        x[i] = M.x(t[i]);
        y[i] = M.y(t[i]);
    }

    for( size_t i {0}; i < 3; ++i){
        size_t p = (i + 1) % 3;
        size_t q = (i + 2) % 3;

        b[i] = y[p] - y[q];
        c[i] = x[q] - x[p];
    }

    Matrix3 K{};

    for( size_t i {0}; i < 3; ++i){
        for( size_t j {0}; j < 3; ++j){
            K[i][j] = k / (4.0 * area) * (b[i] * b[j] + c[i] * c[j]);}}

    return K;
}

Matrix3 element_mass(const Mesh2D &M, size_t e, double rho_c){
    double area = M.area(e);

    Matrix3 matrix{};

    for( size_t i {0}; i < 3; ++i){
        for( size_t j {0}; j < 3; ++j){
            matrix[i][j] = (i == j) ? (rho_c * area / 6.0) : (rho_c * area / 12.0);}}

    return matrix;
}

Vector3 element_load(const Mesh2D &M, size_t e, double q){
    double area = M.area(e);

    Vector3 vect{};
    vect.fill(q * area / 3.0);

    return vect;
}


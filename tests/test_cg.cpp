#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "CGSolver.h"

// tridiag(-1, 2, -1) of size n, built from a dense matrix.
static CSR poisson1d(size_t n) {
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        M[i][i] = 2.0;
        if (i > 0)     M[i][i - 1] = -1.0;
        if (i + 1 < n) M[i][i + 1] = -1.0;
    }
    return CSR(M);
}

// -u'' = 1 on (0,1), u(0) = u(1) = 0 has the exact solution u = x(1-x)/2,
// a quadratic: the 3-point scheme is exact on quadratics, so the discrete
// solution equals u at the nodes. Any gap is a bug in your code, not in the
// discretisation.
TEST(Cg, Poisson1dIsExactAtTheNodes) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver solver(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    solver.solve(false);
    DVector x = solver.get_solution();

    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double xi = (i + 1) * h;
        max_error = std::max(max_error, std::fabs(x[i] - xi * (1.0 - xi) / 2.0));
    }
    EXPECT_LT(max_error, 1e-10);

    // A constant right-hand side only touches half of the eigenvectors,
    // so CG finishes in about n/2 iterations.
    EXPECT_GT(solver.get_iterations(), 0.4 * n);
    EXPECT_LT(solver.get_iterations(), 0.6 * n);
}

TEST(Cg, JacobiPcgGivesTheSameAsCgOnPoisson1d) {
    // [1] il problema
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);
    const unsigned max_it = static_cast<unsigned>(10 * n);

    // [2] due solver con gli STESSI dati
    CGSolver cg(A, b, x0, 1e-12, max_it);
    CGSolver pcg(A, b, x0, 1e-12, max_it);
    cg.solve(false);   // CG semplice
    pcg.solve(true);   // CG con Jacobi

    // [3] stesso numero di iterazioni
    EXPECT_EQ(cg.get_iterations(), pcg.get_iterations());

    // [4] stessa soluzione, nodo per nodo
    DVector x_cg = cg.get_solution();
    DVector x_pcg = pcg.get_solution();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_cg[i], x_pcg[i], 1e-12) << "nodo " << i;
    }
}


static double max_distance_from_ones(const DVector& x) {
    double e = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e = std::max(e, std::fabs(x[i] - 1.0));
    }
    return e;
}

TEST(Cg, JacobiPcgNeedsFarFewerIterationsOnBadlyScaledMatrix) {
    // [1] la scala: un numero s[i] per ogni punto, da 1 a circa 31.6
    const size_t n = 400;
    std::vector<double> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::pow(10.0, 1.5 * static_cast<double>(i) / static_cast<double>(n - 1));
    }

    // [2] la matrice B = S * A * S  (A = Poisson), riempita a mano
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        M[i][i] = s[i] * 2.0 * s[i];
        if (i > 0)     M[i][i - 1] = s[i] * (-1.0) * s[i - 1];
        if (i + 1 < n) M[i][i + 1] = s[i] * (-1.0) * s[i + 1];
    }
    CSR B(M);

    // [3] parto dalla RISPOSTA (tutti 1) e calcolo il termine noto b = B * 1
    DVector ones(n, 1.0), x0(n, 0.0);
    DVector b = B * ones;
    const unsigned max_it = static_cast<unsigned>(10 * n);

    // [4] due solver, uno senza e uno con Jacobi
    CGSolver cg(B, b, x0, 1e-12, max_it);
    CGSolver pcg(B, b, x0, 1e-12, max_it);
    cg.solve(false);
    pcg.solve(true);
    std::cout << "CG: " << cg.get_iterations() << " iterazioni, PCG: "
              << pcg.get_iterations() << " iterazioni" << std::endl;

    // [5] controlli
    EXPECT_LT(4 * pcg.get_iterations(), cg.get_iterations());
    EXPECT_LT(max_distance_from_ones(cg.get_solution()), 1e-8);
    EXPECT_LT(max_distance_from_ones(pcg.get_solution()), 1e-8);
}

TEST(Cg, WrongInputsThrow) {
    CSR A = poisson1d(10);   // matrice 10x10, simmetrica: va bene

    // [1] b troppo corto: 9 numeri invece di 10
    EXPECT_THROW(CGSolver(A, DVector(9, 1.0), DVector(10, 0.0), 1e-10, 100), std::invalid_argument);

    // [2] x0 troppo corto: 9 numeri invece di 10
    EXPECT_THROW(CGSolver(A, DVector(10, 1.0), DVector(9, 0.0), 1e-10, 100), std::invalid_argument);

    // [3] matrice NON simmetrica: il CG non puo funzionare
    std::vector<std::vector<double>> N = {{2, -1, 0}, {-3, 2, 0}, {0, 0, 1}};
    CSR not_symmetric(N);
    EXPECT_THROW(CGSolver(not_symmetric, DVector(3, 1.0), DVector(3, 0.0), 1e-10, 100), std::invalid_argument);
}

// ---- Il solver dice se e' arrivato a convergenza -------------------------------------

// Problema che CG risolve senza storie: Poisson 1D con 100 incognite e 10 n iterazioni a disposizione
// (la stessa del primo test). Sia CG semplice sia con Jacobi devono dire "convergenza".
TEST(Cg, ReportsConvergence) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);
    const unsigned max_it = static_cast<unsigned>(10 * n);

    CGSolver cg(A, b, x0, 1e-12, max_it);
    CGSolver pcg(A, b, x0, 1e-12, max_it);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_TRUE(cg.has_converged());
    EXPECT_TRUE(pcg.has_converged());
}

// Lo stesso problema con sole 5 iterazioni: con 100 incognite CG non puo' arrivare a 1e-12.
// Il solver si ferma perche' le iterazioni sono finite (ne fa esattamente 5) e deve dirlo.
TEST(Cg, ReportsFailureWhenTheIterationsRunOut) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver cg(A, b, x0, 1e-12, 5);
    CGSolver pcg(A, b, x0, 1e-12, 5);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_EQ(cg.get_iterations(), 5u);
    EXPECT_EQ(pcg.get_iterations(), 5u);
    EXPECT_FALSE(cg.has_converged());
    EXPECT_FALSE(pcg.has_converged());
}

// La risposta dipende dal RESIDUO alla fine, non dal numero di iterazioni usate.
// Prima si scopre quante iterazioni K servono davvero (calcolo di riferimento). Poi si rifa' lo
// stesso problema con max_it = K: le iterazioni finiscono proprio quando il residuo scende sotto
// soglia, quindi e' convergenza. Con max_it = K - 1 gli manca l'ultima iterazione: non lo e'.
TEST(Cg, ConvergenceDependsOnTheResidualNotOnTheIterationCount) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, h * h), x0(n, 0.0);

    CGSolver reference(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    reference.solve(false);
    ASSERT_TRUE(reference.has_converged());
    const unsigned K = reference.get_iterations();
    ASSERT_GT(K, 1u);

    CGSolver just_in_time(A, b, x0, 1e-12, K);
    just_in_time.solve(false);
    EXPECT_EQ(just_in_time.get_iterations(), K);
    EXPECT_TRUE(just_in_time.has_converged());

    CGSolver one_short(A, b, x0, 1e-12, K - 1);
    one_short.solve(false);
    EXPECT_EQ(one_short.get_iterations(), K - 1);
    EXPECT_FALSE(one_short.has_converged());
}

// Un NaN nel termine noto: il residuo e' NaN, e un NaN non e' ne' > ne' <= di nessun numero.
// Il ciclo di CG non parte nemmeno (0 iterazioni) e la soluzione e' ancora la stima iniziale.
// Il solver NON deve dire "convergenza": un risultato senza senso non e' un risultato.
TEST(Cg, NaNInTheRightHandSideIsNotConvergence) {
    CSR A = poisson1d(10);
    DVector b(10, 1.0), x0(10, 0.0);
    b[3] = std::nan("");

    CGSolver cg(A, b, x0, 1e-10, 100);
    CGSolver pcg(A, b, x0, 1e-10, 100);
    cg.solve(false);
    pcg.solve(true);

    EXPECT_FALSE(cg.has_converged());
    EXPECT_FALSE(pcg.has_converged());
}

// b = 0 e partenza x0 = 0: la soluzione e' esattamente x = 0 e il residuo e' 0 fin dall'inizio.
// Il criterio |r| <= tol |b| diventa 0 <= 0, che e' vero: convergenza a 0 iterazioni.
// (Con "<" al posto di "<=" il solver direbbe di non essere arrivato, su un problema gia' risolto.)
TEST(Cg, ZeroRightHandSideWithZeroGuessConvergesAtOnce) {
    CSR A = poisson1d(10);
    DVector b(10, 0.0), x0(10, 0.0);

    CGSolver cg(A, b, x0, 1e-12, 100);
    cg.solve(false);

    EXPECT_EQ(cg.get_iterations(), 0u);
    EXPECT_TRUE(cg.has_converged());
    for (double xi : cg.get_solution()) {
        EXPECT_EQ(xi, 0.0);
    }
}

// La tolleranza e' RELATIVA a |b|. Qui b e' 1e-14 volte quello del primo test, quindi |b| e' circa
// 1e-17: il residuo iniziale e' gia' piu' piccolo di 1e-12 in valore assoluto, ma rispetto a |b|
// vale 1, cioe' non e' sceso per niente. Con 5 iterazioni il solver non puo' dire "convergenza";
// con 10 n iterazioni invece ci arriva (CG ragiona in modo identico a qualunque scala).
TEST(Cg, ConvergenceIsRelativeToTheSizeOfB) {
    const size_t n = 100;
    const double h = 1.0 / (n + 1);
    CSR A = poisson1d(n);
    DVector b(n, 1e-14 * h * h), x0(n, 0.0);

    CGSolver too_few(A, b, x0, 1e-12, 5);
    too_few.solve(false);
    EXPECT_FALSE(too_few.has_converged());

    CGSolver enough(A, b, x0, 1e-12, static_cast<unsigned>(10 * n));
    enough.solve(false);
    EXPECT_TRUE(enough.has_converged());
}

TEST(Jacobi, ZeroDiagonalAndWrongSizeThrow) {
    // [1] uno 0 sulla diagonale: Jacobi dovrebbe dividere per 0
    std::vector<std::vector<double>> Z = {{0, 1}, {1, 2}};
    CSR zero_diag(Z);
    EXPECT_THROW(JacobiPreconditioner jac(zero_diag), std::invalid_argument);

    // [2] apply() con un vettore di lunghezza sbagliata (3 invece di 4)
    JacobiPreconditioner J(poisson1d(4));
    EXPECT_THROW(J.apply(DVector(3, 1.0)), std::invalid_argument);
}
#include <gtest/gtest.h>

#include <vector>
#include <cmath>
#include "Fem1D.h"


TEST(Fem1d, DirichletWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);
    DVector F_short = {1.0, 2.0};    // 2 numeri invece di 3

    EXPECT_THROW(apply_dirichlet(K, F_short, 0, 10.0), std::invalid_argument);
    EXPECT_THROW(apply_dirichlet(K, F, 3, 10.0), std::invalid_argument);   // i nodi sono 0, 1, 2
}

TEST(Fem1d, UniformMeshHasTheRightNodes) {
    std::vector<double> x = uniform_mesh(2.0, 4);   // 4 elementi tra 0 e 2
    ASSERT_EQ(x.size(), 5u);                         // quindi 5 nodi
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i], 0.5 * i, 1e-14) << "nodo " << i;
    }
}

// Due elementi di lunghezza 1 e 2: nodi 0, 1, 3. Conducibilita k = 2.
TEST(Fem1d, StiffnessMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);

    DMatrix expected = {{ 2.0, -2.0,  0.0},
                        {-2.0,  3.0, -1.0},
                        { 0.0, -1.0,  1.0}};
    ASSERT_EQ(K.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(K[i].size(), 3u);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(K[i][j], expected[i][j], 1e-14) << "K[" << i << "][" << j << "]";
        }
    }
}

// Stessi nodi, sorgente q = 4.
TEST(Fem1d, LoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector F = assemble_load(x, 4.0);

    DVector expected = {2.0, 6.0, 4.0};
    ASSERT_EQ(F.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(F[i], expected[i], 1e-14) << "F[" << i << "]";
    }
}

static void expect_matrix_near(const DMatrix& A, const DMatrix& B, double tol) {
    ASSERT_EQ(A.size(), B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ASSERT_EQ(A[i].size(), B[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) {
            EXPECT_NEAR(A[i][j], B[i][j], tol) << "K[" << i << "][" << j << "]";
        }
    }
}

static void expect_vector_near(const DVector& a, const DVector& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "F[" << i << "]";
    }
}

// Stesso esempio di prima: x = {0, 1, 3}, k = 2, q = 4. Si fissa il nodo 0 a 10.
TEST(Fem1d, DirichletOnFirstNode) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 0, 10.0);

    expect_matrix_near(K, {{1.0,  0.0,  0.0},
                           {0.0,  3.0, -1.0},
                           {0.0, -1.0,  1.0}}, 1e-14);
    expect_vector_near(F, {10.0, 26.0, 4.0}, 1e-14);
}

// Stessi K e F, ma si fissa il nodo di mezzo (indice 1) a 5.
TEST(Fem1d, DirichletOnAMiddleNode) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 1, 5.0);

    expect_matrix_near(K, {{2.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, 1.0}}, 1e-14);
    expect_vector_near(F, {12.0, 5.0, 9.0}, 1e-14);
}

// Si fissano entrambi gli estremi: T(0) = 10 e T(3) = 40. K diventa diagonale e il
// nodo di mezzo si risolve a mano. La soluzione esatta di -k T'' = q e'
// T(x) = -x^2 + 13 x + 10, quindi T(1) = 22.
TEST(Fem1d, DirichletOnBothEndsGivesTheExactMiddleTemperature) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix K = assemble_stiffness(x, 2.0);
    DVector F = assemble_load(x, 4.0);

    apply_dirichlet(K, F, 0, 10.0);
    apply_dirichlet(K, F, 2, 40.0);

    expect_matrix_near(K, {{1.0, 0.0, 0.0},
                           {0.0, 3.0, 0.0},
                           {0.0, 0.0, 1.0}}, 1e-14);
    expect_vector_near(F, {10.0, 66.0, 40.0}, 1e-14);
    EXPECT_NEAR(F[1] / K[1][1], 22.0, 1e-14);
}

static double exact_steady(double x, double L, double k, double q, double a, double b) {
    return a + (b - a) * x / L + q * x * (L - x) / (2.0 * k);
}

// Risolve sui nodi dati e confronta con la soluzione esatta in ogni nodo.
// (In 1D, con k e q costanti, gli elementi lineari danno i valori ESATTI nei nodi, su qualunque
// maglia: per questo la tolleranza e' cosi' stretta e un test rosso vuol dire un errore nel codice.)
// Se qualcosa non torna stampa un solo messaggio: l'errore massimo e dove si trova.
static void check_against_exact(const std::vector<double>& nodes, double k, double q,
                                double a, double b, double tol) {
    DVector T = solve_steady(nodes, k, q, a, b);
    ASSERT_EQ(T.size(), nodes.size());
    const double L = nodes.back() - nodes.front();
    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const double err = std::fabs(T[i] - exact_steady(nodes[i], L, k, q, a, b));
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }
    EXPECT_LT(max_err, tol) << "errore massimo al nodo " << worst << " (x = " << nodes[worst]
                            << "): T = " << T[worst] << ", esatta = "
                            << exact_steady(nodes[worst], L, k, q, a, b);
}

// q = 0: nessuna sorgente, la temperatura e' una retta tra i due estremi (10 e 40).
TEST(Fem1d, SteadyLineBetweenTwoTemperatures) {
    check_against_exact(uniform_mesh(2.0, 20), 3.0, 0.0, 10.0, 40.0, 1e-9);
}

// Estremi a 0 e sorgente uniforme: parabola. Maglia uniforme.
TEST(Fem1d, SteadyParabolaOnUniformMesh) {
    check_against_exact(uniform_mesh(2.0, 50), 3.0, 5.0, 0.0, 0.0, 1e-9);
}

// Tutto insieme (sorgente + estremi diversi) su una maglia NON uniforme: x_i = L (i/N)^1.5.
// I nodi sono piu' fitti vicino a x = 0.
TEST(Fem1d, SteadySourceAndEndsOnGradedMesh) {
    const size_t N = 40;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }
    check_against_exact(x, 3.0, 5.0, 10.0, 40.0, 1e-9);
}

TEST(Fem1d, MassMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DMatrix M = assemble_mass(x, 6.0);

    DMatrix expected = {{2.0, 1.0, 0.0},
                        {1.0, 6.0, 2.0},
                        {0.0, 2.0, 4.0}};
    ASSERT_EQ(M.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(M[i].size(), 3u);
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(M[i][j], expected[i][j], 1e-14) << "M[" << i << "][" << j << "]";
        }
    }
}

// Su una maglia NON uniforme: M e' simmetrica e la somma di tutti i suoi numeri e' la
// capacita termica totale della sbarra, rho_c * L.
TEST(Fem1d, MassIsSymmetricAndSumsToTotalHeatCapacity) {
    const size_t N = 40;
    const double L = 1.7;
    const double rho_c = 2.5;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }

    DMatrix M = assemble_mass(x, rho_c);

    ASSERT_EQ(M.size(), N + 1);
    double sum = 0.0;
    for (size_t i = 0; i < M.size(); ++i) {
        ASSERT_EQ(M[i].size(), N + 1);
        for (size_t j = 0; j < M.size(); ++j) {
            sum += M[i][j];
            EXPECT_NEAR(M[i][j], M[j][i], 1e-14) << "M[" << i << "][" << j << "]";
        }
    }
    EXPECT_NEAR(sum, rho_c * L, 1e-12);
}

// ---- Evoluzione nel tempo (schema theta) ---------------------------------------------

// Fattore di amplificazione dello schema theta sul modo sin(pi x / L) di una maglia uniforme.
// Con T(0) = T(L) = 0 e q = 0, quel vettore e' un autovettore sia di K sia di M:
//   lambda_K = (k/h) (2 - 2 cos(pi h / L)),   lambda_M = (rho_c h / 6) (4 + 2 cos(pi h / L)).
// Quindi a ogni passo la temperatura viene moltiplicata per
//   g = (lambda_M - (1 - theta) dt lambda_K) / (lambda_M + theta dt lambda_K).
static double amplification_factor(double h, double L, double k, double rho_c, double dt, double theta) {
    const double pi = std::acos(-1.0);
    const double c = std::cos(pi * h / L);
    const double lambda_K = (k / h) * (2.0 - 2.0 * c);
    const double lambda_M = (rho_c * h / 6.0) * (4.0 + 2.0 * c);
    return (lambda_M - (1.0 - theta) * dt * lambda_K) / (lambda_M + theta * dt * lambda_K);
}

// Il risultato deve coincidere con g^n * sin(pi x / L) in ogni nodo. Non c'e' errore di
// discretizzazione da giustificare: e' la soluzione ESATTA dello schema discreto, quindi vale 1e-9.
// Si provano tre valori di theta, anche uno diverso da 0.5 e da 1 (cosi' non si puo' scambiare
// theta con 1 - theta senza che il test se ne accorga).
TEST(Fem1d, TransientMatchesTheAmplificationFactor) {
    const double pi = std::acos(-1.0);
    const size_t N = 20;
    const double L = 1.0, k = 2.0, rho_c = 3.0, dt = 0.01;
    const size_t n_steps = 10;
    std::vector<double> x = uniform_mesh(L, N);
    DVector T0(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        T0[i] = std::sin(pi * x[i] / L);
    }

    for (double theta : {1.0, 0.5, 0.3}) {
        DVector T = solve_transient(x, rho_c, k, 0.0, 0.0, 0.0, T0, dt, n_steps, theta);
        ASSERT_EQ(T.size(), N + 1);
        const double g = amplification_factor(L / N, L, k, rho_c, dt, theta);
        double max_err = 0.0;
        for (size_t i = 0; i <= N; ++i) {
            const double err = std::fabs(T[i] - std::pow(g, n_steps) * std::sin(pi * x[i] / L));
            if (err > max_err) {
                max_err = err;
            }
        }
        EXPECT_LT(max_err, 1e-9) << "theta = " << theta << " (g = " << g << ")";
    }
}

// Lo stato stazionario e' un punto fisso dello schema: se parti da una T con K T = F (estremi
// inclusi), dopo un passo, di qualunque ampiezza e con qualunque theta, resti dove sei.
// Maglia non uniforme, sorgente e estremi diversi: controlla F, il termine dt*F e gli estremi.
TEST(Fem1d, TransientKeepsTheSteadyState) {
    const size_t N = 40;
    const double L = 2.0, k = 3.0, rho_c = 1.5, q = 5.0, T_left = 10.0, T_right = 40.0;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
    }
    DVector T_steady = solve_steady(x, k, q, T_left, T_right);

    for (double theta : {0.5, 1.0}) {
        DVector T = solve_transient(x, rho_c, k, q, T_left, T_right, T_steady, 0.37, 1, theta);
        ASSERT_EQ(T.size(), N + 1);
        double max_diff = 0.0;
        for (size_t i = 0; i <= N; ++i) {
            const double diff = std::fabs(T[i] - T_steady[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        EXPECT_LT(max_diff, 1e-9) << "theta = " << theta;
    }
}

// Si parte da una temperatura uniforme (20) diversa dagli estremi (10 e 40): col tempo si
// arriva allo stato stazionario, e gli estremi restano fissi a ogni passo.
// theta = 1 (Eulero all'indietro): smorza anche i modi veloci, quindi 100 passi bastano.
TEST(Fem1d, TransientRelaxesToTheSteadyState) {
    const size_t N = 40;
    const double L = 2.0, k = 3.0, rho_c = 1.5, q = 5.0, T_left = 10.0, T_right = 40.0;
    std::vector<double> x = uniform_mesh(L, N);
    DVector T0(N + 1, 20.0);

    DVector T = solve_transient(x, rho_c, k, q, T_left, T_right, T0, 0.1, 100, 1.0);   // t = 10
    DVector T_steady = solve_steady(x, k, q, T_left, T_right);

    ASSERT_EQ(T.size(), N + 1);
    EXPECT_NEAR(T.front(), T_left, 1e-9);
    EXPECT_NEAR(T.back(), T_right, 1e-9);
    double max_diff = 0.0;
    for (size_t i = 0; i <= N; ++i) {
        const double diff = std::fabs(T[i] - T_steady[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    EXPECT_LT(max_diff, 1e-7);
}

// Input sbagliati: devono lanciare un'eccezione.
TEST(Fem1d, TransientWrongInputsThrow) {
    std::vector<double> x = uniform_mesh(1.0, 4);    // 5 nodi
    DVector T0(5, 0.0);
    DVector T0_short(4, 0.0);                        // 4 valori invece di 5

    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0_short, 0.1, 1, 0.5), std::invalid_argument);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, 0.0, 1, 0.5), std::invalid_argument);   // dt = 0
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, 0.1, 1, 1.5), std::invalid_argument);   // theta > 1
}

// ---- Verifica: errore L2 e ordine di convergenza ------------------------------------

// L'errore e' la funzione lineare a tratti che nei nodi vale e_i = T_num[i] - T_ref[i].
// La sua norma L2 e' sqrt(integrale di e^2) = sqrt(e^T M e), con M la matrice di massa (rho_c = 1).

// Due casi a mano.
//  1) "Cappello": e = (0, 1, 0) su 2 elementi lunghi 1. La funzione sale da 0 a 1 e riscende,
//     quindi integrale di e^2 = 1/3 + 1/3 = 2/3.
//  2) e = (0, 1, 1, 0) su 3 elementi lunghi 1: sale, resta a 1, riscende.
//     integrale di e^2 = 1/3 + 1 + 1/3 = 5/3. (Qui contano anche i termini fuori diagonale di M.)
TEST(Fem1d, L2ErrorMatchesHandComputation) {
    {
        std::vector<double> x = uniform_mesh(2.0, 2);
        DVector T_num = {0.0, 1.0, 0.0};
        DVector T_ref = {0.0, 0.0, 0.0};
        EXPECT_NEAR(l2_error(x, T_num, T_ref), std::sqrt(2.0 / 3.0), 1e-12);
    }
    {
        std::vector<double> x = uniform_mesh(3.0, 3);
        DVector T_num = {0.0, 1.0, 1.0, 0.0};
        DVector T_ref = {0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(l2_error(x, T_num, T_ref), std::sqrt(5.0 / 3.0), 1e-12);
    }
}

// Su una mesh NON uniforme, lunga 2:
//  - errore costante 0.5: integrale = 0.5^2 * 2, quindi la norma e' 0.5 * sqrt(2);
//  - errore e(x) = x (i valori nei nodi sono le x_i): la funzione lineare a tratti e' proprio la
//    retta e(x) = x, quindi integrale di x^2 su [0, 2] = 8/3, qualunque sia la mesh.
TEST(Fem1d, L2ErrorOfConstantAndLinearFieldsOnAGradedMesh) {
    const size_t N = 10;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = 2.0 * std::pow(static_cast<double>(i) / N, 1.5);
    }
    DVector zero(N + 1, 0.0);
    DVector constant(N + 1, 0.5);

    EXPECT_NEAR(l2_error(x, constant, zero), 0.5 * std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(l2_error(x, x, zero), std::sqrt(8.0 / 3.0), 1e-12);
}

// Due campi qualunque (nessuno dei due nullo) su una mesh non uniforme. Si confronta con
// l'integrale calcolato elemento per elemento: su un elemento lungo h, con errori ea (a sinistra)
// e eb (a destra), l'integrale di e^2 e' h (ea^2 + ea eb + eb^2) / 3.
// In piu': campi uguali danno errore 0, e scambiare i due campi non cambia il risultato.
TEST(Fem1d, L2ErrorMatchesTheElementByElementIntegral) {
    const size_t N = 8;
    std::vector<double> x(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = 3.0 * std::pow(static_cast<double>(i) / N, 2.0);
    }
    DVector a = {1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 6.0, 9.0};
    DVector b = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0};

    double integral = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double h = x[i + 1] - x[i];
        const double ea = a[i] - b[i];
        const double eb = a[i + 1] - b[i + 1];
        integral += h * (ea * ea + ea * eb + eb * eb) / 3.0;
    }

    EXPECT_NEAR(l2_error(x, a, b), std::sqrt(integral), 1e-12);
    EXPECT_NEAR(l2_error(x, b, a), std::sqrt(integral), 1e-12);
    EXPECT_NEAR(l2_error(x, a, a), 0.0, 1e-15);
}

// Un campo con un numero di valori diverso da quello dei nodi: eccezione.
TEST(Fem1d, L2ErrorWrongInputsThrow) {
    std::vector<double> x = uniform_mesh(1.0, 4);   // 5 nodi
    DVector ok(5, 0.0);
    DVector too_short(4, 0.0);

    EXPECT_THROW(l2_error(x, too_short, ok), std::invalid_argument);
    EXPECT_THROW(l2_error(x, ok, too_short), std::invalid_argument);
    EXPECT_THROW(l2_error(x, too_short, too_short), std::invalid_argument);   // entrambi sbagliati
}

// Se l'errore va come C h^p, quando h si divide per "ratio" l'errore si divide per ratio^p.
// Esempi a mano:
//   0.04 -> 0.01 con ratio 2:  4 = 2^p, quindi p = 2
//   0.08 -> 0.04 con ratio 2:  2 = 2^p, quindi p = 1
//   0.09 -> 0.01 con ratio 3:  9 = 3^p, quindi p = 2
//   errore che non cambia:     p = 0 (raffinare non serve a niente)
//   errore che CRESCE (0.01 -> 0.04): p = -2 (il calcolo fine e' peggiore)
TEST(Fem1d, ObservedOrderMatchesHandComputation) {
    EXPECT_NEAR(observed_order(0.04, 0.01, 2.0), 2.0, 1e-12);
    EXPECT_NEAR(observed_order(0.08, 0.04, 2.0), 1.0, 1e-12);
    EXPECT_NEAR(observed_order(0.09, 0.01, 3.0), 2.0, 1e-12);
    EXPECT_NEAR(observed_order(0.05, 0.05, 2.0), 0.0, 1e-12);
    EXPECT_NEAR(observed_order(0.01, 0.04, 2.0), -2.0, 1e-12);
}

// Il logaritmo di un errore nullo o negativo non esiste, e "ratio" deve essere > 1
// (e' il rapporto fra il passo grosso e quello fine).
TEST(Fem1d, ObservedOrderWrongInputsThrow) {
    EXPECT_THROW(observed_order(0.0, 0.01, 2.0), std::invalid_argument);     // errore grosso nullo
    EXPECT_THROW(observed_order(0.04, 0.0, 2.0), std::invalid_argument);     // errore fine nullo
    EXPECT_THROW(observed_order(0.04, -0.01, 2.0), std::invalid_argument);   // errore negativo
    EXPECT_THROW(observed_order(0.04, 0.01, 1.0), std::invalid_argument);    // ratio = 1
    EXPECT_THROW(observed_order(0.04, 0.01, 0.5), std::invalid_argument);    // ratio < 1
}

// ---- Problema di verifica: il seno che decade ---------------------------------------

// Il risultato di decaying_sine_error deve coincidere con l'errore L2 fra
//  - la soluzione ESATTA DELLO SCHEMA, g^n sin(pi x_i) (vedi amplification_factor), e
//  - la soluzione esatta del problema, exp(-pi^2 t) sin(pi x_i).
// Cosi' si controllano insieme T0, k, rho_c, L, il tempo finale t = n_steps * dt, theta e l'esponente.
TEST(Fem1d, DecayingSineErrorMatchesTheDiscreteSolution) {
    const double pi = std::acos(-1.0);
    const size_t N = 20, n_steps = 10;
    const double dt = 0.01, t = n_steps * dt;
    std::vector<double> x = uniform_mesh(1.0, N);

    for (double theta : {1.0, 0.5}) {
        const double g = amplification_factor(1.0 / N, 1.0, 1.0, 1.0, dt, theta);
        DVector T_scheme(N + 1), T_exact(N + 1);
        for (size_t i = 0; i <= N; ++i) {
            T_scheme[i] = std::pow(g, n_steps) * std::sin(pi * x[i]);
            T_exact[i] = std::exp(-pi * pi * t) * std::sin(pi * x[i]);
        }
        EXPECT_NEAR(decaying_sine_error(N, dt, n_steps, theta), l2_error(x, T_scheme, T_exact), 1e-9)
            << "theta = " << theta;
    }
}

// Al tempo 0 la temperatura iniziale e' gia' la soluzione esatta: errore nullo.
TEST(Fem1d, DecayingSineErrorIsZeroAtTimeZero) {
    EXPECT_NEAR(decaying_sine_error(10, 0.01, 0, 1.0), 0.0, 1e-14);
}

// Nessun elemento: eccezione. Gli altri due casi li controlla gia' solve_transient.
TEST(Fem1d, DecayingSineErrorWrongInputsThrow) {
    EXPECT_THROW(decaying_sine_error(0, 0.01, 1, 1.0), std::invalid_argument);    // 0 elementi
    EXPECT_THROW(decaying_sine_error(10, 0.0, 1, 1.0), std::invalid_argument);    // dt = 0
    EXPECT_THROW(decaying_sine_error(10, 0.01, 1, 2.0), std::invalid_argument);   // theta > 1
}

// ---- Ordini di convergenza -----------------------------------------------------------
// Questi tre test non cercano errori nuovi (quelli li hanno gia' presi i test precedenti):
// documentano che il solutore ha l'ordine di accuratezza previsto dalla teoria.

// SPAZIO. Crank-Nicolson con passo temporale piccolissimo (l'errore nel tempo e' trascurabile),
// fino a t = 0.1; la mesh passa da 10 a 160 elementi, dimezzando h ogni volta. Attesi: ordine 2.
TEST(Fem1d, SpatialConvergenceIsSecondOrder) {
    const double dt = 1e-4;
    const size_t n_steps = 1000;   // t = 0.1
    double previous = decaying_sine_error(10, dt, n_steps, 0.5);
    for (size_t N : {20, 40, 80, 160}) {
        const double current = decaying_sine_error(N, dt, n_steps, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "N = " << N / 2 << " -> " << N;
        previous = current;
    }
}

// TEMPO. Mesh molto fine, N = 400: l'errore di mesh (circa 1e-6) e' molto piu' piccolo di quello
// temporale (almeno 5e-5), quindi si misura solo l'errore nel tempo. Fino a t = 0.1 con 5, 10, 20 passi
// (dt = 0.02, 0.01, 0.005), dimezzando dt ogni volta.
// Eulero all'indietro (theta = 1): ordine 1.
TEST(Fem1d, TemporalConvergenceOfBackwardEulerIsFirstOrder) {
    const size_t N = 400;
    double previous = decaying_sine_error(N, 0.1 / 5, 5, 1.0);
    for (size_t steps : {10, 20}) {
        const double current = decaying_sine_error(N, 0.1 / steps, steps, 1.0);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 1.0, 0.1) << steps / 2 << " -> " << steps << " passi";
        previous = current;
    }
}

// Crank-Nicolson (theta = 0.5): ordine 2.
TEST(Fem1d, TemporalConvergenceOfCrankNicolsonIsSecondOrder) {
    const size_t N = 400;
    double previous = decaying_sine_error(N, 0.1 / 5, 5, 0.5);
    for (size_t steps : {10, 20}) {
        const double current = decaying_sine_error(N, 0.1 / steps, steps, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.1) << steps / 2 << " -> " << steps << " passi";
        previous = current;
    }
}

// ---- Se CG non converge, i solutori lo dicono ----------------------------------------

// Un NaN nei dati (sorgente o temperatura imposta) arriva fino al termine noto del sistema, e CG
// non puo' convergere. Prima questi casi restituivano una temperatura qualsiasi (zeri) senza nessun
// errore. Ora devono lanciare std::runtime_error: gli argomenti erano accettabili, e' il calcolo
// che e' fallito (invalid_argument e' per gli argomenti sbagliati, che si vedono prima di calcolare).
TEST(Fem1d, SteadyThrowsWhenTheLinearSolverCannotConverge) {
    const double not_a_number = std::nan("");
    std::vector<double> x = uniform_mesh(1.0, 10);

    EXPECT_THROW(solve_steady(x, 1.0, not_a_number, 0.0, 0.0), std::runtime_error);   // sorgente NaN
    EXPECT_THROW(solve_steady(x, 1.0, 5.0, not_a_number, 0.0), std::runtime_error);   // estremo sinistro NaN
    EXPECT_THROW(solve_steady(x, 1.0, 5.0, 0.0, not_a_number), std::runtime_error);   // estremo destro NaN
}

// Lo stesso per l'evoluzione nel tempo. Basta UN passo con un NaN, in uno qualunque dei dati, e deve
// lanciare: con n_steps = 1 il test controlla anche che il primo passo venga verificato.
TEST(Fem1d, TransientThrowsWhenTheLinearSolverCannotConverge) {
    const double not_a_number = std::nan("");
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T0(11, 1.0);
    DVector T0_with_nan = T0;
    T0_with_nan[4] = not_a_number;

    // argomenti: x, rho_c, k, q, T_left, T_right, T0, dt, n_steps, theta
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, not_a_number, 0.0, 0.0, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, not_a_number, 0.0, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, not_a_number, T0, 0.01, 1, 1.0), std::runtime_error);
    EXPECT_THROW(solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0_with_nan, 0.01, 1, 1.0), std::runtime_error);
}

// Il caso realistico: uno schema instabile. Con theta < 0.5 e dt sopra il limite di stabilita' le
// temperature crescono a ogni passo, fino a uscire dalla scala dei double. Prima, in diversi di questi
// calcoli, la funzione restituiva un vettore di NaN e inf come se niente fosse. Ora, per ogni
// combinazione, succede una delle due cose: lancia std::runtime_error, oppure restituisce solo numeri
// finiti. Il test NON dice che il risultato sia giusto (un calcolo instabile puo' dare numeri enormi
// ma finiti): dice che NaN e inf non passano mai in silenzio.
TEST(Fem1d, TransientNeverReturnsNonFiniteNumbersInSilence) {
    const double pi = std::acos(-1.0);
    for (double theta : {0.0, 0.3}) {
        for (size_t N : {10u, 20u}) {
            for (double dt : {0.1, 1.0, 5.0}) {
                std::vector<double> x = uniform_mesh(1.0, N);
                DVector T0(N + 1);
                for (size_t i = 0; i <= N; ++i) {
                    T0[i] = std::sin(pi * x[i]);
                }
                try {
                    DVector T = solve_transient(x, 1.0, 1.0, 0.0, 0.0, 0.0, T0, dt, 400, theta);
                    bool all_finite = true;
                    for (double v : T) {
                        all_finite = all_finite && std::isfinite(v);
                    }
                    EXPECT_TRUE(all_finite) << "theta = " << theta << ", N = " << N << ", dt = " << dt;
                } catch (const std::runtime_error&) {
                    // va bene: il calcolo e' fallito e lo dice
                }
            }
        }
    }
}
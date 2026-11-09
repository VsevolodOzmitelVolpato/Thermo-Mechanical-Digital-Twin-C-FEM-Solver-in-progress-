#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Thermoelastic1D.h"

// Funzione di appoggio: confronta due vettori numero per numero e, se qualcosa non torna,
// dice quale elemento e' sbagliato.
static void expect_vector_near(const DVector& a, const DVector& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "indice " << i;
    }
}

// ---- Forze nodali equivalenti alla dilatazione termica --------------------------------

// Un solo elemento, lungo 2. E = 3, alpha = 5. La temperatura e' 1 grado sopra T_ref = 0 in
// entrambi i nodi, quindi dT = 1 e la forza vale E alpha dT = 15.
// Il calore spinge il nodo sinistro verso sinistra (-15) e il destro verso destra (+15).
TEST(Thermoelastic1d, ThermalLoadOfOneElement) {
    std::vector<double> x = {0.0, 2.0};
    DVector f = assemble_thermal_load(x, 3.0, 5.0, {1.0, 1.0}, 0.0);

    expect_vector_near(f, {-15.0, 15.0}, 1e-12);
}

// Due elementi lunghi 1 e 2 (nodi 0, 1, 3). E = 2 e alpha = 0.5, quindi E alpha = 1.
// T_ref = 10 e T = {10, 20, 40}: gli scarti da T_ref sono {0, 10, 30}.
//   elemento 0: scarto medio (0 + 10) / 2 = 5   -> forza 5:  nodo 0 riceve -5,  nodo 1 riceve +5
//   elemento 1: scarto medio (10 + 30) / 2 = 20 -> forza 20: nodo 1 riceve -20, nodo 2 riceve +20
//   il nodo 1 e' in comune: +5 - 20 = -15
// Gli elementi hanno lunghezze diverse ma nella formula non compare nessuna lunghezza.
TEST(Thermoelastic1d, ThermalLoadMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector f = assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0}, 10.0);

    expect_vector_near(f, {-5.0, -15.0, 20.0}, 1e-12);
}

// Se la temperatura e' ovunque uguale a T_ref la barra e' a riposo: nessuna forza.
// (T_ref e' diverso da 0 apposta: chi dimenticasse di sottrarlo troverebbe delle forze.)
TEST(Thermoelastic1d, NoTemperatureChangeMeansNoForce) {
    std::vector<double> x = {0.0, 1.0, 3.0, 4.0};
    DVector f = assemble_thermal_load(x, 200.0, 12e-6, DVector(4, 300.0), 300.0);

    expect_vector_near(f, DVector(4, 0.0), 1e-12);
}

// Il calore non spinge la barra dall'esterno: le forze sono tutte interne e la loro somma e' 0.
// Prova con una temperatura qualunque su una maglia irregolare (e si controlla anche che le
// forze ci siano: una somma di zeri non dimostrerebbe niente).
TEST(Thermoelastic1d, ThermalLoadsSumToZero) {
    std::vector<double> x = {0.0, 0.1, 0.5, 0.6, 1.5, 2.0};
    DVector T = {20.0, 35.0, 18.0, 90.0, 60.0, 25.0};
    DVector f = assemble_thermal_load(x, 70.0, 2.3e-5, T, 20.0);

    ASSERT_EQ(f.size(), 6u);
    double sum = 0.0;
    for (double fi : f) {
        sum += fi;
    }
    EXPECT_NEAR(sum, 0.0, 1e-12);
    EXPECT_GT(std::fabs(f[0]), 1e-3);
    EXPECT_GT(std::fabs(f[5]), 1e-3);
}

// Input sbagliati: T con un numero di valori diverso da quello dei nodi.
TEST(Thermoelastic1d, ThermalLoadWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};   // 3 nodi

    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0}, 10.0), std::invalid_argument);               // 2 valori
    EXPECT_THROW(assemble_thermal_load(x, 2.0, 0.5, {10.0, 20.0, 40.0, 50.0}, 10.0), std::invalid_argument);   // 4 valori
}

// ---- Spostamento della barra ---------------------------------------------------------

// Lo stesso esempio a mano di prima: x = {0, 1, 3}, E = 2, alpha = 0.5, T_ref = 10, T = {10, 20, 40}.
// Le forze sono f = {-5, -15, 20} e la matrice di rigidezza (E/h per ogni elemento) e'
//     K = [[ 2, -2,  0],
//          [-2,  3, -1],
//          [ 0, -1,  1]]
// L'estremo sinistro e' bloccato (u0 = 0). Restano due equazioni, le righe 1 e 2 di K u = f:
//      3 u1 - u2 = -15
//       -u1 + u2 =  20
// Sommandole: 2 u1 = 5, quindi u1 = 2.5 e u2 = 22.5.
// Controllo: ogni elemento si allunga di alpha dT_medio h, cioe' 0.5*5*1 = 2.5 e 0.5*20*2 = 20.
// Se anche l'estremo destro e' bloccato (u2 = 0) resta una sola equazione: 3 u1 = -15, u1 = -5.
TEST(Thermoelastic1d, ThermalExpansionMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    DVector u_free = solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Free);
    expect_vector_near(u_free, {0.0, 2.5, 22.5}, 1e-9);

    DVector u_fixed = solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Fixed);
    expect_vector_near(u_fixed, {0.0, -5.0, 0.0}, 1e-9);
}

// Barra lunga 1 m (10 elementi), scaldata in modo uniforme da 20 a 120 gradi, estremo destro libero.
// Libera di muoversi, la barra si allunga di alpha dT L = 12e-6 * 100 * 1 = 1.2e-3 m (1.2 mm),
// e il nodo a distanza x si e' spostato di alpha dT x. Il risultato non dipende da E (niente
// ferma la barra, quindi niente forze): si prova con due moduli diversi, acciaio e alluminio.
TEST(Thermoelastic1d, FreeBarElongatesByAlphaDeltaTTimesLength) {
    const double alpha = 12e-6, T_ref = 20.0, T_hot = 120.0;
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), T_hot);

    for (double E : {200e9, 70e9}) {
        DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
        ASSERT_EQ(u.size(), x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            EXPECT_NEAR(u[i], alpha * (T_hot - T_ref) * x[i], 1e-12) << "E = " << E << ", nodo " << i;
        }
        EXPECT_NEAR(u.back(), 1.2e-3, 1e-12) << "E = " << E;
    }
}

// La stessa barra, ma bloccata anche a destra: le pareti la tengono ferma, quindi u = 0 in
// ogni nodo. (La tensione, invece, non e' 0: la calcoliamo al passo 4c.)
TEST(Thermoelastic1d, ClampedBarWithUniformHeatingDoesNotMove) {
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), 120.0);

    DVector u = solve_thermal_expansion(x, 200e9, 12e-6, T, 20.0, RightEnd::Fixed);

    expect_vector_near(u, DVector(x.size(), 0.0), 1e-12);
}

// Temperatura lineare T(x) = 30 + 15 x su una barra lunga 2, con maglia NON uniforme
// (x_i = L (i/N)^1.5). T_ref = 10, quindi dT(x) = 20 + 15 x. E = 3, alpha = 0.02.
//  - Estremo destro libero: u(x) = alpha * integrale di dT = 0.02 (20 x + 7.5 x^2) = 0.4 x + 0.15 x^2.
//  - Estremo destro bloccato: la tensione e' la stessa ovunque e u(L) = 0, e viene
//    u(x) = 0.15 x (x - 2).
// Con T lineare la regola del trapezio su ogni elemento e' esatta, quindi i valori nei nodi sono
// esatti su qualunque maglia: la tolleranza e' stretta e un test rosso vuol dire un errore nel codice.
TEST(Thermoelastic1d, LinearTemperatureOnAGradedMeshIsExactAtTheNodes) {
    const size_t N = 30;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
        T[i] = 30.0 + 15.0 * x[i];
    }

    DVector u_free = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Free);
    DVector u_fixed = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Fixed);

    ASSERT_EQ(u_free.size(), N + 1);
    ASSERT_EQ(u_fixed.size(), N + 1);
    for (size_t i = 0; i <= N; ++i) {
        EXPECT_NEAR(u_free[i], 0.4 * x[i] + 0.15 * x[i] * x[i], 1e-9) << "libero, nodo " << i;
        EXPECT_NEAR(u_fixed[i], 0.15 * x[i] * (x[i] - 2.0), 1e-9) << "bloccato, nodo " << i;
    }
}

// Barra libera, temperatura molto irregolare, 200 elementi di lunghezza diversa.
// Con l'estremo destro libero niente ferma la barra e ogni elemento si dilata per conto suo:
//     u[i] - u[i-1] = alpha * dT_medio * h      (per QUALUNQUE temperatura e QUALUNQUE maglia).
// Quindi u si ricostruisce sommando gli allungamenti elemento per elemento, senza risolvere nulla.
// Il test serve anche a controllare che CG sia stato fatto andare fino in fondo: con una
// temperatura cosi' irregolare il residuo scende piano piano, e fermarsi troppo presto
// (tolleranza troppo larga) da' un errore grande.
TEST(Thermoelastic1d, FreeBarOnARoughTemperatureExpandsElementByElement) {
    const size_t N = 200;
    const double L = 2.0, E = 7.0e10, alpha = 1.5e-5, T_ref = 300.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.3);
        T[i] = 300.0 + 80.0 * std::sin(1.7 * i) + 30.0 * std::cos(0.37 * i * i);
    }

    DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);

    ASSERT_EQ(u.size(), N + 1);
    EXPECT_NEAR(u[0], 0.0, 1e-15);
    double expected = 0.0;   // spostamento del nodo i = somma degli allungamenti dei primi i elementi
    for (size_t i = 1; i <= N; ++i) {
        expected += alpha * (x[i] - x[i - 1]) * ((T[i - 1] - T_ref) + (T[i] - T_ref)) / 2.0;
        EXPECT_NEAR(u[i], expected, 1e-12) << "nodo " << i;
    }
}

// Errore massimo sui nodi per T(x) = sin(pi x) su una barra lunga 1, con T_ref = 0, E = 3,
// alpha = 2 e l'estremo destro libero. Soluzione esatta: u(x) = alpha (1 - cos(pi x)) / pi.
static double sine_expansion_error(size_t N) {
    const double pi = std::acos(-1.0);
    std::vector<double> x = uniform_mesh(1.0, N);
    DVector T(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        T[i] = std::sin(pi * x[i]);
    }

    DVector u = solve_thermal_expansion(x, 3.0, 2.0, T, 0.0, RightEnd::Free);

    double max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double err = std::fabs(u[i] - 2.0 * (1.0 - std::cos(pi * x[i])) / pi);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Qui T non e' lineare, quindi i valori nei nodi non sono piu' esatti. Gli elementi lineari sono
// del secondo ordine: dimezzando h l'errore deve diventare circa 4 volte piu' piccolo.
TEST(Thermoelastic1d, ThermalExpansionIsSecondOrderForASineTemperature) {
    const double e10 = sine_expansion_error(10);
    const double e20 = sine_expansion_error(20);
    const double e40 = sine_expansion_error(40);

    EXPECT_GT(e10, 1e-3);   // l'errore c'e' (non e' zero per caso)
    EXPECT_NEAR(observed_order(e10, e20, 2.0), 2.0, 0.1);
    EXPECT_NEAR(observed_order(e20, e40, 2.0), 2.0, 0.1);
}

// Input sbagliati: devono lanciare std::invalid_argument.
TEST(Thermoelastic1d, ThermalExpansionWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    // T con 2 valori invece di 3
    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, DVector(2, 10.0), 10.0, RightEnd::Free), std::invalid_argument);
    // una barra con un solo nodo non ha elementi
    EXPECT_THROW(solve_thermal_expansion({0.0}, 2.0, 0.5, {10.0}, 10.0, RightEnd::Free), std::invalid_argument);
    // E deve essere positivo: 0, negativo e NaN non vanno bene
    EXPECT_THROW(solve_thermal_expansion(x, 0.0, 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
    EXPECT_THROW(solve_thermal_expansion(x, -2.0, 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
    EXPECT_THROW(solve_thermal_expansion(x, std::nan(""), 0.5, T, 10.0, RightEnd::Free), std::invalid_argument);
}

// Un NaN nella temperatura: il sistema non ha soluzione e CG non puo' convergere.
// Deve dirlo con un'eccezione (std::runtime_error), non restituire spostamenti senza senso.
TEST(Thermoelastic1d, ThermalExpansionThrowsWhenTheLinearSolverCannotConverge) {
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), 100.0);
    T[4] = std::nan("");

    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Free), std::runtime_error);
    EXPECT_THROW(solve_thermal_expansion(x, 2.0, 0.5, T, 10.0, RightEnd::Fixed), std::runtime_error);
}

// ---- Tensione in ogni elemento ---------------------------------------------------------

// Lo stesso esempio a mano di sempre: x = {0, 1, 3}, E = 2, alpha = 0.5, T_ref = 10, T = {10, 20, 40}.
// Lunghezze degli elementi: {1, 2}. Scarti medi di temperatura: {5, 20}.
// Deformazione dovuta al solo calore (alpha per lo scarto medio): {2.5, 10}.
//     sigma = E * ( (u[e+1] - u[e]) / h  -  alpha dT_medio )
//
// a) barra libera, u = {0, 2.5, 22.5}: deformazione totale {2.5, 10}, uguale a quella del calore.
//    La differenza e' 0, quindi sigma = {0, 0}: una barra libera di dilatarsi non ha tensioni.
// b) barra bloccata, u = {0, -5, 0}: deformazione totale {-5, 2.5}, quindi
//    sigma = 2 * (-5 - 2.5) = -15 e sigma = 2 * (2.5 - 10) = -15. Stessa tensione in tutta la barra,
//    ed e' negativa: compressione.
// c) barra che non si muove (u = 0): sigma = 2 * (0 - 2.5) = -5 e sigma = 2 * (0 - 10) = -20.
// d) niente calore (T = T_ref) e u = {0, 1, 4}: deformazioni 1 e 1.5, quindi sigma = {2, 3}.
//    Qui conta solo la parte meccanica.
TEST(Thermoelastic1d, ElementStressMatchesHandComputation) {
    std::vector<double> x = {0.0, 1.0, 3.0};
    DVector T = {10.0, 20.0, 40.0};

    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5, 22.5}), {0.0, 0.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, -5.0, 0.0}), {-15.0, -15.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 0.0, 0.0}), {-5.0, -20.0}, 1e-12);
    expect_vector_near(element_stress(x, 2.0, 0.5, DVector(3, 10.0), 10.0, {0.0, 1.0, 4.0}), {2.0, 3.0}, 1e-12);
}

// Barra d'acciaio lunga 1 m (10 elementi), scaldata da 20 a 120 gradi. E = 200e9 Pa, alpha = 12e-6.
// Bloccata a entrambi i capi non puo' dilatarsi: la parete la schiaccia di quanto il calore la
// vorrebbe allungare, quindi la deformazione meccanica e' -alpha dT e in ogni elemento
//     sigma = -E alpha dT = -200e9 * 12e-6 * 100 = -2.4e8 Pa   (240 MPa di compressione).
// E' dello stesso ordine della tensione di snervamento di un acciaio da costruzione (circa 250 MPa):
// per questo ponti e rotaie hanno i giunti di dilatazione.
// Con l'estremo destro libero la barra si allunga senza fatica e la tensione e' 0.
TEST(Thermoelastic1d, ClampedSteelBarIsCompressedByEAlphaDeltaT) {
    const double E = 200e9, alpha = 12e-6, T_ref = 20.0, T_hot = 120.0;
    const double scale = E * alpha * (T_hot - T_ref);   // 2.4e8 Pa
    std::vector<double> x = uniform_mesh(1.0, 10);
    DVector T(x.size(), T_hot);

    DVector u_fixed = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Fixed);
    DVector s_fixed = element_stress(x, E, alpha, T, T_ref, u_fixed);
    ASSERT_EQ(s_fixed.size(), 10u);
    for (size_t e = 0; e < s_fixed.size(); ++e) {
        EXPECT_NEAR(s_fixed[e], -scale, 1e-8 * scale) << "bloccata, elemento " << e;
    }

    DVector u_free = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
    DVector s_free = element_stress(x, E, alpha, T, T_ref, u_free);
    ASSERT_EQ(s_free.size(), 10u);
    for (size_t e = 0; e < s_free.size(); ++e) {
        EXPECT_NEAR(s_free[e], 0.0, 1e-8 * scale) << "libera, elemento " << e;
    }
}

// Barra libera, con la stessa temperatura irregolare e la stessa maglia (200 elementi di lunghezza
// diversa) del test FreeBarOnARoughTemperatureExpandsElementByElement. Ogni elemento si dilata per
// conto suo, quindi la tensione e' 0 ovunque, per QUALUNQUE temperatura e QUALUNQUE maglia.
// "Circa 0" qui vuol dire meno di 1e-8 volte E alpha 100, cioe' della tensione che avrebbe la stessa
// barra se fosse bloccata e scaldata di 100 gradi.
TEST(Thermoelastic1d, FreeBarHasNoStressOnARoughTemperature) {
    const size_t N = 200;
    const double L = 2.0, E = 7.0e10, alpha = 1.5e-5, T_ref = 300.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.3);
        T[i] = 300.0 + 80.0 * std::sin(1.7 * i) + 30.0 * std::cos(0.37 * i * i);
    }

    DVector u = solve_thermal_expansion(x, E, alpha, T, T_ref, RightEnd::Free);
    DVector sigma = element_stress(x, E, alpha, T, T_ref, u);

    ASSERT_EQ(sigma.size(), N);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(sigma[e], 0.0, 1e-8 * E * alpha * 100.0) << "elemento " << e;
    }
}

// Temperatura lineare T(x) = 30 + 15 x su una barra lunga 2, con la stessa maglia NON uniforme del
// test LinearTemperatureOnAGradedMeshIsExactAtTheNodes, ma con l'estremo destro bloccato.
// T_ref = 10, E = 3, alpha = 0.02, quindi lo scarto e' 20 + 15 x: il calore vorrebbe dilatare ogni
// pezzetto in modo diverso. Eppure la tensione e' la stessa in ogni elemento, perche' le forze
// devono fare equilibrio lungo la barra. Vale -E alpha per lo scarto medio sulla barra:
//     -3 * 0.02 * (20 + 15 * 1) = -2.1.
TEST(Thermoelastic1d, ClampedBarStressIsUniformForALinearTemperatureOnAGradedMesh) {
    const size_t N = 30;
    const double L = 2.0;
    std::vector<double> x(N + 1);
    DVector T(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        x[i] = L * std::pow(static_cast<double>(i) / N, 1.5);
        T[i] = 30.0 + 15.0 * x[i];
    }

    DVector u = solve_thermal_expansion(x, 3.0, 0.02, T, 10.0, RightEnd::Fixed);
    DVector sigma = element_stress(x, 3.0, 0.02, T, 10.0, u);

    ASSERT_EQ(sigma.size(), N);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(sigma[e], -2.1, 1e-8) << "elemento " << e;
    }
}

// Errore massimo sulla tensione per T(x) = sin(pi x) su una barra lunga 1, con T_ref = 0, E = 3,
// alpha = 2 e l'estremo destro bloccato. Tensione esatta: -E alpha per la media di sin(pi x) su
// [0, 1], cioe' -3 * 2 * (2 / pi) = -12 / pi = -3.82.
static double sine_stress_error(size_t N) {
    const double pi = std::acos(-1.0);
    std::vector<double> x = uniform_mesh(1.0, N);
    DVector T(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        T[i] = std::sin(pi * x[i]);
    }

    DVector u = solve_thermal_expansion(x, 3.0, 2.0, T, 0.0, RightEnd::Fixed);
    DVector sigma = element_stress(x, 3.0, 2.0, T, 0.0, u);

    EXPECT_EQ(sigma.size(), N);
    double max_err = 0.0;
    for (double s : sigma) {
        const double err = std::fabs(s + 12.0 / pi);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// T non e' lineare, quindi la tensione calcolata non e' esatta: la media di T sulla barra viene
// fatta con la regola del trapezio, che sbaglia di circa pi / (6 N^2). L'errore sulla tensione e'
// circa pi / N^2: dimezzando h diventa 4 volte piu' piccolo (secondo ordine).
TEST(Thermoelastic1d, ClampedBarStressIsSecondOrderForASineTemperature) {
    const double e10 = sine_stress_error(10);
    const double e20 = sine_stress_error(20);
    const double e40 = sine_stress_error(40);

    EXPECT_GT(e10, 1e-2);   // l'errore c'e' (circa 0.031): non e' zero per caso
    EXPECT_NEAR(observed_order(e10, e20, 2.0), 2.0, 0.1);
    EXPECT_NEAR(observed_order(e20, e40, 2.0), 2.0, 0.1);
}

// Una tensione per ogni elemento: con n nodi gli elementi sono n - 1.
// Un solo elemento (lungo 2, E = 3, alpha = 5, scarto di temperatura 1) tenuto fermo (u = 0):
// sigma = 3 * (0 - 5 * 1) = -15. Con meno di 2 nodi non ci sono elementi: il risultato e' un
// vettore vuoto, senza errori e senza crash.
TEST(Thermoelastic1d, ElementStressHasOneValuePerElement) {
    DVector one = element_stress({0.0, 2.0}, 3.0, 5.0, {1.0, 1.0}, 0.0, {0.0, 0.0});
    ASSERT_EQ(one.size(), 1u);
    EXPECT_NEAR(one[0], -15.0, 1e-12);

    EXPECT_TRUE(element_stress({0.0}, 3.0, 5.0, {1.0}, 0.0, {0.0}).empty());
    EXPECT_TRUE(element_stress({}, 3.0, 5.0, {}, 0.0, {}).empty());
}

// Input sbagliati: T oppure u con un numero di valori diverso da quello dei nodi.
TEST(Thermoelastic1d, ElementStressWrongInputsThrow) {
    std::vector<double> x = {0.0, 1.0, 3.0};   // 3 nodi
    DVector T = {10.0, 20.0, 40.0};
    DVector u = {0.0, 2.5, 22.5};

    EXPECT_THROW(element_stress(x, 2.0, 0.5, DVector(2, 10.0), 10.0, u), std::invalid_argument);          // T con 2 valori
    EXPECT_THROW(element_stress(x, 2.0, 0.5, DVector(4, 10.0), 10.0, u), std::invalid_argument);          // T con 4 valori
    EXPECT_THROW(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5}), std::invalid_argument);                // u con 2 valori
    EXPECT_THROW(element_stress(x, 2.0, 0.5, T, 10.0, {0.0, 2.5, 22.5, 30.0}), std::invalid_argument);    // u con 4 valori
}

// ---- La catena completa: calore -> dilatazione -> tensione -----------------------------

// Esempio a mano, tutto in un colpo. Barra lunga 2 con 2 elementi (x = {0, 1, 2}, h = 1).
//
// CALORE: un passo di Eulero all'indietro (theta = 1), dt = 0.5, rho_c = 3, k = 1.5, q = 2.
// Estremi fissati a 6 e 14, temperatura iniziale T0 = {6, 6, 14}. Le matrici sono
//     M = rho_c/6 * [[2,1,0],[1,4,1],[0,1,2]] = [[1, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 1]]
//     K = k * [[1,-1,0],[-1,2,-1],[0,-1,1]]   = [[1.5, -1.5, 0], [-1.5, 3, -1.5], [0, -1.5, 1.5]]
// e il calore della sorgente e' F = q h/2 * (1, 2, 1) = (1, 2, 1). Il sistema (M + dt K) T = M T0 + dt F
// ha la riga centrale   -0.25 T_0 + 3.5 T_1 - 0.25 T_2 = 0.5*6 + 2*6 + 0.5*14 + 0.5*2 = 23.
// Con T_0 = 6 e T_2 = 14 (estremi fissati) resta 3.5 T_1 = 23 + 1.5 + 3.5 = 28, quindi T_1 = 8.
//     T = {6, 8, 14}
//
// MECCANICA: T_ref = 5, E = 4, alpha = 0.25. Scarti dT = {1, 3, 9}, scarti medi {2, 6}.
//  - estremo destro libero: ogni elemento si allunga di alpha dT_medio h = 0.5 e 1.5,
//    quindi u = {0, 0.5, 2}, e la tensione e' 0.
//  - estremo destro bloccato: le forze sono E alpha dT_medio = {2, 6}, quindi f = {-2, 2-6, 6}.
//    Con u_0 = u_2 = 0 l'equazione del nodo 1 e' E * 2 u_1 = -4, cioe' u_1 = -0.5.
//    Deformazioni totali -0.5 e +0.5, termiche 0.5 e 1.5: sigma = 4 (-0.5-0.5) = -4 e 4 (0.5-1.5) = -4.
static ThermoMechanicalProblem hand_problem() {
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(2.0, 2);
    p.rho_c = 3.0;
    p.k = 1.5;
    p.q = 2.0;
    p.T_left = 6.0;
    p.T_right = 14.0;
    p.T0 = {6.0, 6.0, 14.0};
    p.dt = 0.5;
    p.n_steps = 1;
    p.theta = 1.0;
    p.E = 4.0;
    p.alpha = 0.25;
    p.T_ref = 5.0;
    return p;
}

TEST(Thermoelastic1d, ThermoMechanicalHandComputation) {
    ThermoMechanicalProblem p = hand_problem();

    p.right_end = RightEnd::Free;
    ThermoMechanicalResult free_bar = solve_thermomechanical(p);
    expect_vector_near(free_bar.T, {6.0, 8.0, 14.0}, 1e-9);
    expect_vector_near(free_bar.u, {0.0, 0.5, 2.0}, 1e-9);
    expect_vector_near(free_bar.sigma, {0.0, 0.0}, 1e-9);

    p.right_end = RightEnd::Fixed;
    ThermoMechanicalResult clamped_bar = solve_thermomechanical(p);
    expect_vector_near(clamped_bar.T, {6.0, 8.0, 14.0}, 1e-9);
    expect_vector_near(clamped_bar.u, {0.0, -0.5, 0.0}, 1e-9);
    expect_vector_near(clamped_bar.sigma, {-4.0, -4.0}, 1e-9);
}

// La catena deve dare esattamente quello che danno i tre passi fatti a mano, uno dopo l'altro.
// Qui ogni dato ha un valore diverso e "scomodo" (nessuno uguale a un altro), la maglia non e'
// uniforme e theta = 0.6: se la funzione scambia due argomenti dello stesso tipo (rho_c e k,
// T_left e T_right, E e alpha...) oppure passa la temperatura iniziale al posto di quella
// finale, i numeri cambiano e il test se ne accorge.
TEST(Thermoelastic1d, ThermoMechanicalEqualsTheThreeStepsOneByOne) {
    const size_t N = 12;
    ThermoMechanicalProblem p;
    p.x.resize(N + 1);
    p.T0.resize(N + 1);
    for (size_t i = 0; i <= N; ++i) {
        p.x[i] = 1.7 * std::pow(static_cast<double>(i) / N, 1.2);
        p.T0[i] = 20.0 + 3.0 * std::sin(2.0 * i);
    }
    p.rho_c = 1.7;
    p.k = 2.3;
    p.q = 4.1;
    p.T_left = 12.5;
    p.T_right = 31.5;
    p.T0.front() = p.T_left;
    p.T0.back() = p.T_right;
    p.dt = 0.07;
    p.n_steps = 9;
    p.theta = 0.6;
    p.E = 5.3;
    p.alpha = 0.013;
    p.T_ref = 9.1;

    for (RightEnd end : {RightEnd::Free, RightEnd::Fixed}) {
        p.right_end = end;
        ThermoMechanicalResult r = solve_thermomechanical(p);

        DVector T = solve_transient(p.x, p.rho_c, p.k, p.q, p.T_left, p.T_right, p.T0, p.dt, p.n_steps, p.theta);
        DVector u = solve_thermal_expansion(p.x, p.E, p.alpha, T, p.T_ref, end);
        DVector sigma = element_stress(p.x, p.E, p.alpha, T, p.T_ref, u);

        expect_vector_near(r.T, T, 1e-12);
        expect_vector_near(r.u, u, 1e-12);
        expect_vector_near(r.sigma, sigma, 1e-12);
    }
}

// Il problema "seno che decade", gia' usato per la temperatura: barra lunga 1, rho_c = 1, k = 1,
// nessuna sorgente, estremi a 0, T0 = sin(pi x), quindi T(x, t) = exp(-pi^2 t) sin(pi x).
// Con T_ref = 0, E = 3 e alpha = 2 la catena ha due risultati esatti, che si ottengono integrando
// T lungo la barra (l'integrale di sin(pi x) tra 0 e 1 vale 2/pi):
//  - estremo destro libero: la punta si sposta di alpha * integrale = 2 alpha exp(-pi^2 t) / pi;
//  - estremo destro bloccato: la tensione e' la stessa ovunque e vale
//    -E alpha * (media di T) = -2 E alpha exp(-pi^2 t) / pi.
static ThermoMechanicalProblem decaying_sine_problem(size_t N, double dt, size_t n_steps, double theta,
                                                     RightEnd right_end) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(1.0, N);
    p.T0.resize(p.x.size());
    for (size_t i = 0; i < p.x.size(); ++i) {
        p.T0[i] = std::sin(pi * p.x[i]);
    }
    p.rho_c = 1.0;
    p.k = 1.0;
    p.q = 0.0;
    p.T_left = 0.0;
    p.T_right = 0.0;
    p.dt = dt;
    p.n_steps = n_steps;
    p.theta = theta;
    p.E = 3.0;
    p.alpha = 2.0;
    p.T_ref = 0.0;
    p.right_end = right_end;
    return p;
}

// Errore (in valore assoluto) dello spostamento della punta, con l'estremo destro libero.
static double tip_error(size_t N, double dt, size_t n_steps, double theta) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalResult r = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, theta, RightEnd::Free));
    EXPECT_EQ(r.u.size(), N + 1);
    if (r.u.size() != N + 1) {
        return 1.0;   // senza spostamenti non c'e' niente da confrontare (il test e' gia' rosso)
    }
    const double t = n_steps * dt;
    return std::fabs(r.u.back() - 2.0 * 2.0 * std::exp(-pi * pi * t) / pi);
}

// A t = 0.1 (Crank-Nicolson, 100 elementi, 100 passi) la catena deve dare i due risultati esatti:
//  - barra libera: la punta si sposta di 2 alpha exp(-pi^2 t) / pi = 0.4745. Il nodo sinistro non si
//    muove, e la tensione e' 0 in ogni elemento (calcolata con la STESSA temperatura finale da cui
//    viene lo spostamento: con la temperatura iniziale non verrebbe 0);
//  - barra bloccata: la tensione vale -2 E alpha exp(-pi^2 t) / pi = -1.4236 in ogni elemento.
// L'errore di discretizzazione e' circa 1.7e-4 in termini relativi, quindi la tolleranza e' 5e-4.
TEST(Thermoelastic1d, ThermoMechanicalFollowsTheDecayingSine) {
    const double pi = std::acos(-1.0);
    const size_t N = 100, n_steps = 100;
    const double dt = 0.001, t = n_steps * dt;

    ThermoMechanicalResult free_bar = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, 0.5, RightEnd::Free));
    ASSERT_EQ(free_bar.u.size(), N + 1);
    ASSERT_EQ(free_bar.sigma.size(), N);
    const double exact_tip = 2.0 * 2.0 * std::exp(-pi * pi * t) / pi;
    EXPECT_NEAR(free_bar.u.front(), 0.0, 1e-15);
    EXPECT_NEAR(free_bar.u.back(), exact_tip, 5e-4 * exact_tip);
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(free_bar.sigma[e], 0.0, 1e-9) << "libera, elemento " << e;
    }

    ThermoMechanicalResult clamped_bar = solve_thermomechanical(decaying_sine_problem(N, dt, n_steps, 0.5, RightEnd::Fixed));
    ASSERT_EQ(clamped_bar.sigma.size(), N);
    const double exact_stress = -2.0 * 3.0 * 2.0 * std::exp(-pi * pi * t) / pi;
    for (size_t e = 0; e < N; ++e) {
        EXPECT_NEAR(clamped_bar.sigma[e], exact_stress, 5e-4 * std::fabs(exact_stress)) << "bloccata, elemento " << e;
    }
}

// ORDINI. Lo spostamento e la tensione derivano dalla temperatura: ne ereditano l'ordine?
// Crank-Nicolson, N = 10, 20, 40, 80 elementi e N passi fino a t = 0.1 (dt = 0.1 / N: quando
// dimezzi h dimezzi anche dt). L'errore nello spazio e quello nel tempo sono entrambi del secondo
// ordine, quindi l'errore sulla punta deve diventare 4 volte piu' piccolo a ogni raddoppio.
TEST(Thermoelastic1d, ThermoMechanicalIsSecondOrderForCrankNicolson) {
    double previous = tip_error(10, 0.1 / 10, 10, 0.5);
    for (size_t N : {20, 40, 80}) {
        const double current = tip_error(N, 0.1 / N, N, 0.5);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 2.0, 0.05) << "N = " << N / 2 << " -> " << N;
        previous = current;
    }
}

// Eulero all'indietro (theta = 1): primo ordine nel tempo. Mesh molto fine (N = 400), cosi' l'errore
// di mesh e' trascurabile; t = 0.1 con 5, 10, 20 passi.
TEST(Thermoelastic1d, ThermoMechanicalIsFirstOrderInTimeForBackwardEuler) {
    const size_t N = 400;
    double previous = tip_error(N, 0.1 / 5, 5, 1.0);
    for (size_t steps : {10, 20}) {
        const double current = tip_error(N, 0.1 / steps, steps, 1.0);
        EXPECT_NEAR(observed_order(previous, current, 2.0), 1.0, 0.1) << steps / 2 << " -> " << steps << " passi";
        previous = current;
    }
}

// Input sbagliati: gli errori dei tre passi non devono sparire. Si parte dal problema a mano
// (valido) e si rovina un dato alla volta.
TEST(Thermoelastic1d, ThermoMechanicalWrongInputsThrow) {
    const ThermoMechanicalProblem good = hand_problem();
    EXPECT_NO_THROW(solve_thermomechanical(good));

    ThermoMechanicalProblem p = good;   // un T0 con un valore in meno (dal calore)
    p.T0.pop_back();
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // dt = 0 (dal calore)
    p.dt = 0.0;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // theta fuori da [0, 1] (dal calore)
    p.theta = 1.5;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // E = 0 e E = NaN (dalla meccanica)
    p.E = 0.0;
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);
    p.E = std::nan("");
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // una barra con un solo nodo non ha elementi (dalla meccanica)
    p.x = {0.0};
    p.T0 = {6.0};
    EXPECT_THROW(solve_thermomechanical(p), std::invalid_argument);

    p = good;                           // un NaN nei dati: CG non converge (dal calore)
    p.T0[1] = std::nan("");
    EXPECT_THROW(solve_thermomechanical(p), std::runtime_error);
}
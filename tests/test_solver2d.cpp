#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Mesh2D.h"
#include "Solver2D.h"

// ---- Temperatura stazionaria con le temperature imposte: tutto insieme ----------------------------------------
//
// solve_steady_2d(mesh, k, f, fixed_nodes, fixed_values) mette insieme i pezzi dei passi prima:
//   1. K = la matrice di rigidezza della mesh, con la conducibilita k;
//   2. A = K con le righe e le colonne dei nodi fissati sostituite da quelle dell'identita';
//   3. b = f - K g^, con i valori imposti sulle righe dei nodi fissati;
//   4. T = la soluzione di A T = b, con il gradiente coniugato e il precondizionatore di Jacobi (tolleranza 1e-12,
//      al piu' 10 n iterazioni, come nei solutori 1D).
// f e' il vettore dei carichi (n numeri, uno per nodo): per una sorgente uniforme q e' assemble_load_2d(mesh, q).
// fixed_values[q] e' la temperatura imposta al nodo fixed_nodes[q].
//
// Il calcolo parte da T0 = g^ (i valori imposti sui nodi fissati e 0 sugli altri). La riga di un nodo fissato d di
// A T = b e' T[d] = g[d] e T0 la soddisfa gia': il suo residuo e' 0 e il CG non la tocca mai (la matrice A e' a
// blocchi, il precondizionatore di Jacobi non mescola le righe), quindi i nodi fissati escono con la temperatura
// imposta ESATTAMENTE. Partendo da T0 = 0 uscirebbero con un errore di circa 1e-13.
//
// Serve almeno un nodo fissato: senza, K e' singolare (le sue righe sommano a 0, quindi se T risolve K T = f,
// anche T + c lo risolve, per ogni c). Lancia std::invalid_argument se k non e' > 0 (NaN compreso), se f non ha
// un numero per nodo, se fixed_nodes e' vuoto, se fixed_values ha un numero di elementi diverso da fixed_nodes,
// se un nodo fissato e' >= n o compare due volte, o se un nodo NON fissato non sta in nessun triangolo (la sua riga
// di K e' tutta 0). Lancia std::runtime_error se il CG non converge.

namespace {

using Dense = std::vector<std::vector<double>>;
using Nodes = std::vector<std::size_t>;

// Confronta due vettori elemento per elemento, con la tolleranza assoluta tol.
void expect_vector(const DVector& got, const DVector& want, double tol, const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ": elemento " << i;
    }
}

// La CSR scritta come tabella piena n x n, letta con get(i, j).
Dense to_dense(const CSR& A) {
    const std::size_t n = A.size();
    Dense out(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            out[i][j] = A.get(i, j);
        }
    }
    return out;
}

// Risolve A x = b (tabella piena) con l'eliminazione di Gauss e lo scambio delle righe. E' lenta (n^3) ma non usa
// ne' il CG ne' nessuna funzione di questo passo o dei due prima: serve da "oracolo".
DVector gauss_solve(Dense A, DVector b) {
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::fabs(A[r][c]) > std::fabs(A[pivot][c])) {
                pivot = r;
            }
        }
        std::swap(A[c], A[pivot]);
        std::swap(b[c], b[pivot]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double ratio = A[r][c] / A[c][c];
            for (std::size_t j = c; j < n; ++j) {
                A[r][j] -= ratio * A[c][j];
            }
            b[r] -= ratio * b[c];
        }
    }
    DVector x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
    return x;
}

// La temperatura calcolata "a mano": i nodi fissati valgono il valore imposto e per ogni nodo libero i
//     somma sui nodi liberi j di K(i, j) T[j] = f[i] - somma sui nodi fissati d di K(i, d) g[d]
// e' un sistema solo nei nodi liberi, risolto con l'eliminazione di Gauss.
DVector dense_solution(const Mesh2D& m, double k, const DVector& f, const Nodes& fixed, const DVector& values) {
    const std::size_t n = m.num_nodes();
    const Dense K = to_dense(assemble_stiffness_2d(m, k));
    std::vector<bool> is_fixed(n, false);
    DVector T(n, 0.0);
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        is_fixed[fixed[q]] = true;
        T[fixed[q]] = values[q];
    }
    Nodes free_nodes;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_fixed[i]) {
            free_nodes.push_back(i);
        }
    }
    const std::size_t nf = free_nodes.size();
    Dense A(nf, DVector(nf, 0.0));
    DVector b(nf, 0.0);
    for (std::size_t a = 0; a < nf; ++a) {
        b[a] = f[free_nodes[a]];
        for (std::size_t c = 0; c < nf; ++c) {
            A[a][c] = K[free_nodes[a]][free_nodes[c]];
        }
        for (std::size_t q = 0; q < fixed.size(); ++q) {
            b[a] -= K[free_nodes[a]][fixed[q]] * values[q];
        }
    }
    if (nf > 0) {
        const DVector x = gauss_solve(A, b);
        for (std::size_t a = 0; a < nf; ++a) {
            T[free_nodes[a]] = x[a];
        }
    }
    return T;
}

// Un numero "a caso" in [0, 1): un piccolo generatore congruenziale, cosi' i test danno gli stessi numeri con
// qualunque compilatore. `state` e' lo stato del generatore, cambia a ogni chiamata.
double next_random(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
}

// Una mesh come rectangle_mesh ma con i nodi INTERNI spostati a caso, in x e in y, di al massimo il 20% del passo
// della griglia. Con il 20% ogni triangolo resta antiorario (il costruttore di Mesh2D lo controlla e lancerebbe
// un'eccezione); la mesh invece e' storta: triangoli di forme e aree diverse, con angoli ottusi (oltre 120 gradi).
// Il bordo non si sposta: e' ancora il rettangolo.
Mesh2D perturbed_rectangle(double lx, double ly, std::size_t nx, std::size_t ny, unsigned long long seed) {
    const Mesh2D base = rectangle_mesh(lx, ly, nx, ny);
    const std::size_t n = base.num_nodes();
    std::vector<bool> on_boundary(n, false);
    for (std::size_t d : boundary_nodes(base)) {
        on_boundary[d] = true;
    }
    const double hx = lx / static_cast<double>(nx);
    const double hy = ly / static_cast<double>(ny);
    unsigned long long state = seed;
    std::vector<double> xs(n);
    std::vector<double> ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = base.x(i);
        ys[i] = base.y(i);
        if (!on_boundary[i]) {
            xs[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hx;
            ys[i] += (2.0 * next_random(state) - 1.0) * 0.2 * hy;
        }
    }
    std::vector<Triangle> triangles;
    for (std::size_t e = 0; e < base.num_triangles(); ++e) {
        triangles.push_back(base.triangle(e));
    }
    return Mesh2D(xs, ys, triangles);
}

// Il triangolo grande con un nodo dentro (ESEMPIO 8 delle mesh): non e' un rettangolo.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// Il quadrilatero irregolare con un nodo dentro (ESEMPIO 39): quattro triangoli di area diversa.
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
}

// La mesh dell'ESEMPIO 42: i nodi 0 e 5 non stanno in nessun triangolo.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

// Mesh di tutte le forme: un rettangolo, il triangolo con un nodo dentro, il quadrilatero irregolare, due mesh
// storte e il quadrato di due triangoli.
std::vector<Mesh2D> varied_meshes() {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(perturbed_rectangle(2.0, 1.0, 8, 6, 5));
    meshes.push_back(perturbed_rectangle(1.0, 1.0, 12, 12, 31));
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));
    return meshes;
}

// Quattro insiemi di nodi fissati, validi per ogni mesh connessa: il solo nodo 0, il solo ultimo nodo, tutto il
// bordo, un nodo si' e uno no.
std::vector<Nodes> some_fixed_sets(const Mesh2D& m) {
    const std::size_t n = m.num_nodes();
    Nodes first;
    first.push_back(0);
    Nodes last;
    last.push_back(n - 1);
    Nodes every_second;
    for (std::size_t i = 0; i < n; i += 2) {
        every_second.push_back(i);
    }
    std::vector<Nodes> sets;
    sets.push_back(first);
    sets.push_back(last);
    sets.push_back(boundary_nodes(m));
    sets.push_back(every_second);
    return sets;
}

// Un carico non uniforme, con numeri positivi e negativi (nodi che scaldano e nodi che raffreddano).
DVector mixed_load(std::size_t n) {
    DVector f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = 4.0 * std::sin(1.3 * static_cast<double>(i)) + static_cast<double>(i % 3);
    }
    return f;
}

// Valori imposti tutti diversi fra loro, uno per nodo della lista.
DVector varied_values(const Nodes& nodes) {
    DVector values;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
        values.push_back(10.0 + 7.0 * std::cos(0.7 * static_cast<double>(q)) + 0.5 * static_cast<double>(nodes[q]));
    }
    return values;
}

}  // namespace

// ESEMPIO 69. Le due celle quadrate di sempre (rectangle_mesh(2, 1, 2, 1), k = 2) con la sorgente q = 6, cioe' con
// il carico f = (2, 3, 1, 1, 3, 2). I nodi 0, 1, 2 sono in basso e 3, 4, 5 in alto; i nodi 1 e 4 stanno nel
// mezzo. La riga 1 di K e' (-1, 4, -1, 0, -2, 0) e la riga 4 e' (0, -2, 0, -1, 4, -1).
//   (a) Lato sinistro (nodi 0 e 3) a 10 e lato destro (nodi 2 e 5) a 30. I nodi liberi sono 1 e 4 e le loro due
//       righe dicono (i fissati passano a destra: 3 di carico piu' 10 + 30)
//           4 T1 - 2 T4 = 43        e        -2 T1 + 4 T4 = 43
//       cioe' T1 = T4 = 21.5. E' anche la formula esatta della lastra, T(x) = 10 + 20 x / 2 + 6 x (2 - x) / (2 * 2),
//       in x = 1.
//   (b) Tutti fissati tranne il nodo 1, con T0 = 10, T2 = 30, T3 = 10, T4 = 18, T5 = 30. La riga 1 dice
//           4 T1 = f1 + 1 * T0 + 1 * T2 + 2 * T4 = 3 + 10 + 30 + 36 = 79       cioe' T1 = 19.75
//       (la temperatura di un nodo e' una media pesata dei vicini, piu' un quarto del suo carico).
//   (c) Tutti fissati tranne il nodo 4, con T0 = 10, T1 = 20, T2 = 30, T3 = 10, T5 = 30. La riga 4 dice
//           4 T4 = f4 + 2 * T1 + 1 * T3 + 1 * T5 = 3 + 40 + 10 + 30 = 83       cioe' T4 = 20.75
TEST(Solver2d, TwoSquareCellsMatchHandComputedTemperatures) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector f = assemble_load_2d(m, 6.0);
    expect_vector(f, {2.0, 3.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "carico");

    expect_vector(solve_steady_2d(m, 2.0, f, {0, 2, 3, 5}, {10.0, 30.0, 10.0, 30.0}),
                  {10.0, 21.5, 30.0, 10.0, 21.5, 30.0}, 1e-9, "caso (a)");
    expect_vector(solve_steady_2d(m, 2.0, f, {0, 2, 3, 4, 5}, {10.0, 30.0, 10.0, 18.0, 30.0}),
                  {10.0, 19.75, 30.0, 10.0, 18.0, 30.0}, 1e-9, "caso (b)");
    expect_vector(solve_steady_2d(m, 2.0, f, {0, 1, 2, 3, 5}, {10.0, 20.0, 30.0, 10.0, 30.0}),
                  {10.0, 20.0, 30.0, 10.0, 20.75, 30.0}, 1e-9, "caso (c)");
}

// ESEMPIO 70. La mesh dell'ESEMPIO 42: due triangoli (il quadrato unitario tagliato lungo la diagonale 1-3) e due
// nodi lontani, 0 e 5, che non stanno in nessun triangolo. Con k = 2 e q = 12 il carico e' f = (0, 4, 2, 4, 2, 0).
// K ha 2 sulla diagonale dei nodi 1, 2, 3, 4, ha -1 fra i nodi vicini (1-2, 2-3, 3-4, 4-1) e 0 fra 1-3 e 2-4.
// Con il nodo 1 fissato a 10, le righe dei nodi 2, 3, 4 dicono
//         2 T2 - T3 = 2 + 10        2 T3 - T2 - T4 = 4        2 T4 - T3 = 2 + 10.
// Per simmetria T2 = T4 = s, quindi T3 = 2 s - 12 e 2 (2 s - 12) - 2 s = 4, cioe' s = 14 e T3 = 16.
// I nodi 0 e 5 hanno la riga di K tutta 0: nessun triangolo li collega a niente. Se sono fissati valgono il valore
// imposto e basta (qui 99 e -5). Se NON lo sono il problema e' singolare (la loro temperatura puo' essere qualunque)
// e la funzione deve lanciare std::invalid_argument: la diagonale di A sarebbe 0 e il precondizionatore di Jacobi
// lo rifiuta, senza che la funzione debba controllare niente.
TEST(Solver2d, NodesOutsideEveryTriangleMustBeFixed) {
    const Mesh2D m = mesh_with_nodes_outside_every_triangle();
    const DVector f = assemble_load_2d(m, 12.0);
    expect_vector(f, {0.0, 4.0, 2.0, 4.0, 2.0, 0.0}, 1e-12, "carico");

    expect_vector(solve_steady_2d(m, 2.0, f, {0, 1, 5}, {99.0, 10.0, -5.0}), {99.0, 10.0, 14.0, 16.0, 14.0, -5.0},
                  1e-9, "tutti i nodi isolati fissati");

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 1}, {99.0, 10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1, 5}, {10.0, -5.0}), std::invalid_argument);
}

// ESEMPIO 71. Due problemi di cui si conosce la soluzione esatta, su rettangoli di forme diverse (celle non quadrate
// comprese): gli stessi dell'ESEMPIO 66, ma adesso si risolve davvero e si confronta T con la formula, nodo per
// nodo. k = 2.5 e la sorgente e' q = 120.
//   (a) Una lastra lunga L fra x = 0 e x = L, con la temperatura imposta sul lato sinistro (20) e sul lato destro
//       (80); il lato in basso e quello in alto sono liberi (nessun flusso). Il problema -k T'' = q ha la soluzione
//           T(x) = T_sinistra + (T_destra - T_sinistra) x / L + q x (L - x) / (2 k).
//   (b) Tutto il bordo fissato alla temperatura di
//           T(x, y) = 50 - q (x^2 + y^2) / (4 k),
//       per cui -k (T_xx + T_yy) = -k (-q / (2k) - q / (2k)) = q.
// Su queste mesh K coincide con lo schema a cinque punti delle differenze finite, che e' esatto per i polinomi di
// secondo grado: la soluzione esatta, calcolata nei nodi, risolve il sistema discreto. Quindi T deve uscire uguale
// alla formula (a meno dell'errore del CG), non "vicina": e' un controllo sul numero, non sull'ordine di convergenza.
TEST(Solver2d, ExactSolutionsOfTwoSimpleProblemsAreReproduced) {
    struct Grid {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Grid grids[] = {{1.0, 1.0, 2, 2}, {3.0, 1.5, 4, 3}, {2.0, 0.5, 8, 2}, {1.0, 2.0, 5, 7}, {4.0, 1.0, 6, 1}};
    const double k = 2.5;
    const double q = 120.0;

    for (const Grid& g : grids) {
        SCOPED_TRACE(std::to_string(g.nx) + " x " + std::to_string(g.ny) + " celle, " + std::to_string(g.lx) + " x " +
                     std::to_string(g.ly));
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const std::size_t n = m.num_nodes();

        Nodes sides;
        DVector side_values;
        DVector slab(n);
        DVector paraboloid(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = m.x(i);
            const double y = m.y(i);
            slab[i] = 20.0 + (80.0 - 20.0) * x / g.lx + q * x * (g.lx - x) / (2.0 * k);
            paraboloid[i] = 50.0 - q * (x * x + y * y) / (4.0 * k);
            if (x < 1e-12 || x > g.lx - 1e-12) {
                sides.push_back(i);
                side_values.push_back(slab[i]);
            }
        }
        const DVector f = assemble_load_2d(m, q);
        expect_vector(solve_steady_2d(m, k, f, sides, side_values), slab, 1e-9, "lastra");

        const Nodes boundary = boundary_nodes(m);
        DVector boundary_values;
        for (std::size_t d : boundary) {
            boundary_values.push_back(paraboloid[d]);
        }
        expect_vector(solve_steady_2d(m, k, f, boundary, boundary_values), paraboloid, 1e-9, "paraboloide");
    }
}

// ESEMPIO 72. Il "patch test", il controllo classico di un codice agli elementi finiti: un campo LINEARE
//     T(x, y) = a + b x + c y
// con sorgente nulla (f = 0) deve essere riprodotto esattamente su QUALSIASI mesh di triangoli, anche storta.
// Si impone il valore del campo su tutto il bordo e si pretende che i nodi interni escano uguali al campo. (Una
// funzione lineare sta fra quelle che gli elementi lineari rappresentano senza errore, e risolve -k Laplaciano(T) = 0.)
// Sulle mesh dell'ESEMPIO 71 la soluzione esatta e' un polinomio di secondo grado e la mesh e' regolare; qui la mesh
// e' storta (triangoli di forme e aree diverse, angoli ottusi) e k cambia da un campo all'altro (anche 0.37 e 250).
TEST(Solver2d, ALinearFieldIsReproducedOnIrregularMeshes) {
    const std::vector<Mesh2D> meshes = varied_meshes();
    const double ks[] = {0.37, 250.0, 1.0, 3.5, 12.0, 0.8};
    const double as[] = {20.0, -5.0, 0.0, 100.0, 7.0, 1.0};
    const double bs[] = {3.0, 40.0, 1.0, -8.0, 0.5, 2.0};
    const double cs[] = {-2.0, 11.0, 1.0, 5.0, 9.0, -3.0};

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh numero " + std::to_string(index));
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();

        DVector field(n);
        for (std::size_t i = 0; i < n; ++i) {
            field[i] = as[index] + bs[index] * m.x(i) + cs[index] * m.y(i);
        }
        const Nodes boundary = boundary_nodes(m);
        DVector values;
        for (std::size_t d : boundary) {
            values.push_back(field[d]);
        }
        expect_vector(solve_steady_2d(m, ks[index], DVector(n, 0.0), boundary, values), field, 1e-9);
    }
}

// ESEMPIO 73. Il confronto con un "oracolo": la stessa temperatura calcolata in un altro modo, lento e semplice,
// che non usa il CG ne' A ne' b. Si scrive K come tabella piena, si scrivono a mano le equazioni dei soli nodi
// liberi
//     somma sui nodi liberi j di K(i, j) T[j] = f[i] - somma sui nodi fissati d di K(i, d) g[d]
// e si risolvono con l'eliminazione di Gauss.
// Le mesh sono quelle di sempre, comprese le storte; i nodi fissati sono quattro insiemi diversi (il solo nodo 0, il
// solo ultimo nodo, tutto il bordo, un nodo si' e uno no); il carico non e' uniforme (positivo e negativo) e i valori
// imposti sono tutti diversi fra loro. Con un solo nodo fissato il CG lavora molto (quasi 90 iterazioni su 169 nodi).
// La tolleranza e' 1e-9. Il CG si ferma quando il residuo e' piccolo, non quando lo e' l'errore: l'errore sulla
// temperatura e' il residuo diviso per un numero piccolo, quindi molto piu' grande della tolleranza del CG. Misurato su
// questi casi, la differenza massima dall'oracolo e'
//     tolleranza del CG:       1e-8      1e-10     1e-11     1e-12     1e-14
//     differenza massima:      5e-7      5e-9      1.3e-9    7e-11     4e-12
// Con la tolleranza 1e-12 (quella dei solutori 1D) il test passa con un margine di 14 volte; con 1e-11 o piu' larga no.
TEST(Solver2d, MatchesTheSlowDenseSolutionOnEveryKindOfMesh) {
    const std::vector<Mesh2D> meshes = varied_meshes();

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const DVector f = mixed_load(m.num_nodes());
        const std::vector<Nodes> sets = some_fixed_sets(m);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh numero " + std::to_string(index) + ", insieme di nodi numero " + std::to_string(s));
            const DVector values = varied_values(sets[s]);
            expect_vector(solve_steady_2d(m, 2.5, f, sets[s], values), dense_solution(m, 2.5, f, sets[s], values),
                          1e-9);
        }
    }
}

// ESEMPIO 74. I nodi fissati escono con la temperatura imposta ESATTAMENTE, uguale bit per bit (ASSERT_EQ, non
// ASSERT_NEAR), e l'ordine in cui si elencano i nodi non conta.
//   - Su una mesh storta di 80 nodi, con tutto il bordo fissato (32 nodi) a valori "scomodi" come 37.25 + 0.1 q +
//     sin(q), ogni nodo fissato deve valere il suo valore. Un CG che parte da zero ci arriva solo alla fine, con un
//     errore di circa 1e-13 (37.25 esce come 37.24999999999973); uno che parte da g^ non tocca mai quei nodi.
//   - Gli stessi nodi con gli stessi valori, elencati in ordine inverso, danno la stessa temperatura: ogni valore
//     sta nella POSIZIONE della lista in cui sta il suo nodo.
//   - Tutti i nodi fissati, in disordine: la temperatura e' la lista dei valori, nodo per nodo.
TEST(Solver2d, ImposedTemperaturesComeBackExactlyInAnyOrder) {
    const Mesh2D m = perturbed_rectangle(2.0, 1.0, 9, 7, 3);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 80u);
    const DVector f = assemble_load_2d(m, 5.0);

    const Nodes fixed = boundary_nodes(m);
    DVector values;
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        values.push_back(37.25 + 0.1 * static_cast<double>(q) + std::sin(static_cast<double>(q)));
    }
    const DVector T = solve_steady_2d(m, 1.7, f, fixed, values);
    ASSERT_EQ(T.size(), n);
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        ASSERT_EQ(T[fixed[q]], values[q]) << std::setprecision(17) << "nodo fissato " << fixed[q] << ": e' uscito "
                                          << T[fixed[q]] << " invece di " << values[q];
    }

    const Nodes reversed_nodes(fixed.rbegin(), fixed.rend());
    const DVector reversed_values(values.rbegin(), values.rend());
    expect_vector(solve_steady_2d(m, 1.7, f, reversed_nodes, reversed_values), T, 1e-12, "ordine inverso");

    const Mesh2D two_cells = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector all = solve_steady_2d(two_cells, 2.0, assemble_load_2d(two_cells, 6.0), {4, 0, 5, 2, 1, 3},
                                        {40.0, 0.5, 50.0, 20.0, 10.0, 30.0});
    ASSERT_EQ(all.size(), 6u);
    const DVector want = {0.5, 10.0, 20.0, 30.0, 40.0, 50.0};
    for (std::size_t i = 0; i < 6; ++i) {
        ASSERT_EQ(all[i], want[i]) << "tutti fissati, nodo " << i;
    }
}

// ESEMPIO 75. Il bilancio del calore. Le righe di K sommano a 0 (una temperatura costante non fa passare calore) e K
// e' simmetrica, quindi la somma di tutte le n righe di K T e' 0. Nei nodi liberi K T = f; nei nodi fissati d, la
// quantita'
//     f[d] - (K T)[d] = (calore prodotto in d) - (calore che da d passa ai vicini)
// e' il calore che il nodo fissato manda all'esterno. La somma di tutte le righe da'
//     somma sui nodi fissati d di (f[d] - (K T)[d]) = somma su TUTTI i nodi di f
// cioe' tutto il calore prodotto nella lastra esce dai nodi fissati: niente si perde e niente si crea. Il test lo
// controlla su ogni mesh, con quattro insiemi di nodi fissati e un carico con parti che scaldano e parti che
// raffreddano (la somma dei carichi puo' essere anche negativa o vicina a 0).
TEST(Solver2d, TheFixedNodesRemoveExactlyTheHeatThatIsProduced) {
    const std::vector<Mesh2D> meshes = varied_meshes();

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const CSR K = assemble_stiffness_2d(m, 2.5);
        const DVector f = mixed_load(n);
        double produced = 0.0;
        double size = 1.0;
        for (double x : f) {
            produced += x;
            size += std::fabs(x);
        }
        const std::vector<Nodes> sets = some_fixed_sets(m);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh numero " + std::to_string(index) + ", insieme di nodi numero " + std::to_string(s));
            const DVector T = solve_steady_2d(m, 2.5, f, sets[s], varied_values(sets[s]));
            const DVector KT = K * T;
            double removed = 0.0;
            for (std::size_t d : sets[s]) {
                removed += f[d] - KT[d];
            }
            EXPECT_NEAR(removed, produced, 1e-9 * size);
        }
    }
}

// ESEMPIO 76. Il principio del massimo, un'altra legge della fisica: senza sorgente (f = 0) la temperatura di ogni
// nodo sta fra la minima e la massima temperatura imposta (il calore va dal caldo al freddo: nessun nodo diventa piu'
// caldo di tutti i vicini); con una sorgente che scalda (f >= 0) nessun nodo scende sotto la minima imposta, e con
// una che raffredda (f <= 0) nessuno sale sopra la massima.
// Nel problema discreto vale quando i numeri fuori dalla diagonale di K sono tutti <= 0 (ogni nodo e' "tirato" verso
// i vicini). Sui rettangoli di rectangle_mesh succede (anche con celle non quadrate: i triangoli hanno un angolo
// retto e due acuti), ma NON su ogni mesh: K(i, j) diventa positivo se i due angoli opposti al lato i-j sommano
// piu' di 180 gradi, e sulle mesh storte dell'ESEMPIO 72 succede (con angoli ottusi di 120-140 gradi). Per questo
// il test e' solo sui rettangoli.
TEST(Solver2d, TemperaturesStayBetweenTheImposedOnesOnRectangles) {
    struct Grid {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Grid grids[] = {{1.0, 1.0, 10, 10}, {3.0, 1.0, 12, 4}, {1.0, 2.0, 3, 9}};
    unsigned long long state = 2024;

    for (const Grid& g : grids) {
        SCOPED_TRACE(std::to_string(g.nx) + " x " + std::to_string(g.ny) + " celle");
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const std::size_t n = m.num_nodes();
        const Nodes boundary = boundary_nodes(m);
        DVector values;
        for (std::size_t q = 0; q < boundary.size(); ++q) {
            values.push_back(20.0 + 80.0 * next_random(state));
        }
        const double lowest = *std::min_element(values.begin(), values.end());
        const double highest = *std::max_element(values.begin(), values.end());
        const double slack = 1e-9;

        const DVector none = solve_steady_2d(m, 1.5, DVector(n, 0.0), boundary, values);
        const DVector heating = solve_steady_2d(m, 1.5, assemble_load_2d(m, 5.0), boundary, values);
        const DVector cooling = solve_steady_2d(m, 1.5, assemble_load_2d(m, -5.0), boundary, values);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_GE(none[i], lowest - slack) << "senza sorgente, nodo " << i;
            EXPECT_LE(none[i], highest + slack) << "senza sorgente, nodo " << i;
            EXPECT_GE(heating[i], lowest - slack) << "sorgente che scalda, nodo " << i;
            EXPECT_LE(cooling[i], highest + slack) << "sorgente che raffredda, nodo " << i;
        }
    }
}

// ESEMPIO 77. Gli errori che sono colpa di chi chiama. Sulle due celle dell'ESEMPIO 69 (6 nodi) lanciano
// std::invalid_argument:
//   - k = 0, negativa o NaN: anche quando TUTTI i nodi sono fissati (allora A e' l'identita' e nessun altro
//     controllo se ne accorgerebbe);
//   - f con un numero di elementi diverso da 6 (5, 7, nessuno);
//   - nessun nodo fissato, sia con un carico non nullo sia con f = 0 (con f = 0 il CG partirebbe da T = 0, con
//     residuo 0, e "convergerebbe" subito, restituendo una temperatura senza senso senza lamentarsi);
//   - fixed_values con un numero di elementi diverso da fixed_nodes, in tutti e due i versi;
//   - un nodo fissato uguale a 6 (l'ultimo e' 5) o piu' grande, anche enorme (un miliardo);
//   - un nodo ripetuto, con lo stesso valore o con valori diversi.
// Vanno bene invece: un solo nodo fissato (qualunque), i nodi fissati in disordine e tutti i nodi fissati.
// Il controllo di un indice deve venire PRIMA di usarlo per leggere o scrivere (per esempio per costruire il punto di
// partenza del CG): un indice di un miliardo usato per primo fa schiantare il programma invece di lanciare
// l'eccezione.
TEST(Solver2d, ThrowsInvalidArgumentOnBadInput) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const DVector f = assemble_load_2d(m, 6.0);
    const DVector zero(6, 0.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {3}, {10.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {5}, {10.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {4, 0, 1}, {1.0, 2.0, 3.0}));
    EXPECT_NO_THROW(solve_steady_2d(m, 2.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    EXPECT_THROW(solve_steady_2d(m, 0.0, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, -2.0, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, nan, f, {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 0.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, -2.0, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, nan, f, {0, 1, 2, 3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                 std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(5, 1.0), {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(7, 1.0), {0, 2}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, DVector(), {0, 2}, {10.0, 30.0}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {}, {}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, zero, {}, {}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0}, {}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {6}, {10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 100}, {10.0, 30.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {1000000000}, {10.0}), std::invalid_argument);

    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {2, 2}, {10.0, 10.0}), std::invalid_argument);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2, 3, 2}, {10.0, 30.0, 10.0, 31.0}), std::invalid_argument);
}

// ESEMPIO 78. I problemi che non hanno una soluzione, e che il CG non riesce a risolvere: la funzione lancia
// std::runtime_error (e non restituisce un vettore sbagliato o pieno di NaN).
//   - Un NaN nel carico, o in un valore imposto: i confronti con un NaN sono tutti falsi e il CG si ferma subito,
//     non convergente.
//   - Una mesh di DUE pezzi separati (due triangoli senza nodi in comune) con un nodo fissato in uno solo, e una
//     sorgente di calore: il pezzo senza nessun nodo fissato si scalda per sempre, uno stato stazionario non
//     esiste (matematicamente, K di quel pezzo e' singolare e il sistema non ha soluzione). Il CG non converge mai.
//     Con un nodo fissato in OGNI pezzo invece il problema e' risolubile: nessuna eccezione.
// (Senza sorgente quel pezzo avrebbe soluzioni a volonta' e il CG ne restituirebbe una, T = 0: il controllo "almeno un
// nodo fissato per pezzo" non si fa, e' compito di chi chiama.)
TEST(Solver2d, ThrowsRuntimeErrorWhenTheSystemCannotBeSolved) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    DVector f = assemble_load_2d(m, 6.0);

    DVector with_nan = f;
    with_nan[4] = nan;
    EXPECT_THROW(solve_steady_2d(m, 2.0, with_nan, {0, 2, 3, 5}, {10.0, 30.0, 10.0, 30.0}), std::runtime_error);
    EXPECT_THROW(solve_steady_2d(m, 2.0, f, {0, 2, 3, 5}, {10.0, nan, 10.0, 30.0}), std::runtime_error);

    const Mesh2D two_pieces({0.0, 1.0, 0.0, 3.0, 4.0, 3.0}, {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, {{0, 1, 2}, {3, 4, 5}});
    const DVector heat = assemble_load_2d(two_pieces, 1.0);
    EXPECT_THROW(solve_steady_2d(two_pieces, 1.0, heat, {0}, {5.0}), std::runtime_error);
    EXPECT_THROW(solve_steady_2d(two_pieces, 1.0, heat, {4}, {5.0}), std::runtime_error);
    EXPECT_NO_THROW(solve_steady_2d(two_pieces, 1.0, heat, {0, 4}, {5.0, 7.0}));
}

// ESEMPIO 79. Una mesh di 60 x 60 celle (3721 nodi), il paraboloide dell'ESEMPIO 71 (tutto il bordo fissato,
// k = 1, q = 10, T = -q (x^2 + y^2) / 4). Il CG fa circa 200 iterazioni (sulle mesh degli altri test mai piu' di
// 100): abbastanza da vedere se la tolleranza e il numero massimo di iterazioni vanno bene su un problema di taglia
// vera. Il risultato deve coincidere con la formula esatta a meno di 1e-8 (l'errore vero e' circa 5e-12); la funzione
// ci mette circa due decimi di secondo (e se il CG non converge gira fino a 10 n iterazioni prima di lanciare
// l'eccezione: una ventina di secondi).
TEST(Solver2d, ALargeMeshIsSolvedAccurately) {
    const std::size_t cells = 60;
    const double k = 1.0;
    const double q = 10.0;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 3721u);

    DVector exact(n);
    for (std::size_t i = 0; i < n; ++i) {
        exact[i] = -q * (m.x(i) * m.x(i) + m.y(i) * m.y(i)) / (4.0 * k);
    }
    const Nodes boundary = boundary_nodes(m);
    ASSERT_EQ(boundary.size(), 240u);
    DVector values;
    for (std::size_t d : boundary) {
        values.push_back(exact[d]);
    }

    const DVector T = solve_steady_2d(m, k, assemble_load_2d(m, q), boundary, values);
    ASSERT_EQ(T.size(), n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(T[i] - exact[i]));
    }
    EXPECT_LT(worst, 1e-8) << "l'errore massimo e' " << worst;
}
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Dirichlet2D.h"
#include "Mesh2D.h"
#include "Triplets.h"

// ---- Temperature imposte (Dirichlet): la matrice ----------------------------------------------------------
//
// Il sistema K T = f ha una riga per nodo. Se la temperatura di alcuni nodi e' imposta, T[d] = g[d] per i nodi d
// "fissati", per quei nodi l'incognita e' gia' nota. Chiamiamo D l'insieme dei nodi fissati e F gli altri, i
// liberi. Le righe dei nodi liberi dicono
//     K_FF T_F + K_FD g = f_F        cioe'        K_FF T_F = f_F - K_FD g
// (K_FF e' K ristretta ai nodi liberi, K_FD collega i liberi ai fissati), mentre le righe dei nodi fissati
// dicono solo T_D = g. Per avere ancora un sistema quadrato n x n e SIMMETRICO (il CG lo richiede) si tengono
// tutte le n righe: quelle dei nodi fissati diventano righe dell'identita' e le loro colonne si azzerano (il
// pezzo K_FD g passa al termine noto: e' il passo dopo). La nuova matrice e'
//
//     A = [ K_FF  0 ]        prima i nodi liberi, poi i fissati
//         [  0    I ]
//
// e dipende solo da QUALI nodi sono fissati, non dai valori g. Si costruisce sulla lista di terne di K:
//   - si buttano le terne (r, c, v) con la riga r o la colonna c fissata;
//   - per ogni nodo fissato d si aggiunge la terna (d, d, 1).
// dirichlet_matrix_triplets_2d(n, triplets, fixed_nodes) restituisce la nuova lista; quella ricevuta non cambia.
// Le terne che restano sono quelle di prima, NON sommate, e quelle con valore 0 NON sono tolte. Lancia
// std::invalid_argument se un nodo fissato e' >= n, se un nodo compare due volte in fixed_nodes (due terne
// (d, d, 1) si sommerebbero a 2) o se una terna ricevuta ha la riga o la colonna >= n.

namespace {

using Dense = std::vector<std::vector<double>>;
using Nodes = std::vector<std::size_t>;

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

// Controlla che A sia la matrice `want`, elemento per elemento, e che occupi esattamente nnz posti.
void expect_matrix(const CSR& A, const Dense& want, std::size_t nnz, const std::string& what = "") {
    ASSERT_EQ(A.size(), want.size()) << what << " dimensione";
    EXPECT_EQ(A.get_val_size(), nnz) << what << " posti occupati";
    for (std::size_t i = 0; i < want.size(); ++i) {
        for (std::size_t j = 0; j < want.size(); ++j) {
            EXPECT_DOUBLE_EQ(A.get(i, j), want[i][j]) << what << " elemento (" << i << ", " << j << ")";
        }
    }
}

// I valori delle terne che stanno in posizione (row, col), in ordine crescente.
std::vector<double> values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col) {
    std::vector<double> out;
    for (const Triplet& t : triplets) {
        const auto& [r, c, v] = t;
        if (r == row && c == col) {
            out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Confronta i valori delle terne in posizione (row, col) con `want` (in ordine crescente).
void expect_values_at(const std::vector<Triplet>& triplets, std::size_t row, std::size_t col,
                      const std::vector<double>& want) {
    const std::vector<double> got = values_at(triplets, row, col);
    ASSERT_EQ(got.size(), want.size()) << "posizione (" << row << ", " << col << ")";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_DOUBLE_EQ(got[i], want[i]) << "posizione (" << row << ", " << col << "), valore numero " << i;
    }
}

// Quante posizioni (riga, colonna) DIVERSE compaiono nella lista, in una matrice n x n.
std::size_t distinct_places(const std::vector<Triplet>& triplets, std::size_t n) {
    std::vector<std::size_t> places;
    for (const Triplet& t : triplets) {
        const auto& [r, c, v] = t;
        places.push_back(r * n + c);
    }
    std::sort(places.begin(), places.end());
    places.erase(std::unique(places.begin(), places.end()), places.end());
    return places.size();
}

// Il "conto lento" della matrice: la tabella K con le righe e le colonne dei nodi fissati azzerate e 1 sulla
// loro diagonale.
Dense slow_dirichlet(Dense K, const Nodes& fixed) {
    for (std::size_t d : fixed) {
        for (std::size_t i = 0; i < K.size(); ++i) {
            K[d][i] = 0.0;
            K[i][d] = 0.0;
        }
        K[d][d] = 1.0;
    }
    return K;
}

// Quanti posti deve occupare la matrice: uno per ogni nodo fissato (la sua diagonale) e uno per ogni posizione
// (riga, colonna) DIVERSA delle terne con riga e colonna entrambe libere.
std::size_t expected_places(const std::vector<Triplet>& trip, std::size_t n, const Nodes& fixed) {
    std::vector<bool> is_fixed(n, false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    std::vector<Triplet> both_free;
    for (const Triplet& t : trip) {
        const auto& [r, c, v] = t;
        if (!is_fixed[r] && !is_fixed[c]) {
            both_free.push_back(t);
        }
    }
    return fixed.size() + distinct_places(both_free, n);
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

}  // namespace

// ESEMPIO 54. Le due celle quadrate dell'ESEMPIO 37 (k = 2), con la temperatura imposta sul lato sinistro (nodi
// 0 e 3) e sul lato destro (nodi 2 e 5). I nodi liberi sono 1 e 4. Delle 36 terne di K restano solo quelle con
// riga E colonna fra i nodi liberi:
//     (1, 1): 3 terne      (1, 4): 2 terne      (4, 1): 2 terne      (4, 4): 3 terne
// cioe' 10 terne, piu' 4 terne (d, d, 1) per i nodi fissati: 14 in tutto. Sommate danno
//
//         [ 1  0  0  0  0  0 ]
//         [ 0  4  0  0 -2  0 ]
//         [ 0  0  1  0  0  0 ]
//     A = [ 0  0  0  1  0  0 ]
//         [ 0 -2  0  0  4  0 ]
//         [ 0  0  0  0  0  1 ]
//
// che occupa 8 posti: i 4 della diagonale dei nodi fissati e i 4 del blocco libero 2 x 2.
TEST(Dirichlet2d, TwoSquareCellsWithTheLeftAndRightSidesFixedMatchHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(m.num_nodes(), trip, {0, 2, 3, 5});

    ASSERT_EQ(out.size(), 14u);
    expect_values_at(out, 0, 0, {1});
    expect_values_at(out, 2, 2, {1});
    expect_values_at(out, 3, 3, {1});
    expect_values_at(out, 5, 5, {1});
    expect_values_at(out, 1, 1, {1, 1, 2});
    expect_values_at(out, 4, 4, {1, 1, 2});
    expect_values_at(out, 1, 4, {-1, -1});
    expect_values_at(out, 4, 1, {-1, -1});
    expect_values_at(out, 0, 1, {});
    expect_values_at(out, 1, 0, {});
    expect_values_at(out, 1, 2, {});
    expect_values_at(out, 2, 1, {});
    expect_values_at(out, 3, 4, {});
    expect_values_at(out, 4, 3, {});

    const Dense want = {{1, 0, 0, 0, 0, 0},
                        {0, 4, 0, 0, -2, 0},
                        {0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0},
                        {0, -2, 0, 0, 4, 0},
                        {0, 0, 0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(m.num_nodes(), out), want, 8);
}

// ESEMPIO 55. Una lista fatta a mano, senza mesh, per vedere cosa succede a OGNI terna. n = 4 e i nodi fissati
// sono 1 e 3. Le 11 terne ricevute (riga, colonna, valore):
//     (0, 0, 1)  (0, 0, 2)  (0, 1, 5)  (1, 0, 5)  (1, 1, 3)  (1, 2, -1)  (2, 1, -1)  (2, 2, 4)  (2, 2, 6)
//     (0, 2, 0)  (3, 3, 7)
// Si buttano le 6 con la riga o la colonna fissata: (0, 1), (1, 0), (1, 1), (1, 2), (2, 1) e (3, 3). Restano 5
// terne, anche quelle con lo stesso posto ((0, 0) due volte e (2, 2) due volte: NON sono sommate) e quella con
// valore 0, piu' (1, 1, 1) e (3, 3, 1): 7 in tutto. Sommate danno una matrice che occupa 5 posti, uno dei quali
// vale 0:
//         [ 3  0  0  0 ]
//     A = [ 0  1  0  0 ]
//         [ 0  0 10  0 ]
//         [ 0  0  0  1 ]
TEST(Dirichlet2d, OnlyTheTripletsOfFixedNodesAreDroppedAndNothingIsSummed) {
    const std::vector<Triplet> trip = {{0, 0, 1.0}, {0, 0, 2.0}, {0, 1, 5.0}, {1, 0, 5.0}, {1, 1, 3.0}, {1, 2, -1.0},
                                       {2, 1, -1.0}, {2, 2, 4.0}, {2, 2, 6.0}, {0, 2, 0.0}, {3, 3, 7.0}};
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(4, trip, {1, 3});

    ASSERT_EQ(out.size(), 7u);
    expect_values_at(out, 0, 0, {1, 2});
    expect_values_at(out, 2, 2, {4, 6});
    expect_values_at(out, 0, 2, {0});
    expect_values_at(out, 1, 1, {1});
    expect_values_at(out, 3, 3, {1});
    expect_values_at(out, 0, 1, {});
    expect_values_at(out, 1, 0, {});
    expect_values_at(out, 1, 2, {});
    expect_values_at(out, 2, 1, {});

    const Dense want = {{3, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 10, 0}, {0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(4, out), want, 5);
}

// ESEMPIO 56. I due casi estremi sulle due celle dell'ESEMPIO 37.
//   - Nessun nodo fissato: la lista resta quella di prima, tutte e 36 le terne, e la matrice e' K.
//   - Tutti i nodi fissati: restano solo le 6 terne (d, d, 1) e la matrice e' l'identita' 6 x 6, con 6 posti.
TEST(Dirichlet2d, NoFixedNodesChangesNothingAndAllFixedLeavesTheIdentity) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);

    const std::vector<Triplet> same = dirichlet_matrix_triplets_2d(6, trip, {});
    ASSERT_EQ(same.size(), 36u);
    EXPECT_EQ(distinct_places(same, 6), 24u);
    const Dense K = {{2, -1, 0, -1, 0, 0},
                     {-1, 4, -1, 0, -2, 0},
                     {0, -1, 2, 0, 0, -1},
                     {-1, 0, 0, 2, -1, 0},
                     {0, -2, 0, -1, 4, -1},
                     {0, 0, -1, 0, -1, 2}};
    expect_matrix(csr_from_triplets(6, same), K, 24);

    const std::vector<Triplet> all = dirichlet_matrix_triplets_2d(6, trip, {0, 1, 2, 3, 4, 5});
    ASSERT_EQ(all.size(), 6u);
    for (std::size_t d = 0; d < 6; ++d) {
        expect_values_at(all, d, d, {1});
    }
    const Dense identity = {{1, 0, 0, 0, 0, 0},
                            {0, 1, 0, 0, 0, 0},
                            {0, 0, 1, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0},
                            {0, 0, 0, 0, 1, 0},
                            {0, 0, 0, 0, 0, 1}};
    expect_matrix(csr_from_triplets(6, all), identity, 6);
}

// ESEMPIO 57. Il risultato su mesh di forme diverse e per insiemi di nodi diversi, confrontato con il "conto
// lento": la tabella piena di K in cui si azzerano a mano le righe e le colonne dei nodi fissati e si mette 1
// sulla loro diagonale. Le mesh: un rettangolo, il triangolo con un nodo dentro, il quadrilatero irregolare, la
// mesh con due nodi fuori da ogni triangolo e il quadrato di due triangoli. Gli insiemi: il solo nodo 0, il solo
// ultimo nodo, tutto il bordo (boundary_nodes), un nodo si' e uno no, tutti i nodi.
// I posti occupati sono: un posto per ogni nodo fissato, piu' le posizioni diverse delle terne con riga e
// colonna entrambe libere (nessuno zero resta nelle righe e colonne dei nodi fissati).
TEST(Dirichlet2d, MatchesTheSlowDenseEliminationOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const std::vector<Triplet> trip = stiffness_triplets_2d(m, 3.5);
        const Dense K = to_dense(csr_from_triplets(n, trip));

        Nodes first;
        first.push_back(0);
        Nodes last;
        last.push_back(n - 1);
        Nodes every_second;
        for (std::size_t i = 0; i < n; i += 2) {
            every_second.push_back(i);
        }
        Nodes everything;
        for (std::size_t i = 0; i < n; ++i) {
            everything.push_back(i);
        }
        std::vector<Nodes> sets;
        sets.push_back(first);
        sets.push_back(last);
        sets.push_back(boundary_nodes(m));
        sets.push_back(every_second);
        sets.push_back(everything);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            SCOPED_TRACE("mesh numero " + std::to_string(index) + ", insieme di nodi numero " + std::to_string(s));
            const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(n, trip, sets[s]);
            expect_matrix(csr_from_triplets(n, out), slow_dirichlet(K, sets[s]), expected_places(trip, n, sets[s]));
        }
    }
}

// ESEMPIO 58. Tutto il bordo di un rettangolo fissato: restano i nodi interni. I posti occupati sono
//     un posto per nodo (l'identita' sui nodi del bordo, la diagonale del blocco libero sugli interni)
//     + 2 per ogni LATO fra due nodi interni (ogni lato occupa (I, J) e (J, I)).
// In un rettangolo di nx x ny celle (nx, ny >= 2) i nodi interni sono (nx - 1)(ny - 1) e i lati fra due nodi
// interni sono
//     orizzontali (nx - 2)(ny - 1)  +  verticali (nx - 1)(ny - 2)  +  diagonali (nx - 2)(ny - 2).
// Per nx = 4, ny = 3: 20 nodi, 6 interni, lati interni 2 * 2 + 3 * 1 + 2 * 1 = 9, posti 20 + 2 * 9 = 38.
// Inoltre le righe e le colonne dei nodi del bordo sono quelle dell'identita'.
TEST(Dirichlet2d, FixingTheWholeBoundaryLeavesOnePlacePerNodeAndTwoPerInnerEdge) {
    const std::size_t sizes[][2] = {{2, 2}, {3, 2}, {2, 4}, {4, 3}, {6, 6}, {10, 7}};

    for (const auto& s : sizes) {
        const std::size_t nx = s[0];
        const std::size_t ny = s[1];
        SCOPED_TRACE(std::to_string(nx) + " x " + std::to_string(ny) + " celle");
        const Mesh2D m = rectangle_mesh(1.0, 1.0, nx, ny);
        const std::size_t nodes = (nx + 1) * (ny + 1);
        const std::size_t inner_edges = (nx - 2) * (ny - 1) + (nx - 1) * (ny - 2) + (nx - 2) * (ny - 2);

        const Nodes boundary = boundary_nodes(m);
        const std::vector<Triplet> trip = stiffness_triplets_2d(m, 1.0);
        const CSR A = csr_from_triplets(nodes, dirichlet_matrix_triplets_2d(nodes, trip, boundary));
        EXPECT_EQ(A.get_val_size(), nodes + 2 * inner_edges);

        for (std::size_t d : boundary) {
            for (std::size_t c = 0; c < nodes; ++c) {
                ASSERT_DOUBLE_EQ(A.get(d, c), d == c ? 1.0 : 0.0) << "riga " << d << ", colonna " << c;
                ASSERT_DOUBLE_EQ(A.get(c, d), d == c ? 1.0 : 0.0) << "riga " << c << ", colonna " << d;
            }
        }
    }
}

// ESEMPIO 59. Gli errori. Con n = 3 i nodi sono 0, 1, 2:
//   - un nodo fissato uguale a n (non esiste: l'ultimo e' n - 1) o piu' grande, anche enorme (un miliardo);
//   - un nodo ripetuto, anche se non e' accanto al suo gemello;
//   - una terna con la riga o la colonna uguale a n, o enorme.
// lanciano std::invalid_argument, anche se la lista dei nodi fissati e' vuota. Il nodo n - 1, la lista vuota e
// tutti i nodi in disordine invece vanno bene.
// L'indice enorme serve a un controllo preciso: se la funzione usa l'indice per leggere la tabella dei nodi
// fissati PRIMA di averlo confrontato con n, legge un posto lontano miliardi di bit dalla tabella e il
// programma si schianta invece di lanciare l'eccezione.
TEST(Dirichlet2d, BadNodesAndBadTripletsThrowInvalidArgument) {
    const std::vector<Triplet> trip = {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {2, 2, 5.0}};

    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {}));
    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {2}));
    EXPECT_NO_THROW(dirichlet_matrix_triplets_2d(3, trip, {2, 0, 1}));

    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {3}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1000}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1000000000}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {0, 7}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {1, 1}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, trip, {0, 2, 1, 2}), std::invalid_argument);

    const std::vector<Triplet> bad_row = {{0, 0, 1.0}, {3, 0, 1.0}};
    const std::vector<Triplet> bad_col = {{0, 0, 1.0}, {0, 3, 1.0}};
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_row, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_col, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_row, {1}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, bad_col, {1}), std::invalid_argument);

    const std::vector<Triplet> huge_row = {{0, 0, 1.0}, {1000000000, 0, 1.0}};
    const std::vector<Triplet> huge_col = {{0, 0, 1.0}, {0, 1000000000, 1.0}};
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, huge_row, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_matrix_triplets_2d(3, huge_col, {}), std::invalid_argument);
}

// ESEMPIO 60. Tre proprieta' che servono al solutore.
//   - A e' simmetrica (is_symmetric(1e-12)): il CG lo controlla e rifiuta le altre.
//   - La diagonale di A e' positiva: il precondizionatore di Jacobi la divide e rifiuta le altre.
//   - L'ordine in cui scrivi i nodi fissati non cambia la matrice: {0, 2, 3, 5} e {5, 0, 3, 2} danno la stessa A.
TEST(Dirichlet2d, TheResultIsSymmetricWithPositiveDiagonalAndDoesNotDependOnTheOrderOfTheNodes) {
    const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
    const std::size_t n = m.num_nodes();
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 3.5);

    std::vector<Nodes> sets;
    sets.push_back(boundary_nodes(m));
    Nodes only_first;
    only_first.push_back(0);
    sets.push_back(only_first);
    Nodes scattered;
    scattered.push_back(n - 1);
    scattered.push_back(0);
    scattered.push_back(7);
    sets.push_back(scattered);

    for (std::size_t s = 0; s < sets.size(); ++s) {
        SCOPED_TRACE("insieme di nodi numero " + std::to_string(s));
        const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, trip, sets[s]));
        EXPECT_TRUE(A.is_symmetric(1e-12));
        const std::vector<double> diagonal = A.diag();
        ASSERT_EQ(diagonal.size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_GT(diagonal[i], 0.0) << "nodo " << i;
        }
    }

    const Mesh2D two = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> t2 = stiffness_triplets_2d(two, 2.0);
    const CSR in_order = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, t2, {0, 2, 3, 5}));
    const CSR shuffled = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, t2, {5, 0, 3, 2}));
    expect_matrix(shuffled, to_dense(in_order), in_order.get_val_size());
}

// ESEMPIO 61. Una mesh di 150 x 150 celle (22801 nodi, 405000 terne) con un nodo su tre fissato (7601 nodi).
// Una ricerca nella lista dei nodi fissati per ogni terna (405000 x 7601 confronti) da' il risultato giusto ma
// ci mette decine di secondi; con una tabella con un'etichetta per nodo ("fissato si'/no"), letta in un passo
// solo, bastano pochi centesimi di secondo. Il test misura il tempo della sola funzione e pretende meno di 2
// secondi: cento volte il tempo giusto, e molto meno della ricerca per ogni terna. Poi controlla il risultato
// con il solo prodotto matrice-vettore: per ogni vettore u, A u vale
//   - sulle righe dei nodi liberi: (K u0)[i], dove u0 e' u con gli elementi dei nodi fissati messi a 0
//     (la riga di A ha zero sulle colonne fissate);
//   - sulle righe dei nodi fissati: u[d] (riga dell'identita').
TEST(Dirichlet2d, LargeMeshWithManyFixedNodes) {
    const std::size_t cells = 150;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 1.0);
    ASSERT_EQ(n, 22801u);
    ASSERT_EQ(trip.size(), 405000u);

    Nodes fixed;
    std::vector<bool> is_fixed(n, false);
    for (std::size_t i = 0; i < n; i += 3) {
        fixed.push_back(i);
        is_fixed[i] = true;
    }
    ASSERT_EQ(fixed.size(), 7601u);

    const auto start = std::chrono::steady_clock::now();
    const std::vector<Triplet> out = dirichlet_matrix_triplets_2d(n, trip, fixed);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 2.0) << "dirichlet_matrix_triplets_2d ha impiegato " << seconds
                            << " s: deve leggere ogni terna una volta sola, senza cercare nella lista dei nodi fissati";
    const CSR A = csr_from_triplets(n, out);
    const CSR K = csr_from_triplets(n, trip);

    std::vector<double> u(n);
    std::vector<double> u0(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = 1.0 + 0.001 * static_cast<double>(i % 97);
        u0[i] = is_fixed[i] ? 0.0 : u[i];
    }
    const std::vector<double> got = A * u;
    const std::vector<double> Ku0 = K * u0;
    ASSERT_EQ(got.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(got[i], is_fixed[i] ? u[i] : Ku0[i], 1e-9) << "riga " << i;
    }
}

// ---- Temperature imposte (Dirichlet): il termine noto -----------------------------------------------------
//
// Il sistema con le temperature imposte e' A T = b. La matrice A e' quella del passo prima; il termine noto b ha
// una riga per nodo:
//   - nodo fissato d: b[d] = g[d], il valore imposto (la riga di A e' quella dell'identita': T[d] = g[d]);
//   - nodo libero i:  b[i] = f[i] - (K_FD g)[i] = f[i] - somma, sui nodi fissati d, di K(i, d) * g[d]
//     (il contributo dei nodi fissati, che e' noto, passa a destra).
// Un doppio ciclo (per ogni nodo libero, per ogni nodo fissato, K.get(i, d)) costa "nodi liberi x nodi fissati"
// letture e ognuna scorre una riga: troppo. Basta UN prodotto matrice-vettore. Sia g^ il vettore di n numeri che
// vale g[d] sui nodi fissati e 0 sugli altri. Per OGNI riga i
//     (K g^)[i] = somma su tutti i nodi j di K(i, j) g^[j] = somma sui nodi fissati d di K(i, d) g[d]
// e sulle righe dei nodi liberi e' proprio (K_FD g)[i]. Quindi
//     b = f - K g^        e poi        b[d] = g[d]  per ogni nodo fissato d
// (sulle righe dei fissati il prodotto non serve: il secondo passo le riscrive). K e' la matrice ORIGINALE, non
// la A del passo prima: A ha gia' azzerato il pezzo K_FD.
// dirichlet_rhs_2d(K, f, fixed_nodes, fixed_values) restituisce b; fixed_values[q] e' il valore imposto al nodo
// fixed_nodes[q]. Lancia std::invalid_argument se K.size() != f.size(), se fixed_values ha un numero di elementi
// diverso da fixed_nodes, se un nodo fissato e' >= n o se un nodo compare due volte in fixed_nodes (anche con lo
// stesso valore).

namespace {

// Confronta due vettori elemento per elemento, con la tolleranza assoluta tol.
void expect_vector(const DVector& got, const DVector& want, double tol, const std::string& what = "") {
    ASSERT_EQ(got.size(), want.size()) << what << ": dimensione";
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << ": elemento " << i;
    }
}

// Il "conto lento" del termine noto, scritto come dice la definizione: per ogni nodo libero i, f[i] meno la somma
// su TUTTI i nodi fissati dei prodotti K(i, d) * g[d]; per ogni nodo fissato, il suo valore.
DVector slow_rhs(const Dense& K, const DVector& f, const Nodes& fixed, const DVector& values) {
    std::vector<bool> is_fixed(f.size(), false);
    for (std::size_t d : fixed) {
        is_fixed[d] = true;
    }
    DVector b = f;
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (!is_fixed[i]) {
            for (std::size_t q = 0; q < fixed.size(); ++q) {
                b[i] -= K[i][fixed[q]] * values[q];
            }
        }
    }
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        b[fixed[q]] = values[q];
    }
    return b;
}

// La temperatura T (una per nodo) e' la soluzione del problema -k Laplaciano(T) = q sulla mesh m, con la
// temperatura imposta sui nodi `fixed` (uguale a T in quei nodi). Se e' cosi', A T - b = 0 in ogni riga: nelle
// righe dei nodi liberi e' l'equazione di K T = f, in quelle dei fissati e' T[d] = g[d].
void expect_solution_solves_the_system(const Mesh2D& m, double k, double q, const Nodes& fixed, const DVector& T) {
    const std::size_t n = m.num_nodes();
    DVector values;
    for (std::size_t d : fixed) {
        values.push_back(T[d]);
    }
    const DVector b = dirichlet_rhs_2d(assemble_stiffness_2d(m, k), assemble_load_2d(m, q), fixed, values);
    const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, stiffness_triplets_2d(m, k), fixed));
    const DVector AT = A * T;
    ASSERT_EQ(b.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(AT[i], b[i], 1e-9) << "riga " << i;
    }
}

}  // namespace

// ESEMPIO 63. Due esempi a mano.
//
// (a) Una barra di 4 nodi, tre elementi uguali con conduttanza 1 (la matrice senza vincoli):
//         [  1 -1  0  0 ]
//     K = [ -1  2 -1  0 ]       f = (7, 1, 1, 9)       nodo 0 fissato a 10, nodo 3 fissato a 40
//         [  0 -1  2 -1 ]
//         [  0  0 -1  1 ]
//   g^ = (10, 0, 0, 40) e K g^ = (10, -10, -40, 40). Nei nodi liberi b = f - K g^: b1 = 1 + 10 = 11 e
//   b2 = 1 + 40 = 41. Nei nodi fissati b e' il valore imposto (i carichi 7 e 9 non contano):
//     b = (10, 11, 41, 40).
//
// (b) Le due celle quadrate degli esempi di prima (k = 2) con la sorgente q = 6, che da' il carico
//     f = (2, 3, 1, 1, 3, 2), il lato sinistro (nodi 0 e 3) a 10 e il lato destro (nodi 2 e 5) a 30. I nodi liberi
//     sono 1 e 4. La riga 1 di K e' (-1, 4, -1, 0, -2, 0): il nodo 1 e' collegato ai fissati 0 e 2, con -1 e -1, e
//     non ai fissati 3 e 5. La riga 4 e' (0, -2, 0, -1, 4, -1): e' collegato ai fissati 3 e 5. Con
//     g^ = (10, 0, 30, 10, 0, 30):
//       (K g^)[1] = -1 * 10 + -1 * 30 = -40         b1 = 3 - (-40) = 43
//       (K g^)[4] = -1 * 10 + -1 * 30 = -40         b4 = 3 - (-40) = 43
//     e b = (10, 43, 30, 10, 43, 30). Con la A del passo prima il sistema ha la soluzione
//     T = (10, 21.5, 30, 10, 21.5, 30): nel nodo 1, 4 * 21.5 - 2 * 21.5 = 43.
TEST(Dirichlet2d, RightHandSideMatchesTwoHandComputedExamples) {
    const CSR bar(Dense{{1, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 1}});
    const DVector load = {7.0, 1.0, 1.0, 9.0};
    expect_vector(dirichlet_rhs_2d(bar, load, {0, 3}, {10.0, 40.0}), {10.0, 11.0, 41.0, 40.0}, 1e-12, "barra");

    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const CSR K = assemble_stiffness_2d(m, 2.0);
    const DVector f = assemble_load_2d(m, 6.0);
    expect_vector(f, {2.0, 3.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "carico");

    const Nodes fixed = {0, 2, 3, 5};
    const DVector b = dirichlet_rhs_2d(K, f, fixed, {10.0, 30.0, 10.0, 30.0});
    expect_vector(b, {10.0, 43.0, 30.0, 10.0, 43.0, 30.0}, 1e-12, "due celle");

    const CSR A = csr_from_triplets(6, dirichlet_matrix_triplets_2d(6, stiffness_triplets_2d(m, 2.0), fixed));
    expect_vector(A * DVector{10.0, 21.5, 30.0, 10.0, 21.5, 30.0}, b, 1e-12, "A per la soluzione");
}

// ESEMPIO 64. I casi estremi sulle due celle dell'ESEMPIO 63 (f = (2, 3, 1, 1, 3, 2)).
//   - Nessun nodo fissato: b = f.
//   - Tutti i nodi fissati, scritti in disordine: b[nodo] = il valore di quel nodo (il valore sta nella STESSA
//     posizione della lista in cui sta il nodo, non nella posizione che il nodo ha nella mesh).
//   - Un solo nodo fissato, con il valore 0: g^ = 0, quindi b = f tranne che nel nodo fissato, che vale 0.
//   - Il carico dei nodi fissati non conta: due vettori f che differiscono solo nei nodi fissati danno lo stesso b.
TEST(Dirichlet2d, RightHandSideWithNoFixedNodesIsTheLoadAndWithAllFixedIsTheValues) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const CSR K = assemble_stiffness_2d(m, 2.0);
    const DVector f = assemble_load_2d(m, 6.0);

    expect_vector(dirichlet_rhs_2d(K, f, {}, {}), f, 1e-12, "nessun nodo fissato");

    const DVector all = dirichlet_rhs_2d(K, f, {4, 0, 5, 2, 1, 3}, {40.0, 0.5, 50.0, 20.0, 10.0, 30.0});
    expect_vector(all, {0.5, 10.0, 20.0, 30.0, 40.0, 50.0}, 1e-12, "tutti i nodi fissati");

    expect_vector(dirichlet_rhs_2d(K, f, {1}, {0.0}), {2.0, 0.0, 1.0, 1.0, 3.0, 2.0}, 1e-12, "nodo 1 a zero");

    DVector other_load = f;
    other_load[0] += 1000.0;
    other_load[5] -= 77.0;
    const Nodes fixed = {0, 2, 3, 5};
    const DVector values = {10.0, 30.0, 10.0, 30.0};
    expect_vector(dirichlet_rhs_2d(K, other_load, fixed, values), dirichlet_rhs_2d(K, f, fixed, values), 1e-12,
                  "carico diverso nei nodi fissati");
}

// ESEMPIO 65. Il risultato su mesh di forme diverse e per insiemi di nodi diversi (gli stessi dell'ESEMPIO 57:
// il solo nodo 0, il solo ultimo nodo, tutto il bordo, un nodo si' e uno no, tutti i nodi), nell'ordine della
// mesh e anche in ordine inverso, con un carico e dei valori imposti tutti diversi fra loro. Il confronto e' con
// il "conto lento": la somma scritta come dice la definizione, un nodo libero alla volta.
// In fondo, una matrice NON simmetrica (non e' una K, ma la formula non lo richiede): b usa K(i, d), con la riga i
// libera e la colonna d fissata, non K(d, i). Con
//         [  4  1 -2  0  3 ]
//         [  0  5  1 -1  2 ]
//     G = [  2 -3  6  1  0 ]     f = (1, 2, 3, 4, 5)     nodo 3 a 10 e nodo 0 a 20
//         [  1  0 -1  7  2 ]
//         [ -2  4  0  1  8 ]
// g^ = (20, 0, 0, 10, 0), G g^ = (80, -10, 50, 90, -30), quindi b = (20, 2 + 10, 3 - 50, 10, 5 + 30) =
// (20, 12, -47, 10, 35). Con la matrice trasposta il nodo 1 darebbe 2 - 20 = -18.
TEST(Dirichlet2d, RightHandSideMatchesTheSlowSumOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());
    meshes.push_back(rectangle_mesh(1.0, 1.0, 1, 1));

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        const Mesh2D& m = meshes[index];
        const std::size_t n = m.num_nodes();
        const CSR K = assemble_stiffness_2d(m, 3.5);
        const Dense dense = to_dense(K);
        DVector f(n);
        for (std::size_t i = 0; i < n; ++i) {
            f[i] = 2.0 + 0.3 * static_cast<double>(i % 7) + 0.01 * static_cast<double>(i);
        }

        Nodes first;
        first.push_back(0);
        Nodes last;
        last.push_back(n - 1);
        Nodes every_second;
        for (std::size_t i = 0; i < n; i += 2) {
            every_second.push_back(i);
        }
        Nodes everything;
        for (std::size_t i = 0; i < n; ++i) {
            everything.push_back(i);
        }
        std::vector<Nodes> sets;
        sets.push_back(first);
        sets.push_back(last);
        sets.push_back(boundary_nodes(m));
        sets.push_back(every_second);
        sets.push_back(everything);

        for (std::size_t s = 0; s < sets.size(); ++s) {
            for (int reversed = 0; reversed < 2; ++reversed) {
                SCOPED_TRACE("mesh numero " + std::to_string(index) + ", insieme di nodi numero " +
                             std::to_string(s) + (reversed ? ", in ordine inverso" : ", in ordine"));
                Nodes nodes = sets[s];
                if (reversed) {
                    std::reverse(nodes.begin(), nodes.end());
                }
                DVector values;
                for (std::size_t q = 0; q < nodes.size(); ++q) {
                    values.push_back(100.0 + 7.0 * static_cast<double>(q) - 0.37 * static_cast<double>(nodes[q]));
                }
                expect_vector(dirichlet_rhs_2d(K, f, nodes, values), slow_rhs(dense, f, nodes, values), 1e-9);
            }
        }
    }

    const Dense general = {{4, 1, -2, 0, 3}, {0, 5, 1, -1, 2}, {2, -3, 6, 1, 0}, {1, 0, -1, 7, 2}, {-2, 4, 0, 1, 8}};
    const DVector load = {1.0, 2.0, 3.0, 4.0, 5.0};
    expect_vector(dirichlet_rhs_2d(CSR(general), load, {3, 0}, {10.0, 20.0}), {20.0, 12.0, -47.0, 10.0, 35.0}, 1e-12,
                  "matrice non simmetrica");
}

// ESEMPIO 66. Due problemi di cui si conosce la soluzione esatta, su rettangoli di forme diverse (celle non
// quadrate comprese). k = 2.5 e la sorgente e' q = 120.
//   (a) Una lastra lunga L fra x = 0 e x = L, con la temperatura imposta sul lato sinistro (20) e sul lato destro
//       (80); il lato in basso e quello in alto sono liberi (nessun flusso). Il problema -k T'' = q ha la soluzione
//           T(x) = T_sinistra + (T_destra - T_sinistra) x / L + q x (L - x) / (2 k).
//   (b) Tutto il bordo fissato alla temperatura di
//           T(x, y) = 50 - q (x^2 + y^2) / (4 k),
//       per cui -k (T_xx + T_yy) = -k (-q / (2k) - q / (2k)) = q.
// Su queste mesh K coincide con lo schema a cinque punti delle differenze finite (le due terne di K sul lato
// obliquo di ogni triangolo valgono 0), che e' esatto per i polinomi di secondo grado: la soluzione esatta,
// calcolata nei nodi, risolve il sistema discreto, e quindi A T = b in ogni riga (con T uguale alla soluzione
// esatta). Un termine noto sbagliato, in un solo nodo, rompe l'uguaglianza.
TEST(Dirichlet2d, TheExactSolutionOfTwoSimpleProblemsSolvesTheSystemWithImposedTemperatures) {
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
        DVector slab(n);
        DVector paraboloid(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = m.x(i);
            const double y = m.y(i);
            slab[i] = 20.0 + (80.0 - 20.0) * x / g.lx + q * x * (g.lx - x) / (2.0 * k);
            paraboloid[i] = 50.0 - q * (x * x + y * y) / (4.0 * k);
            if (x < 1e-12 || x > g.lx - 1e-12) {
                sides.push_back(i);
            }
        }
        {
            SCOPED_TRACE("lastra");
            expect_solution_solves_the_system(m, k, q, sides, slab);
        }
        {
            SCOPED_TRACE("tutto il bordo");
            expect_solution_solves_the_system(m, k, q, boundary_nodes(m), paraboloid);
        }
    }
}

// ESEMPIO 67. Gli errori. Con K di 3 x 3 e f di 3 numeri:
//   - fixed_values con un numero di elementi diverso da fixed_nodes, in tutti e due i versi (anche con una lista
//     vuota);
//   - K e f di dimensioni diverse (f di 2 o di 4 numeri);
//   - un nodo fissato uguale a n = 3 (l'ultimo e' 2) o piu' grande, anche enorme (un miliardo);
//   - un nodo ripetuto, con lo stesso valore o con valori diversi, accanto al gemello o lontano da lui;
// lanciano std::invalid_argument. La lista vuota, il nodo n - 1 e tutti i nodi in disordine invece vanno bene.
// Come per la matrice, il controllo di un indice deve venire PRIMA di usarlo per leggere o scrivere: un indice di
// un miliardo usato per primo fa schiantare il programma invece di lanciare l'eccezione.
TEST(Dirichlet2d, RightHandSideThrowsInvalidArgumentOnBadInput) {
    const CSR K(Dense{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}});
    const DVector f = {1.0, 2.0, 3.0};

    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {}, {}));
    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {2}, {7.0}));
    EXPECT_NO_THROW(dirichlet_rhs_2d(K, f, {2, 0, 1}, {1.0, 2.0, 3.0}));

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 1}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0}, {}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, DVector{1.0, 2.0}, {}, {}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, DVector{1.0, 2.0, 3.0, 4.0}, {}, {}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {3}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1000}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1000000000}, {1.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 7}, {1.0, 2.0}), std::invalid_argument);

    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1, 1}, {5.0, 5.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {1, 1}, {5.0, 6.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {0, 2, 1, 2}, {1.0, 2.0, 3.0, 4.0}), std::invalid_argument);
    EXPECT_THROW(dirichlet_rhs_2d(K, f, {2, 0, 1, 2}, {9.0, 2.0, 3.0, 9.0}), std::invalid_argument);
}

// ESEMPIO 68. Una mesh di 200 x 200 celle (40401 nodi) con un nodo su due fissato (20201 nodi).
// Un doppio ciclo con K.get(i, d) per ogni nodo libero i e ogni nodo fissato d (20200 x 20201 letture, ognuna con
// la scansione di una riga) da' il risultato giusto ma ci mette decine di secondi, e anche un prodotto
// matrice-vettore per ogni nodo fissato; con un solo prodotto matrice-vettore bastano pochi millesimi di secondo.
// Il test misura il tempo della sola funzione e pretende meno di 1 secondo.
// Poi controlla il risultato senza rifare il calcolo: per ogni vettore T che sui nodi fissati vale il valore
// imposto,
//   - sulle righe dei nodi liberi:  (A T - b)[i] = (K T - f)[i]    (K T = K_FF T_F + K_FD g, A T = K_FF T_F);
//   - sulle righe dei nodi fissati: (A T - b)[d] = T[d] - g[d] = 0.
TEST(Dirichlet2d, RightHandSideOnALargeMeshWithManyFixedNodes) {
    const std::size_t cells = 200;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, cells, cells);
    const std::size_t n = m.num_nodes();
    ASSERT_EQ(n, 40401u);
    const CSR K = assemble_stiffness_2d(m, 1.0);
    const DVector f = assemble_load_2d(m, 1.0);

    Nodes fixed;
    DVector values;
    std::vector<bool> is_fixed(n, false);
    for (std::size_t i = 0; i < n; i += 2) {
        fixed.push_back(i);
        values.push_back(1.0 + 0.01 * static_cast<double>(i % 89));
        is_fixed[i] = true;
    }
    ASSERT_EQ(fixed.size(), 20201u);

    const auto start = std::chrono::steady_clock::now();
    const DVector b = dirichlet_rhs_2d(K, f, fixed, values);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 1.0) << "dirichlet_rhs_2d ha impiegato " << seconds
                            << " s: deve fare un solo prodotto matrice-vettore, senza leggere K un elemento alla volta";
    ASSERT_EQ(b.size(), n);

    const CSR A = csr_from_triplets(n, dirichlet_matrix_triplets_2d(n, stiffness_triplets_2d(m, 1.0), fixed));
    DVector T(n);
    for (std::size_t i = 0; i < n; ++i) {
        T[i] = 1.0 + 0.001 * static_cast<double>(i % 97);
    }
    for (std::size_t q = 0; q < fixed.size(); ++q) {
        T[fixed[q]] = values[q];
    }
    const DVector AT = A * T;
    const DVector KT = K * T;
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(AT[i] - b[i], is_fixed[i] ? 0.0 : KT[i] - f[i], 1e-9) << "riga " << i;
    }
}
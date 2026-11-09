#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CSR_and_Vectors.h"
#include "Triplets.h"

// ---- Dalle terne (riga, colonna, valore) alla matrice CSR: csr_from_triplets ---------------------
//
// L'assemblaggio di una matrice agli elementi finiti produce una lista di CONTRIBUTI: ogni elemento
// (ogni triangolo) dice "somma questo numero alla posizione (riga, colonna)". Due elementi che
// condividono dei nodi contribuiscono alle stesse posizioni, quindi nella lista la stessa coppia
// (riga, colonna) compare piu' volte, e i valori vanno SOMMATI. La lista e' in un ordine qualunque.
//
// csr_from_triplets(n, lista) costruisce la matrice n x n che ha in (r, c) la somma dei valori di
// tutte le terne della lista con quella riga e quella colonna:
//   - le terne con la stessa posizione vengono sommate, qualunque sia l'ordine della lista;
//   - una posizione che compare nella lista viene memorizzata anche se la somma fa 0 (lo schema degli
//     elementi non nulli dipende solo da DOVE sono i contributi, non dai loro valori);
//   - le posizioni che non compaiono valgono 0 e non occupano memoria;
//   - una terna fuori dalla matrice (riga o colonna >= n) e una matrice senza righe (n = 0) sono
//     errori: std::invalid_argument, come negli altri costruttori del progetto.
// Il formato CSR e' quello gia' usato dal progetto: per ogni riga, in ordine, i suoi valori e le loro
// colonne; la riga r occupa i posti da row_ptr[r] a row_ptr[r+1] - 1.

namespace {

using Dense = std::vector<std::vector<double>>;

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

}  // namespace

// ESEMPIO 29. Una matrice 3 x 3 a mano, con la lista in disordine e un duplicato:
//     (2, 2, 1)   (0, 0, 2)   (0, 2, 3)   (0, 0, 5)   (1, 1, 4)   (2, 0, 6)
// La posizione (0, 0) compare due volte: 2 + 5 = 7. La matrice e'
//         [ 7  0  3 ]
//         [ 0  4  0 ]          5 posti occupati
//         [ 6  0  1 ]
// In formato CSR, riga per riga con le colonne in ordine:
//     values = [7, 3, 4, 6, 1]     col_indx = [0, 2, 1, 0, 2]     row_ptr = [0, 2, 3, 5]
// (due posti nella riga 0, uno nella riga 1, due nella riga 2: row_ptr si ottiene sommando passo
// passo questi conteggi, partendo da 0).
TEST(Triplets, HandExampleWithUnsortedInputAndADuplicate) {
    const std::vector<Triplet> list = {{2, 2, 1.0}, {0, 0, 2.0}, {0, 2, 3.0},
                                       {0, 0, 5.0}, {1, 1, 4.0}, {2, 0, 6.0}};
    const CSR A = csr_from_triplets(3, list);

    expect_matrix(A, {{7.0, 0.0, 3.0}, {0.0, 4.0, 0.0}, {6.0, 0.0, 1.0}}, 5);

    // Il prodotto con x = (1, 2, 3):  (7 + 9, 8, 6 + 3) = (16, 8, 9).
    const std::vector<double> y = A * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 16.0);
    EXPECT_DOUBLE_EQ(y[1], 8.0);
    EXPECT_DOUBLE_EQ(y[2], 9.0);

    // La diagonale e' (7, 4, 1).
    const std::vector<double> d = A.diag();
    ASSERT_EQ(d.size(), 3u);
    EXPECT_DOUBLE_EQ(d[0], 7.0);
    EXPECT_DOUBLE_EQ(d[1], 4.0);
    EXPECT_DOUBLE_EQ(d[2], 1.0);
}

// ESEMPIO 30. Quattro contributi alla stessa posizione (1, 2), mischiati con altri: 4 + 1 + 8 + 2 = 15.
// Poi due contributi opposti, +2 e -2, alla posizione (0, 1): la somma fa 0 ma la posizione resta
// memorizzata. Infine un contributo che vale 0 gia' da solo, in (2, 0): anche lui resta memorizzato
// (nella matrice di un triangolo rettangolo ci sono termini che valgono esattamente 0, e la loro
// posizione conta lo stesso). In tutto 3 posti occupati, due dei quali contengono 0.
TEST(Triplets, DuplicatesAreSummedWhateverTheirOrderAndZerosAreKept) {
    const CSR A = csr_from_triplets(
        3, {{1, 2, 4.0}, {0, 1, 2.0}, {1, 2, 1.0}, {2, 0, 0.0}, {1, 2, 8.0}, {0, 1, -2.0}, {1, 2, 2.0}});
    expect_matrix(A, {{0.0, 0.0, 0.0}, {0.0, 0.0, 15.0}, {0.0, 0.0, 0.0}}, 3);
}

// ESEMPIO 31. Stessa colonna ma righe diverse: sono posizioni diverse e NON si sommano, anche se nella
// lista ordinata (per riga, poi per colonna) stanno una accanto all'altra. Una matrice 3 x 3 con
// contributi solo nella colonna 1, e con valori non interi (cosi' si vede che i numeri restano double):
//     (2, 1, -2.5)   (0, 1, 5.5)   (1, 1, 7.25)   (1, 1, 3.125)
// La posizione (1, 1) riceve due contributi: 7.25 + 3.125 = 10.375. La matrice e'
//         [ 0   5.5     0 ]
//         [ 0   10.375  0 ]          3 posti occupati
//         [ 0  -2.5     0 ]
// (sono frazioni con denominatore 2, 4 o 8: esatte in binario, quindi si confronta senza tolleranza).
TEST(Triplets, SameColumnInDifferentRowsIsNotMerged) {
    const CSR A = csr_from_triplets(3, {{2, 1, -2.5}, {0, 1, 5.5}, {1, 1, 7.25}, {1, 1, 3.125}});
    expect_matrix(A, {{0.0, 5.5, 0.0}, {0.0, 10.375, 0.0}, {0.0, -2.5, 0.0}}, 3);

    // Il prodotto con x = (1, 2, 3) usa solo la colonna 1:  2 * (5.5, 10.375, -2.5) = (11, 20.75, -5).
    const std::vector<double> y = A * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 11.0);
    EXPECT_DOUBLE_EQ(y[1], 20.75);
    EXPECT_DOUBLE_EQ(y[2], -5.0);
}

// ESEMPIO 32. Righe senza contributi. Una matrice 4 x 4 con contributi solo nelle righe 0 e 3: le righe
// 1 e 2 sono vuote, in mezzo (qui row_ptr = [0, 2, 2, 2, 3]: una riga vuota non occupa posti, quindi
// row_ptr non cambia). Una con un solo contributo nella riga 0: le ultime tre righe sono vuote, in
// fondo (row_ptr = [0, 1, 1, 1, 1]). E una senza nessun contributo: la matrice nulla.
TEST(Triplets, EmptyRowsAndEmptyList) {
    const Dense zeros4(4, std::vector<double>(4, 0.0));

    const CSR A = csr_from_triplets(4, {{3, 3, 1.0}, {0, 1, 2.0}, {0, 0, 3.0}});
    Dense wantA = zeros4;
    wantA[0][0] = 3.0;
    wantA[0][1] = 2.0;
    wantA[3][3] = 1.0;
    expect_matrix(A, wantA, 3, "righe vuote in mezzo:");

    const CSR B = csr_from_triplets(4, {{0, 3, 5.0}});
    Dense wantB = zeros4;
    wantB[0][3] = 5.0;
    expect_matrix(B, wantB, 1, "righe vuote in fondo:");

    const CSR Z = csr_from_triplets(3, {});
    expect_matrix(Z, Dense(3, std::vector<double>(3, 0.0)), 0, "lista vuota:");
    const std::vector<double> y = Z * std::vector<double>{1.0, 2.0, 3.0};
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
}

// ESEMPIO 33. L'ordine della lista non conta. Una lista di 12 terne su una matrice 4 x 4, con 5
// posizioni doppie:
//     (0,0): 1 + 4 = 5     (0,1): -2 + 2 = 0     (1,0): 3 - 3 = 0     (1,1): 6
//     (2,3): 5 + 2 = 7     (3,2): 7              (3,3): -1 + 1 = 0
// cioe' la matrice [[5,0,0,0], [0,6,0,0], [0,0,0,7], [0,0,7,0]] con 7 posti occupati (tre sono zeri
// memorizzati). La stessa lista letta al contrario, e rimescolata 20 volte (sempre con lo stesso seme,
// quindi il test e' ripetibile), da' sempre la stessa matrice. I valori sono interi, cosi' ogni ordine
// di somma da' lo stesso numero esatto.
TEST(Triplets, InputOrderDoesNotMatter) {
    std::vector<Triplet> list = {{0, 0, 1.0}, {0, 1, -2.0}, {1, 0, 3.0}, {0, 0, 4.0}, {2, 3, 5.0}, {3, 3, -1.0},
                                 {1, 0, -3.0}, {2, 3, 2.0}, {3, 2, 7.0}, {0, 1, 2.0}, {1, 1, 6.0}, {3, 3, 1.0}};
    const Dense want = {{5.0, 0.0, 0.0, 0.0}, {0.0, 6.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 7.0}, {0.0, 0.0, 7.0, 0.0}};

    expect_matrix(csr_from_triplets(4, list), want, 7, "ordine dato:");

    std::reverse(list.begin(), list.end());
    expect_matrix(csr_from_triplets(4, list), want, 7, "al contrario:");

    std::mt19937 gen(2024);
    for (int k = 0; k < 20; ++k) {
        std::shuffle(list.begin(), list.end(), gen);
        expect_matrix(csr_from_triplets(4, list), want, 7, "rimescolata " + std::to_string(k) + ":");
    }
}

// ESEMPIO 34. Una catena di 3 sbarre su 4 nodi (0-1, 1-2, 2-3): e' l'assemblaggio 1D che conosci.
// Ogni sbarra a-b contribuisce con la matrice [[1, -1], [-1, 1]], cioe' con 4 terne:
// (a,a,1) (a,b,-1) (b,a,-1) (b,b,1). I nodi 1 e 2 appartengono a due sbarre, quindi le loro
// diagonali ricevono due contributi: 1 + 1 = 2. In tutto 3 * 4 = 12 terne, di cui 2 duplicate, cioe'
// 10 posti occupati:
//         [  1  -1   0   0 ]
//         [ -1   2  -1   0 ]
//         [  0  -1   2  -1 ]
//         [  0   0  -1   1 ]
// La matrice e' simmetrica e ogni riga somma a 0 (una temperatura uniforme non fa passare calore).
TEST(Triplets, AssemblingAChainOfBarsGivesTheTridiagonalMatrix) {
    std::vector<Triplet> list;
    for (std::size_t a = 0; a < 3; ++a) {
        const std::size_t b = a + 1;
        list.push_back({a, a, 1.0});
        list.push_back({a, b, -1.0});
        list.push_back({b, a, -1.0});
        list.push_back({b, b, 1.0});
    }
    ASSERT_EQ(list.size(), 12u);

    const CSR A = csr_from_triplets(4, list);
    expect_matrix(A,
                  {{1.0, -1.0, 0.0, 0.0}, {-1.0, 2.0, -1.0, 0.0}, {0.0, -1.0, 2.0, -1.0}, {0.0, 0.0, -1.0, 1.0}},
                  10);
    EXPECT_TRUE(A.is_symmetric(1e-14));

    const std::vector<double> y = A * std::vector<double>(4, 1.0);
    ASSERT_EQ(y.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(y[i], 0.0) << "riga " << i;
    }
}

// ESEMPIO 35. Una prova "a caso" contro il conto diretto. 1500 contributi su una matrice 20 x 20 (400
// posizioni, quindi quasi ogni posizione riceve piu' contributi, e con valori interi da -3 a 3 molte
// somme fanno 0). Per ogni posizione, il valore atteso e' la somma diretta dei contributi; i posti
// occupati sono tante quante le posizioni diverse che compaiono nella lista. Il seme e' fisso.
TEST(Triplets, MatchesDirectAccumulationOnRandomTriplets) {
    const std::size_t n = 20;
    std::mt19937 gen(12345);
    std::uniform_int_distribution<std::size_t> index(0, n - 1);
    std::uniform_int_distribution<int> value(-3, 3);

    std::vector<Triplet> list;
    Dense want(n, std::vector<double>(n, 0.0));
    std::set<std::pair<std::size_t, std::size_t>> used;
    for (int k = 0; k < 1500; ++k) {
        const std::size_t r = index(gen);
        const std::size_t c = index(gen);
        const double v = value(gen);
        list.push_back({r, c, v});
        want[r][c] += v;
        used.insert({r, c});
    }

    const CSR A = csr_from_triplets(n, list);
    expect_matrix(A, want, used.size());

    // Anche il prodotto con un vettore (interi: la somma e' esatta in ogni ordine).
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) - 7.0;
    }
    const std::vector<double> y = A * x;
    ASSERT_EQ(y.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        double expected = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            expected += want[i][j] * x[j];
        }
        EXPECT_DOUBLE_EQ(y[i], expected) << "riga " << i;
    }
}

// ESEMPIO 36. Cosa non e' ammesso: una matrice senza righe (n = 0) e una terna con riga o colonna >= n,
// da sola o in mezzo ad altre giuste (anche come prima o come ultima della lista). Le posizioni valide
// vanno da 0 a n - 1, ultima compresa.
TEST(Triplets, RejectsNonsense) {
    EXPECT_THROW(csr_from_triplets(0, {}), std::invalid_argument);
    EXPECT_THROW(csr_from_triplets(3, {{3, 0, 1.0}}), std::invalid_argument);                  // riga = n
    EXPECT_THROW(csr_from_triplets(3, {{0, 3, 1.0}}), std::invalid_argument);                  // colonna = n
    EXPECT_THROW(csr_from_triplets(3, {{0, 0, 1.0}, {1000, 1, 1.0}}), std::invalid_argument);  // riga molto grande, ultima
    EXPECT_THROW(csr_from_triplets(3, {{1, 1, 1.0}, {2, 1000, 1.0}}), std::invalid_argument);  // colonna molto grande, ultima
    EXPECT_THROW(csr_from_triplets(3, {{1000, 1, 1.0}, {0, 0, 1.0}}), std::invalid_argument);  // riga molto grande, prima
    EXPECT_THROW(csr_from_triplets(3, {{2, 1000, 1.0}, {1, 1, 1.0}}), std::invalid_argument);  // colonna molto grande, prima
    EXPECT_THROW(csr_from_triplets(3, {{0, 0, 1.0}, {1, 3, 1.0}, {2, 2, 1.0}}), std::invalid_argument);  // in mezzo
    EXPECT_NO_THROW(csr_from_triplets(3, {{2, 2, 1.0}}));                                      // l'ultima posizione valida
}
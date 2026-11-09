#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "Assembly2D.h"
#include "CSR_and_Vectors.h"
#include "Fem2D.h"
#include "Mesh2D.h"

// ---- Dalle matrici dei triangoli alla matrice di tutta la mesh: l'assemblaggio -------------------
//
// element_stiffness(mesh, e, k) da' la matrice 3 x 3 di UN triangolo, con gli indici LOCALI 0, 1, 2.
// La matrice di tutta la mesh, K, e' n x n, con n = numero dei nodi, e i suoi indici sono i NODI della
// mesh: K[I][J] e' il legame fra il nodo I e il nodo J. Si ottiene sommando i contributi di tutti i
// triangoli che hanno sia I sia J fra i loro nodi:
//
//     K[I][J] = somma, sui triangoli e con I e J fra i loro nodi, di Ke[i][j]
//               dove i e j sono le posizioni di I e J nella terna t = mesh.triangle(e)   (t[i] = I, t[j] = J)
//
// Quindi ogni triangolo produce 9 terne (riga, colonna, valore) = (t[i], t[j], Ke[i][j]): riga e colonna
// sono i nodi della mesh, non gli indici locali. Le terne di tutti i triangoli vanno a csr_from_triplets,
// che somma quelle con la stessa posizione. Due nodi che non stanno mai nello stesso triangolo non si
// influenzano: K[I][J] = 0 e non occupa memoria.
//
// assemble_stiffness_2d(mesh, k) restituisce K in formato CSR (n x n, n = mesh.num_nodes()). Non ci sono
// errori da segnalare: la mesh e' gia' stata controllata dal costruttore di Mesh2D.

namespace {

using Dense = std::vector<std::vector<double>>;

// Un rettangolo lx x ly diviso in nx x ny celle, come in rectangle_mesh.
struct Grid {
    double lx, ly;
    std::size_t nx, ny;
};

// Sei rettangoli di forme molto diverse.
std::vector<Grid> sample_grids() {
    return {{1.0, 1.0, 1, 1},     // il quadrato unitario: due triangoli
            {2.0, 1.0, 2, 1},     // le due celle dell'ESEMPIO 37
            {3.0, 1.5, 4, 3},     // celle 0.75 x 0.5
            {1.0, 1.0, 6, 6},     // 72 triangoli
            {0.2, 0.05, 5, 2},    // piccolo (20 cm per 5 cm, in metri)
            {10.0, 0.1, 1, 7}};   // molto schiacciato: celle 10 x 0.0143
}

std::size_t nodes_of(const Grid& g) { return (g.nx + 1) * (g.ny + 1); }

std::string describe(const Grid& g) {
    return "rettangolo " + std::to_string(g.lx) + " x " + std::to_string(g.ly) + " con " + std::to_string(g.nx) +
           " x " + std::to_string(g.ny) + " celle";
}

// Il triangolo grande dell'ESEMPIO 8 delle mesh, con il nodo 3 dentro (area 9/2). Non e' un rettangolo.
Mesh2D triangle_with_interior_node() {
    return Mesh2D({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});
}

// Un quadrilatero irregolare con un nodo dentro: 0 = (0, 0), 1 = (4, 0), 2 = (3, 3), 3 = (-1, 2) e, dentro,
// 4 = (1, 0.7). Quattro triangoli attorno al nodo 4, tutti di forma e area diverse: 1.4, 4.15, 3.6 e 1.35 (in
// tutto 10.5).
Mesh2D irregular_quadrilateral() {
    return Mesh2D({0.0, 4.0, 3.0, -1.0, 1.0}, {0.0, 0.0, 3.0, 2.0, 0.7},
                  {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}});
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

// Il piu' grande valore assoluto sulla diagonale di A: la "scala" della matrice, per fissare le tolleranze.
double largest_diagonal(const CSR& A) {
    double m = 0.0;
    for (double d : A.diag()) {
        m = std::max(m, std::fabs(d));
    }
    return m;
}

// Il campo di temperatura lineare T(x, y) = t0 + gx x + gy y, valutato nei nodi della mesh.
std::vector<double> linear_field(const Mesh2D& mesh, double t0, double gx, double gy) {
    std::vector<double> T;
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        T.push_back(t0 + gx * mesh.x(i) + gy * mesh.y(i));
    }
    return T;
}

// La somma delle aree di tutti i triangoli.
double total_area(const Mesh2D& mesh) {
    double a = 0.0;
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        a += mesh.area(e);
    }
    return a;
}

// Il "conto lento": K in una tabella piena n x n, in cui ogni triangolo somma a mano i suoi 9 numeri nelle
// caselle (t[i], t[j]). Serve solo come confronto, e solo per mesh piccole.
Dense slow_stiffness(const Mesh2D& mesh, double k) {
    Dense D(mesh.num_nodes(), std::vector<double>(mesh.num_nodes(), 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix3 Ke = element_stiffness(mesh, e, k);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                D[t[i]][t[j]] += Ke[i][j];
            }
        }
    }
    return D;
}

// Confronta A con una tabella piena, casella per casella.
void expect_same_as_dense(const CSR& A, const Dense& D) {
    ASSERT_EQ(A.size(), D.size());
    double scale = 0.0;
    for (const auto& row : D) {
        for (double v : row) {
            scale = std::max(scale, std::fabs(v));
        }
    }
    for (std::size_t i = 0; i < D.size(); ++i) {
        for (std::size_t j = 0; j < D.size(); ++j) {
            EXPECT_NEAR(A.get(i, j), D[i][j], 1e-13 * scale) << "elemento (" << i << ", " << j << ")";
        }
    }
}

// L'energia di un campo lineare e l'equilibrio nei nodi interni (ESEMPIO 39). Aggiunge a `interior` il
// numero di nodi interni controllati.
void check_linear_field(const Mesh2D& m, double k, std::size_t& interior) {
    const double t0 = 1.5;
    const double gx = 2.0;
    const double gy = -3.0;

    const CSR K = assemble_stiffness_2d(m, k);
    const std::vector<double> T = linear_field(m, t0, gx, gy);
    ASSERT_EQ(K.size(), T.size());
    const std::vector<double> KT = K * T;
    ASSERT_EQ(KT.size(), T.size());

    // T^T K T = k |grad T|^2 * area
    double energy = 0.0;
    double tmax = 0.0;
    for (std::size_t i = 0; i < T.size(); ++i) {
        energy += T[i] * KT[i];
        tmax = std::max(tmax, std::fabs(T[i]));
    }
    const double expected = k * (gx * gx + gy * gy) * total_area(m);
    EXPECT_NEAR(energy, expected, 1e-9 * expected);

    // (K T)[I] = 0 nei nodi interni
    std::vector<bool> on_boundary(m.num_nodes(), false);
    for (std::size_t node : boundary_nodes(m)) {
        on_boundary[node] = true;
    }
    const double tol = 1e-11 * largest_diagonal(K) * tmax;
    for (std::size_t i = 0; i < KT.size(); ++i) {
        if (!on_boundary[i]) {
            EXPECT_NEAR(KT[i], 0.0, tol) << "nodo interno " << i;
            ++interior;
        }
    }
}

}  // namespace

// ESEMPIO 37. Due celle quadrate 1 x 1 affiancate, rectangle_mesh(2, 1, 2, 1), con k = 2. I nodi, riga per
// riga, e i quattro triangoli (la diagonale di ogni cella va dal nodo in basso a sinistra a quello in alto a
// destra):
//
//         3 ----- 4 ----- 5            T0 = (0, 1, 4)   T1 = (0, 4, 3)      <- cella di sinistra
//         |     / |     / |            T2 = (1, 2, 5)   T3 = (1, 5, 4)      <- cella di destra
//         |   /   |   /   |
//         | /     | /     |
//         0 ----- 1 ----- 2
//
// Ogni triangolo ha area 1/2, quindi k / (4 A) = 1 e K[i][j] = b_i b_j + c_i c_j (la formula dell'ESEMPIO 11).
// I triangoli sono di due tipi: (a, b, c) = (basso-sinistra, basso-destra, alto-destra), per T0 e T2, e
// (a, c, d) = (basso-sinistra, alto-destra, alto-sinistra), per T1 e T3. Con i vertici (0,0), (1,0), (1,1) e
// (0,0), (1,1), (0,1):
//
//     (a, b, c):  b = (-1, 1, 0)  c = (0, -1, 1)          (a, c, d):  b = (0, 1, -1)  c = (-1, 0, 1)
//
//                 [ 1 -1  0 ]                                         [ 1  0 -1 ]
//                 [-1  2 -1 ]                                         [ 0  1 -1 ]
//                 [ 0 -1  1 ]                                         [-1 -1  2 ]
//
// Ora si somma, nodo per nodo. Il nodo 1 sta in T0 (come b: diagonale 2), in T2 (come a: 1) e in T3
// (come a: 1), quindi K[1][1] = 2 + 1 + 1 = 4. La coppia (1, 4) sta in T0 (come b-c: -1) e in T3 (come a-d: -1),
// quindi K[1][4] = -2. Con tutti i nodi:
//
//         [ 2 -1  0 -1  0  0 ]
//         [-1  4 -1  0 -2  0 ]
//     K = [ 0 -1  2  0  0 -1 ]
//         [-1  0  0  2 -1  0 ]
//         [ 0 -2  0 -1  4 -1 ]
//         [ 0  0 -1  0 -1  2 ]
//
// Ogni riga somma a 0. Occupa un posto ogni coppia di nodi che stanno in uno stesso triangolo (compresa la
// coppia di un nodo con se stesso), anche se la somma fa 0: le diagonali delle celle, 0-4 e 1-5, stanno in due
// triangoli ciascuna, con legame 0 in tutti e due (angolo retto), e occupano comunque un posto (come
// nell'ESEMPIO 30). Le coppie che non stanno in nessun triangolo (1-3, 2-4, 0-5...) valgono 0 e non occupano
// niente. In tutto 6 + 2 * 9 = 24 posti: 6 sulla diagonale e 9 coppie fuori diagonale
// (0-1, 0-3, 0-4, 1-2, 1-4, 1-5, 2-5, 3-4, 4-5), ciascuna in due posti.
TEST(Assembly2d, StiffnessOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const Dense want = {{2, -1, 0, -1, 0, 0},
                        {-1, 4, -1, 0, -2, 0},
                        {0, -1, 2, 0, 0, -1},
                        {-1, 0, 0, 2, -1, 0},
                        {0, -2, 0, -1, 4, -1},
                        {0, 0, -1, 0, -1, 2}};
    expect_matrix(assemble_stiffness_2d(m, 2.0), want, 24);
}

// ESEMPIO 38. Due cose che valgono per QUALUNQUE mesh, come per il singolo triangolo (ESEMPIO 15):
//   - K e' simmetrica;
//   - ogni riga somma a zero: se la temperatura e' la stessa in tutti i nodi non passa calore. In formule:
//     la riga I di K per il vettore (1, ..., 1) e' la somma dei contributi dei triangoli attorno a I, e ogni
//     riga di ogni Ke somma a zero.
// Qui sui sei rettangoli di sample_grids: celle quadrate, larghe, strette, piccole, molto schiacciate.
TEST(Assembly2d, StiffnessIsSymmetricAndAConstantTemperatureHasNoFlux) {
    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        const CSR K = assemble_stiffness_2d(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), 2.5);
        ASSERT_EQ(K.size(), nodes_of(g));
        const double tol = 1e-12 * largest_diagonal(K);

        EXPECT_TRUE(K.is_symmetric(tol));

        const std::vector<double> flux = K * std::vector<double>(K.size(), 1.0);
        ASSERT_EQ(flux.size(), K.size());
        for (std::size_t i = 0; i < flux.size(); ++i) {
            EXPECT_NEAR(flux[i], 0.0, tol) << "riga " << i;
        }
    }
}

// ESEMPIO 39. L'energia di un campo di temperatura LINEARE T(x, y) = t0 + gx x + gy y su tutta la mesh: il
// gradiente e' costante (gx, gy) dappertutto, quindi
//     T^T K T = integrale di k |grad T|^2 = k (gx^2 + gy^2) * (area della mesh),
// la somma delle aree dei triangoli. E' lo stesso controllo dell'ESEMPIO 16, ora sulla matrice assemblata
// (T_I e' la temperatura del nodo I).
//
// C'e' di piu'. La riga I di K per T e'
//     (K T)[I] = integrale di k grad T . grad N_I,    sui triangoli che contengono I,
// dove N_I e' la funzione cappello del nodo I: vale 1 in I e scende a 0 sul bordo del pezzo di mesh attorno a I.
// Con T lineare, grad T = (gx, gy) e' costante e si porta fuori dall'integrale:
//     (K T)[I] = k (gx, gy) . integrale di grad N_I.
// Se I e' INTERNO (non di bordo, secondo boundary_nodes) quell'integrale e' 0: N_I vale 0 su tutto il bordo del
// pezzo di mesh, e per il teorema della divergenza l'integrale del gradiente e' l'integrale sul bordo di N_I per
// la normale, cioe' 0. Quindi in ogni nodo interno il bilancio del calore torna a 0, per ogni campo lineare e
// su QUALUNQUE mesh, anche irregolare. Nei nodi di bordo no: li' passa il calore che entra ed esce.
// Questo e' il "patch test", il controllo classico degli elementi finiti, e non usa nessun valore di K
// calcolato a mano. Qui sui sei rettangoli, sul triangolo con un nodo dentro (ESEMPIO 8 delle mesh, area 9/2,
// quindi l'energia e' 2.5 * 13 * 9/2 = 146.25) e su un quadrilatero irregolare con un nodo dentro, fatto di
// quattro triangoli tutti diversi (area 10.5).
TEST(Assembly2d, StiffnessGivesTheEnergyOfALinearFieldAndBalancesTheInteriorNodes) {
    const double k = 2.5;
    std::size_t interior = 0;

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        check_linear_field(rectangle_mesh(g.lx, g.ly, g.nx, g.ny), k, interior);
    }
    {
        SCOPED_TRACE("triangolo con un nodo dentro");
        check_linear_field(triangle_with_interior_node(), k, interior);
    }
    {
        SCOPED_TRACE("quadrilatero irregolare con un nodo dentro");
        check_linear_field(irregular_quadrilateral(), k, interior);
    }
    EXPECT_GT(interior, 30u) << "il controllo dei nodi interni non deve essere vuoto";
}

// ESEMPIO 40. Lo stesso K calcolato per un'altra strada, lenta e semplice: una tabella piena n x n, in cui
// ogni triangolo somma a mano i suoi 9 numeri nelle caselle (t[i], t[j]). Per mesh piccole si puo' fare;
// per quelle grandi la tabella piena non entra in memoria (ESEMPIO 43), ed e' il motivo del formato CSR. I
// due metodi devono dare la stessa matrice in ogni casella, con coefficienti non banali (k = 3.5, 0.7 e 1.3)
// e su tre mesh diverse, l'ultima fatta di triangoli tutti diversi.
TEST(Assembly2d, StiffnessMatchesTheSlowDenseAssembly) {
    {
        SCOPED_TRACE("rettangolo 3 x 1.5 con 4 x 3 celle, k = 3.5");
        const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
        expect_same_as_dense(assemble_stiffness_2d(m, 3.5), slow_stiffness(m, 3.5));
    }
    {
        SCOPED_TRACE("triangolo con un nodo dentro, k = 0.7");
        const Mesh2D m = triangle_with_interior_node();
        expect_same_as_dense(assemble_stiffness_2d(m, 0.7), slow_stiffness(m, 0.7));
    }
    {
        SCOPED_TRACE("quadrilatero irregolare con un nodo dentro, k = 1.3");
        const Mesh2D m = irregular_quadrilateral();
        expect_same_as_dense(assemble_stiffness_2d(m, 1.3), slow_stiffness(m, 1.3));
    }
}

// ESEMPIO 41. Quanti posti occupa K. Il conto e' sulla mesh, non sui numeri: un posto per ogni coppia di nodi
// che stanno in uno stesso triangolo, e due nodi stanno in uno stesso triangolo se e solo se li unisce un
// LATO della mesh. Quindi
//     posti = (un posto per nodo, sulla diagonale) + 2 * (lati)       perche' ogni lato occupa (I, J) e (J, I).
// In un rettangolo di nx x ny celle i nodi sono (nx + 1)(ny + 1) e i lati sono
//     orizzontali nx (ny + 1)  +  verticali (nx + 1) ny  +  diagonali nx ny.
// Per nx = 2, ny = 1: 6 nodi, 4 + 3 + 2 = 9 lati, 6 + 18 = 24 posti, come nell'ESEMPIO 37. Non dipende da k,
// ne' da quanti elementi risultano 0.
TEST(Assembly2d, StiffnessOccupiesOnePlacePerNodeAndTwoPerEdge) {
    const std::size_t sizes[][2] = {{1, 1}, {2, 1}, {1, 2}, {3, 2}, {5, 4}, {1, 9}, {7, 1}};

    for (const auto& s : sizes) {
        const std::size_t nx = s[0];
        const std::size_t ny = s[1];
        const std::size_t nodes = (nx + 1) * (ny + 1);
        const std::size_t edges = nx * (ny + 1) + (nx + 1) * ny + nx * ny;

        for (double k : {1.0, 7.5}) {
            const CSR K = assemble_stiffness_2d(rectangle_mesh(1.0, 1.0, nx, ny), k);
            EXPECT_EQ(K.get_val_size(), nodes + 2 * edges) << nx << " x " << ny << " celle, k = " << k;
        }
    }
}

// ESEMPIO 42. Una mesh in cui due nodi non appartengono a nessun triangolo: 0 = (5, 5) e 5 = (7, 7). I nodi
// 1..4 sono il quadrato unitario con i due triangoli (1, 2, 3) e (1, 3, 4). La matrice ha comunque una riga e
// una colonna per OGNI nodo della mesh (6 x 6), non solo per quelli usati dai triangoli: le righe 0 e 5 sono
// vuote. La dimensione e' mesh.num_nodes(), non si ricava dai triangoli (il nodo piu' grande che usano e' il
// 4, e 4 + 1 farebbe 5). Con k = 2 il blocco dei nodi 1..4 e' il quadrato unitario, 14 posti in tutto:
//
//         [ 0  0  0  0  0  0 ]
//         [ 0  2 -1  0 -1  0 ]
//         [ 0 -1  2 -1  0  0 ]
//     K = [ 0  0 -1  2 -1  0 ]
//         [ 0 -1  0 -1  2  0 ]
//         [ 0  0  0  0  0  0 ]
//
// (i posti 1-3 e 3-1 sono la diagonale della cella: occupati, con valore 0).
TEST(Assembly2d, NodesOutsideEveryTriangleGetEmptyRowsAndColumns) {
    const Mesh2D m({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
    ASSERT_EQ(m.num_nodes(), 6u);

    const Dense want = {{0, 0, 0, 0, 0, 0},
                        {0, 2, -1, 0, -1, 0},
                        {0, -1, 2, -1, 0, 0},
                        {0, 0, -1, 2, -1, 0},
                        {0, -1, 0, -1, 2, 0},
                        {0, 0, 0, 0, 0, 0}};
    expect_matrix(assemble_stiffness_2d(m, 2.0), want, 14);
}

// ESEMPIO 43. Una mesh di 150 x 150 celle sul quadrato unitario: 22801 nodi e 45000 triangoli. Una tabella
// piena avrebbe 22801^2 = 520 milioni di numeri (circa 4 GB); in formato CSR ne bastano 158401 (22801 sulla
// diagonale e 2 * 67800 lati). Se l'assemblaggio costruisse una tabella piena, o fosse quadratico nel numero
// di nodi, questo test non finirebbe. Controlli: i posti (la formula dell'ESEMPIO 41), un campo costante (ogni
// riga somma a 0) e il campo T = x, la cui energia vale k * 1 * 1 = 1 qualunque sia la finezza della mesh.
TEST(Assembly2d, AssemblyOfALargeMeshStaysSparse) {
    const std::size_t n = 150;
    const Mesh2D m = rectangle_mesh(1.0, 1.0, n, n);
    const CSR K = assemble_stiffness_2d(m, 1.0);

    const std::size_t nodes = (n + 1) * (n + 1);
    const std::size_t edges = n * (n + 1) + (n + 1) * n + n * n;
    ASSERT_EQ(K.size(), nodes);
    EXPECT_EQ(K.get_val_size(), nodes + 2 * edges);

    const std::vector<double> flux = K * std::vector<double>(nodes, 1.0);
    for (std::size_t i = 0; i < nodes; ++i) {
        ASSERT_NEAR(flux[i], 0.0, 1e-12) << "riga " << i;
    }

    std::vector<double> T;
    for (std::size_t i = 0; i < nodes; ++i) {
        T.push_back(m.x(i));
    }
    const std::vector<double> KT = K * T;
    double energy = 0.0;
    for (std::size_t i = 0; i < nodes; ++i) {
        energy += T[i] * KT[i];
    }
    EXPECT_NEAR(energy, 1.0, 1e-9);
}

// ---- Massa e carico di tutta la mesh ------------------------------------------------------------
//
// Stessa costruzione di K, con i contributi di altri elementi:
//   - assemble_mass_2d(mesh, rho_c) e' la matrice di massa M (n x n, CSR), con i contributi di element_mass.
//     Occupa gli stessi posti di K: dipendono solo dalla mesh.
//   - assemble_load_2d(mesh, q) e' il vettore dei carichi f (n numeri), con i contributi di element_load.
//     Qui non servono le terne: f e' un vettore, e il numero fe[i] del triangolo e va semplicemente sommato
//     alla casella t[i]. Un nodo che non sta in nessun triangolo riceve 0.

namespace {

// Come slow_stiffness, per la massa.
Dense slow_mass(const Mesh2D& mesh, double rho_c) {
    Dense D(mesh.num_nodes(), std::vector<double>(mesh.num_nodes(), 0.0));
    for (std::size_t e = 0; e < mesh.num_triangles(); ++e) {
        const Triangle& t = mesh.triangle(e);
        const Matrix3 Me = element_mass(mesh, e, rho_c);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                D[t[i]][t[j]] += Me[i][j];
            }
        }
    }
    return D;
}

// Le coordinate x (o y) dei nodi, come vettore.
std::vector<double> coordinate(const Mesh2D& mesh, bool x) {
    std::vector<double> c;
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        c.push_back(x ? mesh.x(i) : mesh.y(i));
    }
    return c;
}

// u^T M v.
double bilinear(const CSR& M, const std::vector<double>& u, const std::vector<double>& v) {
    const std::vector<double> Mv = M * v;
    double s = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        s += u[i] * Mv[i];
    }
    return s;
}

// La mesh dell'ESEMPIO 13: due triangoli di area DIVERSA. 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4);
// T0 = (0, 1, 2) con A = 4 e T1 = (1, 3, 2) con A = 8.
Mesh2D two_triangles_of_different_area() {
    return Mesh2D({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
}

}  // namespace

// ESEMPIO 44. Le stesse due celle dell'ESEMPIO 37, con rho_c = 12. Ogni triangolo ha area 1/2, quindi
// rho_c A / 6 = 1 e rho_c A / 12 = 1/2, e la matrice di OGNI triangolo e'
//         [  1  1/2 1/2 ]
//         [ 1/2  1  1/2 ]
//         [ 1/2 1/2  1  ]
// La somma e' piu' facile di quella di K, perche' i triangoli sono tutti uguali: M[I][I] vale 1 per ogni triangolo
// che contiene I, e M[I][J] vale 1/2 per ogni triangolo che contiene sia I sia J.
//     triangoli di ogni nodo:  0: T0 T1 (2)   1: T0 T2 T3 (3)   2: T2 (1)   3: T1 (1)   4: T0 T1 T3 (3)   5: T2 T3 (2)
//     coppie in DUE triangoli: 0-4 (T0 T1), 1-4 (T0 T3), 1-5 (T2 T3), quindi 1;
//     le altre sei coppie (0-1, 0-3, 1-2, 2-5, 3-4, 4-5) sono in un triangolo solo, quindi 1/2.
//
//         [  2  1/2  0  1/2  1   0  ]
//         [ 1/2  3  1/2  0   1   1  ]
//     M = [  0  1/2  1   0   0  1/2 ]          24 posti, gli stessi di K
//         [ 1/2  0   0   1  1/2  0  ]
//         [  1   1   0  1/2  3  1/2 ]
//         [  0   1  1/2  0  1/2  2  ]
//
// Le righe sommano a 4, 6, 2, 2, 6, 4: per ogni nodo, rho_c * (aree dei suoi triangoli) / 3 = 12 * (n/2) / 3 = 2 n,
// con n il numero dei suoi triangoli. In tutto fa 24 = rho_c * area della mesh = 12 * 2.
TEST(Assembly2d, MassOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const Dense want = {{2, 0.5, 0, 0.5, 1, 0},
                        {0.5, 3, 0.5, 0, 1, 1},
                        {0, 0.5, 1, 0, 0, 0.5},
                        {0.5, 0, 0, 1, 0.5, 0},
                        {1, 1, 0, 0.5, 3, 0.5},
                        {0, 1, 0.5, 0, 0.5, 2}};
    expect_matrix(assemble_mass_2d(m, 12.0), want, 24);
}

// ESEMPIO 45. Due triangoli di AREA DIVERSA, la mesh dell'ESEMPIO 13: nodi 0 = (0, 0), 1 = (4, 0), 2 = (1, 2),
// 3 = (6, 4); T0 = (0, 1, 2) con A = 4 e T1 = (1, 3, 2) con A = 8. Nelle mesh dei rettangoli tutti i triangoli
// sono uguali e una confusione fra un triangolo e l'altro non si vede: qui si' (la massa e il carico di un
// triangolo dipendono solo dalla sua area).
//
// Con rho_c = 6:    T0: rho_c A / 6 = 4 e rho_c A / 12 = 2        T1: rho_c A / 6 = 8 e rho_c A / 12 = 4
// Il nodo 0 sta solo in T0, il nodo 3 solo in T1, i nodi 1 e 2 in tutti e due (diagonale 4 + 8, fuori diagonale
// 2 + 4). I nodi 0 e 3 non stanno mai insieme in un triangolo: 0 e nessun posto.
//
//         [ 4   2   2   0 ]
//     M = [ 2  12   6   4 ]            14 posti
//         [ 2   6  12   4 ]
//         [ 0   4   4   8 ]
//
// Con q = 3 ogni triangolo da' q A / 3 a ciascuno dei suoi nodi: T0 da' 3 * 4 / 3 = 4, T1 da' 3 * 8 / 3 = 8. Quindi
//     f = [4, 4 + 8, 4 + 8, 8] = [4, 12, 12, 8]       (in tutto 36 = q * (4 + 8))
TEST(Assembly2d, MassAndLoadOfTrianglesOfDifferentAreaMatchHandComputation) {
    const Mesh2D m = two_triangles_of_different_area();

    const Dense want = {{4, 2, 2, 0}, {2, 12, 6, 4}, {2, 6, 12, 4}, {0, 4, 4, 8}};
    expect_matrix(assemble_mass_2d(m, 6.0), want, 14);

    const std::vector<double> f = assemble_load_2d(m, 3.0);
    const std::vector<double> want_f = {4.0, 12.0, 12.0, 8.0};
    ASSERT_EQ(f.size(), want_f.size());
    for (std::size_t i = 0; i < want_f.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want_f[i]) << "nodo " << i;
    }
}

// ESEMPIO 46. Cose che valgono per QUALUNQUE rettangolo (qui rho_c = 2.5, sui sei di sample_grids):
//   - M e' simmetrica, e ogni riga somma a un numero positivo (la "massa" concentrata nel nodo);
//   - tutti gli elementi insieme sommano a rho_c * (area della mesh), la capacita termica totale
//     (1^T M 1 = integrale di rho_c);
//   - per due campi LINEARI a(x, y) e b(x, y), con i loro valori nei nodi, a^T M b e' l'integrale ESATTO di
//     rho_c a b: il prodotto ha grado 2, e gli elementi lineari lo integrano esattamente (ESEMPIO 24). Su un
//     rettangolo [0, Lx] x [0, Ly]:
//         integrale di x = Lx^2 Ly / 2     integrale di x^2 = Lx^3 Ly / 3     integrale di x y = Lx^2 Ly^2 / 4
//     Quindi  x^T M 1,  x^T M x  e  x^T M y  devono dare questi numeri, per rho_c, con QUALUNQUE numero di
//     celle: non sono approssimazioni che migliorano con la mesh, sono esatti.
TEST(Assembly2d, MassIsSymmetricPositiveAndIntegratesLinearFieldsExactly) {
    const double rho_c = 2.5;

    for (const Grid& g : sample_grids()) {
        SCOPED_TRACE(describe(g));
        const Mesh2D m = rectangle_mesh(g.lx, g.ly, g.nx, g.ny);
        const CSR M = assemble_mass_2d(m, rho_c);
        ASSERT_EQ(M.size(), nodes_of(g));

        const std::vector<double> one(M.size(), 1.0);
        const std::vector<double> x = coordinate(m, true);
        const std::vector<double> y = coordinate(m, false);
        const double area = g.lx * g.ly;
        const double tol = 1e-12 * rho_c * area;

        EXPECT_TRUE(M.is_symmetric(1e-12 * largest_diagonal(M)));
        const std::vector<double> lumped = M * one;
        for (std::size_t i = 0; i < lumped.size(); ++i) {
            EXPECT_GT(lumped[i], 0.0) << "riga " << i;
        }

        EXPECT_NEAR(bilinear(M, one, one), rho_c * area, tol);
        EXPECT_NEAR(bilinear(M, x, one), rho_c * g.lx * g.lx * g.ly / 2.0, tol * g.lx);
        EXPECT_NEAR(bilinear(M, x, x), rho_c * g.lx * g.lx * g.lx * g.ly / 3.0, tol * g.lx * g.lx);
        EXPECT_NEAR(bilinear(M, x, y), rho_c * g.lx * g.lx * g.ly * g.ly / 4.0, tol * g.lx * g.ly);
    }
}

// ESEMPIO 47. Come l'ESEMPIO 40, per la massa: lo stesso M col conto lento (tabella piena), su tre mesh
// (rho_c = 3.5, 0.7 e 1.3), l'ultima fatta di triangoli tutti di area diversa. Inoltre la massa occupa gli stessi
// posti della rigidezza: dipendono dalla mesh, non dall'elemento.
TEST(Assembly2d, MassMatchesTheSlowDenseAssemblyAndHasTheSamePlacesAsStiffness) {
    {
        SCOPED_TRACE("rettangolo 3 x 1.5 con 4 x 3 celle, rho_c = 3.5");
        const Mesh2D m = rectangle_mesh(3.0, 1.5, 4, 3);
        const CSR M = assemble_mass_2d(m, 3.5);
        expect_same_as_dense(M, slow_mass(m, 3.5));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
    {
        SCOPED_TRACE("triangolo con un nodo dentro, rho_c = 0.7");
        const Mesh2D m = triangle_with_interior_node();
        const CSR M = assemble_mass_2d(m, 0.7);
        expect_same_as_dense(M, slow_mass(m, 0.7));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
    {
        SCOPED_TRACE("quadrilatero irregolare con un nodo dentro, rho_c = 1.3");
        const Mesh2D m = irregular_quadrilateral();
        const CSR M = assemble_mass_2d(m, 1.3);
        expect_same_as_dense(M, slow_mass(m, 1.3));
        EXPECT_EQ(M.get_val_size(), assemble_stiffness_2d(m, 1.0).get_val_size());
    }
}

// ESEMPIO 48. Il carico delle stesse due celle dell'ESEMPIO 37, con q = 6. Ogni triangolo ha area 1/2 e da'
// q A / 3 = 1 a ciascuno dei suoi tre nodi. Quindi f[I] e' il numero dei triangoli del nodo I (ESEMPIO 44):
//     f = [2, 3, 1, 1, 3, 2]
// e in tutto fa 12 = q * area della mesh = 6 * 2: il calore prodotto dalla sorgente in tutta la lastra.
TEST(Assembly2d, LoadOfTwoSquareCellsMatchesHandComputation) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    const std::vector<double> f = assemble_load_2d(m, 6.0);
    const std::vector<double> want = {2.0, 3.0, 1.0, 1.0, 3.0, 2.0};
    ASSERT_EQ(f.size(), want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want[i]) << "nodo " << i;
    }
}

// ESEMPIO 49. Due cose che valgono per QUALUNQUE mesh (qui q = 7, rho_c = 2.5):
//   - il calore totale: la somma di f e' q * (area della mesh);
//   - il carico e la massa sono legati, come per il triangolo (ESEMPIO 27): f[I] = q / rho_c * (riga I di M)
//     sommata, cioe' f = (q / rho_c) * (M per il vettore di tutti 1).
// Con q negativo (un pozzo di calore) tutto cambia segno. Sui sei rettangoli, sul triangolo con un nodo dentro,
// sul quadrilatero irregolare e sui due triangoli di area diversa.
TEST(Assembly2d, LoadAddsUpToTheTotalHeatAndIsTheRowSumOfTheMass) {
    const double rho_c = 2.5;

    std::vector<Mesh2D> meshes;
    for (const Grid& g : sample_grids()) {
        meshes.push_back(rectangle_mesh(g.lx, g.ly, g.nx, g.ny));
    }
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh numero " + std::to_string(index));
        const Mesh2D& m = meshes[index];
        const double area = total_area(m);
        const std::vector<double> lumped = assemble_mass_2d(m, rho_c) * std::vector<double>(m.num_nodes(), 1.0);

        for (double q : {7.0, -7.0}) {
            SCOPED_TRACE("q = " + std::to_string(q));
            const std::vector<double> f = assemble_load_2d(m, q);
            ASSERT_EQ(f.size(), m.num_nodes());
            const double tol = 1e-12 * std::fabs(q) * area;

            double total = 0.0;
            for (std::size_t i = 0; i < f.size(); ++i) {
                total += f[i];
                EXPECT_NEAR(f[i], q / rho_c * lumped[i], tol) << "nodo " << i;
            }
            EXPECT_NEAR(total, q * area, tol);
        }
    }
}

// ESEMPIO 50. La mesh dell'ESEMPIO 42, con i nodi 0 e 5 fuori da ogni triangolo. Con rho_c = 12 e q = 6, i
// due triangoli (1, 2, 3) e (1, 3, 4) hanno area 1/2, quindi la matrice di ogni triangolo e' quella dell'ESEMPIO
// 44 e il carico per nodo e' 1:
//         [ 0   0   0   0   0  0 ]
//         [ 0   2  1/2  1  1/2 0 ]
//         [ 0  1/2  1  1/2  0  0 ]                       f = [0, 2, 1, 2, 1, 0]
//     M = [ 0   1  1/2  2  1/2 0 ]       14 posti
//         [ 0  1/2  0  1/2  1  0 ]
//         [ 0   0   0   0   0  0 ]
// Le dimensioni sono quelle della mesh (6), i nodi isolati hanno riga vuota in M e carico 0.
TEST(Assembly2d, MassAndLoadKeepOneRowPerNodeEvenForNodesOutsideEveryTriangle) {
    const Mesh2D m({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
    ASSERT_EQ(m.num_nodes(), 6u);

    const Dense want = {{0, 0, 0, 0, 0, 0},
                        {0, 2, 0.5, 1, 0.5, 0},
                        {0, 0.5, 1, 0.5, 0, 0},
                        {0, 1, 0.5, 2, 0.5, 0},
                        {0, 0.5, 0, 0.5, 1, 0},
                        {0, 0, 0, 0, 0, 0}};
    expect_matrix(assemble_mass_2d(m, 12.0), want, 14);

    const std::vector<double> f = assemble_load_2d(m, 6.0);
    const std::vector<double> want_f = {0.0, 2.0, 1.0, 2.0, 1.0, 0.0};
    ASSERT_EQ(f.size(), want_f.size());
    for (std::size_t i = 0; i < want_f.size(); ++i) {
        EXPECT_DOUBLE_EQ(f[i], want_f[i]) << "nodo " << i;
    }
}

// ---- Le terne prima di sommarle -------------------------------------------------------------------
//
// assemble_stiffness_2d e assemble_mass_2d restituiscono una CSR, e una CSR non si modifica: non offre nessun
// modo di cambiare una riga. Le condizioni di Dirichlet (passo successivo) cambiano righe e colonne di K, quindi
// vanno applicate PRIMA, sulla lista delle terne (riga, colonna, valore) da cui nasce la CSR. Per questo le due
// liste diventano pubbliche:
//   - stiffness_triplets_2d(mesh, k): le terne di K;
//   - mass_triplets_2d(mesh, rho_c): le terne di M.
// Sono i contributi dei triangoli, ancora da sommare: 9 terne per triangolo, (t[i], t[j], Ke[i][j]) con i e j da
// 0 a 2, in un ordine qualunque. Le terne con la stessa posizione NON sono sommate (lo fa csr_from_triplets) e
// quelle con valore 0 NON sono tolte. Per chi le usa, assemble_stiffness_2d e assemble_mass_2d non cambiano: sono
// la CSR di queste liste.

namespace {

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

// La mesh dell'ESEMPIO 42: i nodi 0 e 5 non stanno in nessun triangolo.
Mesh2D mesh_with_nodes_outside_every_triangle() {
    return Mesh2D({5.0, 0.0, 1.0, 1.0, 0.0, 7.0}, {5.0, 0.0, 0.0, 1.0, 1.0, 7.0}, {{1, 2, 3}, {1, 3, 4}});
}

}  // namespace

// ESEMPIO 51. Le terne di K per le due celle dell'ESEMPIO 37 (k = 2). Ogni triangolo ne da' 9 e i triangoli sono
// 4, quindi 36 terne, ma le posizioni diverse sono solo 24 (i 24 posti di K): le altre 12 terne cadono su posti
// gia' presi da un altro triangolo. Con le matrici dei triangoli dell'ESEMPIO 37:
//     (1, 1): T0 da' 2, T2 da' 1 e T3 da' 1       2 + 1 + 1 = 4 = K[1][1]
//     (1, 4): T0 da' -1 e T3 da' -1               -2 = K[1][4], e lo stesso in (4, 1)
//     (0, 4): T0 e T1 danno 0 tutti e due         il posto c'e' lo stesso e le terne vanno conservate
//     (2, 2): solo T2, che da' 2
//     (1, 3): nessuna (1 e 3 non stanno mai nello stesso triangolo)
// Sommate con csr_from_triplets, le 36 terne danno la K dell'ESEMPIO 37.
TEST(Assembly2d, StiffnessTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = stiffness_triplets_2d(m, 2.0);

    ASSERT_EQ(trip.size(), 36u);
    EXPECT_EQ(distinct_places(trip, 6), 24u);
    expect_values_at(trip, 1, 1, {1, 1, 2});
    expect_values_at(trip, 1, 4, {-1, -1});
    expect_values_at(trip, 4, 1, {-1, -1});
    expect_values_at(trip, 0, 4, {0, 0});
    expect_values_at(trip, 2, 2, {2});
    expect_values_at(trip, 1, 3, {});

    const Dense want = {{2, -1, 0, -1, 0, 0},
                        {-1, 4, -1, 0, -2, 0},
                        {0, -1, 2, 0, 0, -1},
                        {-1, 0, 0, 2, -1, 0},
                        {0, -2, 0, -1, 4, -1},
                        {0, 0, -1, 0, -1, 2}};
    expect_matrix(csr_from_triplets(m.num_nodes(), trip), want, 24);
}

// ESEMPIO 52. Lo stesso per la massa, con rho_c = 12 (ESEMPIO 44): 36 terne in 24 posizioni diverse. Ogni
// triangolo ha la matrice [1 1/2 1/2; 1/2 1 1/2; 1/2 1/2 1], quindi
//     (1, 1): T0, T2 e T3 danno 1 ciascuno         1 + 1 + 1 = 3 = M[1][1]
//     (0, 4): T0 e T1 danno 1/2 ciascuno           1 = M[0][4]
//     (2, 5): solo T2, che da' 1/2
//     (1, 3): nessuna
// e sommate danno la M dell'ESEMPIO 44.
TEST(Assembly2d, MassTripletsAreTheContributionsOfEveryTriangle) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);
    const std::vector<Triplet> trip = mass_triplets_2d(m, 12.0);

    ASSERT_EQ(trip.size(), 36u);
    EXPECT_EQ(distinct_places(trip, 6), 24u);
    expect_values_at(trip, 1, 1, {1, 1, 1});
    expect_values_at(trip, 0, 4, {0.5, 0.5});
    expect_values_at(trip, 4, 0, {0.5, 0.5});
    expect_values_at(trip, 2, 5, {0.5});
    expect_values_at(trip, 1, 3, {});

    const Dense want = {{2, 0.5, 0, 0.5, 1, 0},
                        {0.5, 3, 0.5, 0, 1, 1},
                        {0, 0.5, 1, 0, 0, 0.5},
                        {0.5, 0, 0, 1, 0.5, 0},
                        {1, 1, 0, 0.5, 3, 0.5},
                        {0, 1, 0.5, 0, 0.5, 2}};
    expect_matrix(csr_from_triplets(m.num_nodes(), trip), want, 24);
}

// ESEMPIO 53. Le due liste su mesh di forme diverse, confrontate con il conto lento dell'ESEMPIO 40 (la tabella
// piena in cui ogni triangolo somma a mano i suoi 9 numeri):
//   - 9 terne per triangolo;
//   - tutti gli indici minori del numero dei nodi (altrimenti csr_from_triplets lancia std::invalid_argument);
//   - sommate, danno la stessa K (e la stessa M) del conto lento.
// Le mesh: un rettangolo, il triangolo con un nodo dentro, il quadrilatero irregolare, due triangoli di area
// diversa e la mesh con due nodi fuori da ogni triangolo.
TEST(Assembly2d, TripletsSumToTheSlowDenseAssemblyOnEveryKindOfMesh) {
    std::vector<Mesh2D> meshes;
    meshes.push_back(rectangle_mesh(3.0, 1.5, 4, 3));
    meshes.push_back(triangle_with_interior_node());
    meshes.push_back(irregular_quadrilateral());
    meshes.push_back(two_triangles_of_different_area());
    meshes.push_back(mesh_with_nodes_outside_every_triangle());

    for (std::size_t index = 0; index < meshes.size(); ++index) {
        SCOPED_TRACE("mesh numero " + std::to_string(index));
        const Mesh2D& m = meshes[index];

        const std::vector<Triplet> stiff = stiffness_triplets_2d(m, 3.5);
        ASSERT_EQ(stiff.size(), 9 * m.num_triangles());
        expect_same_as_dense(csr_from_triplets(m.num_nodes(), stiff), slow_stiffness(m, 3.5));

        const std::vector<Triplet> mass = mass_triplets_2d(m, 0.7);
        ASSERT_EQ(mass.size(), 9 * m.num_triangles());
        expect_same_as_dense(csr_from_triplets(m.num_nodes(), mass), slow_mass(m, 0.7));
    }
}
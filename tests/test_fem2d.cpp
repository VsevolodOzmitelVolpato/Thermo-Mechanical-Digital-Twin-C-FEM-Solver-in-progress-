#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Fem2D.h"
#include "Mesh2D.h"

// ---- Matrice di rigidezza di un triangolo: element_stiffness ------------------------------------
//
// element_stiffness(mesh, e, k) e' la matrice 3 x 3 del triangolo e, per un materiale di
// conducibilita k (lastra di spessore 1). K[i][j], con i e j = 0, 1, 2 i tre nodi del triangolo
// nell'ordine di mesh.triangle(e), vale
//
//     K[i][j] = integrale sul triangolo di  k grad(N_i) . grad(N_j)  dA
//
// dove N_i e' la funzione lineare che vale 1 nel nodo i e 0 negli altri due. Una funzione lineare ha
// il gradiente costante, quindi l'integrale e' semplicemente  k * (grad N_i . grad N_j) * A.
//
// Il gradiente: N_i vale 0 su tutto il lato opposto al nodo i, quindi grad(N_i) e' perpendicolare a
// quel lato, e vale 1 / (altezza da i). In coordinate, con p = (i+1) % 3 e q = (i+2) % 3 (i due nodi
// che seguono i, in cerchio):
//
//     b_i = y_p - y_q        c_i = x_q - x_p        grad(N_i) = (b_i, c_i) / (2 A)
//     K[i][j] = k / (4 A) * (b_i b_j + c_i c_j)

namespace {

// I vertici di un triangolo, in senso antiorario.
struct Tri {
    double x0, y0, x1, y1, x2, y2;
};

// Una mesh con quel solo triangolo (nodi 0, 1, 2 in quest'ordine).
Mesh2D mesh_of(const Tri& t) {
    return Mesh2D({t.x0, t.x1, t.x2}, {t.y0, t.y1, t.y2}, {{0, 1, 2}});
}

// Il triangolo t ingrandito di s, ruotato di theta (radianti) attorno all'origine e spostato di (dx, dy).
Tri moved(const Tri& t, double s, double theta, double dx, double dy) {
    const double c = std::cos(theta);
    const double n = std::sin(theta);
    Tri r;
    r.x0 = s * (c * t.x0 - n * t.y0) + dx;
    r.y0 = s * (n * t.x0 + c * t.y0) + dy;
    r.x1 = s * (c * t.x1 - n * t.y1) + dx;
    r.y1 = s * (n * t.x1 + c * t.y1) + dy;
    r.x2 = s * (c * t.x2 - n * t.y2) + dx;
    r.y2 = s * (n * t.x2 + c * t.y2) + dy;
    return r;
}

// Sei triangoli antiorari, di forme e dimensioni molto diverse.
std::vector<Tri> sample_triangles() {
    return {{0.0, 0.0, 1.0, 0.0, 0.0, 1.0},         // rettangolo (ESEMPIO 11)
            {0.0, 0.0, 4.0, 0.0, 1.0, 2.0},         // scaleno (ESEMPIO 12)
            {1.0, 1.0, 5.0, 2.0, 2.0, 4.0},         // scaleno, lontano dall'origine
            {-1.5, 2.0, 0.5, -1.0, 3.0, 1.5},       // con coordinate negative
            {0.0, 0.0, 10.0, 0.0, 5.0, 0.1},        // molto schiacciato
            {0.0, 0.0, 0.004, 0.0, 0.001, 0.002}};  // piccolo (millimetri)
}

// Il piu' grande degli elementi di K in valore assoluto (per fissare le tolleranze).
double largest(const Matrix3& K) {
    double m = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            m = std::max(m, std::fabs(v));
        }
    }
    return m;
}

// Confronta due matrici elemento per elemento.
void expect_matrix_near(const Matrix3& got, const Matrix3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(got[i][j], want[i][j], tol) << what << " elemento (" << i << ", " << j << ")";
        }
    }
}

// La matrice dell'ESEMPIO 12, calcolata a mano.
Matrix3 scalene_stiffness() {
    return {{{13.0 / 16, -1.0 / 16, -12.0 / 16},
             {-1.0 / 16, 5.0 / 16, -4.0 / 16},
             {-12.0 / 16, -4.0 / 16, 16.0 / 16}}};
}

}  // namespace

// ESEMPIO 11. Il triangolo con i nodi 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), a mano. Qui le N_i si vedono
// a occhio: N_1 = x, N_2 = y e N_0 = 1 - x - y (ognuna vale 1 nel proprio nodo e 0 negli altri due).
// Quindi, con area A = 1/2,
//     grad N_0 = (-1, -1)     grad N_1 = (1, 0)     grad N_2 = (0, 1)
// e K[i][j] = k A (grad N_i . grad N_j):
//     K[0][0] = k/2 * 2 = k      K[0][1] = k/2 * (-1) = -k/2      K[0][2] = -k/2
//     K[1][1] = k/2 * 1 = k/2    K[1][2] = k/2 * 0 = 0            K[2][2] = k/2
// (le altre tre per simmetria). Con le formule dei coefficienti si trova lo stesso: b = (-1, 1, 0) e
// c = (-1, 0, 1), quindi grad N_i = (b_i, c_i) / (2 A) = (b_i, c_i).
TEST(Fem2d, StiffnessOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    const Matrix3 k1 = {{{1.0, -0.5, -0.5}, {-0.5, 0.5, 0.0}, {-0.5, 0.0, 0.5}}};
    expect_matrix_near(element_stiffness(m, 0, 1.0), k1, 1e-14, "k = 1:");

    const Matrix3 k3 = {{{3.0, -1.5, -1.5}, {-1.5, 1.5, 0.0}, {-1.5, 0.0, 1.5}}};
    expect_matrix_near(element_stiffness(m, 0, 3.0), k3, 1e-14, "k = 3:");
}

// ESEMPIO 12. Un triangolo qualunque: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2). Area A = 1/2 [4 * 2 - 1 * 0] = 4.
// Per ogni nodo i, p e q sono i due nodi che seguono (in cerchio):
//     nodo 0:  b = y_1 - y_2 = 0 - 2 = -2     c = x_2 - x_1 = 1 - 4 = -3
//     nodo 1:  b = y_2 - y_0 = 2 - 0 =  2     c = x_0 - x_2 = 0 - 1 = -1
//     nodo 2:  b = y_0 - y_1 = 0 - 0 =  0     c = x_1 - x_0 = 4 - 0 =  4
// (le b sommano a 0 e anche le c: un controllo gratis). Con 4 A = 16, K[i][j] = (b_i b_j + c_i c_j) / 16:
//     K[0][0] = (4 + 9) / 16 = 13/16     K[0][1] = (-4 + 3) / 16 = -1/16     K[0][2] = (0 - 12) / 16 = -12/16
//     K[1][1] = (4 + 1) / 16 = 5/16      K[1][2] = (0 - 4) / 16 = -4/16
//     K[2][2] = (0 + 16) / 16 = 1
// Ogni riga somma a zero: 13 - 1 - 12 = 0, -1 + 5 - 4 = 0, -12 - 4 + 16 = 0.
TEST(Fem2d, StiffnessOfAScaleneTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 4.0, 0.0, 1.0, 2.0});
    expect_matrix_near(element_stiffness(m, 0, 1.0), scalene_stiffness(), 1e-14);
}

// ESEMPIO 13. Due triangoli diversi che condividono un lato: serve a controllare che si usino i nodi e
// l'area del triangolo richiesto (e non quelli del primo). Nodi: 0 = (0, 0), 1 = (4, 0), 2 = (1, 2),
// 3 = (6, 4). Triangoli: T0 = (0, 1, 2), quello dell'ESEMPIO 12 (A = 4), e T1 = (1, 3, 2) (A = 8).
// Per T1 i nodi locali 0, 1, 2 sono i nodi 1, 3, 2 della mesh, cioe' (4, 0), (6, 4), (1, 2):
//     nodo 0:  b = 4 - 2 =  2     c = 1 - 6 = -5
//     nodo 1:  b = 2 - 0 =  2     c = 4 - 1 =  3
//     nodo 2:  b = 0 - 4 = -4     c = 6 - 4 =  2
// Con 4 A = 32:
//     K[0][0] = (4 + 25) / 32 = 29/32     K[0][1] = (4 - 15) / 32 = -11/32     K[0][2] = (-8 - 10) / 32 = -18/32
//     K[1][1] = (4 + 9) / 32 = 13/32      K[1][2] = (-8 + 6) / 32 = -2/32
//     K[2][2] = (16 + 4) / 32 = 20/32
TEST(Fem2d, StiffnessUsesTheNodesAndTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});
    EXPECT_NEAR(m.area(0), 4.0, 1e-14);   // la mesh e' quella giusta
    EXPECT_NEAR(m.area(1), 8.0, 1e-14);

    expect_matrix_near(element_stiffness(m, 0, 1.0), scalene_stiffness(), 1e-14, "triangolo 0:");

    const Matrix3 want1 = {{{29.0 / 32, -11.0 / 32, -18.0 / 32},
                            {-11.0 / 32, 13.0 / 32, -2.0 / 32},
                            {-18.0 / 32, -2.0 / 32, 20.0 / 32}}};
    expect_matrix_near(element_stiffness(m, 1, 1.0), want1, 1e-14, "triangolo 1:");
}

// ESEMPIO 14. Una cella larga 2 e alta 1, divisa in due triangoli da rectangle_mesh(2, 1, 1, 1): nodi
// 0 = (0, 0), 1 = (2, 0), 2 = (0, 1), 3 = (2, 1); T0 = (0, 1, 3) e T1 = (0, 3, 2), tutti e due di area 1.
// Stesso calcolo di prima, con 4 A = 4:
//     T0:  b = (-1, 1, 0)  c = (0, -2, 2)         T1:  b = (0, 1, -1)  c = (-2, 0, 2)
//
//          T0:   1/4  -1/4    0                        T1:    1     0    -1
//               -1/4   5/4   -1                               0    1/4  -1/4
//                 0    -1     1                              -1   -1/4   5/4
//
// Il legame fra due nodi e' forte lungo il lato corto (-1) e debole lungo quello lungo (-1/4).
TEST(Fem2d, StiffnessOfTheTwoTrianglesOfAWideCell) {
    const Mesh2D m = rectangle_mesh(2.0, 1.0, 1, 1);

    const Matrix3 t0 = {{{0.25, -0.25, 0.0}, {-0.25, 1.25, -1.0}, {0.0, -1.0, 1.0}}};
    const Matrix3 t1 = {{{1.0, 0.0, -1.0}, {0.0, 0.25, -0.25}, {-1.0, -0.25, 1.25}}};
    expect_matrix_near(element_stiffness(m, 0, 1.0), t0, 1e-14, "T0:");
    expect_matrix_near(element_stiffness(m, 1, 1.0), t1, 1e-14, "T1:");
}

// ESEMPIO 15. Due cose che valgono per QUALUNQUE triangolo, e hanno una ragione fisica.
//   - K e' simmetrica: K[i][j] = K[j][i] (il legame fra due nodi e' lo stesso nei due versi).
//   - Ogni riga somma a zero: se la temperatura e' uguale in tutti i nodi non passa calore. In formule:
//     le tre N_i sommano a 1 dappertutto, quindi i loro tre gradienti sommano a 0.
TEST(Fem2d, StiffnessIsSymmetricAndEachRowSumsToZero) {
    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangolo numero " + std::to_string(n++));
        const Matrix3 K = element_stiffness(mesh_of(t), 0, 2.5);
        const double tol = 1e-12 * largest(K);

        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(K[i][j], K[j][i], tol) << "elemento (" << i << ", " << j << ")";
            }
            EXPECT_NEAR(K[i][0] + K[i][1] + K[i][2], 0.0, tol) << "riga " << i;
        }
    }
}

// ESEMPIO 16. L'energia di un campo di temperatura LINEARE T(x, y) = t0 + gx x + gy y, il cui gradiente
// e' costante (gx, gy), e' l'integrale di k |grad T|^2 sul triangolo, cioe' k A (gx^2 + gy^2). Con le
// temperature T_i dei tre nodi, la matrice la calcola come  T^T K T = somma di T_i K[i][j] T_j.
// E' un controllo di tutta K per un'altra strada, senza usare le formule dei coefficienti b e c.
TEST(Fem2d, StiffnessGivesTheEnergyOfALinearField) {
    const double k = 2.5;
    // t0, gx, gy
    const double fields[][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {2.0, 1.0, -3.0}, {-1.0, 0.5, 0.25}};

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangolo numero " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 K = element_stiffness(m, 0, k);

        for (const auto& f : fields) {
            const double T[3] = {f[0] + f[1] * t.x0 + f[2] * t.y0, f[0] + f[1] * t.x1 + f[2] * t.y1,
                                 f[0] + f[1] * t.x2 + f[2] * t.y2};
            double energy = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    energy += T[i] * K[i][j] * T[j];
                }
            }
            const double expected = k * m.area(0) * (f[1] * f[1] + f[2] * f[2]);
            EXPECT_NEAR(energy, expected, 1e-8 * expected) << "campo con t0 = " << f[0] << ", gx = " << f[1]
                                                          << ", gy = " << f[2];
        }
    }
}

// ESEMPIO 17. K non cambia se il triangolo si sposta, si ruota o si ingrandisce. Spostare e ruotare e'
// ovvio (il materiale e' isotropo). Ingrandire di un fattore s moltiplica b e c per s e l'area per
// s^2, quindi b^2 / A resta uguale: in 2D la conduttanza di una lastra non dipende dalla sua
// grandezza (un quadrato di lato 1 mm e uno di lato 1 m conducono ugualmente, a pari spessore).
TEST(Fem2d, StiffnessDoesNotChangeWithPositionSizeOrRotation) {
    // s (ingrandimento), theta (rotazione in radianti), dx e dy (spostamento)
    const double moves[][4] = {{1.0, 0.0, 3.7, -2.2},                // solo spostato
                               {250.0, 0.0, 0.0, 0.0},               // 250 volte piu' grande
                               {0.01, 0.0, 0.0, 0.0},                // 100 volte piu' piccolo
                               {1.0, 1.5707963267948966, 0.0, 0.0},  // ruotato di 90 gradi
                               {2.5, 0.5235987755982988, -3.0, 4.0}};  // 30 gradi, ingrandito, spostato

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        const Matrix3 base = element_stiffness(mesh_of(t), 0, 1.0);
        for (const auto& mv : moves) {
            SCOPED_TRACE("triangolo numero " + std::to_string(n) + ", s = " + std::to_string(mv[0]) +
                         ", theta = " + std::to_string(mv[1]));
            const Mesh2D other = mesh_of(moved(t, mv[0], mv[1], mv[2], mv[3]));
            expect_matrix_near(element_stiffness(other, 0, 1.0), base, 1e-9 * largest(base));
        }
        ++n;
    }
}

// ESEMPIO 18. I tre nodi di un triangolo si possono elencare partendo da un vertice qualunque, purche'
// si resti in senso antiorario: (0, 1, 2), (1, 2, 0) e (2, 0, 1) sono lo stesso triangolo. La matrice
// deve solo cambiare l'ordine di righe e colonne: nella mesh con (1, 2, 0) il nodo locale i e' quello
// che nella prima era il nodo locale (i + 1) % 3, quindi  K'[i][j] = K[(i+1)%3][(j+1)%3].
TEST(Fem2d, StiffnessFollowsTheNumberingOfTheNodes) {
    const std::vector<double> xs = {0.0, 4.0, 1.0};   // il triangolo dell'ESEMPIO 12
    const std::vector<double> ys = {0.0, 0.0, 2.0};
    const Matrix3 a = element_stiffness(Mesh2D(xs, ys, {{0, 1, 2}}), 0, 1.0);
    const Matrix3 b = element_stiffness(Mesh2D(xs, ys, {{1, 2, 0}}), 0, 1.0);
    const Matrix3 c = element_stiffness(Mesh2D(xs, ys, {{2, 0, 1}}), 0, 1.0);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(b[i][j], a[(i + 1) % 3][(j + 1) % 3], 1e-14) << "(1, 2, 0), elemento (" << i << ", " << j << ")";
            EXPECT_NEAR(c[i][j], a[(i + 2) % 3][(j + 2) % 3], 1e-14) << "(2, 0, 1), elemento (" << i << ", " << j << ")";
        }
    }
}

// ESEMPIO 19. Il risultato non dipende da dove sta il triangolo nella mesh ne' dai numeri dei suoi nodi:
// per ogni triangolo di una mesh grande, la matrice e' uguale a quella dello stesso triangolo preso da
// solo (stessi vertici, nodi 0, 1, 2).
TEST(Fem2d, StiffnessDoesNotDependOnWhereTheTriangleSitsInTheMesh) {
    const Mesh2D big = rectangle_mesh(2.5, 0.5, 5, 3);   // 30 triangoli

    for (std::size_t e = 0; e < big.num_triangles(); ++e) {
        const Triangle& t = big.triangle(e);
        const Tri alone = {big.x(t[0]), big.y(t[0]), big.x(t[1]), big.y(t[1]), big.x(t[2]), big.y(t[2])};
        const Matrix3 want = element_stiffness(mesh_of(alone), 0, 1.0);
        expect_matrix_near(element_stiffness(big, e, 1.0), want, 1e-13 * largest(want),
                           "triangolo " + std::to_string(e) + ":");
    }
}

// ESEMPIO 20. Un triangolo che non esiste: std::out_of_range, come per Mesh2D::triangle.
TEST(Fem2d, StiffnessOfAnUnknownTriangleThrows) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});   // ha solo il triangolo 0
    EXPECT_NO_THROW(element_stiffness(m, 0, 1.0));
    EXPECT_THROW(element_stiffness(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_stiffness(m, 1000, 1.0), std::out_of_range);
}

// ---- Massa e carico di un triangolo: element_mass, element_load ---------------------------------
//
// element_mass(mesh, e, rho_c) e' la matrice 3 x 3 del triangolo e, per un materiale di capacita
// termica volumetrica rho_c (densita * calore specifico), lastra di spessore 1:
//
//     M[i][j] = integrale sul triangolo di  rho_c N_i N_j  dA
//
// element_load(mesh, e, q) e' il vettore dei tre termini noti, per una sorgente di calore q
// (potenza per unita' di volume) uniforme nel triangolo:
//
//     f[i] = integrale sul triangolo di  q N_i  dA
//
// Come si calcolano a mano, con A = area del triangolo:
//   - Il grafico di N_i sopra il triangolo e' una piramide: base il triangolo, altezza 1 sopra il
//     nodo i. Il volume di una piramide e' base * altezza / 3, quindi  integrale di N_i dA = A / 3  e
//     f[i] = q A / 3: ogni nodo riceve un terzo del calore totale q A.
//   - Le tre N_i sommano a 1, quindi ogni riga di M somma a  rho_c * integrale di N_i = rho_c A / 3.
//   - I tre nodi sono equivalenti: ogni triangolo si ottiene da ogni altro con una trasformazione
//     affine che manda N_i in N_i e moltiplica le aree per un fattore fisso. Quindi M[i][i] = a rho_c A
//     e M[i][j] = b rho_c A (i diverso da j), con gli stessi a e b per tutti i triangoli, e a + 2 b = 1/3.
//     Basta un integrale per trovarli. Nel triangolo (0, 0), (1, 0), (0, 1), dove N_1 = x e A = 1/2:
//         a A = integrale di x^2 dA = integrale su x da 0 a 1 di  x^2 (1 - x) dx = 1/3 - 1/4 = 1/12
//     (per ogni x la y va da 0 a 1 - x), cioe' a = 1/6, e poi b = (1/3 - 1/6) / 2 = 1/12. Quindi
//
//         M = rho_c A / 12 * [[2, 1, 1],           f = q A / 3 * [1, 1, 1]
//                             [1, 2, 1],
//                             [1, 1, 2]]
//
// In 1D era lo stesso: rho_c h / 6 * [[2, 1], [1, 2]] per la massa e q h / 2 per nodo per il carico.

namespace {

// Confronta due vettori elemento per elemento.
void expect_vector_near(const Vector3& got, const Vector3& want, double tol, const std::string& what = "") {
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << what << " elemento " << i;
    }
}

// La somma di tutti i nove elementi di K.
double sum_of_entries(const Matrix3& K) {
    double s = 0.0;
    for (const auto& row : K) {
        for (double v : row) {
            s += v;
        }
    }
    return s;
}

}  // namespace

// ESEMPIO 21. Il triangolo con i nodi (0, 0), (1, 0), (0, 1), A = 1/2, dove N_1 = x, N_2 = y e
// N_0 = 1 - x - y. Con rho_c = 1 si calcola ogni elemento direttamente:
//     M[1][1] = integrale di x^2 dA = integrale su x da 0 a 1 di  x^2 (1 - x) dx = 1/3 - 1/4 = 1/12
//     M[1][2] = integrale di x y dA = integrale su x da 0 a 1 di  x (1 - x)^2 / 2 dx = 1/24
// (per ogni x la y va da 0 a 1 - x). La formula rho_c A / 12 = 1/24 da' lo stesso: diagonale
// 2/24 = 1/12 e fuori 1/24. Con rho_c = 6 tutto e' moltiplicato per 6: 1/2 sulla diagonale e 1/4 fuori.
TEST(Fem2d, MassOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    const Matrix3 want1 = {{{1.0 / 12, 1.0 / 24, 1.0 / 24},
                            {1.0 / 24, 1.0 / 12, 1.0 / 24},
                            {1.0 / 24, 1.0 / 24, 1.0 / 12}}};
    expect_matrix_near(element_mass(m, 0, 1.0), want1, 1e-14, "rho_c = 1:");

    const Matrix3 want6 = {{{0.5, 0.25, 0.25}, {0.25, 0.5, 0.25}, {0.25, 0.25, 0.5}}};
    expect_matrix_near(element_mass(m, 0, 6.0), want6, 1e-14, "rho_c = 6:");
}

// ESEMPIO 22. La massa dipende solo dall'AREA (e da rho_c), non dalla forma. Qui i due triangoli
// dell'ESEMPIO 13 (nodi 0 = (0, 0), 1 = (4, 0), 2 = (1, 2), 3 = (6, 4); T0 = (0, 1, 2) con A = 4 e
// T1 = (1, 3, 2) con A = 8), con rho_c = 1.5:
//     T0:  rho_c A / 12 = 1.5 * 4 / 12 = 1/2    ->  M = 1/2 * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
//     T1:  rho_c A / 12 = 1.5 * 8 / 12 = 1      ->  M =       [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
// Serve a controllare che si usi l'area del triangolo richiesto (e non quella del primo).
TEST(Fem2d, MassUsesTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});

    const Matrix3 want0 = {{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}}};
    const Matrix3 want1 = {{{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}}};
    expect_matrix_near(element_mass(m, 0, 1.5), want0, 1e-14, "triangolo 0:");
    expect_matrix_near(element_mass(m, 1, 1.5), want1, 1e-14, "triangolo 1:");
}

// ESEMPIO 23. Cose che valgono per QUALUNQUE triangolo (qui rho_c = 2.5):
//   - M e' simmetrica e tutti i suoi elementi sono positivi (N_i N_j > 0 dentro il triangolo);
//   - ogni riga somma a rho_c A / 3, perche' le tre N_i sommano a 1;
//   - tutti i nove elementi insieme sommano a rho_c A, la capacita termica del triangolo (e' l'integrale
//     di rho_c (N_0 + N_1 + N_2)^2 = rho_c);
//   - fuori dalla diagonale c'e' la meta' della diagonale, sempre: la forma del triangolo non conta.
TEST(Fem2d, MassIsSymmetricPositiveAndSumsToTheHeatCapacity) {
    const double rho_c = 2.5;

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangolo numero " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const double area = m.area(0);
        const Matrix3 M = element_mass(m, 0, rho_c);
        const double tol = 1e-12 * rho_c * area;

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(M[i][0] + M[i][1] + M[i][2], rho_c * area / 3.0, tol) << "riga " << i;
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_GT(M[i][j], 0.0) << "elemento (" << i << ", " << j << ")";
                EXPECT_NEAR(M[i][j], M[j][i], tol) << "elemento (" << i << ", " << j << ")";
                EXPECT_NEAR(M[i][j], (i == j ? 1.0 : 0.5) * M[0][0], tol) << "elemento (" << i << ", " << j << ")";
            }
        }
        EXPECT_NEAR(sum_of_entries(M), rho_c * area, tol);
    }
}

// ESEMPIO 24. Un controllo di tutta M per un'altra strada. Tre valori T_0, T_1, T_2 nei nodi
// definiscono un campo LINEARE T(x, y) nel triangolo, e  T^T M T = somma di T_i M[i][j] T_j  e'
// l'integrale di rho_c T^2. T^2 e' un polinomio di grado 2, e per i polinomi di grado 2 l'integrale su
// un triangolo vale ESATTAMENTE area / 3 per la somma dei valori nei tre PUNTI MEDI dei lati. In un
// punto medio un campo lineare vale la media dei valori ai due estremi del lato. Quindi
//     T^T M T = rho_c A / 3 * [ ((T_0 + T_1) / 2)^2 + ((T_1 + T_2) / 2)^2 + ((T_2 + T_0) / 2)^2 ]
TEST(Fem2d, MassGivesTheIntegralOfTheSquareOfALinearField) {
    const double rho_c = 2.5;
    const double fields[][3] = {{1.0, 0.0, 0.0},  {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                                {1.0, 1.0, 1.0},  {1.0, -1.0, 0.0}, {2.0, -3.0, 5.0}};

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangolo numero " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 M = element_mass(m, 0, rho_c);

        for (const auto& T : fields) {
            double energy = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    energy += T[i] * M[i][j] * T[j];
                }
            }
            const double mid01 = 0.5 * (T[0] + T[1]);
            const double mid12 = 0.5 * (T[1] + T[2]);
            const double mid20 = 0.5 * (T[2] + T[0]);
            const double expected = rho_c * m.area(0) / 3.0 * (mid01 * mid01 + mid12 * mid12 + mid20 * mid20);
            EXPECT_NEAR(energy, expected, 1e-12 * expected)
                << "campo (" << T[0] << ", " << T[1] << ", " << T[2] << ")";
        }
    }
}

// ESEMPIO 25. Il triangolo (0, 0), (1, 0), (0, 1) ha A = 1/2, quindi q A / 3 = q / 6 per ogni nodo:
//     q = 6: 1 a testa (calore totale 6 * 1/2 = 3, un terzo per nodo);     q = 3: 1/2;
//     q = -6: -1 a testa (un pozzo di calore: il segno si conserva);        q = 0: niente.
TEST(Fem2d, LoadOfTheRightTriangleMatchesHandComputation) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

    expect_vector_near(element_load(m, 0, 6.0), {{1.0, 1.0, 1.0}}, 1e-14, "q = 6:");
    expect_vector_near(element_load(m, 0, 3.0), {{0.5, 0.5, 0.5}}, 1e-14, "q = 3:");
    expect_vector_near(element_load(m, 0, -6.0), {{-1.0, -1.0, -1.0}}, 1e-14, "q = -6:");
    expect_vector_near(element_load(m, 0, 0.0), {{0.0, 0.0, 0.0}}, 1e-14, "q = 0:");
}

// ESEMPIO 26. Come l'ESEMPIO 22, con i due triangoli dell'ESEMPIO 13 e q = 1.5:
//     T0 (A = 4):  q A / 3 = 1.5 * 4 / 3 = 2 per nodo        T1 (A = 8):  q A / 3 = 1.5 * 8 / 3 = 4 per nodo
TEST(Fem2d, LoadUsesTheAreaOfTheRequestedTriangle) {
    const Mesh2D m({0.0, 4.0, 1.0, 6.0}, {0.0, 0.0, 2.0, 4.0}, {{0, 1, 2}, {1, 3, 2}});

    expect_vector_near(element_load(m, 0, 1.5), {{2.0, 2.0, 2.0}}, 1e-14, "triangolo 0:");
    expect_vector_near(element_load(m, 1, 1.5), {{4.0, 4.0, 4.0}}, 1e-14, "triangolo 1:");
}

// ESEMPIO 27. Il carico e la massa sono legati. Le tre N_i sommano a 1, quindi
//     f[i] = integrale di q N_i = (q / rho_c) * integrale di rho_c N_i (N_0 + N_1 + N_2)
//          = (q / rho_c) * (M[i][0] + M[i][1] + M[i][2])
// cioe' il carico del nodo i e' la somma della riga i della massa, per q / rho_c. In piu' i tre nodi
// ricevono lo stesso calore, e il totale e' q A, per qualunque forma del triangolo.
TEST(Fem2d, LoadIsTheRowSumOfTheMassAndAddsUpToTheTotalHeat) {
    const double rho_c = 2.5;
    const double q = 7.0;

    std::size_t n = 0;
    for (const Tri& t : sample_triangles()) {
        SCOPED_TRACE("triangolo numero " + std::to_string(n++));
        const Mesh2D m = mesh_of(t);
        const Matrix3 M = element_mass(m, 0, rho_c);
        const Vector3 f = element_load(m, 0, q);
        const double tol = 1e-12 * q * m.area(0);

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(f[i], q / rho_c * (M[i][0] + M[i][1] + M[i][2]), tol) << "nodo " << i;
        }
        EXPECT_NEAR(f[0], f[1], tol);
        EXPECT_NEAR(f[1], f[2], tol);
        EXPECT_NEAR(f[0] + f[1] + f[2], q * m.area(0), tol);
    }
}

// ESEMPIO 28. Un triangolo che non esiste: std::out_of_range, per tutte e due le funzioni.
TEST(Fem2d, MassAndLoadOfAnUnknownTriangleThrow) {
    const Mesh2D m = mesh_of({0.0, 0.0, 1.0, 0.0, 0.0, 1.0});   // ha solo il triangolo 0
    EXPECT_NO_THROW(element_mass(m, 0, 1.0));
    EXPECT_NO_THROW(element_load(m, 0, 1.0));
    EXPECT_THROW(element_mass(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_mass(m, 1000, 1.0), std::out_of_range);
    EXPECT_THROW(element_load(m, 1, 1.0), std::out_of_range);
    EXPECT_THROW(element_load(m, 1000, 1.0), std::out_of_range);
}
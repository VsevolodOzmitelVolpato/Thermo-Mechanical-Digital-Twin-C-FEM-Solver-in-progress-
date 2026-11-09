#include <gtest/gtest.h>
#include "Mesh2D.h"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include "Mesh2D.h"

// ---- La maglia 2D: nodi, triangoli, aree -----------------------------------------------

// ESEMPIO 1. Il quadrato unitario diviso in due triangoli dalla diagonale 0-2:
//
//     3 ---------- 2        nodo 0 = (0, 0)     nodo 1 = (1, 0)
//     |          /  |        nodo 2 = (1, 1)     nodo 3 = (0, 1)
//     |   T1   /    |
//     |      /  T0  |        T0 = nodi (0, 1, 2)      T1 = nodi (0, 2, 3)
//     |    /        |
//     |  /          |
//     0 ---------- 1
//
// Area di T0: gli spigoli che escono dal nodo 0 sono (1, 0) e (1, 1), quindi
//     area = 1/2 * (1*1 - 1*0) = 1/2   (positiva: 0 -> 1 -> 2 gira a sinistra).
// Area di T1: gli spigoli sono (1, 1) e (0, 1), quindi area = 1/2 * (1*1 - 0*1) = 1/2.
// Insieme coprono il quadrato, che ha area 1.
static Mesh2D unit_square() {
    return Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}});
}

TEST(Mesh2d, SquareInTwoTrianglesMatchesHandComputation) {
    Mesh2D m = unit_square();

    EXPECT_EQ(m.num_nodes(), 4u);
    EXPECT_EQ(m.num_triangles(), 2u);

    const double ex[4] = {0.0, 1.0, 1.0, 0.0};
    const double ey[4] = {0.0, 0.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(m.x(i), ex[i]) << "nodo " << i;
        EXPECT_DOUBLE_EQ(m.y(i), ey[i]) << "nodo " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 2}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 2, 3}));
    EXPECT_NEAR(m.area(0), 0.5, 1e-15);
    EXPECT_NEAR(m.area(1), 0.5, 1e-15);
}

// ESEMPIO 2. Un quadrilatero irregolare, con i quattro nodi tutti diversi e i due triangoli di area
// diversa (cosi' si accorge se un'area viene attribuita al triangolo sbagliato):
//     nodo 0 = (0, 0)   nodo 1 = (4, 0)   nodo 2 = (3, 2)   nodo 3 = (0, 3)
//     T0 = (0, 1, 2): spigoli da 0 = (4, 0) e (3, 2)  ->  area = 1/2 * (4*2 - 3*0) = 4
//     T1 = (0, 2, 3): spigoli da 0 = (3, 2) e (0, 3)  ->  area = 1/2 * (3*3 - 0*2) = 4.5
// Il quadrilatero ha area 8.5: la formula dei lacci, 1/2 * (0*0-4*0 + 4*2-3*0 + 3*3-0*2 + 0*0-0*3) = 17/2.
TEST(Mesh2d, IrregularQuadrilateralHasUnequalAreas) {
    Mesh2D m({0.0, 4.0, 3.0, 0.0}, {0.0, 0.0, 2.0, 3.0}, {{0, 1, 2}, {0, 2, 3}});

    EXPECT_DOUBLE_EQ(m.x(1), 4.0);
    EXPECT_DOUBLE_EQ(m.y(1), 0.0);
    EXPECT_DOUBLE_EQ(m.x(2), 3.0);
    EXPECT_DOUBLE_EQ(m.y(2), 2.0);
    EXPECT_DOUBLE_EQ(m.x(3), 0.0);
    EXPECT_DOUBLE_EQ(m.y(3), 3.0);

    EXPECT_NEAR(m.area(0), 4.0, 1e-14);
    EXPECT_NEAR(m.area(1), 4.5, 1e-14);
    EXPECT_NEAR(m.area(0) + m.area(1), 8.5, 1e-14);
}

// ESEMPIO 3. Un triangolo qualunque, con nodi (1, 1), (4, 2), (2, 5): spigoli da 0 = (3, 1) e (1, 4),
// area = 1/2 * (3*4 - 1*1) = 5.5. Lo stesso triangolo, elencato partendo da un altro vertice
// (1, 2, 0) oppure (2, 0, 1), gira ancora a sinistra e ha la stessa area.
TEST(Mesh2d, AreaDoesNotDependOnWhichCornerComesFirst) {
    const std::vector<double> xs = {1.0, 4.0, 2.0};
    const std::vector<double> ys = {1.0, 2.0, 5.0};

    for (const Triangle& t : {Triangle{0, 1, 2}, Triangle{1, 2, 0}, Triangle{2, 0, 1}}) {
        Mesh2D m(xs, ys, std::vector<Triangle>{t});
        EXPECT_NEAR(m.area(0), 5.5, 1e-14) << "triangolo (" << t[0] << ", " << t[1] << ", " << t[2] << ")";
    }
}

// Le regole del costruttore. Si parte da un triangolo valido, (0, 0) (1, 0) (0, 1), e si rompe una
// regola alla volta.
TEST(Mesh2d, WrongInputsThrow) {
    const std::vector<double> xs = {0.0, 1.0, 0.0};
    const std::vector<double> ys = {0.0, 0.0, 1.0};
    const std::vector<Triangle> good = {{0, 1, 2}};
    EXPECT_NO_THROW(Mesh2D(xs, ys, good));

    // regola 1: xs e ys con numeri di valori diversi. Il triangolo {0, 1, 2} usa solo nodi che
    // esistono in tutti e due i vettori e ha area buona: l'unica regola rotta e' la numero 1.
    EXPECT_THROW(Mesh2D(xs, std::vector<double>{0.0, 0.0, 1.0, 5.0}, good), std::invalid_argument);
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, 1.0, 0.0, 5.0}, ys, good), std::invalid_argument);

    // regola 2: nessun triangolo
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{}), std::invalid_argument);

    // regola 3: un nodo che non esiste. Ci sono 3 nodi (0, 1, 2): il numero 3 e' gia' fuori.
    // Lo si prova in ognuna delle tre posizioni del triangolo.
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{3, 1, 2}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 3, 2}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 1, 3}}), std::invalid_argument);
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 1, 99}}), std::invalid_argument);

    // regola 4: triangolo orario. (0, 2, 1) = (0, 0) (0, 1) (1, 0): 1/2 * (0*0 - 1*1) = -1/2.
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 2, 1}}), std::invalid_argument);
    // regola 4: nodo ripetuto (area 0)
    EXPECT_THROW(Mesh2D(xs, ys, std::vector<Triangle>{{0, 0, 1}}), std::invalid_argument);
    // regola 4: tre nodi allineati, (0, 0) (1, 1) (2, 2): area 0
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 1.0, 2.0}, good),
                 std::invalid_argument);
    // regola 4: una coordinata NaN (non e' ne' > 0 ne' <= 0: va rifiutata lo stesso)
    EXPECT_THROW(Mesh2D(std::vector<double>{0.0, std::nan(""), 0.0}, ys, good), std::invalid_argument);

    // I triangoli vanno controllati TUTTI, non solo il primo: sul quadrato unitario
    // il secondo triangolo (0, 3, 2) gira a destra.
    EXPECT_NO_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 3}}));
    EXPECT_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 3, 2}}),
                 std::invalid_argument);
    // e il secondo triangolo con un nodo che non esiste
    EXPECT_THROW(Mesh2D({0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}, {{0, 1, 2}, {0, 2, 4}}),
                 std::invalid_argument);
}

// Con at() un indice sbagliato non legge fuori dai bordi in silenzio: lancia std::out_of_range.
// (E' il tipo di errore che in element_stress faceva andare in crash i test.)
TEST(Mesh2d, AccessorsCheckTheirBounds) {
    Mesh2D m = unit_square();   // 4 nodi, 2 triangoli

    EXPECT_NO_THROW(m.x(3));
    EXPECT_NO_THROW(m.y(3));
    EXPECT_NO_THROW(m.triangle(1));
    EXPECT_NO_THROW(m.area(1));

    EXPECT_THROW(m.x(4), std::out_of_range);
    EXPECT_THROW(m.y(4), std::out_of_range);
    EXPECT_THROW(m.triangle(2), std::out_of_range);
    EXPECT_THROW(m.area(2), std::out_of_range);
}

// ---- Mesh rettangolare strutturata: rectangle_mesh --------------------------------------------
//
// rectangle_mesh(Lx, Ly, nx, ny) divide il rettangolo [0, Lx] x [0, Ly] in nx x ny celle uguali, e
// ogni cella in 2 triangoli, con la diagonale dall'angolo in basso a sinistra a quello in alto a
// destra. Le regole di numerazione, per qualunque nx e ny:
//   - nodi: riga per riga, dal basso verso l'alto; in ogni riga da sinistra a destra. Il nodo della
//     colonna i (da 0 a nx) e della riga j (da 0 a ny) ha il numero  j*(nx+1) + i  e le coordinate
//     (i*Lx/nx, j*Ly/ny);
//   - celle: nello stesso ordine (riga per riga, da sinistra a destra). Gli angoli di una cella sono
//     a = in basso a sinistra, b = in basso a destra, c = in alto a destra, d = in alto a sinistra;
//   - triangoli: la cella numero k da' i triangoli numero 2k = (a, b, c) e numero 2k+1 = (a, c, d).

// ESEMPIO 4. Due celle in orizzontale, una in verticale: Lx = 2, Ly = 1, nx = 2, ny = 1.
//
//     3 --------- 4 --------- 5
//     |        /  |        /  |
//     |  T1  /    |  T3  /    |
//     |    /  T0  |    /  T2  |
//     |  /        |  /        |
//     0 --------- 1 --------- 2
//
// Nodi: 0 = (0, 0)  1 = (1, 0)  2 = (2, 0)  3 = (0, 1)  4 = (1, 1)  5 = (2, 1).
// Cella 0: (a, b, c, d) = (0, 1, 4, 3), quindi T0 = (0, 1, 4) e T1 = (0, 4, 3).
// Cella 1: (a, b, c, d) = (1, 2, 5, 4), quindi T2 = (1, 2, 5) e T3 = (1, 5, 4).
// Ogni triangolo e' meta' di un quadrato di lato 1, quindi ha area 1/2; in tutto 4 * 1/2 = 2 = Lx*Ly.
TEST(Mesh2d, RectangleTwoByOneMatchesHandComputation) {
    Mesh2D m = rectangle_mesh(2.0, 1.0, 2, 1);

    EXPECT_EQ(m.num_nodes(), 6u);
    EXPECT_EQ(m.num_triangles(), 4u);

    const double ex[6] = {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    const double ey[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(m.x(i), ex[i], 1e-14) << "nodo " << i;
        EXPECT_NEAR(m.y(i), ey[i], 1e-14) << "nodo " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 4}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 4, 3}));
    EXPECT_EQ(m.triangle(2), (Triangle{1, 2, 5}));
    EXPECT_EQ(m.triangle(3), (Triangle{1, 5, 4}));

    for (std::size_t e = 0; e < 4; ++e) {
        EXPECT_NEAR(m.area(e), 0.5, 1e-14) << "triangolo " << e;
    }
}

// ESEMPIO 5. Celle non quadrate e nx diverso da ny: Lx = 3, Ly = 1, nx = 3, ny = 2.
// I passi sono dx = 3/3 = 1 e dy = 1/2 = 0.5. Ogni riga ha nx+1 = 4 nodi, quindi il nodo
// della colonna i e della riga j e' il numero 4j + i:
//
//     8 --------- 9 -------- 10 -------- 11        riga j = 2   y = 1
//     |        /  |        /  |        /  |
//     |  T7  /    |  T9  /    | T11  /    |
//     |    /  T6  |    /  T8  |    /  T10 |
//     |  /        |  /        |  /        |
//     4 --------- 5 --------- 6 --------- 7        riga j = 1   y = 0.5
//     |        /  |        /  |        /  |
//     |  T1  /    |  T3  /    |  T5  /    |
//     |    /  T0  |    /  T2  |    /  T4  |
//     |  /        |  /        |  /        |
//     0 --------- 1 --------- 2 --------- 3        riga j = 0   y = 0
//
// Le celle sono 6, riga per riga. La cella 3 (colonna 0, riga 1) ha (a, b, c, d) = (4, 5, 9, 8),
// quindi T6 = (4, 5, 9) e T7 = (4, 9, 8). L'ultima, la cella 5, ha (6, 7, 11, 10), quindi
// T10 = (6, 7, 11) e T11 = (6, 11, 10). Ogni triangolo ha area 1/2 * 1 * 0.5 = 0.25.
TEST(Mesh2d, RectangleWithUnequalSidesPlacesTheNodesCorrectly) {
    Mesh2D m = rectangle_mesh(3.0, 1.0, 3, 2);

    EXPECT_EQ(m.num_nodes(), 12u);
    EXPECT_EQ(m.num_triangles(), 12u);

    const double ex[12] = {0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0};
    const double ey[12] = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(m.x(i), ex[i], 1e-14) << "nodo " << i;
        EXPECT_NEAR(m.y(i), ey[i], 1e-14) << "nodo " << i;
    }

    EXPECT_EQ(m.triangle(0), (Triangle{0, 1, 5}));
    EXPECT_EQ(m.triangle(1), (Triangle{0, 5, 4}));
    EXPECT_EQ(m.triangle(6), (Triangle{4, 5, 9}));
    EXPECT_EQ(m.triangle(7), (Triangle{4, 9, 8}));
    EXPECT_EQ(m.triangle(10), (Triangle{6, 7, 11}));
    EXPECT_EQ(m.triangle(11), (Triangle{6, 11, 10}));

    for (std::size_t e = 0; e < 12; ++e) {
        EXPECT_NEAR(m.area(e), 0.25, 1e-14) << "triangolo " << e;
    }
}

// Conta, per ogni lato (la coppia dei suoi due nodi, in ordine crescente), in quanti triangoli compare.
static std::map<std::pair<std::size_t, std::size_t>, int> edge_counts(const Mesh2D& m) {
    std::map<std::pair<std::size_t, std::size_t>, int> counts;
    for (std::size_t e = 0; e < m.num_triangles(); ++e) {
        const Triangle& t = m.triangle(e);
        for (std::size_t k = 0; k < 3; ++k) {
            std::size_t a = t[k];
            std::size_t b = t[(k + 1) % 3];
            if (a > b) {
                std::swap(a, b);
            }
            ++counts[{a, b}];
        }
    }
    return counts;
}

// ESEMPIO 6. Cose che devono valere per QUALUNQUE rettangolo e QUALUNQUE numero di celle, anche dove
// non si puo' calcolare tutto a mano. Con dx = Lx/nx e dy = Ly/ny:
//   - i nodi sono (nx+1)(ny+1) e i triangoli 2 nx ny;
//   - ogni triangolo e' meta' di una cella: area 1/2 dx dy. Quindi insieme coprono il rettangolo
//     senza buchi e senza sovrapposizioni: somma delle aree = Lx Ly;
//   - i nodi stanno dentro il rettangolo e ne toccano tutti e quattro i lati;
//   - la mesh e' "conforme": ogni lato appartiene a 2 triangoli (lato interno) oppure a 1 (lato sul
//     bordo), mai a 3 o piu'. I lati di bordo sono 2 (nx + ny); i lati in tutto sono
//     nx (ny+1) orizzontali + ny (nx+1) verticali + nx ny diagonali = 3 nx ny + nx + ny.
TEST(Mesh2d, RectangleCoversTheDomainExactlyOnce) {
    struct Case {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Case cases[] = {{1.0, 1.0, 1, 1}, {1.0, 1.0, 4, 4}, {2.5, 0.5, 5, 3}, {0.1, 7.0, 1, 9}, {3.0, 2.0, 7, 1}};

    for (const Case& c : cases) {
        SCOPED_TRACE("nx = " + std::to_string(c.nx) + ", ny = " + std::to_string(c.ny));
        Mesh2D m = rectangle_mesh(c.lx, c.ly, c.nx, c.ny);
        const double dx = c.lx / static_cast<double>(c.nx);
        const double dy = c.ly / static_cast<double>(c.ny);

        EXPECT_EQ(m.num_nodes(), (c.nx + 1) * (c.ny + 1));
        EXPECT_EQ(m.num_triangles(), 2 * c.nx * c.ny);

        double total_area = 0.0;
        for (std::size_t e = 0; e < m.num_triangles(); ++e) {
            EXPECT_NEAR(m.area(e), 0.5 * dx * dy, 1e-12 * dx * dy) << "triangolo " << e;
            total_area += m.area(e);
        }
        EXPECT_NEAR(total_area, c.lx * c.ly, 1e-12 * c.lx * c.ly);

        double xmin = m.x(0);
        double xmax = m.x(0);
        double ymin = m.y(0);
        double ymax = m.y(0);
        for (std::size_t n = 1; n < m.num_nodes(); ++n) {
            xmin = std::min(xmin, m.x(n));
            xmax = std::max(xmax, m.x(n));
            ymin = std::min(ymin, m.y(n));
            ymax = std::max(ymax, m.y(n));
        }
        EXPECT_NEAR(xmin, 0.0, 1e-12 * c.lx);
        EXPECT_NEAR(xmax, c.lx, 1e-12 * c.lx);
        EXPECT_NEAR(ymin, 0.0, 1e-12 * c.ly);
        EXPECT_NEAR(ymax, c.ly, 1e-12 * c.ly);

        const auto counts = edge_counts(m);
        std::size_t boundary_edges = 0;
        for (const auto& entry : counts) {
            const int times = entry.second;
            EXPECT_TRUE(times == 1 || times == 2) << "il lato (" << entry.first.first << ", "
                                                   << entry.first.second << ") sta in " << times << " triangoli";
            if (times == 1) {
                ++boundary_edges;
            }
        }
        EXPECT_EQ(boundary_edges, 2 * (c.nx + c.ny));
        EXPECT_EQ(counts.size(), 3 * c.nx * c.ny + c.nx + c.ny);
    }
}

// Ingressi sbagliati: lati non positivi o NaN, nessuna cella.
TEST(Mesh2d, RectangleWrongInputsThrow) {
    EXPECT_NO_THROW(rectangle_mesh(1.0, 1.0, 1, 1));

    EXPECT_THROW(rectangle_mesh(0.0, 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, 0.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(-1.0, 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, -1.0, 2, 2), std::invalid_argument);
    // Due lati negativi insieme darebbero una mesh "valida" (nel terzo quadrante), con tutti i
    // triangoli antiorari. Ma un rettangolo con lati negativi non ha senso: va rifiutato.
    EXPECT_THROW(rectangle_mesh(-1.0, -1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(std::nan(""), 1.0, 2, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, std::nan(""), 2, 2), std::invalid_argument);

    EXPECT_THROW(rectangle_mesh(1.0, 1.0, 0, 2), std::invalid_argument);
    EXPECT_THROW(rectangle_mesh(1.0, 1.0, 2, 0), std::invalid_argument);
}

// ---- Nodi di bordo: boundary_nodes -------------------------------------------------------------
//
// boundary_nodes(mesh) restituisce i numeri dei nodi che stanno sul bordo della mesh, in ordine
// crescente e ognuno una volta sola. Un nodo e' di bordo se e' un estremo di un lato di bordo, e un
// lato (il lato di un triangolo) e' di bordo se appartiene a UN SOLO triangolo: un lato interno ha un
// triangolo da ciascuna parte, quindi ne ha due. Il criterio non guarda le coordinate, quindi vale per
// qualunque forma, anche con dei buchi.

// ESEMPIO 7. Il quadrato unitario dell'ESEMPIO 1, a mano. I lati dei due triangoli T0 = (0, 1, 2) e
// T1 = (0, 2, 3), e in quanti triangoli compaiono:
//     (0, 1): 1     (1, 2): 1     (0, 2): 2  <- la diagonale e' interna     (2, 3): 1     (0, 3): 1
// I lati di bordo sono i quattro con 1, e toccano i nodi 0, 1, 2, 3: sono tutti di bordo.
//
// Nelle mesh dei rettangoli un nodo e' interno se non sta sul perimetro. Con 2 x 2 celle (9 nodi,
// numerati riga per riga) l'unico interno e' il nodo 4, al centro. Con 3 x 2 celle (12 nodi) sono
// interni il 5 e il 6, cioe' le colonne 1 e 2 della riga 1.
TEST(Mesh2d, BoundaryNodesOfSmallMeshesMatchHandComputation) {
    EXPECT_EQ(boundary_nodes(unit_square()), (std::vector<std::size_t>{0, 1, 2, 3}));

    EXPECT_EQ(boundary_nodes(rectangle_mesh(1.0, 1.0, 2, 2)), (std::vector<std::size_t>{0, 1, 2, 3, 5, 6, 7, 8}));

    EXPECT_EQ(boundary_nodes(rectangle_mesh(3.0, 1.0, 3, 2)),
              (std::vector<std::size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 11}));
}

// ESEMPIO 8. Un triangolo con un nodo dentro. Non e' un rettangolo, quindi niente scorciatoie con le
// coordinate. Nodi: 0 = (0, 0), 1 = (3, 0), 2 = (0, 3) e 3 = (1, 1), che sta dentro. Tre triangoli
// attorno al nodo 3: (0, 1, 3), (1, 2, 3), (2, 0, 3), ciascuno di area 3/2 (in tutto 9/2, come il
// triangolo grande). I lati (0, 1), (1, 2), (2, 0) sono in un solo triangolo; i tre lati che escono
// dal nodo 3 sono in due. Quindi i nodi di bordo sono 0, 1, 2 e il nodo 3 e' interno.
TEST(Mesh2d, BoundaryNodesOfATriangleWithAnInteriorNode) {
    Mesh2D m({0.0, 3.0, 0.0, 1.0}, {0.0, 0.0, 3.0, 1.0}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}});

    EXPECT_NEAR(m.area(0) + m.area(1) + m.area(2), 4.5, 1e-14);   // la mesh e' quella giusta
    EXPECT_EQ(boundary_nodes(m), (std::vector<std::size_t>{0, 1, 2}));
}

// ESEMPIO 9. Un quadrato 3 x 3 con un buco quadrato 1 x 1 al centro: i nodi 0..3 sono sul quadrato
// esterno e 4..7 su quello interno, tutti e due numerati in senso antiorario.
//
//     3 ----------------- 2
//     | \               / |
//     |   7 --------- 6   |
//     |   |           |   |
//     |   |   (buco)  |   |
//     |   |           |   |
//     |   4 --------- 5   |
//     | /               \ |
//     0 ----------------- 1
//
// La corona e' fatta di 4 trapezi, ognuno diviso in 2 triangoli: (0, 1, 5) (0, 5, 4) - (1, 2, 6)
// (1, 6, 5) - (2, 3, 7) (2, 7, 6) - (3, 0, 4) (3, 4, 7). L'area in tutto e' 9 - 1 = 8. Anche i lati
// del buco hanno un solo triangolo (dall'altra parte non c'e' niente): i nodi 4, 5, 6, 7 sono di
// bordo come gli altri.
TEST(Mesh2d, BoundaryNodesIncludeTheOnesAroundAHole) {
    Mesh2D m({0.0, 3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 1.0}, {0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0},
             {{0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}});

    double total_area = 0.0;
    for (std::size_t e = 0; e < m.num_triangles(); ++e) {
        total_area += m.area(e);
    }
    EXPECT_NEAR(total_area, 8.0, 1e-14);   // la mesh e' quella giusta

    EXPECT_EQ(boundary_nodes(m), (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

// ESEMPIO 10. Sui rettangoli c'e' un secondo modo, indipendente, per sapere quali nodi sono di bordo:
// quelli che toccano uno dei quattro lati, cioe' con x = 0, x = Lx, y = 0 oppure y = Ly. Il risultato
// di boundary_nodes, che guarda solo i triangoli, deve coincidere con questo per qualunque dimensione.
// In piu': i nodi di bordo di nx x ny celle sono 2 (nx + ny), e la lista e' in ordine strettamente
// crescente (quindi senza ripetizioni).
TEST(Mesh2d, BoundaryNodesMatchTheGeometryOfRectangles) {
    struct Case {
        double lx;
        double ly;
        std::size_t nx;
        std::size_t ny;
    };
    const Case cases[] = {{1.0, 1.0, 1, 1}, {1.0, 1.0, 4, 4}, {2.5, 0.5, 5, 3},
                          {0.1, 7.0, 1, 9}, {3.0, 2.0, 7, 1}, {2.0, 2.0, 6, 6}};

    for (const Case& c : cases) {
        SCOPED_TRACE("nx = " + std::to_string(c.nx) + ", ny = " + std::to_string(c.ny));
        Mesh2D m = rectangle_mesh(c.lx, c.ly, c.nx, c.ny);
        const std::vector<std::size_t> found = boundary_nodes(m);

        const double tol_x = 1e-9 * c.lx;
        const double tol_y = 1e-9 * c.ly;
        std::vector<std::size_t> expected;
        for (std::size_t n = 0; n < m.num_nodes(); ++n) {
            const bool on_a_vertical_side = std::fabs(m.x(n)) < tol_x || std::fabs(m.x(n) - c.lx) < tol_x;
            const bool on_a_horizontal_side = std::fabs(m.y(n)) < tol_y || std::fabs(m.y(n) - c.ly) < tol_y;
            if (on_a_vertical_side || on_a_horizontal_side) {
                expected.push_back(n);
            }
        }

        EXPECT_EQ(found, expected);
        EXPECT_EQ(found.size(), 2 * (c.nx + c.ny));
        for (std::size_t k = 1; k < found.size(); ++k) {
            EXPECT_LT(found[k - 1], found[k]) << "posizione " << k;
        }
    }
}
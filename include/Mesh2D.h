#ifndef MESH2D_H
#define MESH2D_H

#include <array>
#include <cstddef>
#include <vector>

// Un triangolo e' la terna dei numeri dei suoi 3 nodi (posizioni nella lista dei nodi).
using Triangle = std::array<std::size_t, 3>;

// Maglia 2D di triangoli: i nodi, ognuno con le sue coordinate (x, y), e i triangoli.
//
// A differenza di ThermoMechanicalProblem, qui i dati devono rispettare delle regole che valgono
// SEMPRE. Per questo e' una classe: i dati sono privati, solo il costruttore li imposta, e lo fa
// dopo aver controllato le regole. Chi usa la maglia (assemblaggio, condizioni al contorno...)
// puo' fidarsi senza ricontrollare.
//   1. xs e ys hanno lo stesso numero di valori: il nodo i sta in (xs[i], ys[i]);
//   2. c'e' almeno un triangolo;
//   3. i 3 nodi di ogni triangolo esistono (numeri minori del numero di nodi);
//   4. ogni triangolo e' antiorario e non degenere, cioe' la sua area con segno e' > 0:
//          area = 1/2 * [ (x1 - x0) (y2 - y0) - (x2 - x0) (y1 - y0) ]
//      Il segno e' + se i nodi 0, 1, 2 si susseguono girando a sinistra, - se girano a destra,
//      e l'area e' 0 se i tre nodi sono allineati (o se due coincidono).
class Mesh2D {
private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<Triangle> triangles_;
    std::vector<double> areas_;   // areas_[e] = area del triangolo e, calcolata una volta sola nel costruttore
    
public:
    Mesh2D(const std::vector<double>& xs, const std::vector<double>& ys,
           const std::vector<Triangle>& triangles);

    std::size_t num_nodes() const;
    std::size_t num_triangles() const;

    // Accesso con controllo dell'indice: lanciano std::out_of_range se node (o e) non esiste.
    double x(std::size_t node) const;                 // coordinata x del nodo
    double y(std::size_t node) const;                 // coordinata y del nodo
    const Triangle& triangle(std::size_t e) const;    // i 3 nodi del triangolo e
    double area(std::size_t e) const;                 // area del triangolo e (sempre > 0)

    Mesh2D rectangle_mesh(double xl, double yl, size_t x, size_t y);

};

Mesh2D rectangle_mesh(double xl, double yl, size_t x, size_t y);

std::vector<size_t> boundary_nodes(const Mesh2D &M);

#endif
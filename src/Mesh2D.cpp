#include "Mesh2D.h"
#include <stdexcept>
#include <vector>
#include <map>
#include <set>
#include <utility>

Mesh2D::Mesh2D(const std::vector<double> &x, const std::vector<double> &y, const std::vector<Triangle> &t) :
               x_(x), y_(y), triangles_(t){
                if(x.size() != y.size()){
                    throw std::invalid_argument("the two vectors have different sizes");}
                if(t.empty()){
                    throw std::invalid_argument("there is no triangle");}

                for(const auto &i:t){
                    for(const auto &j:i){
                        if(j >= x.size()){
                            throw std::invalid_argument("invalid number in the triangle");}}

                    double area = 0.5 * ((x[i[1]] - x[i[0]]) *(y[i[2]] - y[i[0]]) - (x[i[2]] - x[i[0]])*(y[i[1]] - y[i[0]]));
                    if(!(area > 0.0)){
                        throw std::invalid_argument("one or more triangles are not counterclockwise or degenerate");}
                    areas_.push_back(area);
                }
            }

std::size_t Mesh2D::num_nodes() const { return x_.size(); }

std::size_t Mesh2D::num_triangles() const { return triangles_.size(); }

double Mesh2D::x(std::size_t node) const { return x_.at(node); }

double Mesh2D::y(std::size_t node) const { return y_.at(node); }

const Triangle& Mesh2D::triangle(std::size_t e) const {
    return triangles_.at(e);
}
double Mesh2D::area(std::size_t node) const { return areas_.at(node); }

Mesh2D rectangle_mesh(double lx, double ly, size_t nx, size_t ny){
    if(!(lx > 0.0) or !(ly > 0.0)){
        throw std::invalid_argument("the lenght of one or both vectors is null or negative");
    }
    if( !(nx > 0 ) or !( ny > 0)){
        throw std::invalid_argument("the numbers of cells are invalid");
    }

    double dx = lx/nx;
    double dy = ly/ny;
    std::vector<double> xvect, yvect;

    for(size_t j {0}; j <= ny; ++j){
        for( size_t i {0}; i <= nx; ++i){
            xvect.push_back(dx*i);
            yvect.push_back(dy*j);}}

    std::vector<Triangle> tr;

    for(size_t j {0}; j < ny; ++j){
        for( size_t i {0}; i < nx; ++i){
            size_t a = j * (nx + 1) + i;
            size_t b = a + 1;
            size_t d = a + (nx + 1);
            size_t c = d + 1;

            Triangle tr1{a, b, c};
            Triangle tr2{a, c, d};

            tr.push_back(tr1);
            tr.push_back(tr2);}}

    return Mesh2D(xvect, yvect, tr);
}

std::vector<size_t> boundary_nodes(const Mesh2D &M){

    std::map<std::pair<std::size_t, std::size_t>, int> counts;

    for(size_t i {0}; i < M.num_triangles(); ++i){
        const Triangle &t = M.triangle(i);

        for(size_t k {0}; k < 3; ++k){
            size_t a = t[k];
            size_t b = t[(k+1)%3];

            if( a > b ){
                std::swap(a,b);}
            
            std::pair<std::size_t, std::size_t> key = {a,b};

            counts[key] +=1;
        }
    }
    
    std::set<size_t> s;

    for (const auto &entry: counts){
        if (entry.second == 1){
            s.insert(entry.first.first);
            s.insert(entry.first.second);}
    }

    std::vector<size_t> v(s.begin(), s.end());

    return v;
}
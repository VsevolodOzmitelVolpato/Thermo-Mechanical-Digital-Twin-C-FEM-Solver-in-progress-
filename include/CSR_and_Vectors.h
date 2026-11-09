#ifndef CSR_AND_VECTORS_H
#define CSR_AND_VECTORS_H
#include <vector>

class CSR {
private:

   std::vector<double> values;
   std::vector<int> col_indx;
   std::vector<int> row_ptr;

   size_t n;

public:
   CSR() = default;
   CSR(const std::vector<std::vector<double>> &Mat);
   CSR(const std::vector<double> &a, const std::vector<int> &b, const std::vector<int> &c);

   std::vector<double> operator*(const std::vector<double> &v) const;
   std::vector<double> diag() const;

   size_t get_val_size() const;
   double get(size_t i, size_t j) const;
   bool is_symmetric(const double &toll) const;
   size_t size() const;
};

typedef std::vector<double> DVector;

DVector operator+(const DVector &v1, const DVector &v2);
DVector operator-(const DVector &v3, const DVector &v5);
DVector operator*(const double &d, const DVector &v4);
double  dot(const DVector &v6, const DVector &v7);
double norm(const DVector &a);

#endif
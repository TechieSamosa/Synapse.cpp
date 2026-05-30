// =============================================================================
//  Matrix.cpp  —  Synapse.cpp
// =============================================================================
#include "Matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
// Box-Muller transform: produce one standard-normal sample
static double boxMuller() {
    static bool hasSpare = false;
    static double spare;
    if (hasSpare) { hasSpare = false; return spare; }
    double u, v, s;
    do {
        u = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    double mul = std::sqrt(-2.0 * std::log(s) / s);
    spare = v * mul;
    hasSpare = true;
    return u * mul;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Private memory management
// ─────────────────────────────────────────────────────────────────────────────
void Matrix::alloc(int r, int c) {
    rows = r; cols = c;
    data = new double[r * c]();   // zero-initialised
}

void Matrix::dealloc() {
    delete[] data;
    data = nullptr;
    rows = cols = 0;
}

void Matrix::copyFrom(const Matrix& other) {
    alloc(other.rows, other.cols);
    std::memcpy(data, other.data, sizeof(double) * rows * cols);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Constructors / Destructor
// ─────────────────────────────────────────────────────────────────────────────
Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

Matrix::Matrix(int r, int c, double init) {
    alloc(r, c);
    fill(init);
}

Matrix::Matrix(const Matrix& o) { copyFrom(o); }

Matrix::Matrix(Matrix&& o) noexcept : rows(o.rows), cols(o.cols), data(o.data) {
    o.rows = o.cols = 0;
    o.data = nullptr;
}

Matrix::~Matrix() { dealloc(); }

// ─────────────────────────────────────────────────────────────────────────────
//  Assignment
// ─────────────────────────────────────────────────────────────────────────────
Matrix& Matrix::operator=(const Matrix& o) {
    if (this != &o) { dealloc(); copyFrom(o); }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& o) noexcept {
    if (this != &o) {
        dealloc();
        rows = o.rows; cols = o.cols; data = o.data;
        o.rows = o.cols = 0; o.data = nullptr;
    }
    return *this;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element access
// ─────────────────────────────────────────────────────────────────────────────
double& Matrix::at(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols)
        throw std::out_of_range("Matrix::at index out of range");
    return data[r * cols + c];
}

double Matrix::at(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols)
        throw std::out_of_range("Matrix::at index out of range");
    return data[r * cols + c];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Shape utilities
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::transpose() const {
    Matrix out(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out.data[c * rows + r] = data[r * cols + c];
    return out;
}

Matrix Matrix::reshape(int nr, int nc) const {
    if (nr * nc != rows * cols)
        throw std::invalid_argument("reshape: element count mismatch");
    Matrix out(nr, nc);
    std::memcpy(out.data, data, sizeof(double) * rows * cols);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise arithmetic
// ─────────────────────────────────────────────────────────────────────────────
#define MATRIX_BINARY_OP(OP) \
    if (rows != b.rows || cols != b.cols) \
        throw std::invalid_argument("Matrix shape mismatch"); \
    Matrix out(rows, cols); \
    for (int i = 0; i < rows * cols; ++i) out.data[i] = data[i] OP b.data[i]; \
    return out;

Matrix Matrix::operator+(const Matrix& b) const { MATRIX_BINARY_OP(+) }
Matrix Matrix::operator-(const Matrix& b) const { MATRIX_BINARY_OP(-) }
Matrix Matrix::operator*(const Matrix& b) const { MATRIX_BINARY_OP(*) }
Matrix Matrix::operator/(const Matrix& b) const { MATRIX_BINARY_OP(/) }

Matrix& Matrix::operator+=(const Matrix& b) { *this = *this + b; return *this; }
Matrix& Matrix::operator-=(const Matrix& b) { *this = *this - b; return *this; }
Matrix& Matrix::operator*=(const Matrix& b) { *this = *this * b; return *this; }

// Scalar operations
Matrix Matrix::operator+(double s) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] + s;
    return out;
}
Matrix Matrix::operator-(double s) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] - s;
    return out;
}
Matrix Matrix::operator*(double s) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] * s;
    return out;
}
Matrix Matrix::operator/(double s) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] / s;
    return out;
}

Matrix& Matrix::operator+=(double s) { for (int i=0;i<rows*cols;++i) data[i]+=s; return *this; }
Matrix& Matrix::operator-=(double s) { for (int i=0;i<rows*cols;++i) data[i]-=s; return *this; }
Matrix& Matrix::operator*=(double s) { for (int i=0;i<rows*cols;++i) data[i]*=s; return *this; }
Matrix& Matrix::operator/=(double s) { for (int i=0;i<rows*cols;++i) data[i]/=s; return *this; }

Matrix operator+(double s, const Matrix& m) { return m + s; }
Matrix operator*(double s, const Matrix& m) { return m * s; }

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix multiplication  (dot product / matmul)
//  out[i,j] = sum_k  A[i,k] * B[k,j]
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::dot(const Matrix& b) const {
    if (cols != b.rows)
        throw std::invalid_argument(
            "dot: incompatible shapes " + shape() + " x " + b.shape());
    Matrix out(rows, b.cols, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < cols; ++k) {
            double aik = data[i * cols + k];
            for (int j = 0; j < b.cols; ++j)
                out.data[i * b.cols + j] += aik * b.data[k * b.cols + j];
        }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reductions
// ─────────────────────────────────────────────────────────────────────────────
double Matrix::sum() const {
    double s = 0.0;
    for (int i = 0; i < rows*cols; ++i) s += data[i];
    return s;
}

double Matrix::mean() const { return sum() / (rows * cols); }

double Matrix::max() const {
    double m = data[0];
    for (int i = 1; i < rows*cols; ++i) if (data[i] > m) m = data[i];
    return m;
}

double Matrix::min() const {
    double m = data[0];
    for (int i = 1; i < rows*cols; ++i) if (data[i] < m) m = data[i];
    return m;
}

Matrix Matrix::sum(int axis) const {
    if (axis == 0) {                        // collapse rows → (1, cols)
        Matrix out(1, cols, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[c] += data[r * cols + c];
        return out;
    } else {                                // collapse cols → (rows, 1)
        Matrix out(rows, 1, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[r] += data[r * cols + c];
        return out;
    }
}

Matrix Matrix::mean(int axis) const {
    if (axis == 0) return sum(0) / (double)rows;
    else           return sum(1) / (double)cols;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise math functions
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::apply(double (*func)(double)) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = func(data[i]);
    return out;
}

Matrix Matrix::exp()    const { return apply(std::exp); }
Matrix Matrix::log()    const { return apply(std::log); }
Matrix Matrix::sqrt()   const { return apply(std::sqrt); }
Matrix Matrix::square() const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i) out.data[i] = data[i] * data[i];
    return out;
}
Matrix Matrix::abs() const { return apply(std::abs); }

Matrix Matrix::clip(double lo, double hi) const {
    Matrix out(rows, cols);
    for (int i = 0; i < rows*cols; ++i)
        out.data[i] = std::max(lo, std::min(hi, data[i]));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Broadcast add/mul
//  Supports (1,cols) broadcast over rows, and (rows,1) broadcast over cols.
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::broadcastAdd(const Matrix& rhs) const {
    Matrix out(rows, cols);
    if (rhs.rows == 1 && rhs.cols == cols) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[r*cols+c] = data[r*cols+c] + rhs.data[c];
    } else if (rhs.cols == 1 && rhs.rows == rows) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[r*cols+c] = data[r*cols+c] + rhs.data[r];
    } else {
        throw std::invalid_argument("broadcastAdd: incompatible shapes");
    }
    return out;
}

Matrix Matrix::broadcastMul(const Matrix& rhs) const {
    Matrix out(rows, cols);
    if (rhs.rows == 1 && rhs.cols == cols) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[r*cols+c] = data[r*cols+c] * rhs.data[c];
    } else if (rhs.cols == 1 && rhs.rows == rows) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.data[r*cols+c] = data[r*cols+c] * rhs.data[r];
    } else {
        throw std::invalid_argument("broadcastMul: incompatible shapes");
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Initialisation
// ─────────────────────────────────────────────────────────────────────────────
void Matrix::fill(double val) {
    for (int i = 0; i < rows*cols; ++i) data[i] = val;
}

void Matrix::randomUniform(double lo, double hi) {
    for (int i = 0; i < rows*cols; ++i)
        data[i] = lo + ((double)std::rand() / RAND_MAX) * (hi - lo);
}

void Matrix::randomNormal(double mean, double stddev) {
    for (int i = 0; i < rows*cols; ++i)
        data[i] = mean + stddev * boxMuller();
}

// Xavier uniform:  U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
void Matrix::xavierUniform(int fanIn, int fanOut) {
    double limit = std::sqrt(6.0 / (fanIn + fanOut));
    randomUniform(-limit, limit);
}

// He normal:  N(0, sqrt(2/fan_in))
void Matrix::heNormal(int fanIn) {
    randomNormal(0.0, std::sqrt(2.0 / fanIn));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Argmax
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::argmax(int axis) const {
    if (axis == 0) {                 // index of max in each column
        Matrix out(1, cols, 0.0);
        for (int c = 0; c < cols; ++c) {
            int best = 0;
            double bestVal = data[c];
            for (int r = 1; r < rows; ++r)
                if (data[r*cols+c] > bestVal) { bestVal = data[r*cols+c]; best = r; }
            out.data[c] = (double)best;
        }
        return out;
    } else {                         // index of max in each row
        Matrix out(rows, 1, 0.0);
        for (int r = 0; r < rows; ++r) {
            int best = 0;
            double bestVal = data[r*cols];
            for (int c = 1; c < cols; ++c)
                if (data[r*cols+c] > bestVal) { bestVal = data[r*cols+c]; best = c; }
            out.data[r] = (double)best;
        }
        return out;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Concatenation
// ─────────────────────────────────────────────────────────────────────────────
Matrix Matrix::hstack(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows)
        throw std::invalid_argument("hstack: row count mismatch");
    Matrix out(a.rows, a.cols + b.cols);
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c)
            out.data[r*out.cols + c] = a.data[r*a.cols + c];
        for (int c = 0; c < b.cols; ++c)
            out.data[r*out.cols + a.cols + c] = b.data[r*b.cols + c];
    }
    return out;
}

Matrix Matrix::vstack(const Matrix& a, const Matrix& b) {
    if (a.cols != b.cols)
        throw std::invalid_argument("vstack: column count mismatch");
    Matrix out(a.rows + b.rows, a.cols);
    std::memcpy(out.data,               a.data, sizeof(double)*a.rows*a.cols);
    std::memcpy(out.data + a.rows*a.cols, b.data, sizeof(double)*b.rows*b.cols);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  I/O
// ─────────────────────────────────────────────────────────────────────────────
std::string Matrix::shape() const {
    return "(" + std::to_string(rows) + "," + std::to_string(cols) + ")";
}

void Matrix::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " " << shape() << ":\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "  [";
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(5)
                      << data[r*cols+c];
            if (c < cols-1) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
}

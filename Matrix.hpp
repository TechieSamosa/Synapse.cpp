// =============================================================================
//  Matrix.hpp  —  Synapse.cpp
//  A self-contained Matrix class for raw C++ ML math.
//  No external libraries. Only <cmath>, <cstdlib>, <stdexcept>, <iostream>.
// =============================================================================
#pragma once

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix — row-major 2-D array stored on the heap.
//  Every operator returns a NEW matrix (value semantics).
// ─────────────────────────────────────────────────────────────────────────────
class Matrix {
public:
    int rows, cols;
    double* data;   // flat row-major storage: data[r*cols + c]

    // ── Constructors / destructor ─────────────────────────────────────────────
    Matrix();
    Matrix(int rows, int cols, double init = 0.0);
    Matrix(const Matrix& other);            // deep copy
    Matrix(Matrix&& other) noexcept;        // move
    ~Matrix();

    // ── Assignment ────────────────────────────────────────────────────────────
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;

    // ── Element access ────────────────────────────────────────────────────────
    double& at(int r, int c);
    double  at(int r, int c) const;

    // ── Shape utilities ───────────────────────────────────────────────────────
    Matrix transpose() const;
    Matrix reshape(int newRows, int newCols) const;

    // ── Arithmetic (element-wise) ─────────────────────────────────────────────
    Matrix operator+(const Matrix& b) const;
    Matrix operator-(const Matrix& b) const;
    Matrix operator*(const Matrix& b) const;   // element-wise (Hadamard)
    Matrix operator/(const Matrix& b) const;

    Matrix& operator+=(const Matrix& b);
    Matrix& operator-=(const Matrix& b);
    Matrix& operator*=(const Matrix& b);

    // ── Scalar arithmetic ─────────────────────────────────────────────────────
    Matrix operator+(double s) const;
    Matrix operator-(double s) const;
    Matrix operator*(double s) const;
    Matrix operator/(double s) const;

    Matrix& operator+=(double s);
    Matrix& operator-=(double s);
    Matrix& operator*=(double s);
    Matrix& operator/=(double s);

    // ── Matrix multiplication ─────────────────────────────────────────────────
    Matrix dot(const Matrix& b) const;

    // ── Reduction ─────────────────────────────────────────────────────────────
    double sum() const;
    double mean() const;
    double max() const;
    double min() const;

    // Sum/mean along axis: axis=0 → collapse rows → (1, cols)
    //                      axis=1 → collapse cols → (rows, 1)
    Matrix sum(int axis) const;
    Matrix mean(int axis) const;

    // ── Element-wise math ─────────────────────────────────────────────────────
    Matrix apply(double (*func)(double)) const;
    Matrix exp()   const;
    Matrix log()   const;
    Matrix sqrt()  const;
    Matrix square() const;
    Matrix abs()   const;
    Matrix clip(double lo, double hi) const;

    // ── Broadcasting add/mul (vector broadcast over matrix) ──────────────────
    //   rhs must be (1,cols) for broadcast over rows
    //       or (rows,1) for broadcast over cols
    Matrix broadcastAdd(const Matrix& rhs) const;
    Matrix broadcastMul(const Matrix& rhs) const;

    // ── Initialisation ────────────────────────────────────────────────────────
    void fill(double val);
    void randomUniform(double lo, double hi);
    void randomNormal(double mean = 0.0, double stddev = 1.0);
    void xavierUniform(int fanIn, int fanOut);
    void heNormal(int fanIn);

    // ── Comparison / concatenation ────────────────────────────────────────────
    Matrix argmax(int axis) const;   // returns index of max per row/col
    static Matrix hstack(const Matrix& a, const Matrix& b);  // cat cols
    static Matrix vstack(const Matrix& a, const Matrix& b);  // cat rows

    // ── I/O ───────────────────────────────────────────────────────────────────
    void print(const std::string& name = "") const;
    std::string shape() const;

private:
    void alloc(int r, int c);
    void dealloc();
    void copyFrom(const Matrix& other);
};

// Non-member scalar-on-left operators
Matrix operator+(double s, const Matrix& m);
Matrix operator*(double s, const Matrix& m);

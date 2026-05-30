// =============================================================================
//  main.cpp  —  Synapse.cpp
//  Demonstrations of the hand-rolled MLP:
//    1. XOR  (classic non-linearly separable problem)
//    2. Circle dataset  (binary classification)
//    3. Sine regression
//
//  Compile:
//    g++ -O2 -std=c++17 main.cpp Matrix.cpp MLP.cpp -o synapse
//  Run:
//    ./synapse
// =============================================================================
#include "MLP.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
//  Utility : print a horizontal separator
// ─────────────────────────────────────────────────────────────────────────────
static void separator(const std::string& title = "") {
    std::cout << "\n";
    if (title.empty()) {
        std::cout << std::string(60, '=') << "\n";
    } else {
        int pad = (58 - (int)title.size()) / 2;
        std::cout << std::string(60, '=') << "\n";
        std::cout << std::string(pad, ' ') << title << "\n";
        std::cout << std::string(60, '=') << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demo 1 — XOR problem
//  Input  : (x1, x2) ∈ {0,1}²
//  Output : x1 XOR x2  ∈ {0,1}
//  A single-hidden-layer MLP can solve this; a linear model cannot.
// ─────────────────────────────────────────────────────────────────────────────
void demoXOR() {
    separator("DEMO 1 — XOR (Binary Classification)");

    // Dataset: 4 canonical XOR examples repeated 50× each → 200 samples
    const int REPEAT = 50;
    const int N      = 4 * REPEAT;

    Matrix X(N, 2), Y(N, 1);
    double xorData[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double xorLabel[4]   = {0,1,1,0};

    for (int rep = 0; rep < REPEAT; ++rep)
        for (int i = 0; i < 4; ++i) {
            int row = rep*4 + i;
            X.at(row, 0) = xorData[i][0];
            X.at(row, 1) = xorData[i][1];
            Y.at(row, 0) = xorLabel[i];
        }

    // Architecture: 2 → 8 → 1  (hidden ReLU, output Sigmoid)
    MLP net(2, {
        LayerConfig{8,  Activation::ReLU,    0.0},
        LayerConfig{1,  Activation::Sigmoid,  0.0}
    });

    net.printSummary();

    TrainConfig cfg;
    cfg.epochs       = 500;
    cfg.batchSize    = 32;
    cfg.learningRate = 0.01;
    cfg.loss         = Loss::BinaryCrossEntropy;
    cfg.optimizer    = Optimizer::Adam;
    cfg.weightInit   = WeightInit::Xavier;
    cfg.verbose      = true;
    cfg.printEvery   = 100;

    net.fit(X, Y, cfg);

    double acc = net.accuracy(X, Y);
    std::cout << "\n[XOR] Final training accuracy: "
              << std::fixed << std::setprecision(2) << acc * 100.0 << "%\n";

    // Show predictions on the 4 canonical inputs
    Matrix test(4, 2);
    for (int i = 0; i < 4; ++i) {
        test.at(i,0) = xorData[i][0];
        test.at(i,1) = xorData[i][1];
    }
    Matrix out = net.predict(test);

    std::cout << "\n  Input          Expected   Predicted\n";
    std::cout << "  ─────────────────────────────────────\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  (" << (int)xorData[i][0] << ", " << (int)xorData[i][1] << ")  →   "
                  << "  " << (int)xorLabel[i]
                  << "         " << std::fixed << std::setprecision(4) << out.at(i,0)
                  << "  (" << (out.at(i,0) >= 0.5 ? 1 : 0) << ")\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demo 2 — Circle dataset  (non-linear binary classification)
//  Points inside radius 0.5 → class 1, outside → class 0.
// ─────────────────────────────────────────────────────────────────────────────
void demoCircle() {
    separator("DEMO 2 — Circle Dataset (Non-Linear Binary)");

    const int N = 400;
    Matrix X(N, 2), Y(N, 1);

    std::srand(42);
    for (int i = 0; i < N; ++i) {
        double x = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        double y = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
        X.at(i, 0) = x;
        X.at(i, 1) = y;
        Y.at(i, 0) = (std::sqrt(x*x + y*y) < 0.6) ? 1.0 : 0.0;
    }

    // Architecture: 2 → 16 → 8 → 1
    MLP net(2, {
        LayerConfig{16, Activation::ReLU,    0.0},
        LayerConfig{8,  Activation::ReLU,    0.0},
        LayerConfig{1,  Activation::Sigmoid,  0.0}
    });

    net.printSummary();

    TrainConfig cfg;
    cfg.epochs       = 300;
    cfg.batchSize    = 32;
    cfg.learningRate = 0.005;
    cfg.loss         = Loss::BinaryCrossEntropy;
    cfg.optimizer    = Optimizer::Adam;
    cfg.weightInit   = WeightInit::Xavier;
    cfg.verbose      = true;
    cfg.printEvery   = 75;

    net.fit(X, Y, cfg);

    double acc = net.accuracy(X, Y);
    std::cout << "\n[Circle] Final training accuracy: "
              << std::fixed << std::setprecision(2) << acc * 100.0 << "%\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demo 3 — Sine regression
//  Learn to approximate  f(x) = sin(2πx)  on [0,1].
// ─────────────────────────────────────────────────────────────────────────────
void demoSineRegression() {
    separator("DEMO 3 — Sine Regression");

    const double PI = 3.14159265358979;
    const int N     = 300;

    Matrix X(N, 1), Y(N, 1);
    for (int i = 0; i < N; ++i) {
        double x   = (double)i / N;
        // Add small noise to make it realistic
        double noise = (((double)std::rand() / RAND_MAX) - 0.5) * 0.05;
        X.at(i, 0) = x;
        Y.at(i, 0) = std::sin(2.0 * PI * x) + noise;
    }

    // Architecture: 1 → 64 → 64 → 1  (tanh hidden, linear output)
    MLP net(1, {
        LayerConfig{64, Activation::Tanh,   0.0},
        LayerConfig{64, Activation::Tanh,   0.0},
        LayerConfig{1,  Activation::Linear, 0.0}
    });

    net.printSummary();

    TrainConfig cfg;
    cfg.epochs       = 500;
    cfg.batchSize    = 32;
    cfg.learningRate = 0.001;
    cfg.loss         = Loss::MSE;
    cfg.optimizer    = Optimizer::Adam;
    cfg.weightInit   = WeightInit::Xavier;
    cfg.verbose      = true;
    cfg.printEvery   = 100;

    net.fit(X, Y, cfg);

    double finalMSE = net.mse(X, Y);
    std::cout << "\n[Sine] Final MSE: " << std::scientific << std::setprecision(4) << finalMSE << "\n";

    // Sample 10 evenly-spaced predictions
    std::cout << "\n  x       y_true       y_pred\n";
    std::cout << "  ──────────────────────────────\n";
    for (int i = 0; i <= 9; ++i) {
        double x = (double)i / 9.0;
        Matrix xi(1, 1); xi.at(0,0) = x;
        Matrix yi = net.predict(xi);
        double true_y = std::sin(2.0 * PI * x);
        std::cout << "  " << std::fixed << std::setprecision(3) << x
                  << "   " << std::setw(10) << true_y
                  << "   " << std::setw(10) << yi.at(0,0) << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demo 4 — Multi-class classification (3-class)
//  Three Gaussian blobs — tests Softmax + Categorical Cross-Entropy.
// ─────────────────────────────────────────────────────────────────────────────
void demoMultiClass() {
    separator("DEMO 4 — Three Gaussian Blobs (Multi-Class)");

    const int N_PER_CLASS = 100;
    const int N = 3 * N_PER_CLASS;

    // Blob centres
    double cx[3] = {-1.0, 1.0, 0.0};
    double cy[3] = { 0.0, 0.0, 1.5};

    Matrix X(N, 2), Y(N, 3);
    Y.fill(0.0);

    for (int cls = 0; cls < 3; ++cls) {
        for (int i = 0; i < N_PER_CLASS; ++i) {
            int row = cls * N_PER_CLASS + i;
            // Box-Muller noise
            double angle  = ((double)std::rand() / RAND_MAX) * 6.2832;
            double radius = std::sqrt(-2.0 * std::log(std::max(1e-9,(double)std::rand()/RAND_MAX))) * 0.3;
            X.at(row, 0) = cx[cls] + radius * std::cos(angle);
            X.at(row, 1) = cy[cls] + radius * std::sin(angle);
            Y.at(row, cls) = 1.0;   // one-hot
        }
    }

    // Architecture: 2 → 32 → 16 → 3  (Softmax output)
    MLP net(2, {
        LayerConfig{32, Activation::ReLU,    0.0},
        LayerConfig{16, Activation::ReLU,    0.0},
        LayerConfig{3,  Activation::Softmax, 0.0}
    });

    net.printSummary();

    TrainConfig cfg;
    cfg.epochs       = 400;
    cfg.batchSize    = 32;
    cfg.learningRate = 0.003;
    cfg.loss         = Loss::CategoricalCrossEntropy;
    cfg.optimizer    = Optimizer::Adam;
    cfg.weightInit   = WeightInit::Xavier;
    cfg.verbose      = true;
    cfg.printEvery   = 100;

    net.fit(X, Y, cfg);

    double acc = net.accuracy(X, Y);
    std::cout << "\n[Blobs] Final training accuracy: "
              << std::fixed << std::setprecision(2) << acc * 100.0 << "%\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::srand((unsigned)std::time(nullptr));

    std::cout << "\n";
    std::cout << "  ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗\n";
    std::cout << "  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝\n";
    std::cout << "  ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  \n";
    std::cout << "  ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  \n";
    std::cout << "  ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗\n";
    std::cout << "  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝\n";
    std::cout << "\n  MLP from Scratch in C++  |  No Libraries  |  Pure Math + Memory\n";
    std::cout << std::string(60, '-') << "\n";

    demoXOR();
    demoCircle();
    demoSineRegression();
    demoMultiClass();

    separator("All demos complete!");
    return 0;
}
// =============================================================================
//  MLP.hpp  —  Synapse.cpp
//  A fully hand-rolled Multi-Layer Perceptron (MLP / feedforward neural net).
//
//  Features
//  --------
//  • Arbitrary depth and width (vector<int> topology)
//  • Activation functions  : ReLU, Leaky-ReLU, Sigmoid, Tanh, Linear
//  • Loss functions        : MSE, Binary Cross-Entropy, Categorical Cross-Entropy
//  • Optimizers            : SGD (+ momentum), Adam
//  • Weight initialization : Xavier-uniform, He-normal
//  • Batch training        : mini-batch forward + backprop
//  • Dropout regularisation (during training)
//  • Gradient clipping
//  • Train / eval mode switching
// =============================================================================
#pragma once

#include "Matrix.hpp"
#include <vector>
#include <string>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
//  Enumerations
// ─────────────────────────────────────────────────────────────────────────────
enum class Activation { ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Softmax };
enum class Loss       { MSE, BinaryCrossEntropy, CategoricalCrossEntropy };
enum class Optimizer  { SGD, Adam };
enum class WeightInit { Xavier, He, RandomUniform, RandomNormal };

// ─────────────────────────────────────────────────────────────────────────────
//  LayerConfig — describes a single dense layer
// ─────────────────────────────────────────────────────────────────────────────
struct LayerConfig {
    int        units;
    Activation activation = Activation::ReLU;
    double     dropout    = 0.0;   // dropout probability (0 = off)
};

// ─────────────────────────────────────────────────────────────────────────────
//  TrainConfig — hyperparameters bundle
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig {
    int       epochs        = 100;
    int       batchSize     = 32;
    double    learningRate  = 0.001;
    Loss      loss          = Loss::MSE;
    Optimizer optimizer     = Optimizer::Adam;
    WeightInit weightInit   = WeightInit::Xavier;
    double    momentum      = 0.9;   // SGD momentum  / Adam beta1
    double    beta2         = 0.999; // Adam beta2
    double    epsilon       = 1e-8;  // Adam epsilon
    double    gradClip      = 0.0;   // 0 = disabled
    bool      verbose       = true;
    int       printEvery    = 10;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Layer — one fully-connected layer with its own parameters & gradients
// ─────────────────────────────────────────────────────────────────────────────
struct Layer {
    // ── Parameters ────────────────────────────────────────────────────────────
    Matrix W;           // weight matrix  (inFeatures, units)
    Matrix b;           // bias vector    (1, units)

    // ── Gradients ─────────────────────────────────────────────────────────────
    Matrix dW;
    Matrix db;

    // ── Adam moment estimates ──────────────────────────────────────────────────
    Matrix mW, vW;      // 1st & 2nd moment for W
    Matrix mb, vb;      // 1st & 2nd moment for b

    // ── SGD momentum ──────────────────────────────────────────────────────────
    Matrix velW, velb;

    // ── Cached activations (needed for backprop) ───────────────────────────────
    Matrix Z;           // pre-activation  (batch, units)
    Matrix A;           // post-activation (batch, units)
    Matrix dropMask;    // dropout mask    (batch, units)

    // ── Config ────────────────────────────────────────────────────────────────
    Activation activation;
    double     dropout;
    int        inFeatures;
    int        units;
};

// ─────────────────────────────────────────────────────────────────────────────
//  MLP — the neural network
// ─────────────────────────────────────────────────────────────────────────────
class MLP {
public:
    // ── Construction ──────────────────────────────────────────────────────────
    // topology: list of LayerConfig objects, from first hidden to output layer.
    // inputSize: number of input features.
    MLP(int inputSize, const std::vector<LayerConfig>& topology);

    // ── Forward pass ──────────────────────────────────────────────────────────
    // X : (batch, inputSize)
    // returns : (batch, outputUnits)
    Matrix forward(const Matrix& X);

    // ── Loss computation ──────────────────────────────────────────────────────
    // predictions : (batch, outputUnits),  targets : (batch, outputUnits)
    double computeLoss(const Matrix& predictions, const Matrix& targets, Loss lossType);

    // ── Backward pass ─────────────────────────────────────────────────────────
    void backward(const Matrix& X,
                  const Matrix& targets,
                  Loss          lossType,
                  double        gradClip = 0.0);

    // ── Parameter update ──────────────────────────────────────────────────────
    void updateSGD (double lr, double momentum);
    void updateAdam(double lr, double beta1, double beta2, double eps);

    // ── Training loop ─────────────────────────────────────────────────────────
    void fit(const Matrix& X, const Matrix& Y, const TrainConfig& cfg);

    // ── Inference ─────────────────────────────────────────────────────────────
    Matrix predict(const Matrix& X);

    // ── Mode ──────────────────────────────────────────────────────────────────
    void setTraining(bool training);

    // ── Metrics ───────────────────────────────────────────────────────────────
    double accuracy(const Matrix& X, const Matrix& Y); // classification
    double mse     (const Matrix& X, const Matrix& Y); // regression

    // ── Diagnostics ───────────────────────────────────────────────────────────
    void printSummary() const;
    std::vector<double> getLossHistory() const { return lossHistory_; }

private:
    int                 inputSize_;
    std::vector<Layer>  layers_;
    bool                isTraining_ = true;
    int                 adamStep_   = 0;
    std::vector<double> lossHistory_;

    // ── Activation helpers ────────────────────────────────────────────────────
    Matrix applyActivation (const Matrix& Z, Activation act) const;
    Matrix activationGrad  (const Matrix& Z, const Matrix& A, Activation act) const;

    // ── Softmax ───────────────────────────────────────────────────────────────
    Matrix softmax(const Matrix& Z) const;

    // ── Weight init ───────────────────────────────────────────────────────────
    void initWeights(Layer& layer, WeightInit init);
};

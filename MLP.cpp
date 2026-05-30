// =============================================================================
//  MLP.cpp  —  Synapse.cpp
//  Implementation of the MLP class.
//  Pure C++ — no external ML libraries used anywhere.
// =============================================================================
#include "MLP.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────
MLP::MLP(int inputSize, const std::vector<LayerConfig>& topology)
    : inputSize_(inputSize)
{
    if (topology.empty())
        throw std::invalid_argument("MLP: topology must have at least one layer");

    int inFeat = inputSize;

    for (const auto& cfg : topology) {
        Layer layer;
        layer.units      = cfg.units;
        layer.activation = cfg.activation;
        layer.dropout    = cfg.dropout;
        layer.inFeatures = inFeat;

        // Weights + biases (will be properly initialised in fit() or manually)
        layer.W  = Matrix(inFeat, cfg.units);
        layer.b  = Matrix(1,      cfg.units, 0.0);
        layer.dW = Matrix(inFeat, cfg.units, 0.0);
        layer.db = Matrix(1,      cfg.units, 0.0);

        // Adam moments
        layer.mW  = Matrix(inFeat, cfg.units, 0.0);
        layer.vW  = Matrix(inFeat, cfg.units, 0.0);
        layer.mb  = Matrix(1,      cfg.units, 0.0);
        layer.vb  = Matrix(1,      cfg.units, 0.0);

        // SGD velocity
        layer.velW = Matrix(inFeat, cfg.units, 0.0);
        layer.velb = Matrix(1,      cfg.units, 0.0);

        layers_.push_back(std::move(layer));
        inFeat = cfg.units;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Weight initialisation
// ─────────────────────────────────────────────────────────────────────────────
void MLP::initWeights(Layer& layer, WeightInit init) {
    switch (init) {
        case WeightInit::Xavier:
            layer.W.xavierUniform(layer.inFeatures, layer.units); break;
        case WeightInit::He:
            layer.W.heNormal(layer.inFeatures); break;
        case WeightInit::RandomUniform:
            layer.W.randomUniform(-0.5, 0.5); break;
        case WeightInit::RandomNormal:
            layer.W.randomNormal(0.0, 0.01); break;
    }
    layer.b.fill(0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Activation functions
// ─────────────────────────────────────────────────────────────────────────────
Matrix MLP::softmax(const Matrix& Z) const {
    // Numerically stable: subtract row-max before exp
    Matrix out(Z.rows, Z.cols);
    for (int r = 0; r < Z.rows; ++r) {
        double rowMax = Z.data[r * Z.cols];
        for (int c = 1; c < Z.cols; ++c)
            if (Z.data[r*Z.cols+c] > rowMax) rowMax = Z.data[r*Z.cols+c];

        double rowSum = 0.0;
        for (int c = 0; c < Z.cols; ++c) {
            out.data[r*Z.cols+c] = std::exp(Z.data[r*Z.cols+c] - rowMax);
            rowSum += out.data[r*Z.cols+c];
        }
        for (int c = 0; c < Z.cols; ++c)
            out.data[r*Z.cols+c] /= rowSum;
    }
    return out;
}

Matrix MLP::applyActivation(const Matrix& Z, Activation act) const {
    switch (act) {
        case Activation::ReLU:
            return Z.apply([](double x){ return x > 0.0 ? x : 0.0; });

        case Activation::LeakyReLU:
            return Z.apply([](double x){ return x > 0.0 ? x : 0.01 * x; });

        case Activation::Sigmoid:
            return Z.apply([](double x){ return 1.0 / (1.0 + std::exp(-x)); });

        case Activation::Tanh:
            return Z.apply([](double x){ return std::tanh(x); });

        case Activation::Linear:
            return Z;   // identity

        case Activation::Softmax:
            return softmax(Z);
    }
    return Z;
}

// Derivative of activation w.r.t. Z (element-wise).
// For softmax the full Jacobian is handled in the loss gradient, so we return ones here.
Matrix MLP::activationGrad(const Matrix& Z, const Matrix& A, Activation act) const {
    switch (act) {
        case Activation::ReLU:
            return Z.apply([](double x){ return x > 0.0 ? 1.0 : 0.0; });

        case Activation::LeakyReLU:
            return Z.apply([](double x){ return x > 0.0 ? 1.0 : 0.01; });

        case Activation::Sigmoid:
            // σ'(z) = σ(z)·(1−σ(z)) = A·(1−A)
            return A * (A * (-1.0) + 1.0);

        case Activation::Tanh:
            // tanh'(z) = 1 − tanh²(z) = 1 − A²
            return A.square() * (-1.0) + 1.0;

        case Activation::Linear:
        case Activation::Softmax:
            // Softmax gradient is folded into loss gradient (see backward())
            return Matrix(Z.rows, Z.cols, 1.0);
    }
    return Matrix(Z.rows, Z.cols, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mode switching
// ─────────────────────────────────────────────────────────────────────────────
void MLP::setTraining(bool training) { isTraining_ = training; }

// ─────────────────────────────────────────────────────────────────────────────
//  Forward pass
//  X : (batch, inputSize)  →  returns (batch, outputUnits)
// ─────────────────────────────────────────────────────────────────────────────
Matrix MLP::forward(const Matrix& X) {
    Matrix current = X;

    for (auto& layer : layers_) {
        // ── Linear transform  Z = X·W + b ─────────────────────────────────────
        layer.Z = current.dot(layer.W).broadcastAdd(layer.b);

        // ── Activation ────────────────────────────────────────────────────────
        layer.A = applyActivation(layer.Z, layer.activation);

        // ── Dropout (only during training) ────────────────────────────────────
        if (isTraining_ && layer.dropout > 0.0) {
            layer.dropMask = Matrix(layer.A.rows, layer.A.cols);
            double keep = 1.0 - layer.dropout;
            for (int i = 0; i < layer.dropMask.rows * layer.dropMask.cols; ++i) {
                double r = (double)std::rand() / RAND_MAX;
                layer.dropMask.data[i] = (r < keep) ? (1.0 / keep) : 0.0;
            }
            layer.A = layer.A * layer.dropMask;
        } else {
            // During eval, dropMask is all-ones (no scaling)
            layer.dropMask = Matrix(layer.A.rows, layer.A.cols, 1.0);
        }

        current = layer.A;
    }

    return current;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Loss computation
// ─────────────────────────────────────────────────────────────────────────────
double MLP::computeLoss(const Matrix& preds, const Matrix& targets, Loss lossType) {
    int n = preds.rows;
    switch (lossType) {

        case Loss::MSE: {
            Matrix diff = preds - targets;
            return diff.square().sum() / (2.0 * n);
        }

        case Loss::BinaryCrossEntropy: {
            // −(y·log(p) + (1−y)·log(1−p))
            double eps = 1e-12;
            double loss = 0.0;
            for (int i = 0; i < preds.rows * preds.cols; ++i) {
                double p = std::max(eps, std::min(1.0 - eps, preds.data[i]));
                double y = targets.data[i];
                loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
            }
            return loss / n;
        }

        case Loss::CategoricalCrossEntropy: {
            // −sum( y·log(p) )
            double eps = 1e-12;
            double loss = 0.0;
            for (int i = 0; i < preds.rows * preds.cols; ++i) {
                double p = std::max(eps, preds.data[i]);
                loss += -targets.data[i] * std::log(p);
            }
            return loss / n;
        }
    }
    return 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward pass  (full backpropagation)
// ─────────────────────────────────────────────────────────────────────────────
void MLP::backward(const Matrix& X,
                   const Matrix& targets,
                   Loss          lossType,
                   double        gradClip)
{
    int n = targets.rows;
    const Matrix& outputA = layers_.back().A;  // final predictions

    // ── Compute initial gradient dL/dA (output layer) ─────────────────────────
    Matrix dA(outputA.rows, outputA.cols);

    switch (lossType) {
        case Loss::MSE:
            // dL/dA = (A - Y) / n
            dA = (outputA - targets) / (double)n;
            break;

        case Loss::BinaryCrossEntropy:
            // dL/dA = (A - Y) / (A*(1-A)*n)   ... but combined with sigmoid grad
            // For sigmoid + BCE the combined gradient is simply (A−Y)/n
            dA = (outputA - targets) / (double)n;
            break;

        case Loss::CategoricalCrossEntropy:
            // For softmax + CCE the combined gradient is (A−Y)/n
            dA = (outputA - targets) / (double)n;
            break;
    }

    // ── Backpropagate through layers (reverse order) ───────────────────────────
    for (int l = (int)layers_.size() - 1; l >= 0; --l) {
        Layer& layer = layers_[l];

        // Apply dropout mask gradient
        dA = dA * layer.dropMask;

        // ── dL/dZ = dL/dA ⊙ σ'(Z) ────────────────────────────────────────────
        // Special case: for the output layer with BCE+Sigmoid or CCE+Softmax,
        // the combined gradient is already dA = (A−Y)/n, so we skip extra multiplication.
        Matrix dZ(dA.rows, dA.cols);
        bool isCombinedGrad =
            (l == (int)layers_.size() - 1) &&
            ((lossType == Loss::BinaryCrossEntropy && layer.activation == Activation::Sigmoid) ||
             (lossType == Loss::CategoricalCrossEntropy && layer.activation == Activation::Softmax));

        if (isCombinedGrad) {
            dZ = dA;
        } else {
            Matrix actGrad = activationGrad(layer.Z, layer.A, layer.activation);
            dZ = dA * actGrad;
        }

        // ── Get input to this layer (either X or previous layer's A) ───────────
        const Matrix& layerInput = (l == 0) ? X : layers_[l-1].A;

        // ── dL/dW = layerInput^T · dZ  /  (already normalised by n above) ──────
        layer.dW = layerInput.transpose().dot(dZ);

        // ── dL/db = sum over batch dim ─────────────────────────────────────────
        layer.db = dZ.sum(0);

        // ── Gradient clipping (by L2 norm) ────────────────────────────────────
        if (gradClip > 0.0) {
            auto clipGrad = [&](Matrix& g) {
                double norm = std::sqrt(g.square().sum());
                if (norm > gradClip) g *= (gradClip / norm);
            };
            clipGrad(layer.dW);
            clipGrad(layer.db);
        }

        // ── dL/dA_prev = dZ · W^T  (propagate to previous layer) ─────────────
        if (l > 0) dA = dZ.dot(layer.W.transpose());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SGD update  (with momentum)
//  velocity = momentum * velocity − lr * grad
//  W        = W + velocity
// ─────────────────────────────────────────────────────────────────────────────
void MLP::updateSGD(double lr, double momentum) {
    for (auto& layer : layers_) {
        layer.velW = layer.velW * momentum - layer.dW * lr;
        layer.velb = layer.velb * momentum - layer.db * lr;
        layer.W   += layer.velW;
        layer.b   += layer.velb;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Adam update
//  m = β₁·m + (1−β₁)·g
//  v = β₂·v + (1−β₂)·g²
//  m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
//  W = W − lr · m̂ / (√v̂ + ε)
// ─────────────────────────────────────────────────────────────────────────────
void MLP::updateAdam(double lr, double beta1, double beta2, double eps) {
    ++adamStep_;
    double bc1 = 1.0 - std::pow(beta1, adamStep_);   // bias correction 1
    double bc2 = 1.0 - std::pow(beta2, adamStep_);   // bias correction 2

    for (auto& layer : layers_) {
        // ── Weights ───────────────────────────────────────────────────────────
        layer.mW = layer.mW * beta1 + layer.dW * (1.0 - beta1);
        layer.vW = layer.vW * beta2 + layer.dW.square() * (1.0 - beta2);

        Matrix mWhat = layer.mW / bc1;
        Matrix vWhat = layer.vW / bc2;
        layer.W -= (mWhat / (vWhat.sqrt() + eps)) * lr;

        // ── Biases ────────────────────────────────────────────────────────────
        layer.mb = layer.mb * beta1 + layer.db * (1.0 - beta1);
        layer.vb = layer.vb * beta2 + layer.db.square() * (1.0 - beta2);

        Matrix mbhat = layer.mb / bc1;
        Matrix vbhat = layer.vb / bc2;
        layer.b -= (mbhat / (vbhat.sqrt() + eps)) * lr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Training loop — mini-batch gradient descent
// ─────────────────────────────────────────────────────────────────────────────
void MLP::fit(const Matrix& X, const Matrix& Y, const TrainConfig& cfg) {
    // Initialise weights
    for (auto& layer : layers_)
        initWeights(layer, cfg.weightInit);

    adamStep_ = 0;
    lossHistory_.clear();

    int n = X.rows;

    // Shuffle index array
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    setTraining(true);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        // Shuffle indices each epoch
        for (int i = n - 1; i > 0; --i) {
            int j = std::rand() % (i + 1);
            std::swap(idx[i], idx[j]);
        }

        double epochLoss = 0.0;
        int    numBatches = 0;

        for (int start = 0; start < n; start += cfg.batchSize) {
            int end = std::min(start + cfg.batchSize, n);
            int bs  = end - start;

            // ── Build mini-batch matrices ──────────────────────────────────────
            Matrix Xb(bs, X.cols), Yb(bs, Y.cols);
            for (int i = 0; i < bs; ++i) {
                int row = idx[start + i];
                for (int c = 0; c < X.cols; ++c) Xb.data[i*X.cols+c] = X.data[row*X.cols+c];
                for (int c = 0; c < Y.cols; ++c) Yb.data[i*Y.cols+c] = Y.data[row*Y.cols+c];
            }

            // ── Forward ───────────────────────────────────────────────────────
            Matrix preds = forward(Xb);

            // ── Loss ──────────────────────────────────────────────────────────
            double batchLoss = computeLoss(preds, Yb, cfg.loss);
            epochLoss += batchLoss;
            ++numBatches;

            // ── Backward ──────────────────────────────────────────────────────
            backward(Xb, Yb, cfg.loss, cfg.gradClip);

            // ── Update ────────────────────────────────────────────────────────
            if (cfg.optimizer == Optimizer::SGD)
                updateSGD(cfg.learningRate, cfg.momentum);
            else
                updateAdam(cfg.learningRate, cfg.momentum, cfg.beta2, cfg.epsilon);
        }

        double avgLoss = epochLoss / numBatches;
        lossHistory_.push_back(avgLoss);

        if (cfg.verbose && (epoch + 1) % cfg.printEvery == 0) {
            std::cout << "Epoch [" << std::setw(5) << epoch + 1 << "/" << cfg.epochs << "]"
                      << "  Loss: " << std::fixed << std::setprecision(6) << avgLoss
                      << "\n";
        }
    }

    setTraining(false);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Inference
// ─────────────────────────────────────────────────────────────────────────────
Matrix MLP::predict(const Matrix& X) {
    setTraining(false);
    return forward(X);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Metrics
// ─────────────────────────────────────────────────────────────────────────────
double MLP::accuracy(const Matrix& X, const Matrix& Y) {
    Matrix preds = predict(X);
    int correct = 0, total = X.rows;

    if (Y.cols == 1) {
        // Binary: threshold at 0.5
        for (int r = 0; r < total; ++r) {
            int pred_class  = preds.data[r] >= 0.5 ? 1 : 0;
            int true_class  = Y.data[r] >= 0.5      ? 1 : 0;
            if (pred_class == true_class) ++correct;
        }
    } else {
        // Multi-class: argmax
        for (int r = 0; r < total; ++r) {
            int predIdx = 0, trueIdx = 0;
            double predMax = preds.data[r*preds.cols], trueMax = Y.data[r*Y.cols];
            for (int c = 1; c < preds.cols; ++c) {
                if (preds.data[r*preds.cols+c] > predMax) { predMax = preds.data[r*preds.cols+c]; predIdx = c; }
                if (Y.data[r*Y.cols+c]          > trueMax) { trueMax = Y.data[r*Y.cols+c];          trueIdx = c; }
            }
            if (predIdx == trueIdx) ++correct;
        }
    }

    return (double)correct / total;
}

double MLP::mse(const Matrix& X, const Matrix& Y) {
    Matrix preds = predict(X);
    return computeLoss(preds, Y, Loss::MSE);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Summary
// ─────────────────────────────────────────────────────────────────────────────
void MLP::printSummary() const {
    auto actName = [](Activation a) -> std::string {
        switch(a) {
            case Activation::ReLU:      return "ReLU";
            case Activation::LeakyReLU: return "LeakyReLU";
            case Activation::Sigmoid:   return "Sigmoid";
            case Activation::Tanh:      return "Tanh";
            case Activation::Linear:    return "Linear";
            case Activation::Softmax:   return "Softmax";
        }
        return "?";
    };

    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout <<   "║          Synapse.cpp  —  MLP Architecture            ║\n";
    std::cout <<   "╠══════════════════════════════════════════════════════╣\n";
    std::cout <<   "║  Input size : " << std::setw(37) << inputSize_ << " ║\n";
    std::cout <<   "╠══════════════════════════════════════════════════════╣\n";

    long long totalParams = 0;
    for (int l = 0; l < (int)layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        long long params = (long long)layer.W.rows * layer.W.cols + layer.b.cols;
        totalParams += params;
        std::cout << "║  Layer " << l+1 << "  : "
                  << std::setw(3) << layer.inFeatures << " → " << std::setw(3) << layer.units
                  << "  |  " << std::setw(10) << actName(layer.activation)
                  << "  |  params: " << std::setw(7) << params
                  << "  ║\n";
    }

    std::cout <<   "╠══════════════════════════════════════════════════════╣\n";
    std::cout <<   "║  Total trainable params : " << std::setw(27) << totalParams << " ║\n";
    std::cout <<   "╚══════════════════════════════════════════════════════╝\n\n";
}

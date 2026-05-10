#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense> // For high-performance linear algebra operations (SIMD-optimized)

using namespace std;

// 1. DATA LOADING UTILITIES

vector<float> load_bias_csv(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) {
        cerr << "Error: Unable to open bias file " << filename << endl;
        return result;
    }
    while (getline(file, line)) {
        if (!line.empty()) result.push_back(stof(line));
    }
    return result;
}

vector<float> load_weight_csv_flat(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) {
        cerr << "Error: Unable to open weight file " << filename << endl;
        return result;
    }
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            if (!val.empty()) result.push_back(stof(val));
        }
    }
    return result;
}

// 2. CORE MATHEMATICS (LOW-LATENCY OPERATIONS WITH EIGEN)

void apply_relu(vector<float> &vec) {
    for (float &val : vec) {
        val = std::max(0.0f, val);
    }
}

/**
 * Performs a Linear Layer computation: Z = W * X + b.
 * - Uses Eigen::Map to map existing C++ memory (Zero Copy / Zero Allocation).
 * - Utilizes AVX/SIMD CPU instructions for parallel computation.
 * - .noalias() avoids temporary matrix allocations during multiplication.
 */
void linear_layer(const vector<float> &W, const vector<float> &X, const vector<float> &b, vector<float> &Z, int rows, int cols) {
    // Map raw 1D pointers to Eigen Matrices/Vectors
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matW(W.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> vecX(X.data(), cols);
    Eigen::Map<const Eigen::VectorXf> vecB(b.data(), rows);
    Eigen::Map<Eigen::VectorXf> vecZ(Z.data(), rows);

    // Highly optimized SIMD matrix multiplication
    vecZ.noalias() = matW * vecX + vecB;
}

// 3. EXECUTION ENGINE

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <S> <K> <T> <r> <sigma> (standardized values)" << endl;
        cerr << "Example: " << argv[0] << " -0.43 0.27 -0.75 -1.18 1.34" << endl;
        return 1;
    }

    vector<float> input_features;
    for (int i = 1; i <= 5; ++i) {
        input_features.push_back(stof(argv[i]));
    }

    string path = "model_weights/";

    vector<float> W1 = load_weight_csv_flat(path + "fc1.weight.csv");
    vector<float> b1 = load_bias_csv(path + "fc1.bias.csv");

    vector<float> W2 = load_weight_csv_flat(path + "fc2.weight.csv");
    vector<float> b2 = load_bias_csv(path + "fc2.bias.csv");

    vector<float> W3 = load_weight_csv_flat(path + "fc3.weight.csv");
    vector<float> b3 = load_bias_csv(path + "fc3.bias.csv");

    vector<float> W_out = load_weight_csv_flat(path + "output_layer.weight.csv");
    vector<float> b_out = load_bias_csv(path + "output_layer.bias.csv");

    if (W1.empty() || b_out.empty()) {
        cerr << "Fatal Error: Model weights not found in " << path << endl;
        return 1;
    }

    vector<float> z1(64, 0.0f);
    vector<float> z2(64, 0.0f);
    vector<float> z3(64, 0.0f);
    vector<float> output(1, 0.0f);

    // --- MICRO-BENCHMARKING SETUP ---
    int num_warmup = 1000;
    int num_runs = 10000;

    // 1. WARMUP (Fill L1/L2 Cache, wake up CPU)
    for(int i = 0; i < num_warmup; i++) {
        linear_layer(W1, input_features, b1, z1, 64, 5);
        apply_relu(z1);
        linear_layer(W2, z1, b2, z2, 64, 64);
        apply_relu(z2);
        linear_layer(W3, z2, b3, z3, 64, 64);
        apply_relu(z3);
        linear_layer(W_out, z3, b_out, output, 1, 64);
    }

    // 2. CRITICAL PATH (Averaged over 10,000 iterations)
    auto start_time = chrono::high_resolution_clock::now();

    for(int i = 0; i < num_runs; i++) {
        linear_layer(W1, input_features, b1, z1, 64, 5);
        apply_relu(z1);
        linear_layer(W2, z1, b2, z2, 64, 64);
        apply_relu(z2);
        linear_layer(W3, z2, b3, z3, 64, 64);
        apply_relu(z3);
        linear_layer(W_out, z3, b_out, output, 1, 64);
    }

    auto end_time = chrono::high_resolution_clock::now();
    
    // Calculate average latency
    auto total_duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
    long long avg_latency = total_duration.count() / num_runs;

    cout << "Predicted Option Price : " << output[0] << endl;
    cout << "Average Latency (" << num_runs << " runs) : " << avg_latency << " ns" << endl;

    return 0;
}
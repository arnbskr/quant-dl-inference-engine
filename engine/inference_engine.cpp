#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// 1. DATA LOADING UTILITIES

/**
 * Loads a 1D vector from a CSV file (used for biases).
 */
vector<float> load_bias_csv(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Unable to open bias file " << filename << endl;
        return result;
    }

    while (getline(file, line)) {
        if (!line.empty()) {
            result.push_back(stof(line));
        }
    }
    return result;
}

/**
 * Loads a 2D matrix into a single flat 1D vector.
 * This maximizes CPU cache hits by ensuring memory contiguity.
 */
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
            if (!val.empty()) {
                result.push_back(stof(val));
            }
        }
    }
    return result;
}

// 2. CORE MATHEMATICS (LOW-LATENCY OPERATIONS)

/**
 * Applies the Rectified Linear Unit (ReLU) activation function in-place.
 */
void apply_relu(vector<float> &vec) {
    for (float &val : vec) {
        val = std::max(0.0f, val);
    }
}

/**
 * Performs a Linear Layer computation: Z = W * X + b.
 * - Optimized for 1D flat weight matrices.
 * - Pre-allocated output buffer Z to avoid heap allocations.
 */
void linear_layer(const vector<float> &W, const vector<float> &X, const vector<float> &b, vector<float> &Z, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        int row_offset = i * cols;
        
        for (int j = 0; j < cols; ++j) {
            // Contiguous memory access for performance
            sum += W[row_offset + j] * X[j];
        }
        Z[i] = sum + b[i];
    }
}

// 3. EXECUTION ENGINE

int main(int argc, char* argv[]) {
    // Basic argument check: Expecting 5 standardized market parameters
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <S> <K> <T> <r> <sigma> (standardized values)" << endl;
        cerr << "Example: " << argv[0] << " -0.43 0.27 -0.75 -1.18 1.34" << endl;
        return 1;
    }

    // Parse input features from command line
    vector<float> input_features;
    for (int i = 1; i <= 5; ++i) {
        input_features.push_back(stof(argv[i]));
    }

    string path = "model_weights/";

    // 1. Loading model parameters into RAM
    vector<float> W1 = load_weight_csv_flat(path + "fc1.weight.csv");
    vector<float> b1 = load_bias_csv(path + "fc1.bias.csv");

    vector<float> W2 = load_weight_csv_flat(path + "fc2.weight.csv");
    vector<float> b2 = load_bias_csv(path + "fc2.bias.csv");

    vector<float> W3 = load_weight_csv_flat(path + "fc3.weight.csv");
    vector<float> b3 = load_bias_csv(path + "fc3.bias.csv");

    vector<float> W_out = load_weight_csv_flat(path + "output_layer.weight.csv");
    vector<float> b_out = load_bias_csv(path + "output_layer.bias.csv");

    // Fatal error if weights could not be loaded
    if (W1.empty() || b_out.empty()) {
        cerr << "Fatal Error: Model weights not found in " << path << endl;
        return 1;
    }

    // 2. Pre-allocate inference buffers
    // This avoids slow 'malloc' calls during the critical execution path
    vector<float> z1(64, 0.0f);
    vector<float> z2(64, 0.0f);
    vector<float> z3(64, 0.0f);
    vector<float> output(1, 0.0f);

    // CRITICAL PATH START
    auto start_time = chrono::high_resolution_clock::now();

    // Layer 1: 5 -> 64
    linear_layer(W1, input_features, b1, z1, 64, 5);
    apply_relu(z1);

    // Layer 2: 64 -> 64
    linear_layer(W2, z1, b2, z2, 64, 64);
    apply_relu(z2);

    // Layer 3: 64 -> 64
    linear_layer(W3, z2, b3, z3, 64, 64);
    apply_relu(z3);

    // Output Layer: 64 -> 1
    linear_layer(W_out, z3, b_out, output, 1, 64);

    auto end_time = chrono::high_resolution_clock::now();
    // CRITICAL PATH END
    
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);

    cout << "Predicted Option Price : " << output[0] << endl;
    cout << "Inference Latency      : " << duration.count() << " ns" << endl;

    return 0;
}
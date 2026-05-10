#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// 1. DATA LOADING

vector<float> load_bias_csv(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return result;
    }

    while (getline(file, line)) {
        result.push_back(stof(line));
    }
    return result;
}

// Load W matrix as a single contiguous 1D vector (Cache Optimization)
vector<float> load_weight_csv_flat(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return result;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string val;

        while (getline(ss, val, ',')) {
            result.push_back(stof(val));
        }
    }
    return result;
}

// 2. NETWORK MATHEMATICS (LOW LATENCY OPERATIONS)

void apply_relu(vector<float> &vec) {
    for (float &val : vec) {
        val = std::max(0.0f, val);
    }
}

// Optimized Forward Pass:
// - W is a 1D vector (CPU Cache optimization)
// - Z is passed by reference (Zero allocation during inference)
void linear_layer(const vector<float> &W, const vector<float> &X, const vector<float> &b, vector<float> &Z, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        int row_offset = i * cols; // Calculated once per row
        
        for (int j = 0; j < cols; ++j) {
            // Contiguous memory access W[row_offset + j]
            sum += W[row_offset + j] * X[j];
        }
        Z[i] = sum + b[i];
    }
}

// 3. MAIN ENGINE

int main() {
    string path = "model_weights/";

    // 1. Load weights in flat 1D format
    vector<float> W1 = load_weight_csv_flat(path + "fc1.weight.csv");
    vector<float> b1 = load_bias_csv(path + "fc1.bias.csv");

    vector<float> W2 = load_weight_csv_flat(path + "fc2.weight.csv");
    vector<float> b2 = load_bias_csv(path + "fc2.bias.csv");

    vector<float> W3 = load_weight_csv_flat(path + "fc3.weight.csv");
    vector<float> b3 = load_bias_csv(path + "fc3.bias.csv");

    vector<float> W_out = load_weight_csv_flat(path + "output_layer.weight.csv");
    vector<float> b_out = load_bias_csv(path + "output_layer.bias.csv");

    if (W1.empty() || b_out.empty()) {
        cerr << "Fatal Error: Failed to load weights." << endl;
        return 1;
    }

    vector<float> input_features = load_bias_csv(path + "current_input.csv");
    if (input_features.size() != 5) {
        cerr << "Fatal Error: Invalid or missing input file." << endl;
        return 1;
    }

    // 2. Pre-allocate memory buffers (before the timer), ensures no heap allocation (malloc/new) happens during trading logic.
    vector<float> z1(64, 0.0f);
    vector<float> z2(64, 0.0f);
    vector<float> z3(64, 0.0f);
    vector<float> output(1, 0.0f);

    // INFERENCE START & TIMER
    auto start_time = chrono::high_resolution_clock::now();

    // Layer 1: 5 inputs -> 64 outputs
    linear_layer(W1, input_features, b1, z1, 64, 5);
    apply_relu(z1);

    // Layer 2: 64 inputs -> 64 outputs
    linear_layer(W2, z1, b2, z2, 64, 64);
    apply_relu(z2);

    // Layer 3: 64 inputs -> 64 outputs
    linear_layer(W3, z2, b3, z3, 64, 64);
    apply_relu(z3);

    // Output Layer: 64 inputs -> 1 output (Linear Activation)
    linear_layer(W_out, z3, b_out, output, 1, 64);

    // INFERENCE END & TIMER
    auto end_time = chrono::high_resolution_clock::now();
    
    // Measure in nanoseconds
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);

    cout << "Predicted Option Price : " << output[0] << endl;
    cout << "Inference Latency      : " << duration.count() << " ns" << endl;

    return 0;
}
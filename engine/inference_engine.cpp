#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;

// 1. DATA LOADING UTILITIES (BINARY ZERO-PARSING)

/**
 * Loads a binary file directly into a float vector.
 * No string parsing (stof), extremely fast memory mapping.
 */
vector<float> load_binary_1d(const string &filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) return {};
    
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    return {};
}

// 2. CORE MATHEMATICS (LOW-LATENCY OPERATIONS)

void apply_relu(vector<float> &vec) {
    for (float &val : vec) val = std::max(0.0f, val);
}

void linear_layer(const vector<float> &W, const vector<float> &X, const vector<float> &b, vector<float> &Z, int rows, int cols) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matW(W.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> vecX(X.data(), cols);
    Eigen::Map<const Eigen::VectorXf> vecB(b.data(), rows);
    Eigen::Map<Eigen::VectorXf> vecZ(Z.data(), rows);
    vecZ.noalias() = matW * vecX + vecB;
}

// 3. EXECUTION ENGINE (HYBRID MODE)

int main(int argc, char* argv[]) {
    bool is_batch_mode = (argc == 1); 
    
    if (!is_batch_mode && argc != 7) {
        cerr << "Usage Single Mode: " << argv[0] << " <S> <K> <T> <r> <log_moneyness> <sqrt_T>" << endl;
        return 1;
    }

    string path = "model_weights/";

    // Load Neural Network Weights (BINARY)
    vector<float> W1 = load_binary_1d(path + "fc1.weight.bin");
    vector<float> b1 = load_binary_1d(path + "fc1.bias.bin");
    vector<float> W2 = load_binary_1d(path + "fc2.weight.bin");
    vector<float> b2 = load_binary_1d(path + "fc2.bias.bin");
    vector<float> W3 = load_binary_1d(path + "fc3.weight.bin");
    vector<float> b3 = load_binary_1d(path + "fc3.bias.bin");
    vector<float> W_out = load_binary_1d(path + "output_layer.weight.bin");
    vector<float> b_out = load_binary_1d(path + "output_layer.bias.bin");

    vector<float> scaler_mean = load_binary_1d(path + "scaler_mean.bin");
    vector<float> scaler_scale = load_binary_1d(path + "scaler_scale.bin");

    if (W1.empty() || scaler_mean.empty()) {
        cerr << "Fatal Error: Model weights or scaler parameters not found (Ensure .bin files exist)." << endl;
        return 1;
    }

    vector<float> z1(64, 0.0f), z2(64, 0.0f), z3(64, 0.0f), output(1, 0.0f);

    // PATH A : STREAMLIT BATCH MODE
    if (is_batch_mode) {
        // Read the entire batch instantly in binary
        vector<float> flat_batch_inputs = load_binary_1d("batch_inputs.bin");
        if (flat_batch_inputs.empty()) return 1;
        
        int num_options = flat_batch_inputs.size() / 6;

        // 1. WARMUP
        vector<float> dummy_input(6, 0.5f);
        for(int i = 0; i < 5000; i++) {
            linear_layer(W1, dummy_input, b1, z1, 64, 6);
            apply_relu(z1);
            linear_layer(W2, z1, b2, z2, 64, 64);
            apply_relu(z2);
            linear_layer(W3, z2, b3, z3, 64, 64);
            apply_relu(z3);
            linear_layer(W_out, z3, b_out, output, 1, 64);
        }

        // Open binary output stream
        ofstream outfile("batch_outputs.bin", ios::binary);
        vector<float> scaled_input(6, 0.0f);

        // CRITICAL PATH START
        auto start_time = chrono::high_resolution_clock::now();

        for (int opt = 0; opt < num_options; ++opt) {
            for(int i = 0; i < 6; ++i) {
                scaled_input[i] = (flat_batch_inputs[opt * 6 + i] - scaler_mean[i]) / scaler_scale[i];
            }

            linear_layer(W1, scaled_input, b1, z1, 64, 6);
            apply_relu(z1);
            linear_layer(W2, z1, b2, z2, 64, 64);
            apply_relu(z2);
            linear_layer(W3, z2, b3, z3, 64, 64);
            apply_relu(z3);
            linear_layer(W_out, z3, b_out, output, 1, 64);

            // Write raw float bits to file
            outfile.write(reinterpret_cast<const char*>(&output[0]), sizeof(float));
        }

        auto end_time = chrono::high_resolution_clock::now();
        // CRITICAL PATH END

        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);
        cout << "Latency:" << duration.count() << "ns" << endl;
    } 
    // PATH B : TERMINAL SINGLE MODE
    else {
        vector<float> input_features;
        for (int i = 1; i <= 6; ++i) input_features.push_back(stof(argv[i]));

        vector<float> scaled_input(6, 0.0f);
        for(int i = 0; i < 6; ++i) scaled_input[i] = (input_features[i] - scaler_mean[i]) / scaler_scale[i];

        auto start_time = chrono::high_resolution_clock::now();

        linear_layer(W1, scaled_input, b1, z1, 64, 6);
        apply_relu(z1);
        linear_layer(W2, z1, b2, z2, 64, 64);
        apply_relu(z2);
        linear_layer(W3, z2, b3, z3, 64, 64);
        apply_relu(z3);
        linear_layer(W_out, z3, b_out, output, 1, 64);

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);

        cout << "Predicted Option Price : " << output[0] << " $" << endl;
        cout << "Inference Latency      : " << duration.count() << " ns" << endl;
    }

    return 0;
}
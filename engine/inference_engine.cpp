#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;

// 1. DATA LOADING UTILITIES

vector<float> load_csv_1d(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) return result;
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            if (!val.empty()) result.push_back(stof(val));
        }
    }
    return result;
}

vector<vector<float>> load_csv_2d(const string &filename) {
    vector<vector<float>> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) return result;
    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            if (!val.empty()) row.push_back(stof(val));
        }
        if (!row.empty()) result.push_back(row);
    }
    return result;
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
    bool is_batch_mode = (argc == 1); // Streamlit calls it without arguments
    
    // Check arguments for Single Mode (Expecting 6 features now)
    if (!is_batch_mode && argc != 7) {
        cerr << "Usage for Single Mode: " << argv[0] << " <S> <K> <T> <r> <log_moneyness> <sqrt_T>" << endl;
        cerr << "Usage for Batch Mode : " << argv[0] << " (Reads batch_inputs.csv automatically)" << endl;
        return 1;
    }

    string path = "model_weights/";

    // Load Neural Network Weights
    vector<float> W1 = load_csv_1d(path + "fc1.weight.csv");
    vector<float> b1 = load_csv_1d(path + "fc1.bias.csv");
    vector<float> W2 = load_csv_1d(path + "fc2.weight.csv");
    vector<float> b2 = load_csv_1d(path + "fc2.bias.csv");
    vector<float> W3 = load_csv_1d(path + "fc3.weight.csv");
    vector<float> b3 = load_csv_1d(path + "fc3.bias.csv");
    vector<float> W_out = load_csv_1d(path + "output_layer.weight.csv");
    vector<float> b_out = load_csv_1d(path + "output_layer.bias.csv");

    // Load Scikit-Learn Scaler parameters
    vector<float> scaler_mean = load_csv_1d(path + "scaler_mean.csv");
    vector<float> scaler_scale = load_csv_1d(path + "scaler_scale.csv");

    if (W1.empty() || scaler_mean.empty()) {
        cerr << "Fatal Error: Model weights or scaler parameters not found." << endl;
        return 1;
    }

    // Pre-allocate Memory Buffers
    vector<float> z1(64, 0.0f), z2(64, 0.0f), z3(64, 0.0f), output(1, 0.0f);

    // PATH A : STREAMLIT BATCH MODE
    if (is_batch_mode) {
        vector<vector<float>> batch_inputs = load_csv_2d("batch_inputs.csv");
        if (batch_inputs.empty()) {
            cerr << "Fatal Error: batch_inputs.csv not found or empty." << endl;
            return 1;
        }

        // 1. WARMUP (Fill L1/L2 Cache, wake up CPU from idle state)
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

        ofstream outfile("batch_outputs.csv");
        
        // 2. PRE-ALLOCATE SCALING BUFFER (Outside the loop!)
        vector<float> scaled_input(6, 0.0f);

        // CRITICAL PATH START
        auto start_time = chrono::high_resolution_clock::now();

        for (const auto& row : batch_inputs) {
            // Overwrite existing buffer (Zero Allocation)
            for(int i = 0; i < 6; ++i) {
                scaled_input[i] = (row[i] - scaler_mean[i]) / scaler_scale[i];
            }

            // Forward Pass
            linear_layer(W1, scaled_input, b1, z1, 64, 6);
            apply_relu(z1);
            linear_layer(W2, z1, b2, z2, 64, 64);
            apply_relu(z2);
            linear_layer(W3, z2, b3, z3, 64, 64);
            apply_relu(z3);
            linear_layer(W_out, z3, b_out, output, 1, 64);

            outfile << output[0] << "\n";
        }

        auto end_time = chrono::high_resolution_clock::now();
        // CRITICAL PATH END

        auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time);

        // Required print format for Streamlit parsing
        cout << "Latency:" << duration.count() << "ns" << endl;
    }
    // PATH B : TERMINAL SINGLE MODE
    else {
        vector<float> input_features;
        for (int i = 1; i <= 6; ++i) {
            input_features.push_back(stof(argv[i]));
        }

        vector<float> scaled_input(6, 0.0f);
        for(int i = 0; i < 6; ++i) {
            scaled_input[i] = (input_features[i] - scaler_mean[i]) / scaler_scale[i];
        }

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
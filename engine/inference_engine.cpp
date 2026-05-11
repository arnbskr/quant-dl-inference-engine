#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

using namespace std;

// 1. DATA LOADING UTILITIES (BINARY)
vector<float> load_binary_1d(const string &filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) return {};
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) return buffer;
    return {};
}

// 2. CORE MATHEMATICS
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

// 3. TCP SOCKET DAEMON ENGINE
int main() {
    string path = "model_weights/";
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
        cerr << "Fatal Error: Missing .bin weights." << endl; return 1;
    }

    // TCP Server Initialization
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) return 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(5555);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) return 1;
    if (listen(server_fd, 3) < 0) return 1;

    cout << "C++ Inference Daemon listening on local port 5555..." << endl;

    vector<float> z1(64, 0.0f), z2(64, 0.0f), z3(64, 0.0f), output(1, 0.0f);
    vector<float> scaled_input(6, 0.0f);

    while(true) {
        new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (new_socket < 0) continue;

        int32_t num_options;
        if (read(new_socket, &num_options, sizeof(int32_t)) <= 0) { close(new_socket); continue; }

        vector<float> flat_batch_inputs(num_options * 6);
        size_t bytes_to_read = num_options * 6 * sizeof(float);
        size_t total_read = 0;
        char* ptr = reinterpret_cast<char*>(flat_batch_inputs.data());
        
        while (total_read < bytes_to_read) {
            ssize_t bytes = read(new_socket, ptr + total_read, bytes_to_read - total_read);
            if (bytes <= 0) break;
            total_read += bytes;
        }

        vector<float> batch_outputs(num_options);

        // CRITICAL PATH START
        auto start_time = chrono::high_resolution_clock::now();

        for (int opt_idx = 0; opt_idx < num_options; ++opt_idx) {
            for(int i = 0; i < 6; ++i) {
                scaled_input[i] = (flat_batch_inputs[opt_idx * 6 + i] - scaler_mean[i]) / scaler_scale[i];
            }
            linear_layer(W1, scaled_input, b1, z1, 64, 6);
            apply_relu(z1);
            linear_layer(W2, z1, b2, z2, 64, 64);
            apply_relu(z2);
            linear_layer(W3, z2, b3, z3, 64, 64);
            apply_relu(z3);
            linear_layer(W_out, z3, b_out, output, 1, 64);
            batch_outputs[opt_idx] = output[0];
        }

        auto end_time = chrono::high_resolution_clock::now();
        // CRITICAL PATH END

        int64_t latency_ns = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();

        // Write raw bits directly into the socket stream back to Python
        send(new_socket, &latency_ns, sizeof(int64_t), 0);
        send(new_socket, batch_outputs.data(), num_options * sizeof(float), 0);
        
        close(new_socket);
    }
    return 0;
}
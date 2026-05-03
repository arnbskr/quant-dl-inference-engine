#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

using namespace std;

// 1. Fonctions de chargement des données (CSV) optimisées pour la performance

vector<float> load_1d_csv(const string &filename) {
    vector<float> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) {
        cerr << "Erreur : Impossible d'ouvrir " << filename << endl;
        exit(1);
    }
    while (getline(file, line)) {
        result.push_back(stof(line));
    }
    return result;
}

vector<vector<float>> load_2d_csv(const string &filename) {
    vector<vector<float>> result;
    ifstream file(filename);
    string line;
    if (!file.is_open()) {
        cerr << "Erreur : Impossible d'ouvrir " << filename << endl;
        exit(1);
    }
    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            row.push_back(stof(val));
        }
        result.push_back(row);
    }
    return result;
}

// 2. Opérations de base du réseau de neurones optimisées pour la performance

// Passage par référence (&) pour modifier sur place sans copier la mémoire
void apply_relu(vector<float> &vec) {
    for (float &val : vec) {
        val = std::max(0.0f, val);
    }
}

// Passage par référence constante (const &) pour éviter les copies inutiles
vector<float> linear_layer(const vector<vector<float>> &W, const vector<float> &X, const vector<float> &b) {
    vector<float> Z(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < X.size(); ++j) {
            sum += W[i][j] * X[j];
        }
        Z[i] = sum + b[i];
    }
    return Z;
}

// 3. Moteur d'inférence principal optimisé pour la performance

int main(int argc, char* argv[]) {
    // Le binaire prend désormais les paramètres directement en argument
    if (argc != 5) {
        cerr << "Usage: ./inference_engine <S> <K> <T> <r>" << endl;
        cerr << "Exemple: ./inference_engine 150.0 155.0 0.5 0.02" << endl;
        return 1;
    }

    // Récupération des inputs bruts du marché
    vector<float> raw_input = {
        stof(argv[1]), stof(argv[2]), stof(argv[3]), stof(argv[4])
    };

    string path = "model_weights/";

    // 1. Chargement du scaler
    vector<float> scaler_mean = load_1d_csv(path + "scaler_mean.csv");
    vector<float> scaler_scale = load_1d_csv(path + "scaler_scale.csv");

    // 2. Chargement des poids
    vector<vector<float>> W1 = load_2d_csv(path + "fc1.weight.csv");
    vector<float> b1 = load_1d_csv(path + "fc1.bias.csv");
    vector<vector<float>> W2 = load_2d_csv(path + "fc2.weight.csv");
    vector<float> b2 = load_1d_csv(path + "fc2.bias.csv");
    vector<vector<float>> W3 = load_2d_csv(path + "fc3.weight.csv");
    vector<float> b3 = load_1d_csv(path + "fc3.bias.csv");
    vector<vector<float>> W_out = load_2d_csv(path + "output_layer.weight.csv");
    vector<float> b_out = load_1d_csv(path + "output_layer.bias.csv");

    // Début de l'inférence, chronométré pour mesurer la latence
    auto start_time = chrono::high_resolution_clock::now();

    // 3. Standardisation des inputs (scaling)
    vector<float> scaled_input(4);
    for (size_t i = 0; i < 4; ++i) {
        scaled_input[i] = (raw_input[i] - scaler_mean[i]) / scaler_scale[i];
    }

    // 4. Inférence dans le MLP
    vector<float> z1 = linear_layer(W1, scaled_input, b1);
    apply_relu(z1);
    vector<float> z2 = linear_layer(W2, z1, b2);
    apply_relu(z2);
    vector<float> z3 = linear_layer(W3, z2, b3);
    apply_relu(z3);
    vector<float> output = linear_layer(W_out, z3, b_out);

    auto end_time = chrono::high_resolution_clock::now();
    // Fin de l'inférence, calcul de la durée

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

    // Formatage spécial pour que le script Python/Streamlit puisse lire la sortie facilement
    cout << "PRICE:" << output[0] << ",LATENCY:" << duration.count() << endl;

    return 0;
}
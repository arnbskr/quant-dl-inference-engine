#!/bin/bash

# Low-latency quantitative inference pipeline for Deep Learning in C++ with a Streamlit interface

echo -e "\e[1;36m[1/3] Checking model weights...\e[0m"

# Check if model_weights directory exists, otherwise train the model
if [ ! -d "model_weights" ]; then
    echo -e "\e[33mInitial training required. Launching train_model.py...\e[0m"
    python research/train_model.py
    if [ $? -ne 0 ]; then
        echo -e "\e[31mError during model training.\e[0m"
        exit 1
    fi
else
    echo -e "\e[32m✔ Weights found in model_weights/\e[0m"
fi

echo -e "\e[1;36m[2/3] Compiling C++ Engine (O3 Optimization)...\e[0m"
g++ -O3 engine/inference_engine.cpp -o engine/inference_engine
if [ $? -ne 0 ]; then
    echo -e "\e[31mC++ Compilation Error.\e[0m"
    exit 1
fi
echo -e "\e[32m✔ Compilation successful.\e[0m"

echo -e "\e[1;36m[3/3] Launching Live Trading Interface (Streamlit)...\e[0m"
echo -e "\e[33mPress Ctrl+C to stop the server.\e[0m"
streamlit run dashboard/app.py
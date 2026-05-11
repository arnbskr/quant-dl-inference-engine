#!/bin/bash

# Safety: Kill any lingering Daemon processes stuck in the background
trap "pkill -f inference_engine" EXIT
pkill -f inference_engine 2>/dev/null

echo -e "\e[1;36m[1/3] Checking model weights...\e[0m"
if [ ! -d "model_weights" ]; then
    echo -e "\e[33mInitial training required. Launching train_model.py...\e[0m"
    python research/train_model.py
    if [ $? -ne 0 ]; then exit 1; fi
fi

echo -e "\e[1;36m[2/3] Compiling C++ Engine (O3 Optimization)...\e[0m"
g++ -O3 -march=native -I/usr/include/eigen3 engine/inference_engine.cpp -o engine/inference_engine
if [ $? -ne 0 ]; then exit 1; fi

echo -e "\e[1;36m[3/3] Launching C++ Daemon & Streamlit...\e[0m"
# Launch the C++ Daemon in the background
./engine/inference_engine &
sleep 1 # Allow 1 second for the TCP server to initialize and bind to the port

echo -e "\e[33mPress Ctrl+C to stop the server and the daemon.\e[0m"
streamlit run dashboard/app.py
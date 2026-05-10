# Quant Deep Learning Inference Engine & Arbitrage Scanner

**Low-Latency C++ Deep Learning Inference Engine for Non-Linear Options Pricing**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

This project demonstrates the design and deployment of an end-to-end quantitative development pipeline. It bridges the gap between the predictive power of deep neural networks (MLPs) and the ultra-low latency requirements of High-Frequency Trading (HFT) and Statistical Arbitrage.

## Project Overview

Derivatives pricing often faces a dilemma: using fast but imperfect analytical formulas (e.g., Black-Scholes), or precise but computationally expensive stochastic/local volatility models. 

This project tackles this bottleneck through a hybrid architecture:
1. **Modeling (Python / PyTorch):** Training a Multi-Layer Perceptron (MLP) on synthetic data to approximate complex, non-linear pricing functions (e.g., Volatility Smiles).
2. **Inference Engine (Native C++):** Exporting model weights to a custom, lightweight C++ inference engine that executes the forward pass entirely independently of heavy ML frameworks.
3. **Live Application (Streamlit):** A real-time dashboard fetching live market data (via Yahoo Finance), running the C++ engine for theoretical pricing, and scanning for arbitrage opportunities (model vs. market spreads) in microseconds.

## Repository Structure

```text
├── app.py                     # Streamlit Dashboard (Live market scanner interface)
├── inference_engine.cpp       # Native low-latency inference engine
├── train_model.py             # Data generation, PyTorch MLP training, and weight export
├── plot_engineering.py        # Latency visualization script
├── plot_results.py            # AI performance visualization script
├── run.sh                     # Automation script (Compilation & Execution)
├── requirements.txt           # Python dependencies
└── model_weights/             # Exported tensors and parameters (auto-generated)
    ├── fc1.weight.csv
    ├── scaler_mean.csv
    └── ...
```

## Current Status & Low-Latency Roadmap

**Current Performance:** The engine currently achieves a microsecond-level pure inference latency (~3.00 µs per option) using standard STL containers.

While this is extremely fast for standard applications, true HFT requires nanosecond-scale latency. **The codebase is actively being refactored to implement the following quantitative engineering standards:**

* [ ] **Memory Management:** Transitioning from heap allocation (`std::vector`) to strict stack allocation (`std::array` or raw arrays) to eliminate dynamic memory overhead.
* [ ] **Cache Line Optimization:** Flattening 2D weight matrices into contiguous 1D arrays to maximize L1/L2 CPU cache hits and prevent pointer chasing.
* [ ] **Vectorization (SIMD):** Replacing naive matrix multiplication loops with SIMD intrinsic instructions (AVX2/AVX-512) for single-clock-cycle parallel computing.
* [ ] **Binary Serialization:** Replacing `.csv` parsing with a binary weight format to drastically reduce engine initialization time.

## Prerequisites

* **Python 3.8+**
* **C++ Compiler** supporting C++17 (e.g., `g++` or `clang++`).
* OS: Linux, macOS, or Windows (via WSL/MinGW).

## Installation & Quick Start

### Step 1: Clone and Setup Environment

It is highly recommended to use a Python virtual environment.

```bash
git clone [https://github.com/arnbskr/quant-dl-inference-engine.git](https://github.com/arnbskr/quant-dl-inference-engine.git)
cd quant-dl-inference-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train Model & Export Weights

Generate the synthetic options data, train the PyTorch MLP, and export the weights into the `model_weights/` directory.

```bash
python train_model.py
```

### Step 3: Compile the C++ Inference Engine

Compile the native engine with the maximum optimization flag (`-O3`).

```bash
g++ -O3 inference_engine.cpp -o inference_engine
```

### Step 4: Run the Live Arbitrage Scanner

Launch the Streamlit dashboard to see the MLOps pipeline scan live market anomalies.

```bash
streamlit run app.py
```

## References

This system architecture was inspired by the following quantitative research papers:

* Culkin, R., & Das, S. R. (2017). *Machine Learning in Finance: The Case of Deep Learning for Option Pricing.* Santa Clara University.
* Ke, A., & Yang, A. (2019). *Option Pricing with Deep Learning.* Stanford University.

---

*Developed by Arnold Baskar - 2026*
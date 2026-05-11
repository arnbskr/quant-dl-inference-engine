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
1. **Research & Modeling (Python / PyTorch):** Training a Multi-Layer Perceptron (MLP) on synthetic data to approximate complex, non-linear pricing functions (e.g., Volatility Smiles).
2. **Inference Engine (Native C++):** Exporting model weights to a custom, lightweight C++ inference engine that executes the forward pass entirely independently of heavy ML frameworks.
3. **Live Application (Streamlit):** A real-time dashboard fetching live market data (via Yahoo Finance), communicating with the C++ Daemon via TCP Sockets, and scanning for arbitrage opportunities (model vs. market spreads) in microseconds.

## Repository Structure

```text
quant-dl-inference-engine/
├── assets/                    # Images and plots for documentation
├── dashboard/                 # Live market scanner interface
│   └── app.py
├── engine/                    # Native low-latency inference engine
│   └── inference_engine.cpp
├── research/                  # Data generation, PyTorch training, and analytics
│   ├── train_model.py
│   ├── plot_engineering.py
│   └── plot_results.py
├── model_weights/             # Exported binary tensors (.bin)
├── run.sh                     # Automation script (Compilation, Daemon & UI)
└── requirements.txt           # Python dependencies
```

## Performance & Low-Latency Engineering

The engine achieves an ultra-low pure inference latency of **< 500 nanoseconds** per option on a standard CPU, and an End-to-End system latency (Python UI to C++ and back) of **~25 microseconds**.

This high-frequency trading (HFT) standard was achieved by implementing strict quantitative engineering practices:

* [x] **Memory Management:** Zero dynamic allocation (no `malloc`/`new`) during the critical execution path via pre-allocated buffers and `Eigen::Map`.
* [x] **Cache Line Optimization:** Complete elimination of `std::vector<vector<float>>` pointer chasing by flattening 2D weight matrices into contiguous 1D arrays, maximizing L1/L2 CPU cache hits.
* [x] **Vectorization (SIMD):** Integration of the `Eigen` library and `-march=native` compiler flags to execute SIMD intrinsic instructions (AVX/AVX2) for single-clock-cycle parallel computing.
* [x] **Zero-Overhead IPC (Inter-Process Communication):** Replaced slow disk I/O and CSV parsing with a persistent C++ TCP Daemon exchanging raw binary data (`np.float32` <-> `float`) with Python via local sockets.
* [x] **Micro-Benchmarking:** Implementation of CPU warm-up cycles to mitigate OS jitter and cold-cache penalties during performance measurement.

## Prerequisites

* **Python 3.8+**
* **C++ Compiler** supporting C++17 (e.g., `g++` or `clang++`)
* **Eigen3 Library** (C++ template library for linear algebra)
* *Linux (Fedora/RHEL):* `sudo dnf install eigen3-devel`
* *Linux (Ubuntu/Debian):* `sudo apt install libeigen-dev`
* *macOS:* `brew install eigen`

## Installation & Quick Start

### Step 1: Clone and Setup Environment

It is highly recommended to use a Python virtual environment.

```bash
git clone https://github.com/arnbskr/quant-dl-inference-engine.git
cd quant-dl-inference-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run the Pipeline (Automated Way)

The easiest way to start the project is to use the provided bash script. It will automatically train the model (if weights are missing), compile the C++ engine, launch the TCP Daemon in the background, and open the Streamlit scanner.

```bash
./run.sh
```

---

### Alternative: Manual Execution (For Developers)

If you prefer to run each component manually to test the latency at each step:

**1. Train Model & Export Binary Weights:**

```bash
python research/train_model.py
```

**2. Compile the C++ Inference Engine:**

```bash
g++ -O3 -march=native -I/usr/include/eigen3 engine/inference_engine.cpp -o engine/inference_engine
```

**3. Test the C++ Engine in Single-Shot Mode (CLI):**
*(Requires 6 standardized market parameters)*

```bash
./engine/inference_engine -0.43 0.27 -0.75 -1.18 1.34 0.88
```

**4. Launch the C++ Daemon & Streamlit GUI:**

```bash
./engine/inference_engine & 
streamlit run dashboard/app.py
```

## References

This system architecture was inspired by the following quantitative research papers:

* Culkin, R., & Das, S. R. (2017). *Machine Learning in Finance: The Case of Deep Learning for Option Pricing.* Santa Clara University.
* Ke, A., & Yang, A. (2019). *Option Pricing with Deep Learning.* Stanford University.

---

*Developed by Arnold Baskar - 2026*
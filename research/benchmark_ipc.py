import socket
import struct
import time
import numpy as np

# Benchmark parameters
NUM_RUNS = 10000
NUM_OPTIONS_PER_BATCH = 27 # Simulates the Streamlit GUI payload
PORT = 5555

print(f"Launching HFT Benchmark: {NUM_RUNS} iterations...")

# Standardized dummy data (6 inputs per option)
dummy_batch = np.random.randn(NUM_OPTIONS_PER_BATCH, 6).astype(np.float32)
batch_bytes = dummy_batch.tobytes()

cpp_latencies_ns = []
e2e_latencies_us = []

def send_request():
    """Function simulating the exact behavior of the Streamlit interface"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', PORT))
    
    # 1. Send batch size
    s.sendall(struct.pack('i', NUM_OPTIONS_PER_BATCH))
    # 2. Send binary payload
    s.sendall(batch_bytes)
    
    # 3. Receive C++ internal latency
    latency_data = s.recv(8)
    cpp_lat_ns = struct.unpack('q', latency_data)[0]
    
    # 4. Receive predicted prices
    prices_data = b""
    expected = NUM_OPTIONS_PER_BATCH * 4
    while len(prices_data) < expected:
        packet = s.recv(expected - len(prices_data))
        if not packet: break
        prices_data += packet
        
    s.close()
    return cpp_lat_ns

try:
    # WARMUP (Wake up CPU cache and network kernel to prevent cold-start jitter)
    for _ in range(1000):
        send_request()

    # REAL BENCHMARK
    for _ in range(NUM_RUNS):
        start_e2e = time.perf_counter()
        
        cpp_lat_ns = send_request()
            
        end_e2e = time.perf_counter()
        
        cpp_latencies_ns.append(cpp_lat_ns)
        # Calculate End-to-End latency per option
        e2e_latencies_us.append(((end_e2e - start_e2e) * 1_000_000) / NUM_OPTIONS_PER_BATCH)

    # STATISTICS CALCULATION (Percentiles)
    cpp_array = np.array(cpp_latencies_ns) / NUM_OPTIONS_PER_BATCH # ns/opt
    e2e_array = np.array(e2e_latencies_us)                         # us/opt

    print("\n" + "="*50)
    print("BENCHMARK RESULTS (Per Option)")
    print("="*50)
    print(f"🔹 PURE C++ LATENCY (AI Inference)")
    print(f"   Min : {np.min(cpp_array):.0f} ns")
    print(f"   P50 (Median) : {np.percentile(cpp_array, 50):.0f} ns")
    print(f"   P90 : {np.percentile(cpp_array, 90):.0f} ns")
    print(f"   P99 : {np.percentile(cpp_array, 99):.0f} ns")
    
    print(f"\n🔹 END-TO-END LATENCY (IPC + TCP + Python)")
    print(f"   Min : {np.min(e2e_array):.2f} µs")
    print(f"   P50 (Median) : {np.percentile(e2e_array, 50):.2f} µs")
    print(f"   P90 : {np.percentile(e2e_array, 90):.2f} µs")
    print(f"   P99 : {np.percentile(e2e_array, 99):.2f} µs")
    print("="*50)

except Exception as e:
    print(f"Connection Error. Ensure the C++ Daemon is running (./engine/inference_engine &). Error: {e}")
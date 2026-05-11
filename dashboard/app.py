import streamlit as st
import yfinance as yf
import pandas as pd
import subprocess
import numpy as np
import time
from datetime import datetime
import os
import socket
import struct

st.set_page_config(page_title="Quant Anomaly Scanner", layout="wide")
st.title("Pricing Anomaly Scanner (Batch Mode)")
st.markdown("*C++ Deep Learning Inference Engine (Critical Latency)*")

st.sidebar.header("Market Parameters")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
threshold = st.sidebar.slider("Anomaly Detection Threshold ($)", min_value=0.10, max_value=2.00, value=0.50, step=0.10)

if st.sidebar.button("Launch Live Scanner", type="primary"):
    with st.spinner(f"Connecting to market for {ticker_symbol}..."):
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        expirations = ticker.options
        
        if not expirations:
            st.error(f"No options found for ticker {ticker_symbol}.")
        else:
            exp_date = expirations[0] 
            opt = ticker.option_chain(exp_date)
            calls = opt.calls
            
            calls = calls[(calls['strike'] > current_price * 0.8) & (calls['strike'] < current_price * 1.2)]
            
            days_to_exp = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days
            T_real = max(days_to_exp / 365.0, 0.01)
            r_fixed = 0.04 
            
            batch_data = []
            valid_calls = []
            
            for index, row in calls.iterrows():
                market_price = (row['bid'] + row['ask']) / 2.0
                if market_price <= 0.01: continue
                
                log_moneyness = np.log(current_price / row['strike'])
                sqrt_T = np.sqrt(T_real)
                
                batch_data.append([current_price, row['strike'], T_real, r_fixed, log_moneyness, sqrt_T])
                valid_calls.append((row['strike'], market_price))
            
            if batch_data:
                try:
                    start_e2e = time.perf_counter()
                    
                    # Connect to the C++ Daemon via local TCP Sockets
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(('127.0.0.1', 5555))
                    
                    # 1. Send the number of options (32-bit Integer)
                    num_options = len(batch_data)
                    s.sendall(struct.pack('i', num_options))
                    
                    # 2. Send the binary stream directly to RAM
                    batch_bytes = np.array(batch_data, dtype=np.float32).tobytes()
                    s.sendall(batch_bytes)
                    
                    # 3. Receive the C++ latency (64-bit Integer)
                    latency_data = s.recv(8)
                    latency_batch_ns = struct.unpack('q', latency_data)[0]
                    
                    # 4. Receive the array of predicted prices (32-bit Float)
                    prices_data = b""
                    expected_bytes = num_options * 4
                    while len(prices_data) < expected_bytes:
                        packet = s.recv(expected_bytes - len(prices_data))
                        if not packet: break
                        prices_data += packet
                        
                    ai_prices = np.frombuffer(prices_data, dtype=np.float32)
                    s.close()
                    
                    end_e2e = time.perf_counter()
                    e2e_latency_us = (end_e2e - start_e2e) * 1_000_000
                    
                    results = []
                    for i, (strike, market_price) in enumerate(valid_calls):
                        ai_price = ai_prices[i]
                        spread = abs(market_price - ai_price)
                        results.append({
                            "Ticker": ticker_symbol,
                            "Strike ($)": strike,
                            "Market Price ($)": market_price,
                            "AI Model Price ($)": ai_price,
                            "Spread ($)": spread
                        })
                    
                    df = pd.DataFrame(results)
                    nb_options = len(df)
                    
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("Options Scanned", nb_options)
                    col3.metric("Anomalies", len(df[df['Spread ($)'] > threshold]))
                    col4.metric("Pure C++ Latency", f"{latency_batch_ns / nb_options:.0f} ns/opt")
                    col5.metric("End-to-End Latency", f"{e2e_latency_us / nb_options:.0f} µs/opt")
                    st.markdown("---")
                    
                    st.subheader(f"Live Options Feed (Expiration: {exp_date})")
                    
                    def highlight_anomalies(row):
                        if row['Spread ($)'] > threshold:
                            return ['background-color: rgba(46, 204, 113, 0.3)'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = df.style.apply(highlight_anomalies, axis=1).format({
                        "Strike ($)": "{:.2f}",
                        "Market Price ($)": "{:.2f}",
                        "AI Model Price ($)": "{:.2f}",
                        "Spread ($)": "{:.2f}"
                    })
                    
                    st.dataframe(styled_df, width="stretch", hide_index=True)
                    
                except ConnectionRefusedError:
                    st.error("C++ Daemon is not running. Please start the engine/inference_engine background process.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

            if os.path.exists("batch_inputs.bin"): os.remove("batch_inputs.bin")
            if os.path.exists("batch_outputs.bin"): os.remove("batch_outputs.bin")
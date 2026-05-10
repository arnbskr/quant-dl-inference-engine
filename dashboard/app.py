import streamlit as st
import yfinance as yf
import pandas as pd
import subprocess
import numpy as np
import time
from datetime import datetime
import os

st.set_page_config(page_title="Scanner d'Anomalies Quant", layout="wide")
st.title("⚡ Scanner d'Anomalies de Pricing (Batch Mode)")
st.markdown("*Moteur d'inférence Deep Learning C++ (Latence critique)*")

st.sidebar.header("⚙️ Paramètres du Marché")
ticker_symbol = st.sidebar.text_input("Ticker de l'action", value="AAPL").upper()
threshold = st.sidebar.slider("Seuil de détection d'anomalie ($)", min_value=0.10, max_value=2.00, value=0.50, step=0.10)

if st.sidebar.button("Lancer le Scanner Live", type="primary"):
    with st.spinner(f"Connexion au marché pour {ticker_symbol}..."):
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        expirations = ticker.options
        
        if not expirations:
            st.error(f"Aucune option trouvée pour le ticker {ticker_symbol}.")
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
                np.savetxt("batch_inputs.csv", batch_data, delimiter=",")
                
                try:
                    # Chronomètre End-to-End (Python inclut I/O et OS overhead)
                    start_e2e = time.perf_counter()
                    
                    process = subprocess.run(["./inference_engine"], capture_output=True, text=True, check=True)
                    output = process.stdout.strip()
                    
                    end_e2e = time.perf_counter()
                    e2e_latency_us = (end_e2e - start_e2e) * 1_000_000
                    
                    latency_batch_us = int(output.split(":")[1])
                    
                    ai_prices = np.loadtxt("batch_outputs.csv")
                    if ai_prices.ndim == 0:
                        ai_prices = [float(ai_prices)]
                    
                    results = []
                    for i, (strike, market_price) in enumerate(valid_calls):
                        ai_price = ai_prices[i]
                        spread = abs(market_price - ai_price)
                        results.append({
                            "Action": ticker_symbol,
                            "Strike ($)": strike,
                            "Prix Marché ($)": market_price,
                            "Prix Modèle IA ($)": ai_price,
                            "Écart (Spread)": spread
                        })
                    
                    df = pd.DataFrame(results)
                    nb_options = len(df)
                    
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Prix Actuel", f"{current_price:.2f} $")
                    col2.metric("Options Évaluées", nb_options)
                    col3.metric("Anomalies", len(df[df['Écart (Spread)'] > threshold]))
                    col4.metric("Latence C++ Pure", f"{latency_batch_us / nb_options:.2f} µs/opt")
                    col5.metric("Latence End-to-End", f"{e2e_latency_us / nb_options:.0f} µs/opt")
                    st.markdown("---")
                    
                    st.subheader(f"📡 Flux de cotation (Échéance : {exp_date})")
                    
                    def highlight_anomalies(row):
                        if row['Écart (Spread)'] > threshold:
                            return ['background-color: rgba(46, 204, 113, 0.3)'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = df.style.apply(highlight_anomalies, axis=1).format({
                        "Strike ($)": "{:.2f}",
                        "Prix Marché ($)": "{:.2f}",
                        "Prix Modèle IA ($)": "{:.2f}",
                        "Écart (Spread)": "{:.2f}"
                    })
                    
                    st.dataframe(styled_df, width="stretch", hide_index=True)
                    
                except Exception as e:
                    st.error(f"Erreur d'exécution C++ : {e}")

            if os.path.exists("batch_inputs.csv"): os.remove("batch_inputs.csv")
            if os.path.exists("batch_outputs.csv"): os.remove("batch_outputs.csv")
import streamlit as st
import yfinance as yf
import pandas as pd
import subprocess
from datetime import datetime

# Configuration de la page Streamlit
st.set_page_config(page_title="Scanner d'Anomalies Quant", layout="wide")
st.title("⚡ Scanner d'Anomalies de Pricing")
st.markdown("*Moteur d'inférence Deep Learning C++ (Latence critique)*")

# Barre latérale pour les paramètres du marché
st.sidebar.header("⚙️ Paramètres du Marché")
ticker_symbol = st.sidebar.text_input("Ticker de l'action", value="AAPL").upper()
threshold = st.sidebar.slider("Seuil de détection d'anomalie ($)", min_value=0.10, max_value=2.00, value=0.50, step=0.10)

if st.sidebar.button("Lancer le Scanner Live", type="primary"):
    
    with st.spinner(f"Connexion au marché pour {ticker_symbol}..."):
        # 1. Téléchargement des données réelles
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        expirations = ticker.options
        
        if not expirations:
            st.error(f"Aucune option trouvée pour le ticker {ticker_symbol}.")
        else:
            exp_date = expirations[0] # On prend l'échéance la plus proche
            opt = ticker.option_chain(exp_date)
            calls = opt.calls
            
            # On filtre pour ne garder que les options proches du prix (At/Near The Money)
            calls = calls[(calls['strike'] > current_price * 0.8) & (calls['strike'] < current_price * 1.2)]
            
            # Calcul du temps à l'échéance (T) en années
            days_to_exp = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days
            T_real = max(days_to_exp / 365.0, 0.01)
            r_fixed = 0.04 # Taux sans risque simulé à 4%
            
            results = []
            total_latency = 0
            
            # 2. Boucle de pricing via C++
            for index, row in calls.iterrows():
                S = current_price
                K = row['strike']
                market_price = (row['bid'] + row['ask']) / 2.0
                
                if market_price <= 0.01:
                    continue # On ignore les options sans liquidité
                
                # Python appelle le binaire C++ compilé
                cmd = ["./inference_engine", str(S), str(K), str(T_real), str(r_fixed)]
                try:
                    # Exécution de la commande système
                    process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    output = process.stdout.strip()
                    
                    # Décodage de la sortie du C++ (PRICE:X.XX,LATENCY:YY)
                    if "PRICE:" in output and "LATENCY:" in output:
                        parts = output.split(",")
                        ai_price = float(parts[0].split(":")[1])
                        latency_us = int(parts[1].split(":")[1])
                        
                        spread = abs(market_price - ai_price)
                        total_latency += latency_us
                        
                        results.append({
                            "Action": ticker_symbol,
                            "Strike ($)": K,
                            "Prix Marché ($)": market_price,
                            "Prix Modèle IA ($)": ai_price,
                            "Écart (Spread)": spread,
                            "Latence C++ (µs)": latency_us
                        })
                except Exception as e:
                    st.error(f"Erreur d'exécution C++ : {e}")
                    break
            
            # 3. Affichage des résultats
            if results:
                df = pd.DataFrame(results)
                
                # KPIs (Indicateurs clés)
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Prix Actuel Action", f"{current_price:.2f} $")
                col2.metric("Options Évaluées", len(df))
                col3.metric("Anomalies Détectées", len(df[df['Écart (Spread)'] > threshold]))
                col4.metric("Latence Inférence Moy.", f"{total_latency / len(df):.0f} µs")
                st.markdown("---")
                
                st.subheader(f"📡 Flux de cotation (Échéance : {exp_date})")
                
                # Fonction pour colorer les anomalies en vert (opportunités)
                def highlight_anomalies(row):
                    if row['Écart (Spread)'] > threshold:
                        return ['background-color: rgba(46, 204, 113, 0.3)'] * len(row)
                    return [''] * len(row)
                
                # Formatage propre du tableau
                styled_df = df.style.apply(highlight_anomalies, axis=1).format({
                    "Strike ($)": "{:.2f}",
                    "Prix Marché ($)": "{:.2f}",
                    "Prix Modèle IA ($)": "{:.2f}",
                    "Écart (Spread)": "{:.2f}",
                    "Latence C++ (µs)": "{:.0f}"
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
#!/bin/bash

# Pipeline d'inférence quantifiée à basse latence pour le Deep Learning en C++ avec une interface Streamlit

echo -e "\e[1;36m[1/3] Vérification des poids du modèle...\e[0m"

# Vérifie si le dossier model_weights existe, sinon on entraîne le modèle
if [ ! -d "model_weights" ]; then
    echo -e "\e[33mEntraînement initial requis. Lancement de train_model.py...\e[0m"
    python research/train_model.py
    if [ $? -ne 0 ]; then
        echo -e "\e[31mErreur lors de l'entraînement du modèle.\e[0m"
        exit 1
    fi
else
    echo -e "\e[32m✔ Poids trouvés dans model_weights/\e[0m"
fi

echo -e "\e[1;36m[2/3] Compilation du moteur C++ (O3 Optimization)...\e[0m"
g++ -O3 engine/inference_engine.cpp -o engine/inference_engine
if [ $? -ne 0 ]; then
    echo -e "\e[31mErreur de compilation C++.\e[0m"
    exit 1
fi
echo -e "\e[32m✔ Compilation réussie.\e[0m"

echo -e "\e[1;36m[3/3] Lancement de l'interface de Trading Live (Streamlit)...\e[0m"
echo -e "\e[33mAppuyez sur Ctrl+C pour arrêter le serveur.\e[0m"
streamlit run dashboard/app.py
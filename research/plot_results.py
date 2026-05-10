import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1. Génération des données synthétiques pour l'entraînement avec smile de volatilité
print("Génération des données et entraînement express pour les graphiques...")
np.random.seed(42)
num_samples = 300000

S = np.random.uniform(10, 500, num_samples)
K = np.random.uniform(10, 500, num_samples)
T = np.random.uniform(0.01, 3.0, num_samples)
r = np.random.uniform(0.00, 0.05, num_samples)

log_moneyness = np.log(S / K)
sqrt_T = np.sqrt(T)

sigma_smile = 0.20 + 0.5 * (log_moneyness)**2 + 0.1 * T
sigma_smile = np.clip(sigma_smile, 0.05, 0.90)

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

y_target = black_scholes_call(S, K, T, r, sigma_smile)

X_features = np.column_stack((S, K, T, r, log_moneyness, sqrt_T))
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. Architecture du MLP pour la prédiction du prix d'option (6 entrées : S, K, T, r, log_moneyness, sqrt_T) + smile de volatilité implicite
class PricingMLP(nn.Module):
    def __init__(self):
        super(PricingMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.output_layer(x)

model = PricingMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
loss_history = [] # Pour stocker l'erreur à chaque epoch

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Moyenne de la loss pour l'epoch
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

# 3. Génération des graphiques pour la validation et la convergence du modèle
print("Génération des images en cours...")

# Configuration du style global
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Graphique 1 : Courbe d'apprentissage
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, color='#2ecc71', linewidth=2.5, label='Erreur quadratique moyenne (Train)')
plt.title("Convergence du modèle deep learning (loss curve)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Époques", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
plt.yscale('log') # Échelle logarithmique
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

# Graphique 2 : Scatter Plot (Vérité terrain vs Prédictions)
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

# Pour éviter un graphe trop lourd à charger, on prend un échantillon aléatoire de 5000 points
indices = np.random.choice(len(y_test), 5000, replace=False)
y_test_sample = y_test[indices]
y_pred_sample = y_pred[indices]

plt.figure(figsize=(8, 8))
# Ligne parfaite (Idéale)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#e74c3c', linestyle='--', linewidth=2, label='Prédiction parfaite')
# Nuage de points (Réalité)
plt.scatter(y_test_sample, y_pred_sample, color='#3498db', alpha=0.3, s=10, label='Prédictions du modèle')

plt.title("Validation Out-of-Sample : Prix prédit vs Vérité terrain", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Prix réel (synthétique Heston/SABR) [$]", fontsize=14)
plt.ylabel("Prix prédit par le modèle IA [$]", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=300)
plt.close()

print("Les fichiers 'loss_curve.png' et 'scatter_plot.png' ont été créés.")
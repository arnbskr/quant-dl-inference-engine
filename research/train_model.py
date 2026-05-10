import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# 1. Génération des données synthétiques pour l'entraînement avec smile de volatilité
np.random.seed(42)
num_samples = 300000

# Paramètres du marché (S, K, T, r)
S = np.random.uniform(10, 500, num_samples)
K = np.random.uniform(10, 500, num_samples)
T = np.random.uniform(0.01, 3.0, num_samples)
r = np.random.uniform(0.00, 0.05, num_samples)

# Feature Engineering : Ajout de la Moneyness, Log-Moneyness et Racine du temps
log_moneyness = np.log(S / K)
sqrt_T = np.sqrt(T)

# On utilise log_moneyness pour générer le smile, c'est plus stable mathématiquement
sigma_smile = 0.20 + 0.5 * (log_moneyness)**2 + 0.1 * T
sigma_smile = np.clip(sigma_smile, 0.05, 0.90)

# Fonction Black-Scholes pour générer la vérité terrain
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

y_target = black_scholes_call(S, K, T, r, sigma_smile)
X_features = np.column_stack((S, K, T, r, log_moneyness, sqrt_T))

# 2. Préparation des données pour PyTorch (train/test split + scaling)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 3. Architecture du MLP pour la prédiction du prix d'option (6 entrées : S, K, T, r, log_moneyness, sqrt_T) + smile de volatilité implicite
class PricingMLP(nn.Module):
    def __init__(self):
        super(PricingMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=64) # 6 entrées
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

# 4. Entraînement du modèle
print("Début de l'entraînement...")
epochs = 50
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Export des poids du modèle et du scaler pour utilisation en C++
os.makedirs("model_weights", exist_ok=True)
for name, param in model.named_parameters():
    np.savetxt(f"model_weights/{name}.csv", param.detach().numpy(), delimiter=",")

np.savetxt("model_weights/scaler_mean.csv", scaler_X.mean_, delimiter=",")
np.savetxt("model_weights/scaler_scale.csv", scaler_X.scale_, delimiter=",")
print("Poids et Scaler exportés avec succès.")

# 6. Évaluation du modèle sur les données de test (Out-of-Sample)
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nPerformances sur données de test (Out-of-Sample) :")
    print(f"RMSE : {rmse:.4f} $")
    print(f"MAE  : {mae:.4f} $")
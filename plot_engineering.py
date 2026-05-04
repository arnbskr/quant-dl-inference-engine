import matplotlib.pyplot as plt

# Données d'ingénierie
configurations = ['Python (PyTorch)', 'C++ (Appel unitaire)', 'C++ (Batch inference)']
latencies = [114.0, 23.0, 5.77]
colors = ['#e74c3c', '#f39c12', '#2ecc71'] # Rouge (lent), Orange (moyen), Vert (ultra-rapide)

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, ax = plt.subplots(figsize=(9, 6))

# Création des barres
bars = ax.bar(configurations, latencies, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# Ajout des annotations sur les barres
for bar, latency in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
            f"{latency} µs",
            ha='center', va='bottom', fontweight='bold', fontsize=14)

# Personnalisation des axes et des titres
plt.title("Comparaison de la latence d'inférence (par option)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Latence (microsecondes)", fontsize=14)
plt.ylim(0, 130)

# Nettoyage visuel
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Sauvegarde de l'image
plt.savefig("latency_comparison.png", dpi=300)
print("L'image 'latency_comparison.png' a été générée avec succès.")
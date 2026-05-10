import matplotlib.pyplot as plt

# Engineering data
configurations = ['Python (PyTorch)', 'C++ (Single Call)', 'C++ (Batch Inference)']
latencies = [114.0, 23.0, 5.77]
colors = ['#e74c3c', '#f39c12', '#2ecc71'] 

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, ax = plt.subplots(figsize=(9, 6))

# Creating bars
bars = ax.bar(configurations, latencies, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# Adding annotations on bars
for bar, latency in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
            f"{latency} µs",
            ha='center', va='bottom', fontweight='bold', fontsize=14)

# Axes and titles customization
plt.title("Inference Latency Comparison (per option)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Latency (microseconds)", fontsize=14)
plt.ylim(0, 130)

# Visual cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Saving the image
plt.savefig("assets/latency_comparison.png", dpi=300)
print("Image 'latency_comparison.png' successfully generated in assets folder.")
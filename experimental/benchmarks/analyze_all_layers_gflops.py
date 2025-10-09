import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("output_tmp", exist_ok=True)

# 1. Read gflops data
df = pd.read_csv("all_layers_gflops.csv")
gflops = df["gflops"]

# 2. Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(gflops, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("GFLOPS")
plt.ylabel("Count")
plt.title("GFLOPS Distribution")
plt.tight_layout()
plt.savefig("output_tmp/all_layers_gflops_hist.png")
plt.close()

# 3. Plot boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(gflops, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.ylabel("GFLOPS")
plt.title("GFLOPS Boxplot")
plt.tight_layout()
plt.savefig("output_tmp/all_layers_gflops_box.png")
plt.close()
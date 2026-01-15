import pandas as pd
import matplotlib.pyplot as plt

PKL_PATH = "bigearthnet_df.pkl"

# Load
df = pd.read_pickle(PKL_PATH)

# Parse HHMMSS -> hour/minute
t = df["time_str"].astype(str).str.zfill(6)
df["hour"] = t.str.slice(0, 2).astype(int)
df["minute"] = t.str.slice(2, 4).astype(int)

# Keep only 09:00–11:59
mask = (df["hour"] >= 9) & (df["hour"] <= 11)
dfw = df.loc[mask].copy()

# Half-hour bin index within 09:00–11:59: 0..5
dfw["bin"] = (dfw["hour"] - 9) * 2 + (dfw["minute"] // 30)

bin_labels = {
    0: "09:00–09:29",
    1: "09:30–09:59",
    2: "10:00–10:29",
    3: "10:30–10:59",
    4: "11:00–11:29",
    5: "11:30–11:59",
}
order = [bin_labels[i] for i in range(6)]
dfw["bin_label"] = dfw["bin"].map(bin_labels)

# Counts + percent (ordered)
counts = dfw["bin_label"].value_counts().reindex(order, fill_value=0)
perc = (counts / counts.sum() * 100).round(2)

summary = pd.DataFrame({"count": counts, "percent": perc})
print("Total samples in 09:00–11:59:", int(counts.sum()))
print(summary)

# --- Bar plot ---
fig, ax = plt.subplots()
bars = ax.bar(summary.index, summary["count"])

ax.set_title("Class Balance (09:00–11:59, 30-min bins)")
ax.set_xlabel("Time bin")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=30)

# Add percent labels above bars
ymax = summary["count"].max() if len(summary) else 0
ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

for bar, p in zip(bars, summary["percent"].values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f"{p:.2f}%", ha="center", va="bottom")

ax.set_yticklabels([])          # hide y tick labels
ax.tick_params(axis="y", length=0)  # hide y tick marks (optional)

plt.tight_layout()
plt.show()

plt.savefig("class_balance_9_to_12.png", dpi=200, bbox_inches="tight")

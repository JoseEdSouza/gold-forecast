"""Heatmap de acurácia direcional por modelo × ano (horizonte h=30)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "diracc_heatmap_h30.png"

MODELS = {
    "XGBoost":  REPO_ROOT / "outputs" / "02_xgboost" / "yearly_h30.csv",
    "GRU":      REPO_ROOT / "outputs" / "03_gru"     / "yearly_h30.csv",
    "CNN 1D":   REPO_ROOT / "outputs" / "04_cnn"     / "yearly_h30.csv",
    "LSTM":     REPO_ROOT / "outputs" / "05_lstm"    / "yearly_h30.csv",
}

# --- carrega e monta pivot table ---
frames = []
for model, path in MODELS.items():
    df = pd.read_csv(path)
    df["model"] = model
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
pivot = data.pivot(index="model", columns="ano", values="DirAcc%")
pivot = pivot.loc[list(MODELS.keys())]          # ordem definida acima
pivot.columns = [str(int(c)) for c in pivot.columns]

# --- colormap divergente centrado em 50% ---
cmap = mcolors.LinearSegmentedColormap.from_list(
    "diracc",
    [
        (0.0,  "#9C2B2B"),   # vermelho escuro  — 0 %
        (0.35, "#D9534F"),   # vermelho claro   — 35 %
        (0.50, "#F5F0E8"),   # bege neutro      — 50 % (baseline aleatório)
        (0.65, "#5A9E6F"),   # verde médio      — 65 %
        (1.0,  "#1B5E35"),   # verde escuro     — 100 %
    ],
)

fig, ax = plt.subplots(figsize=(9, 3.8))
fig.patch.set_facecolor("white")

im = ax.imshow(
    pivot.values,
    cmap=cmap,
    vmin=30,
    vmax=100,
    aspect="auto",
)

# anotações
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        color = "white" if (val < 42 or val > 80) else "#2C3E50"
        ax.text(
            j, i, f"{val:.1f}%",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=color,
        )

# eixos
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11, fontweight="bold")
ax.tick_params(length=0)

# linha de referência 50% (random walk)
for j in range(len(pivot.columns)):
    ax.axvline(j - 0.5, color="#CCCCCC", linewidth=0.6)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Acurácia Direcional (%)", fontsize=10)
cbar.ax.axhline(50, color="#2C3E50", linewidth=1.2, linestyle="--")
cbar.ax.text(2.8, 50, "50%\n(aleatório)", va="center", fontsize=8, color="#2C3E50")

# títulos
ax.set_title(
    "Acurácia Direcional por Modelo × Ano  —  horizonte h = 30 dias",
    fontsize=13, fontweight="bold", pad=12, color="#2C3E50",
)
ax.set_xlabel("Ano", fontsize=11)

# destaque visual 2022 (crise)
col_2022 = list(pivot.columns).index("2022")
ax.add_patch(plt.Rectangle(
    (col_2022 - 0.5, -0.5), 1, len(pivot.index),
    fill=False, edgecolor="#9C2B2B", linewidth=2.0, linestyle="--", zorder=5,
))
ax.text(
    col_2022, len(pivot.index) - 0.05, "↑ 2022",
    ha="center", va="bottom", fontsize=9, color="#9C2B2B", fontweight="bold",
)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Salvo: {OUTPUT_PATH}")

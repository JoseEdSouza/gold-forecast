#!/usr/bin/env python
# coding: utf-8

# # Predição do preço do ouro com GRU
#
# Versão em `.py` no estilo notebook-export:
# - constantes hardcoded
# - execução linear
# - mesmas features do `gold_features.csv`
# - métricas e plots no mesmo estilo do experimento CNN

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers


from lib import (
    fit_and_scale,
    inverse_predict,
    per_year_metrics,
    plot_learning_curves,
    plot_predicted_vs_actual,
    plot_price_series,
    plot_return_predictions,
    plot_residuals,
    plot_scatter,
    plot_split_regions,
    plot_yearly_perf,
    regression_metrics,
    temporal_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ## 1. Setup

SEED = 42
LOOKBACK = 60
HORIZONS = [5, 15, 30]
GAP = max(HORIZONS)
EPOCHS = 120
BATCH_SIZE = 64
SHOW_PLOTS = True

CSV_PATH = REPO_ROOT / "data" / "processed" / "gold_features.csv"
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts" / "gold_price_gru"
MODELS_DIR = REPO_ROOT / "models"

np.random.seed(SEED)
tf.random.set_seed(SEED)
plt.rcParams.update(
    {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10}
)


# ## 2. Helpers


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"CSV inválido. Colunas encontradas: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    blocked = {"date", "close", "open", "high", "low"}
    feature_cols = [
        col
        for col in df.columns
        if col not in blocked
        and not col.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 1
    ]
    if not feature_cols:
        raise ValueError("Nenhuma feature numérica válida encontrada.")
    return feature_cols


def build_model(n_features: int) -> Model:
    inp = layers.Input(shape=(LOOKBACK, n_features))
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(0.15)(x)
    x = layers.GRU(64)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outs = [layers.Dense(1, name=f"h{h}")(x) for h in HORIZONS]

    model = Model(inp, outs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={f"h{h}": tf.keras.losses.Huber(1.0) for h in HORIZONS},
        loss_weights={"h5": 1.0, "h15": 0.7, "h30": 0.5},
    )
    return model


def predict_returns(model: Model, X: np.ndarray, y_scaler) -> np.ndarray:
    return inverse_predict(model.predict(X, verbose=0), y_scaler).astype(np.float32)


def show_or_close(fig: plt.Figure) -> None:
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ## 3. Carregamento

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = load_processed_data(CSV_PATH)
feature_cols = select_feature_cols(df)

for h in HORIZONS:
    if f"y_{h}" not in df.columns:
        raise ValueError(f"Coluna alvo ausente: y_{h}")

print("TensorFlow", tf.__version__)
print(f"{len(df)} linhas | {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"{len(feature_cols)} features:")
print(feature_cols)


# ## 4. Série histórica completa

fig = plot_price_series(
    df["date"],
    df["close"],
    title="Preço do ouro (US$/oz) — série completa",
    ylabel="US$/oz",
)
fig.savefig(ARTIFACTS_DIR / "price_series.png", bbox_inches="tight")
show_or_close(fig)


# ## 5. Split temporal + janelas

splits = temporal_split(
    df=df,
    feature_cols=feature_cols,
    horizons=HORIZONS,
    lookback=LOOKBACK,
    gap=GAP,
)
for name, ws in splits.items():
    d0 = df["date"].iloc[ws.idx[0]].date()
    d1 = df["date"].iloc[ws.idx[-1]].date()
    print(f"{name:>6}: {ws.X.shape} | {d0} -> {d1}")

fig = plot_split_regions(
    df,
    {name: (ws.X, ws.Y, ws.prices, ws.idx) for name, ws in splits.items()},
    title="Divisão temporal treino / validação / teste",
    ylabel="US$/oz",
)
fig.savefig(ARTIFACTS_DIR / "split_regions.png", bbox_inches="tight")
show_or_close(fig)


# ## 6. Escalonamento

scaled = fit_and_scale(splits, horizons=HORIZONS, train_key="treino")
print("escala aplicada — treino X:", scaled.X["treino"].shape)


# ## 7. Modelo GRU

model = build_model(len(feature_cols))
model.summary()


# ## 8. Treino

history = model.fit(
    scaled.X["treino"],
    scaled.Y_dict("treino"),
    validation_data=(scaled.X["val"], scaled.Y_dict("val")),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
        ),
    ],
)
print(f"Treinou {len(history.history['loss'])} épocas")

fig = plot_learning_curves(history.history, HORIZONS)
fig.savefig(ARTIFACTS_DIR / "learning_curves.png", bbox_inches="tight")
show_or_close(fig)


# ## 9. Avaliação

preds = {
    split: predict_returns(model, scaled.X[split], scaled.y_scaler)
    for split in ["treino", "val", "teste"]
}

tbl = pd.concat(
    [
        regression_metrics(
            scaled.Yreal[split],
            preds[split],
            scaled.prices[split],
            HORIZONS,
            split_name=split,
        )
        for split in ["treino", "val", "teste"]
    ],
    ignore_index=True,
)
tbl.to_csv(ARTIFACTS_DIR / "metrics.csv", index=False)
print(tbl.round(3))

print("\nDESEMPENHO NO TESTE")
t = tbl[tbl["split"] == "teste"].set_index("h")[
    ["MAE", "RMSE", "MAPE%", "DirAcc%", "MAE/naive"]
]
print(t.round(2).to_string())
print("\nMAE/naive < 1 => melhor que o baseline | DirAcc > 50% => acerta direção")


# ## 10. Gráficos de desempenho

dates_te = df["date"].values[scaled.idx["teste"]]
h30_idx = HORIZONS.index(30)

fig = plot_predicted_vs_actual(
    dates_te,
    scaled.Yreal["teste"],
    preds["teste"],
    scaled.prices["teste"],
    HORIZONS,
    ylabel="Preço (US$/oz)",
)
fig.savefig(ARTIFACTS_DIR / "predicted_vs_actual_test.png", bbox_inches="tight")
show_or_close(fig)

fig = plot_scatter(
    scaled.Yreal["teste"],
    preds["teste"],
    scaled.prices["teste"],
    HORIZONS,
)
fig.savefig(ARTIFACTS_DIR / "scatter_test.png", bbox_inches="tight")
show_or_close(fig)

yearly = per_year_metrics(
    scaled.Yreal["teste"],
    preds["teste"],
    scaled.prices["teste"],
    dates_te,
    horizon_idx=h30_idx,
)
yearly.to_csv(ARTIFACTS_DIR / "yearly_h30.csv")

fig = plot_yearly_perf(yearly, horizon=30)
fig.savefig(ARTIFACTS_DIR / "yearly_h30.png", bbox_inches="tight")
show_or_close(fig)

fig = plot_residuals(
    scaled.Yreal["teste"],
    preds["teste"],
    scaled.prices["teste"],
    horizon_idx=h30_idx,
    horizon=30,
)
fig.savefig(ARTIFACTS_DIR / "residuals_h30.png", bbox_inches="tight")
show_or_close(fig)

fig = plot_return_predictions(
    dates_te,
    scaled.Yreal["teste"],
    preds["teste"],
    HORIZONS,
)
fig.savefig(ARTIFACTS_DIR / "returns_test.png", bbox_inches="tight")
show_or_close(fig)


# ## 11. Plots extras de features

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
feature_sets = [
    ["log_ret_1", "log_ret_5", "log_ret_21"],
    ["vol_5", "vol_21", "vol_ratio"],
    ["rsi", "macd_hist", "drawdown_252"],
]
for ax, cols in zip(axes, feature_sets):
    for col in cols:
        if col in df.columns:
            ax.plot(df["date"], df[col], lw=0.9, label=col)
    ax.legend(loc="upper left")
    ax.set_title(" | ".join(cols))
plt.tight_layout()
fig.savefig(ARTIFACTS_DIR / "feature_panels.png", bbox_inches="tight")
show_or_close(fig)

fig, ax = plt.subplots(figsize=(12, 8))
corr_cols = feature_cols[:]
corr = df[corr_cols].corr().round(2)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90)
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols)
ax.set_title("Correlação entre features")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
plt.tight_layout()
fig.savefig(ARTIFACTS_DIR / "feature_correlation.png", bbox_inches="tight")
show_or_close(fig)


# ## 12. Persistência

model.save(MODELS_DIR / "gold_price_gru.keras")

print(f"\nArtefatos salvos em: {ARTIFACTS_DIR}")
print(f"Modelo salvo em: {MODELS_DIR / 'gold_price_gru.keras'}")

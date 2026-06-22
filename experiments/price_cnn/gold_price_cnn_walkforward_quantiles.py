# -*- coding: utf-8 -*-
"""
Predição do ouro — v2: WALK-FORWARD + QUANTIS
==============================================
Evoluções sobre gold_cnn.py (que continua sendo a base de features/janelas):

  1. WALK-FORWARD COM RETREINO ANUAL
     Em vez de um único split 70/15/15, o modelo é retreinado a cada 252
     pregões (~1 ano) com janela de treino expansiva. Cada fold testa apenas
     no ano seguinte ao treino (com embargo p/ evitar vazamento). Motivação:
     na v1 o melhor checkpoint veio na época ~3 — dados financeiros saturam
     rápido e dados recentes valem mais. O walk-forward também dá ~8 anos de
     avaliação out-of-sample em vez de ~3.

  2. PREDIÇÃO POR QUANTIS (q10 / q50 / q90, pinball loss)
     Em vez de um ponto único, cada horizonte prevê três quantis do
     log-retorno. Ganhos:
       - q50 (mediana) é mais robusto que a média ao ruído de cauda;
       - o intervalo [q10, q90] expõe a incerteza: em regime calmo a banda
         estreita, em crise ela abre — a "timidez nos extremos" da v1 vira
         informação explícita em vez de erro oculto;
       - cobertura empírica do intervalo (~80% esperado) vira diagnóstico
         de calibração do modelo.

Uso:
    python experiments/price_cnn/gold_price_cnn_walkforward_quantiles.py --csv data/processed/final_gold_data.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

# Reaproveita carregamento, features, alvos e janelas da v1
from gold_price_cnn import (
    GAP,
    HORIZONS,
    LOOKBACK,
    SEED,
    build_features,
    build_targets,
    load_data,
    make_windows,
)

np.random.seed(SEED)
tf.random.set_seed(SEED)

QUANTILES = [0.10, 0.50, 0.90]
FOLD_LEN = 252  # ~1 ano de pregões por fold de teste
INITIAL_TRAIN = 0.55  # primeiro treino usa 55% da série (~2000 -> ~2014)
EMBARGO = GAP + LOOKBACK
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "final_gold_data.csv"
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"


# ----------------------------------------------------------------------------
# Pinball loss (quantile loss): penaliza assimetricamente acima/abaixo do
# quantil. Minimizá-la faz cada saída convergir para o quantil correspondente.
# ----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
def pinball_loss(y_true, y_pred):
    q = tf.constant(QUANTILES, tf.float32)  # (3,)
    y_true = tf.reshape(y_true, (-1, 1))  # (b,1)
    e = y_true - y_pred  # (b,3)
    return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))


def build_model(n_features: int) -> Model:
    inp = layers.Input(shape=(LOOKBACK, n_features))
    x = layers.Conv1D(64, 5, padding="causal", activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(0.15)(x)
    x = layers.Concatenate()(
        [layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)]
    )
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    # 3 quantis por horizonte
    outs = [layers.Dense(len(QUANTILES), name=f"h{h}")(x) for h in HORIZONS]
    model = Model(inp, outs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={f"h{h}": pinball_loss for h in HORIZONS},
        loss_weights={"h5": 1.0, "h15": 0.7, "h30": 0.5},
    )
    return model


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def walk_forward(df, feature_cols, epochs=40, fold_start=1, fold_end=99):
    n = len(df)
    nf = len(feature_cols)
    train_end = int(n * INITIAL_TRAIN)
    all_rows = []
    fold = 0

    while train_end + EMBARGO < n - 10:
        fold += 1
        test_start = train_end + EMBARGO
        test_end = min(test_start + FOLD_LEN, n)
        if fold > fold_end:
            break
        if fold < fold_start:
            train_end += FOLD_LEN  # avança sem treinar
            continue

        Xtr, Ytr, _, _ = make_windows(df, feature_cols, 0, train_end)
        Xte, Yte, Pte, idx = make_windows(df, feature_cols, test_start, test_end)
        if len(Xte) == 0:
            break

        # validação interna = último ano do treino (p/ early stopping)
        n_val = min(252, len(Xtr) // 5)
        Xva, Yva = Xtr[-n_val:], Ytr[-n_val:]
        Xtr, Ytr = Xtr[:-n_val], Ytr[:-n_val]

        # escala: fit SÓ no treino deste fold
        xs = RobustScaler().fit(Xtr.reshape(-1, nf))
        def sc(X):
            return np.clip(
                    xs.transform(X.reshape(-1, nf)).reshape(X.shape), -8, 8
                ).astype(np.float32)
        Xtr, Xva, Xte = sc(Xtr), sc(Xva), sc(Xte)
        ys = RobustScaler().fit(Ytr)
        def d(Y):
            return {
                    f"h{h}": ys.transform(Y).astype(np.float32)[:, j]
                    for j, h in enumerate(HORIZONS)
                }

        tf.keras.backend.clear_session()
        model = build_model(nf)
        model.fit(
            Xtr,
            d(Ytr),
            validation_data=(Xva, d(Yva)),
            epochs=epochs,
            batch_size=128,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
                ),
            ],
        )

        # predição -> desfaz escala por horizonte (mesmo center/scale p/ os 3 quantis)
        preds = model.predict(Xte, verbose=0)  # lista de (n,3)
        rows = {"date": df["date"].values[idx], "close_t": Pte, "fold": fold}
        for j, h in enumerate(HORIZONS):
            pq = preds[j] * ys.scale_[j] + ys.center_[j]
            pq = np.sort(pq, axis=1)  # evita cruzamento de quantis
            for k, q in enumerate(QUANTILES):
                rows[f"h{h}_q{int(q * 100)}"] = Pte * np.exp(pq[:, k])
            rows[f"h{h}_true"] = Pte * np.exp(Yte[:, j])
        all_rows.append(pd.DataFrame(rows))

        d0 = df["date"].iloc[test_start].date()
        d1 = df["date"].iloc[test_end - 1].date()
        print(
            f"[fold {fold}] treino ate idx {train_end} | teste {d0} -> {d1} "
            f"| {len(Xte)} amostras"
        )
        train_end += FOLD_LEN  # janela expansiva, teste contínuo

    return pd.concat(all_rows, ignore_index=True)


# ----------------------------------------------------------------------------
# Avaliação agregada
# ----------------------------------------------------------------------------
def report(res: pd.DataFrame):
    print("\n===== WALK-FORWARD (todo o periodo out-of-sample) =====")
    for h in HORIZONS:
        y, q50 = res[f"h{h}_true"], res[f"h{h}_q50"]
        q10, q90 = res[f"h{h}_q10"], res[f"h{h}_q90"]
        naive = res["close_t"]
        mae = (y - q50).abs().mean()
        mae_n = (y - naive).abs().mean()
        mape = ((y - q50).abs() / y).mean() * 100
        dacc = (np.sign(q50 - naive) == np.sign(y - naive)).mean() * 100
        cov = ((y >= q10) & (y <= q90)).mean() * 100
        width = ((q90 - q10) / naive).mean() * 100
        print(
            f"h={h:>2}d | MAE ${mae:7.2f} | MAPE {mape:5.2f}% | Dir.Acc {dacc:5.1f}% "
            f"| MAE/naive {mae / mae_n:.3f} | Cobertura80 {cov:5.1f}% "
            f"| Largura banda {width:4.1f}%"
        )

    print("\n--- Por ano (h=30, q50) ---")
    res = res.copy()
    res["ano"] = pd.to_datetime(res["date"]).dt.year
    for ano, g in res.groupby("ano"):
        y, q50, naive = g["h30_true"], g["h30_q50"], g["close_t"]
        ratio = (y - q50).abs().mean() / max((y - naive).abs().mean(), 1e-9)
        dacc = (np.sign(q50 - naive) == np.sign(y - naive)).mean() * 100
        cov = ((y >= g["h30_q10"]) & (y <= g["h30_q90"])).mean() * 100
        print(
            f"{ano} | n={len(g):>3} | MAE/naive {ratio:.3f} | Dir.Acc {dacc:5.1f}% "
            f"| Cobertura {cov:5.1f}%"
        )


def main():
    default_csv = DEFAULT_CSV if DEFAULT_CSV.exists() else None

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(default_csv) if default_csv else None,
        help="caminho do final_gold_data.csv",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", default="1-99", help="ex.: 1-4")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.csv:
        raise ValueError("É necessário especificar o arquivo CSV do dataset.")

    df = load_data(args.csv)
    df = build_features(df)
    df = build_targets(df)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"date", "close", "open", "high", "low"}
        and not c.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 1
    ]
    print(f"[features] {len(feature_cols)}")

    if args.report_only:
        csv_files = sorted(ARTIFACTS_DIR.glob("predicoes_folds_*.csv"))
        res = pd.concat(
            [pd.read_csv(f) for f in csv_files], ignore_index=True
        ).sort_values("date")
        out_path = ARTIFACTS_DIR / "predicoes_walkforward.csv"
        res.to_csv(out_path, index=False)
        report(res)
        print(f"\n[ok] salvo: {out_path}")
        return

    f0, f1 = map(int, args.folds.split("-"))
    res = walk_forward(df, feature_cols, epochs=args.epochs, fold_start=f0, fold_end=f1)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / f"predicoes_folds_{f0:02d}_{f1:02d}.csv"
    res.to_csv(out, index=False)

    # Se rodou todos os folds disponíveis, opcionalmente já gera o report consolidado
    csv_files = sorted(ARTIFACTS_DIR.glob("predicoes_folds_*.csv"))
    if csv_files:
        try:
            res_all = pd.concat(
                [pd.read_csv(f) for f in csv_files], ignore_index=True
            ).sort_values("date")
            report_path = ARTIFACTS_DIR / "predicoes_walkforward.csv"
            res_all.to_csv(report_path, index=False)
            report(res_all)
            print(f"\n[ok] Relatório geral salvo em {report_path}")
        except Exception as e:
            print(f"Nota: Não foi possível consolidar todos os folds ainda: {e}")
            report(res)


if __name__ == "__main__":
    main()

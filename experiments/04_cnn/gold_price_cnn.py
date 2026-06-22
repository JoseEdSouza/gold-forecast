# -*- coding: utf-8 -*-
"""
Predição do preço do ouro com CNN 1D (janelas de 5, 15 e 30 dias)
==================================================================
Dataset alvo: final_gold_data.csv (Kaggle: precious-metals-history-since-2000-with-news)

Decisões de projeto (resumo):
  1. O alvo NÃO é o preço bruto, e sim o LOG-RETORNO acumulado no horizonte h.
     Isso torna a série (quase) estacionária e neutraliza o problema dos
     "outliers estruturais" (crise de 2008, COVID-2020, guerra 2022): o preço
     saiu de ~$280 (2000) para ~$2700+ (2025), então um modelo treinado em
     preço bruto nunca viu os níveis do conjunto de teste.
  2. Escalonamento com RobustScaler (mediana/IQR), ajustado SOMENTE no treino,
     para que caudas pesadas das crises não distorçam a normalização.
  3. Perda Huber: robusta a outliers de retorno (dias de pânico) sem
     descartá-los — crises são sinal, não ruído.
  4. Janela deslizante (sliding window): lookback de 60 dias -> prevê os
     retornos acumulados em t+5, t+15 e t+30 com um único modelo multi-head.
  5. Split temporal (sem shuffle entre conjuntos) + gap entre treino/val/teste
     para evitar vazamento de dados pelas janelas sobrepostas.
  6. Baseline ingênuo (random walk: retorno previsto = 0) para comparação.
     Se o modelo não bater o baseline, ele não aprendeu nada útil.

Uso:
    python experiments/price_cnn/gold_price_cnn.py --csv data/processed/final_gold_data.csv
    python experiments/price_cnn/gold_price_cnn.py --synthetic
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import keras
import tensorflow as tf
from keras import layers, Model, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOOKBACK = 60  # tamanho da janela deslizante (dias)
HORIZONS = [5, 15, 30]  # horizontes de predição
GAP = max(HORIZONS)  # gap entre splits p/ evitar vazamento
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "final_gold_data.csv"
ARTIFACTS_DIR = REPO_ROOT / "outputs" / "04_cnn"
MODELS_DIR = REPO_ROOT / "outputs" / "04_cnn"


# ----------------------------------------------------------------------------
# 1. CARREGAMENTO (flexível quanto aos nomes de colunas do CSV)
# ----------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]

    # Detecta coluna de data
    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if date_col is None:
        raise ValueError(f"Coluna de data não encontrada. Colunas: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})

    # Detecta coluna de preço (close/price/gold...)
    candidates = ["close", "price", "gold", "adj close", "adj_close", "value"]
    price_col = next(
        (
            c
            for c in df.columns
            for cand in candidates
            if cand in c and df[c].dtype != object
        ),
        None,
    )
    if price_col is None:
        raise ValueError(f"Coluna de preço não encontrada. Colunas: {list(df.columns)}")
    df = df.rename(columns={price_col: "close"})

    # Mantém OHLC/volume extras se existirem
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    print(
        f"[load] {len(df)} linhas | {df['date'].min().date()} -> {df['date'].max().date()}"
    )
    return df


def make_synthetic(n=6000) -> pd.DataFrame:
    """Série sintética com regimes/choques (imita 2008 e 2020) p/ smoke test."""
    rng = np.random.default_rng(SEED)
    ret = rng.normal(0.0003, 0.009, n)
    ret[2000:2120] += rng.normal(0.002, 0.03, 120)  # "crise" 1: volatilidade extrema
    ret[4500:4560] += rng.normal(0.004, 0.025, 60)  # "crise" 2
    price = 280 * np.exp(np.cumsum(ret))
    dates = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"date": dates, "close": price})


# ----------------------------------------------------------------------------
# 2. ENGENHARIA DE FEATURES
#    Todas as features são RELATIVAS (retornos, razões, osciladores) — nunca
#    o preço bruto — para que o modelo generalize entre níveis de preço.
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    out["log_ret_1"] = np.log(c / c.shift(1))  # retorno diário
    out["log_ret_5"] = np.log(c / c.shift(5))  # momentum semanal
    out["log_ret_21"] = np.log(c / c.shift(21))  # momentum mensal

    out["vol_5"] = out["log_ret_1"].rolling(5).std()  # volatilidade curta
    out["vol_21"] = out["log_ret_1"].rolling(21).std()  # volatilidade mensal
    out["vol_ratio"] = out["vol_5"] / (
        out["vol_21"] + 1e-9
    )  # regime de vol (>1 = estresse)

    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    out["dist_ma20"] = c / ma20 - 1.0  # distância da média 20d
    out["dist_ma50"] = c / ma50 - 1.0
    out["ma_cross"] = ma20 / ma50 - 1.0  # tendência

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    out["rsi"] = (out["rsi"] - 50) / 50  # centrado em 0

    # MACD (normalizado pelo preço p/ ser comparável em 2000 e 2025)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / c

    # Bollinger %B
    std20 = c.rolling(20).std()
    out["boll_b"] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-9) - 0.5

    # Drawdown em relação à máxima de 1 ano (contexto de regime)
    out["drawdown_252"] = c / c.rolling(252).max() - 1.0

    # Features extras do CSV (ex.: prata, sentimento de notícias) entram aqui
    # automaticamente se forem numéricas e não-OHLC:
    skip = {"close", "date", "open", "high", "low"}
    for col in df.columns:
        if col not in skip and pd.api.types.is_numeric_dtype(df[col]):
            extra = pd.to_numeric(df[col], errors="coerce")
            # transforma níveis em retornos se a coluna parece um preço
            if extra.min() > 0 and extra.std() / (extra.mean() + 1e-9) > 0.1:
                out[f"{col}_ret"] = np.log(extra / extra.shift(1))
            else:
                out[col] = extra

    return out


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Alvo: log-retorno acumulado nos próximos h dias."""
    c = df["close"]
    for h in HORIZONS:
        df[f"y_{h}"] = np.log(c.shift(-h) / c)
    return df


# ----------------------------------------------------------------------------
# 3. JANELA DESLIZANTE + SPLIT TEMPORAL SEM VAZAMENTO
# ----------------------------------------------------------------------------
def make_windows(df, feature_cols, start, end):
    """Gera (X, y, prices, idx) para amostras cujo ÚLTIMO dia da janela está em [start, end)."""
    X, Y, P, IDX = [], [], [], []
    feats = df[feature_cols].values
    ys = df[[f"y_{h}" for h in HORIZONS]].values
    closes = df["close"].values
    for t in range(max(start, LOOKBACK - 1), end):
        if np.isnan(ys[t]).any():
            continue
        win = feats[t - LOOKBACK + 1 : t + 1]
        if np.isnan(win).any():
            continue
        X.append(win)
        Y.append(ys[t])
        P.append(closes[t])
        IDX.append(t)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(P), np.array(IDX)


def temporal_split(n, train=0.70, val=0.15):
    """Índices de corte com GAP entre conjuntos (evita janelas/alvos sobrepostos)."""
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return (0, i1), (i1 + GAP + LOOKBACK, i2), (i2 + GAP + LOOKBACK, n)


# ----------------------------------------------------------------------------
# 4. MODELO: CNN 1D causal com dilatação (campo receptivo cobre os 60 dias)
# ----------------------------------------------------------------------------
def build_model(n_features: int) -> Model:
    # Hiperparâmetros obtidos via BayesianOptimization (outputs/04_cnn/best_hp.json)
    inp = layers.Input(shape=(LOOKBACK, n_features))

    x = layers.Conv1D(96, 5, padding="causal", activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(0.20)(x)

    # Pooling duplo: média (tendência) + máximo (eventos extremos/choques)
    x = layers.Concatenate()(
        [layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)]
    )
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)

    # Uma cabeça por horizonte (representação compartilhada, saídas separadas)
    outs = [layers.Dense(1, name=f"h{h}")(x) for h in HORIZONS]
    model = Model(inp, outs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.00405701014195834),
        loss={f"h{h}": keras.losses.Huber(delta=1.0) for h in HORIZONS},
        loss_weights={"h5": 1.0, "h15": 0.6, "h30": 0.2},
    )
    return model


# ----------------------------------------------------------------------------
# 5. AVALIAÇÃO (em preço reconstruído + acurácia direcional + baseline)
# ----------------------------------------------------------------------------
def evaluate(model, X, Y, prices, y_scaler, label=""):
    preds = model.predict(X, verbose=0)
    preds = np.hstack(preds)  # (n, 3) retornos escalonados
    preds = y_scaler.inverse_transform(preds)  # volta a log-retornos reais

    print(f"\n===== {label} =====")
    rows = []
    for j, j_h in enumerate(HORIZONS):
        y_true_ret, y_pred_ret = Y[:, j], preds[:, j]
        p_true = prices * np.exp(y_true_ret)  # preço real em t+h
        p_pred = prices * np.exp(y_pred_ret)  # preço previsto em t+h
        p_naive = prices  # baseline: preço não muda

        mae = np.mean(np.abs(p_true - p_pred))
        mae_naive = np.mean(np.abs(p_true - p_naive))
        rmse = np.sqrt(np.mean((p_true - p_pred) ** 2))
        mape = np.mean(np.abs((p_true - p_pred) / p_true)) * 100
        dir_acc = np.mean(np.sign(y_pred_ret) == np.sign(y_true_ret)) * 100

        rows.append((j_h, mae, rmse, mape, dir_acc, mae / mae_naive))
        print(
            f"h={j_h:>2}d | MAE ${mae:8.2f} | RMSE ${rmse:8.2f} | MAPE {mape:5.2f}% "
            f"| Dir.Acc {dir_acc:5.1f}% | MAE/naive {mae / mae_naive:.3f} "
            f"{'(MELHOR que baseline)' if mae < mae_naive else '(pior que baseline)'}"
        )
    return rows, preds


# ----------------------------------------------------------------------------
# 6. PIPELINE PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    default_csv = DEFAULT_CSV if DEFAULT_CSV.exists() else None

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(default_csv) if default_csv else None,
        help="caminho do final_gold_data.csv",
    )
    ap.add_argument("--synthetic", action="store_true", help="usar dados sintéticos")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    # Se nenhum CSV for passado ou se não for encontrado, usa sintético por padrão
    df = make_synthetic() if (args.synthetic or not args.csv) else load_data(args.csv)
    df = build_features(df)
    df = build_targets(df)

    feature_cols = [
        c
        for c in df.columns
        if c not in {"date", "close", "open", "high", "low"}
        and not c.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"[features] {len(feature_cols)}: {feature_cols}")

    n = len(df)
    (tr0, tr1), (va0, va1), (te0, te1) = temporal_split(n)
    print(
        f"[split] treino [0:{tr1}] | val [{va0}:{va1}] | teste [{te0}:{n}] (gap={GAP + LOOKBACK})"
    )

    Xtr, Ytr, Ptr, _ = make_windows(df, feature_cols, tr0, tr1)
    Xva, Yva, Pva, _ = make_windows(df, feature_cols, va0, va1)
    Xte, Yte, Pte, idx_te = make_windows(df, feature_cols, te0, te1)
    print(f"[janelas] treino {Xtr.shape} | val {Xva.shape} | teste {Xte.shape}")

    # --- Escalonamento robusto: fit SÓ no treino (anti-vazamento, anti-outlier)
    nf = Xtr.shape[2]
    x_scaler = RobustScaler().fit(Xtr.reshape(-1, nf))

    def scale_x(X):
        return x_scaler.transform(X.reshape(-1, nf)).reshape(X.shape).astype(np.float32)

    Xtr, Xva, Xte = scale_x(Xtr), scale_x(Xva), scale_x(Xte)
    # clip pós-escala: limita a influência de dias de pânico sem removê-los
    Xtr, Xva, Xte = (np.clip(a, -8, 8) for a in (Xtr, Xva, Xte))

    y_scaler = RobustScaler().fit(Ytr)
    Ytr_s = y_scaler.transform(Ytr).astype(np.float32)
    Yva_s = y_scaler.transform(Yva).astype(np.float32)

    def to_dict(Y):
        return {f"h{h}": Y[:, j] for j, h in enumerate(HORIZONS)}

    model = build_model(nf)
    model.summary()

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]
    model.fit(
        Xtr,
        to_dict(Ytr_s),
        validation_data=(Xva, to_dict(Yva_s)),
        epochs=args.epochs,
        batch_size=64,
        callbacks=cbs,
        verbose=2,
    )

    evaluate(model, Xva, Yva, Pva, y_scaler, "VALIDAÇÃO")
    rows, preds = evaluate(model, Xte, Yte, Pte, y_scaler, "TESTE")

    # Salva predições do teste p/ análise posterior
    res = pd.DataFrame({"date": df["date"].values[idx_te], "close_t": Pte})
    for j, h in enumerate(HORIZONS):
        res[f"pred_price_t+{h}"] = Pte * np.exp(preds[:, j])
        res[f"true_price_t+{h}"] = Pte * np.exp(Yte[:, j])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = ARTIFACTS_DIR / "predicoes_teste.csv"
    model_path = MODELS_DIR / "gold_price_cnn.keras"
    res.to_csv(pred_path, index=False)
    model.save(model_path)
    print(f"\n[ok] salvos: {model_path}, {pred_path}")


if __name__ == "__main__":
    main()

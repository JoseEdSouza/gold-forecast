#!/usr/bin/env python
# coding: utf-8
"""
make_dataset.py
---------------
Lê final_gold_data.csv, aplica a mesma engenharia de features de gold_cnn.py
e salva o resultado em data/gold_features.csv.

Uso:
    python make_dataset.py [--input PATH] [--output PATH]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = [5, 15, 30]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "final_gold_data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "gold_features.csv"

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})
    cands = ["close", "price", "gold", "adj close", "adj_close", "value"]
    price_col = next(
        (c for c in df.columns for k in cands if k in c and df[c].dtype != object), None
    )
    df = df.rename(columns={price_col: "close"})
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Engenharia de features
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]

    # retornos logarítmicos
    out["log_ret_1"] = np.log(c / c.shift(1))
    out["log_ret_5"] = np.log(c / c.shift(5))
    out["log_ret_21"] = np.log(c / c.shift(21))

    # volatilidade realizada e razão de stress
    out["vol_5"] = out["log_ret_1"].rolling(5).std()
    out["vol_21"] = out["log_ret_1"].rolling(21).std()
    out["vol_ratio"] = out["vol_5"] / (out["vol_21"] + 1e-9)

    # distância às médias móveis
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    out["dist_ma20"] = c / ma20 - 1
    out["dist_ma50"] = c / ma50 - 1
    out["ma_cross"] = ma20 / ma50 - 1

    # RSI(14) normalizado para [-1, 1]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = ((100 - 100 / (1 + gain / (loss + 1e-9))) - 50) / 50

    # MACD histograma normalizado pelo preço
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / c

    # Bollinger %B centrado em 0
    std20 = c.rolling(20).std()
    out["boll_b"] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-9) - 0.5

    # drawdown de 252 dias
    out["drawdown_252"] = c / c.rolling(252).max() - 1

    # features OHLC (se disponíveis)
    if {"high", "low", "open"}.issubset(out.columns):
        h, l, o = out["high"], out["low"], out["open"]
        out["hl_range"] = (h - l) / c
        out["close_pos"] = (c - l) / (h - l + 1e-9) - 0.5
        out["gap_open"] = np.log(o / c.shift(1))

    # índice de medo por léxico (se há coluna headlines)
    if "headlines" in df.columns:
        fear_pattern = (
            "crisis|crash|war|recession|inflation|default|collapse|panic|fear|"
            "conflict|sanction|bankrupt|plunge|turmoil|slump|selloff|attack|"
            "invasion|pandemic|virus"
        )
        txt = df["headlines"].fillna("").str.lower()
        out["fear_ratio"] = txt.str.count(fear_pattern) / (txt.str.count("/") + 1)
        out["fear_ma5"] = out["fear_ratio"].rolling(5).mean()

    return out


# ---------------------------------------------------------------------------
# Alvos (log-retorno acumulado h dias à frente)
# ---------------------------------------------------------------------------


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    for h in HORIZONS:
        df[f"y_{h}"] = np.log(c.shift(-h) / c)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Gera dataset de features para o ouro")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    print(f"Lendo {args.input} ...")
    df_raw = load_data(args.input)
    print(
        f"  {len(df_raw)} linhas | {df_raw['date'].min().date()} -> {df_raw['date'].max().date()}"
    )

    df = build_targets(build_features(df_raw))

    feature_cols = [
        col
        for col in df.columns
        if col not in {"date", "close", "open", "high", "low"}
        and not col.startswith("y_")
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique() > 1
    ]

    keep = ["date", "close"] + feature_cols + [f"y_{h}" for h in HORIZONS]
    out = df[keep]

    out.to_csv(args.output, index=False)
    print(f"Salvo em {args.output}")
    print(f"  {len(out)} linhas × {len(out.columns)} colunas")
    print(f"  features ({len(feature_cols)}): {feature_cols}")
    nan_summary = out[feature_cols].isna().sum()
    print(
        f"  NaNs por feature (primeiros 10):\n{nan_summary[nan_summary > 0].head(10)}"
    )


if __name__ == "__main__":
    main()

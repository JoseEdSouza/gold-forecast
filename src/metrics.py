"""lib.metrics — métricas de regressão para modelos de forecasting de séries temporais.

Todas as funções operam em **log-retornos** e reconstroem o preço internamente,
mantendo compatibilidade com o target `y_h = log(P[t+h] / P[t])`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    horizons: list[int],
    split_name: str = "",
) -> pd.DataFrame:
    """Calcula MAE, RMSE, MAPE, acurácia direcional e MAE/naive por horizonte.

    Os erros são calculados sobre o **preço reconstruído** `P * exp(y)`, não sobre
    o log-retorno diretamente, para ter uma interpretação em US$/oz.

    Parameters
    ----------
    y_true:
        Log-retornos reais, shape ``(n, H)``.
    y_pred:
        Log-retornos previstos, shape ``(n, H)``.
    prices:
        Preço de fechamento em ``t`` (âncora da previsão), shape ``(n,)``.
    horizons:
        Lista de horizontes em dias, ex.: ``[5, 15, 30]``. Deve ter comprimento H.
    split_name:
        Rótulo do conjunto (ex.: "treino", "val", "teste") — adicionado como coluna.

    Returns
    -------
    pd.DataFrame
        Uma linha por horizonte com colunas:
        ``split, h, MAE, RMSE, MAPE%, DirAcc%, MAE/naive``.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true e y_pred devem ter o mesmo shape; "
            f"recebidos {y_true.shape} e {y_pred.shape}"
        )
    if y_true.shape[1] != len(horizons):
        raise ValueError(
            f"Número de colunas ({y_true.shape[1]}) != len(horizons) ({len(horizons)})"
        )

    rows: list[dict] = []
    for j, h in enumerate(horizons):
        p_true = prices * np.exp(y_true[:, j])
        p_pred = prices * np.exp(y_pred[:, j])
        p_naive = prices  # random-walk: preço não muda

        mae = float(np.mean(np.abs(p_true - p_pred)))
        mae_naive = float(np.mean(np.abs(p_true - p_naive)))
        rmse = float(np.sqrt(np.mean((p_true - p_pred) ** 2)))
        mape = float(np.mean(np.abs((p_true - p_pred) / p_true)) * 100)
        dir_acc = float(np.mean(np.sign(y_pred[:, j]) == np.sign(y_true[:, j])) * 100)

        rows.append(
            {
                "split": split_name,
                "h": h,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE%": mape,
                "DirAcc%": dir_acc,
                "MAE/naive": mae / mae_naive if mae_naive > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def per_year_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    dates: np.ndarray,
    horizon_idx: int,
) -> pd.DataFrame:
    """Agrupa acurácia direcional e MAE/naive por ano calendário.

    Útil para identificar dependência de regime: o modelo costuma performar melhor
    em anos de tendência clara e pior em mercado lateral.

    Parameters
    ----------
    y_true:
        Log-retornos reais, shape ``(n, H)``.
    y_pred:
        Log-retornos previstos, shape ``(n, H)``.
    prices:
        Preço de fechamento em ``t``, shape ``(n,)``.
    dates:
        Array de datas (``numpy.datetime64`` ou conversível), shape ``(n,)``.
    horizon_idx:
        Índice da coluna de horizonte a analisar (ex.: ``2`` para h=30).

    Returns
    -------
    pd.DataFrame
        Indexado por ano, colunas ``DirAcc%`` e ``MAE/naive``.
    """
    p_true = prices * np.exp(y_true[:, horizon_idx])
    p_pred = prices * np.exp(y_pred[:, horizon_idx])

    df = pd.DataFrame(
        {
            "ano": pd.to_datetime(dates).year,
            "dir_ok": np.sign(y_pred[:, horizon_idx])
            == np.sign(y_true[:, horizon_idx]),
            "ae": np.abs(p_true - p_pred),
            "ae_naive": np.abs(p_true - prices),
        }
    )

    grouped = df.groupby("ano").agg(
        DirAcc=("dir_ok", "mean"),
        ae_sum=("ae", "sum"),
        naive_sum=("ae_naive", "sum"),
    )
    grouped["DirAcc%"] = grouped["DirAcc"] * 100
    grouped["MAE/naive"] = grouped["ae_sum"] / grouped["naive_sum"].replace(0, np.nan)

    return grouped[["DirAcc%", "MAE/naive"]]

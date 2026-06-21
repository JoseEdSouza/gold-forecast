"""lib.data — preparação de dados para modelos de forecasting de séries temporais.

Fluxo típico::

    splits = temporal_split(df, feature_cols, horizons=[5, 15, 30], lookback=60)
    scaled = fit_and_scale(splits, train_key="treino")

    # treino
    model.fit(scaled.X["treino"], scaled.Y_dict("treino"), ...)

    # avaliação (log-retornos na escala original)
    metrics = regression_metrics(scaled.Yreal["teste"], y_pred, scaled.prices["teste"], ...)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------


@dataclass
class WindowedSplit:
    """Arrays brutos (não escalonados) de uma janela deslizante para um split."""

    X: np.ndarray       # (n, lookback, n_features) float32
    Y: np.ndarray       # (n, H) float32 — log-retornos
    prices: np.ndarray  # (n,) float64 — preço de fechamento em t
    idx: np.ndarray     # (n,) int — índices originais no DataFrame


@dataclass
class ScaledSplits:
    """Todos os splits escalonados, prontos para treino e avaliação.

    Attributes
    ----------
    X:
        ``{split: array (n, lookback, nf)}`` — janelas escalonadas + clipped.
    Y:
        ``{split: array (n, H)}`` — log-retornos escalonados (alvo de treino).
    Yreal:
        ``{split: array (n, H)}`` — log-retornos na escala original (para métricas).
    prices:
        ``{split: array (n,)}`` — preço âncora em t.
    idx:
        ``{split: array (n,)}`` — índices originais no DataFrame.
    x_scaler:
        ``RobustScaler`` ajustado apenas no split de treino.
    y_scaler:
        ``RobustScaler`` ajustado apenas no split de treino.
    horizons:
        Lista de horizontes usados, na ordem das colunas de Y.
    """

    X: dict[str, np.ndarray]
    Y: dict[str, np.ndarray]
    Yreal: dict[str, np.ndarray]
    prices: dict[str, np.ndarray]
    idx: dict[str, np.ndarray]
    x_scaler: RobustScaler
    y_scaler: RobustScaler
    horizons: list[int]

    def Y_dict(self, split: str) -> dict[str, np.ndarray]:
        """Retorna ``{"h5": array, "h15": array, ...}`` — formato esperado pelo Keras."""
        return {f"h{h}": self.Y[split][:, j] for j, h in enumerate(self.horizons)}

    def inverse_Y(self, split: str) -> np.ndarray:
        """Desfaz a escala de Y para o split dado (usado em predict)."""
        return self.y_scaler.inverse_transform(self.Y[split]).astype(np.float32)


# ---------------------------------------------------------------------------
# Janela deslizante
# ---------------------------------------------------------------------------


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    lookback: int,
    start: int,
    end: int,
) -> WindowedSplit:
    """Cria janelas deslizantes para um intervalo de índices ``[start, end)``.

    Cada amostra é uma matriz ``(lookback, n_features)`` cujo último passo é o
    dia ``t``. Linhas com NaN em qualquer feature ou alvo são descartadas.

    Parameters
    ----------
    df:
        DataFrame com colunas de features, ``"close"`` e ``"y_{h}"`` para cada horizonte.
    feature_cols:
        Lista ordenada de colunas de features a incluir.
    horizons:
        Lista de horizontes em dias (deve haver colunas ``y_{h}`` em ``df``).
    lookback:
        Número de dias históricos por amostra.
    start:
        Índice de início (inclusive) no DataFrame.
    end:
        Índice de fim (exclusivo) no DataFrame.

    Returns
    -------
    WindowedSplit
        Arrays ``X``, ``Y``, ``prices``, ``idx``.
    """
    feats = df[feature_cols].values
    ys = df[[f"y_{h}" for h in horizons]].values
    closes = df["close"].values

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    P_list: list[float] = []
    I_list: list[int] = []

    for t in range(max(start, lookback - 1), end):
        if np.isnan(ys[t]).any():
            continue
        win = feats[t - lookback + 1 : t + 1]
        if np.isnan(win).any():
            continue
        X_list.append(win)
        Y_list.append(ys[t])
        P_list.append(closes[t])
        I_list.append(t)

    return WindowedSplit(
        X=np.array(X_list, dtype=np.float32),
        Y=np.array(Y_list, dtype=np.float32),
        prices=np.array(P_list, dtype=np.float64),
        idx=np.array(I_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Split temporal com gap anti-vazamento
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    lookback: int = 60,
    gap: int | None = None,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_names: tuple[str, str, str] = ("treino", "val", "teste"),
) -> dict[str, WindowedSplit]:
    """Divide o DataFrame em treino / validação / teste com gap anti-vazamento.

    O gap entre conjuntos tem tamanho ``gap + lookback``, garantindo que
    nenhuma janela de teste inclua dias do período de treino.

    Parameters
    ----------
    df:
        DataFrame completo, ordenado cronologicamente, com colunas ``"close"``,
        as features listadas em ``feature_cols`` e ``"y_{h}"`` para cada horizonte.
    feature_cols:
        Colunas de features (na ordem das colunas de X).
    horizons:
        Horizontes em dias; deve haver colunas ``y_{h}`` em ``df``.
    lookback:
        Tamanho da janela deslizante.
    gap:
        Número de dias de margem entre o último dia de um split e o primeiro do
        próximo (além do ``lookback``). Padrão: ``max(horizons)``.
    ratios:
        Proporções ``(treino, val, teste)`` que devem somar ≤ 1. O excesso de
        linhas vai para treino.
    split_names:
        Nomes dos splits no dicionário retornado.

    Returns
    -------
    dict[str, WindowedSplit]
        Um ``WindowedSplit`` para cada split.

    Raises
    ------
    ValueError
        Se ``ratios`` não tiverem 3 elementos ou somarem > 1.
    """
    if len(ratios) != 3 or sum(ratios) > 1.0 + 1e-9:
        raise ValueError(f"ratios deve ter 3 elementos somando ≤ 1; recebido {ratios}")

    if gap is None:
        gap = max(horizons)

    n = len(df)
    margin = gap + lookback

    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))

    spans: dict[str, tuple[int, int]] = {
        split_names[0]: (0, i1),
        split_names[1]: (i1 + margin, i2),
        split_names[2]: (i2 + margin, n),
    }

    result: dict[str, WindowedSplit] = {}
    for name, (start, end) in spans.items():
        ws = make_windows(df, feature_cols, horizons, lookback, start, end)
        result[name] = ws

    return result


# ---------------------------------------------------------------------------
# Escalonamento robusto
# ---------------------------------------------------------------------------


def fit_scalers(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> tuple[RobustScaler, RobustScaler]:
    """Ajusta um ``RobustScaler`` para X e outro para Y usando apenas o treino.

    Parameters
    ----------
    X_train:
        Janelas de treino, shape ``(n, lookback, n_features)``.
    Y_train:
        Log-retornos de treino, shape ``(n, H)``.

    Returns
    -------
    tuple[RobustScaler, RobustScaler]
        ``(x_scaler, y_scaler)`` prontos para serem usados com ``apply_scalers``.
    """
    nf = X_train.shape[2]
    x_scaler = RobustScaler().fit(X_train.reshape(-1, nf))
    y_scaler = RobustScaler().fit(Y_train)
    return x_scaler, y_scaler


def fit_and_scale(
    splits: dict[str, WindowedSplit],
    horizons: list[int],
    train_key: str = "treino",
    clip_value: float = 8.0,
) -> ScaledSplits:
    """Ajusta os scalers no treino e aplica em todos os splits.

    O scaler X usa ``RobustScaler`` (mediana/IQR) seguido de clip em
    ``±clip_value`` — limita influência de dias de pânico sem removê-los.
    O scaler Y normaliza os log-retornos para facilitar a convergência.

    Parameters
    ----------
    splits:
        Dicionário retornado por :func:`temporal_split`.
    horizons:
        Lista de horizontes em dias (ex.: ``[5, 15, 30]``), na mesma ordem
        das colunas de Y usadas em :func:`temporal_split`.
    train_key:
        Chave do split de treino (usado para ajustar os scalers).
    clip_value:
        Valor absoluto máximo após escalonamento de X.

    Returns
    -------
    ScaledSplits
        Todos os splits com X e Y escalonados, Yreal na escala original,
        mais os scalers para uso posterior (e.g., inverter predições).
    """
    train = splits[train_key]
    x_scaler, y_scaler = fit_scalers(train.X, train.Y)
    nf = train.X.shape[2]

    if len(horizons) != train.Y.shape[1]:
        raise ValueError(
            f"len(horizons)={len(horizons)} != colunas de Y ({train.Y.shape[1]})"
        )

    X_scaled: dict[str, np.ndarray] = {}
    Y_scaled: dict[str, np.ndarray] = {}
    Yreal: dict[str, np.ndarray] = {}
    prices: dict[str, np.ndarray] = {}
    idx: dict[str, np.ndarray] = {}

    for name, ws in splits.items():
        X_scaled[name] = np.clip(
            x_scaler.transform(ws.X.reshape(-1, nf)).reshape(ws.X.shape),
            -clip_value,
            clip_value,
        ).astype(np.float32)
        Y_scaled[name] = y_scaler.transform(ws.Y).astype(np.float32)
        Yreal[name] = ws.Y
        prices[name] = ws.prices
        idx[name] = ws.idx

    return ScaledSplits(
        X=X_scaled,
        Y=Y_scaled,
        Yreal=Yreal,
        prices=prices,
        idx=idx,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        horizons=horizons,
    )


def inverse_predict(
    raw_preds: list[np.ndarray] | np.ndarray,
    y_scaler: RobustScaler,
) -> np.ndarray:
    """Converte predições escalonadas do Keras de volta para log-retornos reais.

    Parameters
    ----------
    raw_preds:
        Saída de ``model.predict()``: lista de arrays ``(n, 1)`` (um por cabeça)
        ou array ``(n, H)`` já concatenado.
    y_scaler:
        Scaler ajustado pelo :func:`fit_and_scale`.

    Returns
    -------
    np.ndarray
        Log-retornos na escala original, shape ``(n, H)``.
    """
    if isinstance(raw_preds, list):
        stacked = np.hstack(raw_preds)  # (n, H)
    else:
        stacked = raw_preds
    return y_scaler.inverse_transform(stacked)

"""lib.plotting — visualizações reutilizáveis para experimentos de forecasting.

Todas as funções retornam ``matplotlib.figure.Figure``.
O chamador decide se salva ou exibe (``fig.savefig(...)`` / ``plt.show()``).
Nenhuma função chama ``plt.show()`` internamente.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paleta padrão (pode ser sobrescrita via parâmetro `palette`)
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE: dict[str, str] = {
    "dark": "#1a1a2e",
    "red": "#e94560",
    "blue": "#0f3460",
    "amber": "#f0a500",
    "gray": "#aaaab8",
}

_SPLIT_COLORS: dict[str, str] = {
    "treino": _DEFAULT_PALETTE["blue"],
    "val": _DEFAULT_PALETTE["amber"],
    "teste": _DEFAULT_PALETTE["red"],
}


def _palette(override: dict[str, str] | None) -> dict[str, str]:
    if override is None:
        return _DEFAULT_PALETTE
    return {**_DEFAULT_PALETTE, **override}


# ---------------------------------------------------------------------------
# 1. Série histórica completa
# ---------------------------------------------------------------------------


def plot_price_series(
    dates: pd.Series | np.ndarray,
    prices: pd.Series | np.ndarray,
    title: str = "Série histórica de preços",
    ylabel: str = "Preço",
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Plota a série de preços completa.

    Parameters
    ----------
    dates:
        Datas correspondentes a cada ponto de preço.
    prices:
        Série de preços de fechamento.
    title:
        Título do gráfico.
    ylabel:
        Rótulo do eixo y.
    palette:
        Sobrescreve cores do tema padrão. Chaves aceitas: ``"dark"``.
    figsize:
        Tamanho da figura em polegadas ``(width, height)``.
    """
    pal = _palette(palette)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, prices, color=pal["dark"], lw=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Divisão treino / validação / teste sobre a série
# ---------------------------------------------------------------------------


def plot_split_regions(
    df: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, Any, Any, np.ndarray]],
    date_col: str = "date",
    price_col: str = "close",
    title: str = "Divisão temporal treino / validação / teste",
    ylabel: str = "Preço",
    split_colors: dict[str, str] | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Plota a série completa destacando cada split com cor diferente.

    Parameters
    ----------
    df:
        DataFrame completo com colunas ``date_col`` e ``price_col``.
    splits:
        Dicionário ``{nome: (X, Y, P, idx_array)}``, formato retornado por
        ``make_windows``. Apenas ``idx_array`` (4º elemento) é usado.
    date_col:
        Nome da coluna de data em ``df``.
    price_col:
        Nome da coluna de preço em ``df``.
    title:
        Título do gráfico.
    ylabel:
        Rótulo do eixo y.
    split_colors:
        Dicionário ``{nome_split: cor}``. Padrão: azul/âmbar/vermelho.
    palette:
        Sobrescreve cores do tema base.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)
    colors = {**_SPLIT_COLORS, **(split_colors or {})}

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[date_col], df[price_col], color=pal["gray"], lw=0.8)

    for name, (_, _, _, idx) in splits.items():
        seg = df.iloc[idx[0] : idx[-1] + 1]
        ax.plot(seg[date_col], seg[price_col], color=colors.get(name, "black"), lw=1.1, label=name)

    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Curvas de aprendizado
# ---------------------------------------------------------------------------


def plot_learning_curves(
    history: dict[str, list[float]],
    horizons: list[int] | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Plota perda total e perda de validação por horizonte.

    Parameters
    ----------
    history:
        Dicionário ``model.history.history`` do Keras, ou qualquer dict com
        chaves ``"loss"``, ``"val_loss"`` e, opcionalmente, ``"val_h{N}_loss"``.
    horizons:
        Lista de horizontes para plotar perdas individuais.
        Se ``None``, infere automaticamente das chaves ``val_h*_loss``.
    palette:
        Sobrescreve cores do tema padrão.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)

    if horizons is None:
        horizons = [
            int(k.split("val_h")[1].split("_")[0])
            for k in history
            if k.startswith("val_h") and k.endswith("_loss")
        ]
        horizons = sorted(horizons)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history["loss"], label="treino", color=pal["blue"])
    axes[0].plot(history["val_loss"], label="validação", color=pal["red"])
    axes[0].set_title("Perda total")
    axes[0].set_xlabel("época")
    axes[0].legend()

    for h in horizons:
        key = f"val_h{h}_loss"
        if key in history:
            axes[1].plot(history[key], label=f"val h={h}d")
    axes[1].set_title("Perda de validação por horizonte")
    axes[1].set_xlabel("época")
    axes[1].legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Previsto × Real (série temporal)
# ---------------------------------------------------------------------------


def plot_predicted_vs_actual(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    horizons: list[int],
    ylabel: str = "Preço (US$/oz)",
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (13, 10),
) -> plt.Figure:
    """Plota o preço real e previsto ao longo do tempo para cada horizonte.

    Os log-retornos são convertidos para preço via ``P * exp(y)``.

    Parameters
    ----------
    dates:
        Array de datas do conjunto avaliado, shape ``(n,)``.
    y_true:
        Log-retornos reais, shape ``(n, H)``.
    y_pred:
        Log-retornos previstos, shape ``(n, H)``.
    prices:
        Preço âncora em ``t``, shape ``(n,)``.
    horizons:
        Lista de horizontes em dias.
    ylabel:
        Rótulo do eixo y.
    palette:
        Sobrescreve cores do tema.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)
    H = len(horizons)
    fig, axes = plt.subplots(H, 1, figsize=figsize, sharex=True)
    if H == 1:
        axes = [axes]

    for ax, j, h in zip(axes, range(H), horizons):
        ax.plot(dates, prices * np.exp(y_true[:, j]), color=pal["dark"], lw=1.1, label="real")
        ax.plot(
            dates,
            prices * np.exp(y_pred[:, j]),
            color=pal["red"],
            lw=1.0,
            alpha=0.85,
            label="previsto",
        )
        ax.set_title(f"Horizonte {h} dias — preço em t+{h}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Dispersão previsto × real com R²
# ---------------------------------------------------------------------------


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    horizons: list[int],
    xlabel: str = "real",
    ylabel: str = "previsto",
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (14, 4.5),
) -> plt.Figure:
    """Dispersão previsto × real com linha identidade e R² por horizonte.

    Parameters
    ----------
    y_true:
        Log-retornos reais, shape ``(n, H)``.
    y_pred:
        Log-retornos previstos, shape ``(n, H)``.
    prices:
        Preço âncora em ``t``, shape ``(n,)``.
    horizons:
        Lista de horizontes em dias.
    xlabel:
        Rótulo do eixo x.
    ylabel:
        Rótulo do eixo y.
    palette:
        Sobrescreve cores do tema.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)
    H = len(horizons)
    fig, axes = plt.subplots(1, H, figsize=figsize)
    if H == 1:
        axes = [axes]

    for ax, j, h in zip(axes, range(H), horizons):
        yt = prices * np.exp(y_true[:, j])
        yp = prices * np.exp(y_pred[:, j])
        r2 = r2_score(yt, yp)

        ax.scatter(yt, yp, s=6, alpha=0.35, color=pal["blue"])
        lo = min(float(yt.min()), float(yp.min()))
        hi = max(float(yt.max()), float(yp.max()))
        ax.plot([lo, hi], [lo, hi], color=pal["red"], lw=1)
        ax.set_title(f"h={h}d | R²={r2:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Acurácia direcional e MAE/naive por ano
# ---------------------------------------------------------------------------


def plot_yearly_perf(
    yearly_df: pd.DataFrame,
    horizon: int,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Barras de acurácia direcional e MAE/naive por ano calendário.

    Parameters
    ----------
    yearly_df:
        DataFrame retornado por :func:`lib.metrics.per_year_metrics`,
        indexado por ano, com colunas ``DirAcc%`` e ``MAE/naive``.
    horizon:
        Horizonte em dias — usado apenas no título.
    palette:
        Sobrescreve cores do tema.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    anos = yearly_df.index.astype(str)

    axes[0].bar(anos, yearly_df["DirAcc%"], color=pal["blue"])
    axes[0].axhline(50, color=pal["red"], ls="--", lw=1)
    axes[0].set_title(f"Acurácia direcional % (h={horizon})")
    axes[0].set_xlabel("ano")
    axes[0].set_ylabel("%")

    axes[1].bar(anos, yearly_df["MAE/naive"], color=pal["amber"])
    axes[1].axhline(1, color=pal["red"], ls="--", lw=1)
    axes[1].set_title(f"MAE/naive por ano (h={horizon}) — <1 é melhor")
    axes[1].set_xlabel("ano")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Distribuição dos resíduos
# ---------------------------------------------------------------------------


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    horizon_idx: int,
    horizon: int | None = None,
    bins: int = 50,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """Histograma dos resíduos de preço (real - previsto).

    Um resíduo centrado em zero indica baixo viés; caudas largas revelam
    choques não antecipados pelo modelo.

    Parameters
    ----------
    y_true:
        Log-retornos reais, shape ``(n, H)``.
    y_pred:
        Log-retornos previstos, shape ``(n, H)``.
    prices:
        Preço âncora em ``t``, shape ``(n,)``.
    horizon_idx:
        Índice da coluna de horizonte a analisar.
    horizon:
        Horizonte em dias — usado no título. Inferido de ``horizon_idx`` se omitido.
    bins:
        Número de bins do histograma.
    palette:
        Sobrescreve cores do tema.
    figsize:
        Tamanho da figura.
    """
    pal = _palette(palette)
    resid = (prices * np.exp(y_true[:, horizon_idx])) - (prices * np.exp(y_pred[:, horizon_idx]))

    h_label = f"h={horizon}" if horizon is not None else f"col {horizon_idx}"
    mean, std = float(resid.mean()), float(resid.std())

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(resid, bins=bins, color=pal["blue"], alpha=0.8)
    ax.axvline(0, color=pal["red"], lw=1.2)
    ax.set_title(f"Resíduos (real - previsto), {h_label} | média={mean:.1f}  desv={std:.1f}")
    ax.set_xlabel("US$")
    fig.tight_layout()
    return fig

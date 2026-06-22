"""src — módulos reutilizáveis de dados, métricas e plotagem para experimentos de forecasting."""

from src.data import (
    WindowedSplit,
    ScaledSplits,
    make_windows,
    temporal_split,
    fit_scalers,
    fit_and_scale,
    inverse_predict,
)
from src.metrics import regression_metrics, per_year_metrics
from src.plotting import (
    plot_price_series,
    plot_split_regions,
    plot_learning_curves,
    plot_predicted_vs_actual,
    plot_return_predictions,
    plot_scatter,
    plot_yearly_perf,
    plot_residuals,
)

__all__ = [
    # data
    "WindowedSplit",
    "ScaledSplits",
    "make_windows",
    "temporal_split",
    "fit_scalers",
    "fit_and_scale",
    "inverse_predict",
    # metrics
    "regression_metrics",
    "per_year_metrics",
    # plotting
    "plot_price_series",
    "plot_split_regions",
    "plot_learning_curves",
    "plot_predicted_vs_actual",
    "plot_return_predictions",
    "plot_scatter",
    "plot_yearly_perf",
    "plot_residuals",
]

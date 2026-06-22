"""src — módulos reutilizáveis de dados, métricas e plotagem para experimentos de forecasting."""

import inspect
from pathlib import Path
from typing import Final

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
    # paths
    "ROOT_DIR",
    "resolve_relative_path",
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

# Raiz do repositório — sempre resolvida a partir do local deste arquivo,
# independente de onde o processo (ou kernel Jupyter) foi iniciado.
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent


def resolve_relative_path(relative_path: str) -> Path:
    """
    Resolve an absolute path from a given relative path based on the caller's file location.

    This function determines the absolute path by using the file location of the script
    that calls this function. It inspects the caller's stack frame to retrieve the caller's
    __file__ variable, then constructs the absolute path accordingly.

    :param relative_path: The relative path to be resolved.
    :type relative_path: str
    :return: The absolute path corresponding to the given relative path.
    :rtype: Path
    """
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.frame.f_globals["__file__"]
    return Path(caller_file).parent.joinpath(relative_path).resolve().absolute()


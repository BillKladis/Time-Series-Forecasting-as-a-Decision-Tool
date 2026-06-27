"""
STL decomposition (Cleveland et al. 1990) via statsmodels.

get_stl_decomposition(series, period=52):
  - Returns StlDecomposition namedtuple with: trend, seasonal, residual, dates
  - Use statsmodels STL with robust=True (handles promotional outliers)

seasonal_strength(seasonal, residual):
  - Hyndman & Athanasopoulos: F_S = max(0, 1 - Var(R)/(Var(S+R)))
  - Returns float in [0,1]: 0 = no seasonality, 1 = perfectly seasonal

trend_strength(trend, residual):
  - F_T = max(0, 1 - Var(R)/(Var(T+R)))
  - Returns float in [0,1]

abc_classification(df):
  - Groups by store_id x item_id, computes total sales
  - Sorts by descending total
  - Cumulative share: A = top 80%, B = next 15%, C = bottom 5%
  - Returns DataFrame with columns: store_id, item_id, total_sales, cumulative_pct, class

demand_cv(df):
  - Coefficient of variation per store/item: CV = std/mean * 100
  - Returns DataFrame with store_id, item_id, mean_sales, std_sales, cv_pct, demand_class
    where demand_class: 'Smooth' (CV<20), 'Erratic' (CV 20-50), 'Lumpy' (CV>50)
"""

import numpy as np
import pandas as pd
import warnings
from collections import namedtuple

warnings.filterwarnings("ignore")

StlDecomposition = namedtuple("StlDecomposition", ["trend", "seasonal", "residual", "dates"])


def get_stl_decomposition(series: np.ndarray, period: int = 52, dates=None) -> StlDecomposition:
    """
    Perform STL decomposition on a time series.

    Uses statsmodels STL with robust=True to handle promotional outliers.

    Parameters
    ----------
    series : np.ndarray
        Time series values.
    period : int
        Seasonal period (52 for weekly data).
    dates : array-like, optional
        Corresponding dates for the series.

    Returns
    -------
    StlDecomposition namedtuple with trend, seasonal, residual, dates.
    """
    from statsmodels.tsa.seasonal import STL

    series = np.asarray(series, dtype=float)

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    return StlDecomposition(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        dates=dates,
    )


def seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    """
    Compute seasonal strength following Hyndman & Athanasopoulos.

    F_S = max(0, 1 - Var(R) / Var(S + R))

    Parameters
    ----------
    seasonal : np.ndarray
        Seasonal component from STL decomposition.
    residual : np.ndarray
        Residual component from STL decomposition.

    Returns
    -------
    float in [0, 1]. 0 = no seasonality, 1 = perfectly seasonal.
    """
    seasonal = np.asarray(seasonal, dtype=float)
    residual = np.asarray(residual, dtype=float)

    var_residual = np.var(residual)
    var_seasonal_plus_residual = np.var(seasonal + residual)

    if var_seasonal_plus_residual == 0:
        return 0.0

    fs = max(0.0, 1.0 - var_residual / var_seasonal_plus_residual)
    return float(fs)


def trend_strength(trend: np.ndarray, residual: np.ndarray) -> float:
    """
    Compute trend strength following Hyndman & Athanasopoulos.

    F_T = max(0, 1 - Var(R) / Var(T + R))

    Parameters
    ----------
    trend : np.ndarray
        Trend component from STL decomposition.
    residual : np.ndarray
        Residual component from STL decomposition.

    Returns
    -------
    float in [0, 1]. 0 = no trend, 1 = perfectly trended.
    """
    trend = np.asarray(trend, dtype=float)
    residual = np.asarray(residual, dtype=float)

    var_residual = np.var(residual)
    var_trend_plus_residual = np.var(trend + residual)

    if var_trend_plus_residual == 0:
        return 0.0

    ft = max(0.0, 1.0 - var_residual / var_trend_plus_residual)
    return float(ft)


def abc_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    ABC classification based on total sales volume.

    Groups by store_id x item_id, computes total sales,
    and assigns A/B/C class based on cumulative revenue share:
      A = top 80%, B = next 15%, C = bottom 5%

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: store_id, item_id, sales.

    Returns
    -------
    pd.DataFrame with columns: store_id, item_id, total_sales, cumulative_pct, class
    """
    grouped = (
        df.groupby(["store_id", "item_id"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "total_sales"})
    )

    grouped = grouped.sort_values("total_sales", ascending=False).reset_index(drop=True)

    total_all = grouped["total_sales"].sum()
    grouped["cumulative_pct"] = grouped["total_sales"].cumsum() / total_all * 100

    def assign_class(cum_pct):
        if cum_pct <= 80:
            return "A"
        elif cum_pct <= 95:
            return "B"
        else:
            return "C"

    grouped["class"] = grouped["cumulative_pct"].apply(assign_class)
    grouped["cumulative_pct"] = grouped["cumulative_pct"].round(2)
    grouped["total_sales"] = grouped["total_sales"].round(1)

    return grouped[["store_id", "item_id", "total_sales", "cumulative_pct", "class"]]


def demand_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coefficient of variation per store/item.

    CV = std / mean * 100

    Demand class:
      'Smooth'  : CV < 20
      'Erratic' : CV 20-50
      'Lumpy'   : CV > 50

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: store_id, item_id, sales.

    Returns
    -------
    pd.DataFrame with columns:
        store_id, item_id, mean_sales, std_sales, cv_pct, demand_class
    """
    grouped = df.groupby(["store_id", "item_id"])["sales"].agg(
        mean_sales="mean",
        std_sales="std",
    ).reset_index()

    grouped["cv_pct"] = np.where(
        grouped["mean_sales"] > 0,
        grouped["std_sales"] / grouped["mean_sales"] * 100,
        0.0,
    )

    def classify_demand(cv):
        if cv < 20:
            return "Smooth"
        elif cv <= 50:
            return "Erratic"
        else:
            return "Lumpy"

    grouped["demand_class"] = grouped["cv_pct"].apply(classify_demand)
    grouped["mean_sales"] = grouped["mean_sales"].round(2)
    grouped["std_sales"] = grouped["std_sales"].round(2)
    grouped["cv_pct"] = grouped["cv_pct"].round(2)

    return grouped[["store_id", "item_id", "mean_sales", "std_sales", "cv_pct", "demand_class"]]

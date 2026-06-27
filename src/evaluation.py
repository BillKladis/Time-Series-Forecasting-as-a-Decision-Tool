"""
Rolling-origin (walk-forward) cross-validation for time series.

rolling_origin_cv(series, model_fn, h=8, n_origins=3, min_train=52):
  - Creates n_origins test windows, each of size h
  - Origin k uses training data ending at T - (n_origins - k) * h
  - Returns dict: {'mae': float, 'mape': float, 'winkler': float, 'per_horizon': array}

winkler_score(actual, lower, upper, alpha=0.05):
  - Proper scoring rule for prediction intervals
  - Width penalty + coverage failure penalty:
    W = (U - L) + (2/alpha) * max(L - y, 0) + (2/alpha) * max(y - U, 0)
  - Lower is better; equals width when perfectly calibrated

diebold_mariano(e1, e2, h=1):
  - Harvey et al. (1997) small-sample corrected DM test
  - H0: equal predictive accuracy
  - Returns (DM_stat, p_value)
  - Uses MSE loss differential: d_t = e1_t^2 - e2_t^2
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def winkler_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05) -> float:
    """
    Winkler proper scoring rule for prediction intervals.

    W = (U - L) + (2/alpha) * max(L - y, 0) + (2/alpha) * max(y - U, 0)

    Lower is better; equals interval width when perfectly calibrated.

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    alpha : float
        Significance level (0.05 for 95% intervals).

    Returns
    -------
    float
        Mean Winkler score across all time steps.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower
    penalty_below = (2.0 / alpha) * np.maximum(lower - actual, 0.0)
    penalty_above = (2.0 / alpha) * np.maximum(actual - upper, 0.0)

    scores = width + penalty_below + penalty_above
    return float(np.mean(scores))


def rolling_origin_cv(
    series: np.ndarray,
    model_fn,
    h: int = 8,
    n_origins: int = 3,
    min_train: int = 52,
) -> dict:
    """
    Rolling-origin (walk-forward) cross-validation.

    Creates n_origins test windows, each of size h.
    Origin k uses training data ending at T - (n_origins - k) * h.

    Parameters
    ----------
    series : np.ndarray
        Full time series.
    model_fn : callable
        Function with signature model_fn(train) -> (forecast, lower, upper)
        where forecast, lower, upper are arrays of length h.
    h : int
        Forecast horizon per origin.
    n_origins : int
        Number of rolling origins.
    min_train : int
        Minimum training size; skip origins with fewer training points.

    Returns
    -------
    dict with keys:
        'mae': float
        'mape': float
        'winkler': float
        'per_horizon': np.ndarray of shape (h,) — mean MAE per horizon step
    """
    series = np.asarray(series, dtype=float)
    T = len(series)

    all_errors = []
    all_actuals = []
    all_forecasts = []
    all_lowers = []
    all_uppers = []
    per_horizon_errors = []

    for k in range(n_origins):
        # Training data ends at T - (n_origins - k) * h
        train_end = T - (n_origins - k) * h
        if train_end < min_train:
            continue

        train = series[:train_end]
        test = series[train_end: train_end + h]

        if len(test) < h:
            continue

        try:
            fc, lower, upper = model_fn(train)
            fc = np.asarray(fc, dtype=float)[:h]
            lower = np.asarray(lower, dtype=float)[:h]
            upper = np.asarray(upper, dtype=float)[:h]

            errors = np.abs(fc - test)
            all_errors.extend(errors.tolist())
            all_actuals.extend(test.tolist())
            all_forecasts.extend(fc.tolist())
            all_lowers.extend(lower.tolist())
            all_uppers.extend(upper.tolist())
            per_horizon_errors.append(errors)

        except Exception:
            continue

    if not all_errors:
        return {
            "mae": np.nan,
            "mape": np.nan,
            "winkler": np.nan,
            "per_horizon": np.full(h, np.nan),
        }

    all_errors = np.array(all_errors)
    all_actuals = np.array(all_actuals)
    all_forecasts = np.array(all_forecasts)
    all_lowers = np.array(all_lowers)
    all_uppers = np.array(all_uppers)

    mae = float(np.mean(all_errors))

    mask = all_actuals > 0
    mape = float(np.mean(np.abs((all_forecasts[mask] - all_actuals[mask]) / all_actuals[mask])) * 100) if mask.any() else np.nan

    winkler = winkler_score(all_actuals, all_lowers, all_uppers)

    if per_horizon_errors:
        stacked = np.array(per_horizon_errors)  # shape (n_valid_origins, h)
        per_horizon = np.mean(stacked, axis=0)
    else:
        per_horizon = np.full(h, np.nan)

    return {
        "mae": round(mae, 4),
        "mape": round(mape, 4) if not np.isnan(mape) else np.nan,
        "winkler": round(winkler, 4),
        "per_horizon": per_horizon,
    }


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """
    Harvey et al. (1997) small-sample corrected Diebold-Mariano test.

    H0: equal predictive accuracy between two models.
    Uses MSE loss differential: d_t = e1_t^2 - e2_t^2

    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors from model 1 (actual - forecast).
    e2 : np.ndarray
        Forecast errors from model 2 (actual - forecast).
    h : int
        Forecast horizon (used for Harvey correction).

    Returns
    -------
    tuple (DM_stat, p_value)
        DM_stat: Harvey-corrected test statistic
        p_value: two-sided p-value from t-distribution with n-1 df
    """
    from scipy.stats import t as t_dist

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    n = len(e1)
    if n < 3:
        return np.nan, np.nan

    # Loss differential using MSE
    d = e1 ** 2 - e2 ** 2

    d_bar = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var <= 0 or np.isnan(d_var):
        return np.nan, np.nan

    # Harvey et al. (1997) correction factor
    harvey_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)

    dm_stat = (d_bar / np.sqrt(d_var / n)) * harvey_correction

    # Two-sided p-value from t distribution with n-1 degrees of freedom
    p_value = 2.0 * (1.0 - t_dist.cdf(np.abs(dm_stat), df=n - 1))

    return float(dm_stat), float(p_value)

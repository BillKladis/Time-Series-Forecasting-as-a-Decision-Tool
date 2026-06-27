"""
Extended forecasting models for Stock Sense.

theta_forecast(train, horizon):
  Theta method (Assimakopoulos & Nikolopoulos 2000, M3 winner).
  Equivalent to SES with linear drift (Hyndman & Billah 2003).

sarimax_promo_forecast(train, promo_train, promo_future, horizon):
  SARIMAX(1,1,1)(1,1,0,52) with promotional dummy as exogenous.

ensemble_forecast(forecasts_dict, weights=None):
  Weighted average ensemble.

compute_ensemble_weights(backtest_maes):
  Inverse-MAE weights (Bates & Granger 1969).
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def theta_forecast(train: np.ndarray, horizon: int):
    """
    Theta method (Assimakopoulos & Nikolopoulos 2000, M3 winner).
    Equivalent to SES with linear drift (Hyndman & Billah 2003).

    Steps:
    1. Fit SES (ExponentialSmoothing with trend=None, seasonal=None) to get alpha and level
    2. Compute linear trend from OLS on t = 0..n-1 vs train
    3. Theta forecast = SES_forecast + (trend_slope/2) * h for h=1..horizon
    4. Prediction interval: approximate via residual std * z * sqrt(h)

    Parameters
    ----------
    train : np.ndarray
        Historical time series values.
    horizon : int
        Number of steps ahead to forecast.

    Returns
    -------
    tuple (forecast, lower_95, upper_95)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    train = np.asarray(train, dtype=float)
    n = len(train)

    try:
        # Step 1: Fit SES (simple exponential smoothing)
        ses_model = ExponentialSmoothing(
            train,
            trend=None,
            seasonal=None,
        )
        ses_result = ses_model.fit(optimized=True)

        # SES level forecast (flat from last fitted value)
        ses_fc = ses_result.forecast(horizon)

        # Step 2: OLS linear trend on the training data
        t = np.arange(n, dtype=float)
        coeffs = np.polyfit(t, train, deg=1)  # [slope, intercept]
        slope = coeffs[0]

        # Step 3: Theta forecast = SES_forecast + (slope/2) * h
        h_steps = np.arange(1, horizon + 1, dtype=float)
        fc = ses_fc + (slope / 2.0) * h_steps

        # Step 4: Approximate 95% prediction interval via residual std
        in_sample_fc = ses_result.fittedvalues
        residuals = train - in_sample_fc
        resid_std = float(np.std(residuals, ddof=1))

        z = 1.96
        margin = z * resid_std * np.sqrt(h_steps)

        fc = np.maximum(fc, 0)
        lower = np.maximum(fc - margin, 0)
        upper = fc + margin

        return fc, lower, upper

    except Exception:
        # Fallback: naive mean forecast
        mean_val = float(np.mean(train[-12:]))
        std_val = float(np.std(train[-12:]))
        h_steps = np.arange(1, horizon + 1, dtype=float)
        fc = np.full(horizon, mean_val)
        lower = np.maximum(fc - 1.96 * std_val * np.sqrt(h_steps), 0)
        upper = fc + 1.96 * std_val * np.sqrt(h_steps)
        return fc, lower, upper


def sarimax_promo_forecast(
    train: np.ndarray,
    promo_train: np.ndarray,
    promo_future: np.ndarray,
    horizon: int,
):
    """
    SARIMAX(1,1,1)(1,1,0,52) with promotional dummy as exogenous regressor.

    Parameters
    ----------
    train : np.ndarray
        Historical sales series.
    promo_train : np.ndarray
        Promotional indicator for training period (0/1).
    promo_future : np.ndarray
        Promotional indicator for forecast period. If None or wrong length,
        assumes zeros (no promotions in forecast period).
    horizon : int
        Number of steps ahead to forecast.

    Returns
    -------
    tuple (forecast, lower_95, upper_95, promo_lift_coeff)
        promo_lift_coeff: estimated lift in units per promo week.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = np.asarray(train, dtype=float)
    promo_train = np.asarray(promo_train, dtype=float)

    # Handle future promo values
    if promo_future is None or len(promo_future) != horizon:
        promo_future = np.zeros(horizon, dtype=float)
    else:
        promo_future = np.asarray(promo_future, dtype=float)

    # Try SARIMAX with promo exogenous
    try:
        model = SARIMAX(
            train,
            exog=promo_train.reshape(-1, 1),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)

        promo_lift_coeff = float(result.params.get("x1", result.params.iloc[-1] if "x1" not in result.params.index else result.params["x1"]))

        forecast_obj = result.get_forecast(steps=horizon, exog=promo_future.reshape(-1, 1))
        fc_mean = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values

        return (
            np.maximum(fc_mean.values, 0),
            np.maximum(lower, 0),
            np.maximum(upper, 0),
            round(promo_lift_coeff, 3),
        )

    except Exception:
        pass

    # Fallback: regular SARIMA without promo
    try:
        from src.forecasting import fit_sarima
        fc, lower, upper = fit_sarima(train, horizon)
        return fc, lower, upper, 0.0
    except Exception:
        pass

    # Final fallback: naive
    mean_val = float(np.mean(train[-12:]))
    std_val = float(np.std(train[-12:]))
    fc = np.full(horizon, mean_val)
    lower = np.maximum(fc - 1.96 * std_val, 0)
    upper = fc + 1.96 * std_val
    return fc, lower, upper, 0.0


def ensemble_forecast(forecasts_dict: dict, weights: dict = None):
    """
    Weighted average ensemble of multiple forecasts.

    Parameters
    ----------
    forecasts_dict : dict
        {'model_name': (fc, lower, upper), ...}
        where fc, lower, upper are np.ndarray of the same length.
    weights : dict, optional
        {'model_name': weight, ...}. Default = equal weights.
        Weights are normalized to sum to 1 internally.

    Returns
    -------
    tuple (ensemble_fc, ensemble_lower, ensemble_upper)
    """
    model_names = list(forecasts_dict.keys())

    if not model_names:
        raise ValueError("forecasts_dict is empty.")

    if weights is None:
        # Equal weights
        w = {name: 1.0 / len(model_names) for name in model_names}
    else:
        # Normalize provided weights
        total_w = sum(weights.get(name, 0.0) for name in model_names)
        if total_w <= 0:
            w = {name: 1.0 / len(model_names) for name in model_names}
        else:
            w = {name: weights.get(name, 0.0) / total_w for name in model_names}

    # Determine horizon from first model
    first_fc, first_lower, first_upper = forecasts_dict[model_names[0]]
    horizon = len(first_fc)

    ensemble_fc = np.zeros(horizon)
    ensemble_lower = np.zeros(horizon)
    ensemble_upper = np.zeros(horizon)

    for name in model_names:
        fc, lower, upper = forecasts_dict[name]
        weight = w.get(name, 0.0)
        ensemble_fc += weight * np.asarray(fc, dtype=float)
        ensemble_lower += weight * np.asarray(lower, dtype=float)
        ensemble_upper += weight * np.asarray(upper, dtype=float)

    return ensemble_fc, ensemble_lower, ensemble_upper


def compute_ensemble_weights(backtest_maes: dict) -> dict:
    """
    Compute inverse-MAE weights (Bates & Granger 1969).

    Models with lower MAE receive higher weight.

    Parameters
    ----------
    backtest_maes : dict
        {'model_name': mae_value, ...}
        NaN or None values are excluded.

    Returns
    -------
    dict {'model_name': weight, ...} normalized to sum to 1.
    """
    # Filter out invalid MAE values
    valid = {
        name: mae
        for name, mae in backtest_maes.items()
        if mae is not None and not (isinstance(mae, float) and np.isnan(mae)) and mae > 0
    }

    if not valid:
        # Equal weights for all
        n = len(backtest_maes)
        return {name: 1.0 / n for name in backtest_maes}

    inv_maes = {name: 1.0 / mae for name, mae in valid.items()}
    total = sum(inv_maes.values())
    weights = {name: inv_mae / total for name, inv_mae in inv_maes.items()}

    # For models that were excluded (invalid MAE), assign 0 weight
    for name in backtest_maes:
        if name not in weights:
            weights[name] = 0.0

    return weights

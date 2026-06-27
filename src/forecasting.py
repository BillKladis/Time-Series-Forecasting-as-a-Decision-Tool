"""
Forecasting models for Stock Sense.
MODEL 1: SARIMA (statsmodels SARIMAX)
MODEL 2: ETS (statsforecast AutoETS, with fallback to statsmodels ExponentialSmoothing)
MODEL 3: Theta (Assimakopoulos & Nikolopoulos 2000) — via src.models
MODEL 4: SARIMAX+Promo — via src.models
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def fit_sarima(train: np.ndarray, horizon: int):
    """
    Fit SARIMA(1,1,1)(1,1,0,52) on weekly data.
    Returns (forecast, lower_95, upper_95).
    Falls back to ARIMA(1,1,1) if seasonal fit fails.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Try seasonal SARIMA first
    try:
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        forecast_obj = result.get_forecast(steps=horizon)
        fc_mean = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return np.maximum(fc_mean.values, 0), np.maximum(lower, 0), np.maximum(upper, 0)
    except Exception:
        pass

    # Fallback: ARIMA(1,1,1) without seasonality
    try:
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        forecast_obj = result.get_forecast(steps=horizon)
        fc_mean = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return np.maximum(fc_mean.values, 0), np.maximum(lower, 0), np.maximum(upper, 0)
    except Exception as e:
        # Final fallback: naive mean forecast
        mean_val = float(np.mean(train[-12:]))
        std_val = float(np.std(train[-12:]))
        fc = np.full(horizon, mean_val)
        lower = np.maximum(fc - 1.96 * std_val, 0)
        upper = fc + 1.96 * std_val
        return fc, lower, upper


def fit_ets(train: np.ndarray, horizon: int):
    """
    Fit ETS model. Uses statsforecast AutoETS if available,
    otherwise falls back to statsmodels ExponentialSmoothing.
    Returns (forecast, lower_95, upper_95).
    """
    # Try statsforecast AutoETS
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        n = len(train)
        sf_df = pd.DataFrame({
            "unique_id": ["series1"] * n,
            "ds": pd.date_range(start="2021-01-04", periods=n, freq="W-MON"),
            "y": train,
        })

        sf = StatsForecast(
            models=[AutoETS(season_length=52)],
            freq="W-MON",
            n_jobs=1,
        )
        sf.fit(df=sf_df)
        forecast_df = sf.predict(h=horizon, level=[95])

        fc = forecast_df["AutoETS"].values
        lower = forecast_df["AutoETS-lo-95"].values
        upper = forecast_df["AutoETS-hi-95"].values
        return np.maximum(fc, 0), np.maximum(lower, 0), np.maximum(upper, 0)

    except Exception:
        pass

    # Fallback: statsmodels ExponentialSmoothing
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Need enough data for seasonal period
        seasonal_periods = 52
        if len(train) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            )
        else:
            model = ExponentialSmoothing(
                train,
                trend="add",
            )

        result = model.fit(optimized=True)
        fc = result.forecast(horizon)
        # Compute prediction interval via simulation
        sim = result.simulate(horizon, repetitions=500, error="add")
        lower = np.maximum(np.percentile(sim, 2.5, axis=1), 0)
        upper = np.percentile(sim, 97.5, axis=1)
        return np.maximum(fc.values, 0), lower, upper

    except Exception:
        # Last resort: naive
        mean_val = float(np.mean(train[-12:]))
        std_val = float(np.std(train[-12:]))
        fc = np.full(horizon, mean_val)
        lower = np.maximum(fc - 1.96 * std_val, 0)
        upper = fc + 1.96 * std_val
        return fc, lower, upper


def run_backtest(series: np.ndarray, promo_series: np.ndarray = None):
    """
    Backtest: train on all but last 12 weeks, test on last 12.

    Includes SARIMA, ETS, Theta, and SARIMAX+Promo models.
    Also runs rolling-origin CV (3 origins, h=8) for each model.
    Returns Winkler interval scores and DM test p-values vs SARIMA baseline.

    Parameters
    ----------
    series : np.ndarray
        Full sales time series.
    promo_series : np.ndarray, optional
        Promotional indicator series aligned with `series`.

    Returns
    -------
    dict with all model results, rolling CV results, and Winkler scores.
    """
    from src.evaluation import rolling_origin_cv, winkler_score, diebold_mariano
    from src.models import theta_forecast, sarimax_promo_forecast

    n_test = 12
    train = series[:-n_test]
    test = series[-n_test:]

    if promo_series is not None:
        promo_series = np.asarray(promo_series, dtype=float)
        promo_train = promo_series[:-n_test]
        promo_future = promo_series[-n_test:]
    else:
        promo_train = np.zeros(len(train))
        promo_future = np.zeros(n_test)

    results = {}

    # -----------------------------------------------------------------------
    # SARIMA backtest
    # -----------------------------------------------------------------------
    try:
        fc_sarima, lower_sarima, upper_sarima = fit_sarima(train, n_test)
        mae_sarima = float(np.mean(np.abs(fc_sarima - test)))
        mask = test > 0
        mape_sarima = float(np.mean(np.abs((fc_sarima[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        winkler_sarima = winkler_score(test, lower_sarima, upper_sarima)
        results["sarima"] = {
            "MAE": round(mae_sarima, 2),
            "MAPE": round(mape_sarima, 2),
            "Winkler": round(winkler_sarima, 2),
            "forecast": fc_sarima,
            "lower": lower_sarima,
            "upper": upper_sarima,
            "errors": fc_sarima - test,
        }
    except Exception:
        results["sarima"] = {
            "MAE": None, "MAPE": None, "Winkler": None,
            "forecast": np.full(n_test, np.nan),
            "lower": np.full(n_test, np.nan),
            "upper": np.full(n_test, np.nan),
            "errors": np.full(n_test, np.nan),
        }

    # -----------------------------------------------------------------------
    # ETS backtest
    # -----------------------------------------------------------------------
    try:
        fc_ets, lower_ets, upper_ets = fit_ets(train, n_test)
        mae_ets = float(np.mean(np.abs(fc_ets - test)))
        mask = test > 0
        mape_ets = float(np.mean(np.abs((fc_ets[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        winkler_ets = winkler_score(test, lower_ets, upper_ets)
        dm_stat_ets, dm_p_ets = diebold_mariano(results["sarima"]["errors"], fc_ets - test)
        results["ets"] = {
            "MAE": round(mae_ets, 2),
            "MAPE": round(mape_ets, 2),
            "Winkler": round(winkler_ets, 2),
            "DM_p": round(dm_p_ets, 4) if dm_p_ets is not None and not np.isnan(dm_p_ets) else None,
            "forecast": fc_ets,
            "lower": lower_ets,
            "upper": upper_ets,
            "errors": fc_ets - test,
        }
    except Exception:
        results["ets"] = {
            "MAE": None, "MAPE": None, "Winkler": None, "DM_p": None,
            "forecast": np.full(n_test, np.nan),
            "lower": np.full(n_test, np.nan),
            "upper": np.full(n_test, np.nan),
            "errors": np.full(n_test, np.nan),
        }

    # -----------------------------------------------------------------------
    # Theta backtest
    # -----------------------------------------------------------------------
    try:
        fc_theta, lower_theta, upper_theta = theta_forecast(train, n_test)
        mae_theta = float(np.mean(np.abs(fc_theta - test)))
        mask = test > 0
        mape_theta = float(np.mean(np.abs((fc_theta[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        winkler_theta = winkler_score(test, lower_theta, upper_theta)
        dm_stat_theta, dm_p_theta = diebold_mariano(results["sarima"]["errors"], fc_theta - test)
        results["theta"] = {
            "MAE": round(mae_theta, 2),
            "MAPE": round(mape_theta, 2),
            "Winkler": round(winkler_theta, 2),
            "DM_p": round(dm_p_theta, 4) if dm_p_theta is not None and not np.isnan(dm_p_theta) else None,
            "forecast": fc_theta,
            "lower": lower_theta,
            "upper": upper_theta,
            "errors": fc_theta - test,
        }
    except Exception:
        results["theta"] = {
            "MAE": None, "MAPE": None, "Winkler": None, "DM_p": None,
            "forecast": np.full(n_test, np.nan),
            "lower": np.full(n_test, np.nan),
            "upper": np.full(n_test, np.nan),
            "errors": np.full(n_test, np.nan),
        }

    # -----------------------------------------------------------------------
    # SARIMAX+Promo backtest
    # -----------------------------------------------------------------------
    try:
        fc_sarimax, lower_sarimax, upper_sarimax, promo_lift = sarimax_promo_forecast(
            train, promo_train, promo_future, n_test
        )
        mae_sarimax = float(np.mean(np.abs(fc_sarimax - test)))
        mask = test > 0
        mape_sarimax = float(np.mean(np.abs((fc_sarimax[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        winkler_sarimax = winkler_score(test, lower_sarimax, upper_sarimax)
        dm_stat_sarimax, dm_p_sarimax = diebold_mariano(results["sarima"]["errors"], fc_sarimax - test)
        results["sarimax_promo"] = {
            "MAE": round(mae_sarimax, 2),
            "MAPE": round(mape_sarimax, 2),
            "Winkler": round(winkler_sarimax, 2),
            "DM_p": round(dm_p_sarimax, 4) if dm_p_sarimax is not None and not np.isnan(dm_p_sarimax) else None,
            "promo_lift": promo_lift,
            "forecast": fc_sarimax,
            "lower": lower_sarimax,
            "upper": upper_sarimax,
            "errors": fc_sarimax - test,
        }
    except Exception:
        results["sarimax_promo"] = {
            "MAE": None, "MAPE": None, "Winkler": None, "DM_p": None, "promo_lift": None,
            "forecast": np.full(n_test, np.nan),
            "lower": np.full(n_test, np.nan),
            "upper": np.full(n_test, np.nan),
            "errors": np.full(n_test, np.nan),
        }

    results["n_test"] = n_test
    results["actual"] = test

    # -----------------------------------------------------------------------
    # Rolling-origin cross-validation (3 origins, h=8)
    # -----------------------------------------------------------------------
    h_cv = 8
    n_origins = 3

    # SARIMA rolling CV
    try:
        def sarima_fn(tr):
            return fit_sarima(tr, h_cv)
        cv_sarima = rolling_origin_cv(series, sarima_fn, h=h_cv, n_origins=n_origins, min_train=52)
        results["sarima"]["rolling_cv"] = cv_sarima
    except Exception:
        results["sarima"]["rolling_cv"] = {"mae": None, "mape": None, "winkler": None}

    # ETS rolling CV
    try:
        def ets_fn(tr):
            return fit_ets(tr, h_cv)
        cv_ets = rolling_origin_cv(series, ets_fn, h=h_cv, n_origins=n_origins, min_train=52)
        results["ets"]["rolling_cv"] = cv_ets
    except Exception:
        results["ets"]["rolling_cv"] = {"mae": None, "mape": None, "winkler": None}

    # Theta rolling CV
    try:
        def theta_fn(tr):
            return theta_forecast(tr, h_cv)
        cv_theta = rolling_origin_cv(series, theta_fn, h=h_cv, n_origins=n_origins, min_train=52)
        results["theta"]["rolling_cv"] = cv_theta
    except Exception:
        results["theta"]["rolling_cv"] = {"mae": None, "mape": None, "winkler": None}

    # SARIMAX+Promo rolling CV (uses zero promo for CV simplicity)
    try:
        def sarimax_fn(tr):
            promo_tr = np.zeros(len(tr))
            promo_fut = np.zeros(h_cv)
            fc, lo, hi, _ = sarimax_promo_forecast(tr, promo_tr, promo_fut, h_cv)
            return fc, lo, hi
        cv_sarimax = rolling_origin_cv(series, sarimax_fn, h=h_cv, n_origins=n_origins, min_train=52)
        results["sarimax_promo"]["rolling_cv"] = cv_sarimax
    except Exception:
        results["sarimax_promo"]["rolling_cv"] = {"mae": None, "mape": None, "winkler": None}

    return results

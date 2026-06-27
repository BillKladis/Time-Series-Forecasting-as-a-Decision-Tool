"""
Forecasting models for Stock Sense.
MODEL 1: SARIMA (statsmodels SARIMAX)
MODEL 2: ETS (statsforecast AutoETS, with fallback to statsmodels ExponentialSmoothing)
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


def run_backtest(series: np.ndarray):
    """
    Backtest: train on all but last 12 weeks, test on last 12.
    Returns dict with MAE and MAPE for both models.
    """
    n_test = 12
    train = series[:-n_test]
    test = series[-n_test:]

    results = {}

    # SARIMA backtest
    try:
        fc_sarima, _, _ = fit_sarima(train, n_test)
        mae_sarima = float(np.mean(np.abs(fc_sarima - test)))
        # Avoid division by zero in MAPE
        mask = test > 0
        mape_sarima = float(np.mean(np.abs((fc_sarima[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        results["sarima"] = {
            "MAE": round(mae_sarima, 2),
            "MAPE": round(mape_sarima, 2),
            "forecast": fc_sarima,
        }
    except Exception as e:
        results["sarima"] = {"MAE": None, "MAPE": None, "forecast": np.full(n_test, np.nan)}

    # ETS backtest
    try:
        fc_ets, _, _ = fit_ets(train, n_test)
        mae_ets = float(np.mean(np.abs(fc_ets - test)))
        mask = test > 0
        mape_ets = float(np.mean(np.abs((fc_ets[mask] - test[mask]) / test[mask])) * 100) if mask.any() else 0.0
        results["ets"] = {
            "MAE": round(mae_ets, 2),
            "MAPE": round(mape_ets, 2),
            "forecast": fc_ets,
        }
    except Exception as e:
        results["ets"] = {"MAE": None, "MAPE": None, "forecast": np.full(n_test, np.nan)}

    results["n_test"] = n_test
    results["actual"] = test

    return results

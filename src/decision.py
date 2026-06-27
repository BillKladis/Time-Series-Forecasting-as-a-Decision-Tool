"""
Decision layer for Stock Sense.
Converts forecasts into inventory decisions.
"""

import numpy as np
from scipy.stats import norm


def compute_inventory_decision(
    forecast: np.ndarray,
    historical_series: np.ndarray,
    service_level: float = 0.95,
    lead_time: int = 2,
) -> dict:
    """
    Compute inventory decisions from a forecast.

    Parameters
    ----------
    forecast : np.ndarray
        Point forecast for the horizon period (weekly units).
    historical_series : np.ndarray
        Full historical sales series used to estimate demand variability.
    service_level : float
        Desired service level (0.80 to 0.99).
    lead_time : int
        Lead time in weeks.

    Returns
    -------
    dict with keys:
        safety_stock, rop, order_qty, mean_weekly_demand,
        z_score, interpretation_text
    """
    horizon = len(forecast)

    # Z-score from service level using scipy
    z_score = float(norm.ppf(service_level))

    # Demand statistics
    mean_weekly_demand = float(np.mean(forecast))

    # Use historical demand variability for safety stock — the forecast may be
    # a flat line (e.g. ETS without trend), which would understate variability.
    hist_window = historical_series[-52:] if len(historical_series) >= 52 else historical_series
    sigma_demand = float(np.std(hist_window))
    if sigma_demand == 0:
        sigma_demand = mean_weekly_demand * 0.15  # fallback: 15% CV

    # Standard deviation of demand during lead time
    sigma_lead = sigma_demand * np.sqrt(lead_time)

    # Safety stock formula: Z * sigma_lead
    safety_stock = z_score * sigma_lead
    safety_stock = max(safety_stock, 0.0)

    # Reorder point
    rop = mean_weekly_demand * lead_time + safety_stock

    # Current inventory estimate: 2 weeks of mean historical demand
    mean_hist_demand = float(np.mean(historical_series[-52:])) if len(historical_series) >= 52 else float(np.mean(historical_series))
    current_inventory_estimate = 2 * mean_hist_demand

    # Recommended order quantity
    total_forecast_demand = float(np.sum(forecast))
    order_qty = total_forecast_demand + safety_stock - current_inventory_estimate
    order_qty = max(order_qty, 0.0)

    # Interpretation text
    sl_pct = int(round(service_level * 100))
    interpretation_text = (
        f"To hit {sl_pct}% service level over the next {horizon} weeks "
        f"with a {lead_time}-week lead time, order {int(round(order_qty)):,} units. "
        f"Safety stock: {int(round(safety_stock)):,} units. "
        f"Reorder point: {int(round(rop)):,} units "
        f"(order when inventory drops to this level)."
    )

    return {
        "safety_stock": round(safety_stock, 1),
        "rop": round(rop, 1),
        "order_qty": round(order_qty, 1),
        "mean_weekly_demand": round(mean_weekly_demand, 1),
        "z_score": round(z_score, 3),
        "interpretation_text": interpretation_text,
        "current_inventory_estimate": round(current_inventory_estimate, 1),
        "total_forecast_demand": round(total_forecast_demand, 1),
        "service_level": service_level,
        "lead_time": lead_time,
        "horizon": horizon,
    }

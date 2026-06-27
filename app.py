"""
Stock Sense — Retail Demand Forecasting App
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Stock Sense",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
# Cached forecasting helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_sarima_forecast(store_id: str, item_id: str, horizon: int, n_train: int):
    """Cache SARIMA forecast by store/item/horizon/training length."""
    from src.forecasting import fit_sarima
    df = load_data()
    series = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")["sales"].values
    train = series[:n_train]
    fc, lower, upper = fit_sarima(train, horizon)
    return fc, lower, upper


@st.cache_data(show_spinner=False)
def get_ets_forecast(store_id: str, item_id: str, horizon: int, n_train: int):
    """Cache ETS forecast by store/item/horizon/training length."""
    from src.forecasting import fit_ets
    df = load_data()
    series = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")["sales"].values
    train = series[:n_train]
    fc, lower, upper = fit_ets(train, horizon)
    return fc, lower, upper


@st.cache_data(show_spinner=False)
def get_theta_forecast(store_id: str, item_id: str, horizon: int, n_train: int):
    """Cache Theta forecast by store/item/horizon/training length."""
    from src.models import theta_forecast
    df = load_data()
    series = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")["sales"].values
    train = series[:n_train]
    fc, lower, upper = theta_forecast(train, horizon)
    return fc, lower, upper


@st.cache_data(show_spinner=False)
def get_sarimax_promo_forecast(store_id: str, item_id: str, horizon: int, n_train: int):
    """Cache SARIMAX+Promo forecast by store/item/horizon/training length."""
    from src.models import sarimax_promo_forecast
    df = load_data()
    sub = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")
    series = sub["sales"].values
    promo = sub["is_promo"].values if "is_promo" in sub.columns else np.zeros(len(series))
    train = series[:n_train]
    promo_train = promo[:n_train]
    promo_future = np.zeros(horizon)
    fc, lower, upper, lift = sarimax_promo_forecast(train, promo_train, promo_future, horizon)
    return fc, lower, upper, lift


@st.cache_data(show_spinner=False)
def get_backtest(store_id: str, item_id: str):
    """Cache backtest results by store/item."""
    from src.forecasting import run_backtest
    df = load_data()
    sub = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")
    series = sub["sales"].values
    promo = sub["is_promo"].values if "is_promo" in sub.columns else None
    return run_backtest(series, promo_series=promo)


@st.cache_data(show_spinner=False)
def get_stl_decomposition_cached(store_id: str, item_id: str):
    """Cache STL decomposition by store/item."""
    from src.decomposition import get_stl_decomposition
    df = load_data()
    sub = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")
    series = sub["sales"].values
    dates = sub["date"].values
    result = get_stl_decomposition(series, period=52, dates=dates)
    return result


@st.cache_data(show_spinner=False)
def get_abc_classification():
    """Cache ABC classification (full dataset)."""
    from src.decomposition import abc_classification
    df = load_data()
    return abc_classification(df)


@st.cache_data(show_spinner=False)
def get_demand_cv():
    """Cache demand CV (full dataset)."""
    from src.decomposition import demand_cv
    df = load_data()
    return demand_cv(df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📦 Stock Sense")
st.sidebar.markdown("---")

store_id = st.sidebar.selectbox("Store", ["Store_1", "Store_2", "Store_3"])
item_id = st.sidebar.selectbox("Item", ["Item_1", "Item_2", "Item_3", "Item_4", "Item_5"])
horizon = st.sidebar.select_slider(
    "Forecast Horizon (weeks)",
    options=[4, 8, 12],
    value=8,
)
service_level = st.sidebar.slider(
    "Service Level",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
    format="%.2f",
)
lead_time = st.sidebar.slider(
    "Lead Time (weeks)",
    min_value=1,
    max_value=8,
    value=2,
)

st.sidebar.markdown("---")
st.sidebar.caption("Models: SARIMA | ETS | Theta | SARIMAX+Promo | Ensemble")

# ---------------------------------------------------------------------------
# Load data and prepare series
# ---------------------------------------------------------------------------

df = load_data()
series_df = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date").reset_index(drop=True)
all_dates = series_df["date"].values
all_sales = series_df["sales"].values
n_total = len(all_sales)

# Training data: all data (forecast from end of series)
n_train = n_total

# Forecast dates (future weekly dates after end of training)
last_date = pd.Timestamp(all_dates[-1])
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(weeks=1),
    periods=horizon,
    freq="W-MON",
)

# Historical window: last 52 weeks for display
hist_window = min(52, n_total)
hist_dates = all_dates[-hist_window:]
hist_sales = all_sales[-hist_window:]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecast",
    "📊 Benchmark",
    "🎯 Decision",
    "🔍 Decomposition",
    "🔬 Diagnostics",
    "🗂️ Portfolio",
])

# ===========================================================================
# TAB 1 — FORECAST
# ===========================================================================

with tab1:
    st.title(f"Demand Forecast — {store_id} / {item_id}")

    with st.spinner("Fitting SARIMA model..."):
        fc_sarima, lower_sarima, upper_sarima = get_sarima_forecast(store_id, item_id, horizon, n_train)

    with st.spinner("Fitting ETS model..."):
        fc_ets, lower_ets, upper_ets = get_ets_forecast(store_id, item_id, horizon, n_train)

    with st.spinner("Fitting Theta model..."):
        fc_theta, lower_theta, upper_theta = get_theta_forecast(store_id, item_id, horizon, n_train)

    # Compute ensemble weights from backtest MAEs
    with st.spinner("Computing ensemble..."):
        try:
            bt_for_ensemble = get_backtest(store_id, item_id)
            from src.models import compute_ensemble_weights, ensemble_forecast
            mae_dict = {
                "sarima": bt_for_ensemble["sarima"]["MAE"],
                "ets": bt_for_ensemble["ets"]["MAE"],
                "theta": bt_for_ensemble["theta"]["MAE"],
            }
            ensemble_weights = compute_ensemble_weights(mae_dict)
            forecasts_dict = {
                "sarima": (fc_sarima, lower_sarima, upper_sarima),
                "ets": (fc_ets, lower_ets, upper_ets),
                "theta": (fc_theta, lower_theta, upper_theta),
            }
            fc_ensemble, lower_ensemble, upper_ensemble = ensemble_forecast(forecasts_dict, ensemble_weights)
        except Exception:
            fc_ensemble = (fc_sarima + fc_ets + fc_theta) / 3.0
            ensemble_weights = {"sarima": 1/3, "ets": 1/3, "theta": 1/3}

    # Build Plotly chart
    fig = go.Figure()

    # Historical sales
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_sales,
        mode="lines",
        name="Historical Sales",
        line=dict(color="#2c3e50", width=2),
    ))

    # SARIMA prediction interval (shaded)
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
        y=np.concatenate([upper_sarima, lower_sarima[::-1]]),
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=True,
        name="SARIMA 95% PI",
    ))

    # SARIMA forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=fc_sarima,
        mode="lines+markers",
        name="SARIMA Forecast",
        line=dict(color="#3498db", width=2.5, dash="dash"),
        marker=dict(size=6),
    ))

    # ETS forecast line (no interval shading)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=fc_ets,
        mode="lines+markers",
        name="ETS Forecast",
        line=dict(color="#e67e22", width=2.5, dash="dot"),
        marker=dict(size=6, symbol="diamond"),
    ))

    # Theta forecast line (green dashed)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=fc_theta,
        mode="lines+markers",
        name="Theta Forecast",
        line=dict(color="#27ae60", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="triangle-up"),
    ))

    # Ensemble forecast line (purple, no interval)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=fc_ensemble,
        mode="lines+markers",
        name="Ensemble Forecast",
        line=dict(color="#8e44ad", width=2.5),
        marker=dict(size=7, symbol="star"),
    ))

    # Add vertical line at forecast start
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Forecast Start",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Weekly Sales Forecast: {store_id} / {item_id} ({horizon}-week horizon)",
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Ensemble weights expander
    with st.expander("Ensemble weights (inverse-MAE)"):
        weights_df = pd.DataFrame({
            "Model": list(ensemble_weights.keys()),
            "Weight": [f"{v:.3f}" for v in ensemble_weights.values()],
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.caption("Weights are proportional to 1/MAE from the holdout backtest. Models with lower error receive higher weight.")

    # Metrics row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    last52_mean = float(np.mean(hist_sales))
    last52_std = float(np.std(hist_sales))
    # Trend: compare last 8 weeks vs prior 8 weeks
    if len(hist_sales) >= 16:
        recent_mean = np.mean(hist_sales[-8:])
        prior_mean = np.mean(hist_sales[-16:-8])
        trend_pct = (recent_mean - prior_mean) / prior_mean * 100 if prior_mean > 0 else 0
        trend_dir = f"{'↑' if trend_pct > 1 else '↓' if trend_pct < -1 else '→'} {abs(trend_pct):.1f}%"
    else:
        trend_dir = "—"

    with col1:
        st.metric("52-Week Mean Sales", f"{last52_mean:.1f}")
    with col2:
        st.metric("52-Week Std Dev", f"{last52_std:.1f}")
    with col3:
        st.metric("Trend (last 8 vs prior 8 wks)", trend_dir)
    with col4:
        cv = last52_std / last52_mean * 100 if last52_mean > 0 else 0
        st.metric("Coefficient of Variation", f"{cv:.1f}%")


# ===========================================================================
# TAB 2 — BENCHMARK
# ===========================================================================

with tab2:
    st.header("Backtest Comparison (last 12 weeks + rolling CV)")

    with st.spinner("Running backtest..."):
        bt = get_backtest(store_id, item_id)

    n_test = bt["n_test"]
    test_dates = all_dates[-n_test:]

    # Build extended metrics table
    def safe_val(d, key):
        v = d.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return v

    model_display = {
        "sarima": "SARIMA(1,1,1)(1,1,0,52)",
        "ets": "ETS (Auto)",
        "theta": "Theta",
        "sarimax_promo": "SARIMAX+Promo",
    }

    rows = []
    for key, label in model_display.items():
        m = bt[key]
        cv_data = m.get("rolling_cv", {})
        rows.append({
            "Model": label,
            "MAE (holdout)": safe_val(m, "MAE"),
            "MAPE % (holdout)": safe_val(m, "MAPE"),
            "Winkler (holdout)": safe_val(m, "Winkler"),
            "MAE (rolling CV)": safe_val(cv_data, "mae") if cv_data else "N/A",
            "Winkler (rolling CV)": safe_val(cv_data, "winkler") if cv_data else "N/A",
            "DM p-val vs SARIMA": safe_val(m, "DM_p") if key != "sarima" else "baseline",
        })

    metrics_df = pd.DataFrame(rows)

    st.markdown("#### Model Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Highlight winner on holdout MAE
    valid_maes = {k: bt[k]["MAE"] for k in model_display if bt[k]["MAE"] is not None}
    if valid_maes:
        best_model = min(valid_maes, key=lambda k: valid_maes[k])
        st.success(f"**Best model on holdout MAE:** {model_display[best_model]} (MAE = {valid_maes[best_model]:.2f})")

    st.markdown("---")

    # Backtest chart: actual vs predicted
    col_left, col_right = st.columns([1, 2])

    with col_right:
        fig_bt = go.Figure()

        fig_bt.add_trace(go.Scatter(
            x=test_dates,
            y=bt["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#2c3e50", width=2),
            marker=dict(size=7),
        ))

        fig_bt.add_trace(go.Scatter(
            x=test_dates,
            y=bt["sarima"]["forecast"],
            mode="lines+markers",
            name="SARIMA Predicted",
            line=dict(color="#3498db", width=2, dash="dash"),
            marker=dict(size=6),
        ))

        fig_bt.add_trace(go.Scatter(
            x=test_dates,
            y=bt["ets"]["forecast"],
            mode="lines+markers",
            name="ETS Predicted",
            line=dict(color="#e67e22", width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
        ))

        fig_bt.add_trace(go.Scatter(
            x=test_dates,
            y=bt["theta"]["forecast"],
            mode="lines+markers",
            name="Theta Predicted",
            line=dict(color="#27ae60", width=2, dash="dash"),
            marker=dict(size=6, symbol="triangle-up"),
        ))

        fig_bt.add_trace(go.Scatter(
            x=test_dates,
            y=bt["sarimax_promo"]["forecast"],
            mode="lines+markers",
            name="SARIMAX+Promo",
            line=dict(color="#8e44ad", width=2, dash="longdash"),
            marker=dict(size=6, symbol="square"),
        ))

        fig_bt.update_layout(
            title=f"Holdout Period: Actual vs Predicted — {store_id} / {item_id}",
            xaxis_title="Date",
            yaxis_title="Sales (units)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            template="plotly_white",
        )

        st.plotly_chart(fig_bt, use_container_width=True)

    with col_left:
        st.markdown("#### Promo Lift Coefficient")
        promo_lift = bt["sarimax_promo"].get("promo_lift")
        if promo_lift is not None:
            st.metric("Units lifted per promo week", f"{promo_lift:.2f}")
        else:
            st.info("Promo lift: N/A")

        st.markdown("**DM test** p-values < 0.05 indicate a statistically significant difference in predictive accuracy vs SARIMA baseline.")


# ===========================================================================
# TAB 3 — DECISION
# ===========================================================================

with tab3:
    st.header("Inventory Decision Engine")

    from src.decision import compute_inventory_decision

    # Use SARIMA forecast (already computed above)
    # If SARIMA failed to compute, try to get it
    try:
        fc_sarima_decision, _, _ = get_sarima_forecast(store_id, item_id, horizon, n_train)
    except Exception:
        fc_sarima_decision = np.full(horizon, float(np.mean(all_sales[-12:])))

    decision = compute_inventory_decision(
        forecast=fc_sarima_decision,
        historical_series=all_sales,
        service_level=service_level,
        lead_time=lead_time,
    )

    # Large callout box
    sl_pct = int(round(service_level * 100))
    if decision["order_qty"] > 0:
        st.success(f"**Recommendation:** {decision['interpretation_text']}")
    else:
        st.info(
            f"**Current inventory appears sufficient.** "
            f"Estimated on-hand ({decision['current_inventory_estimate']:,.0f} units) "
            f"covers forecasted demand. No immediate order needed."
        )

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Recommended Order Qty",
            f"{max(int(round(decision['order_qty'])), 0):,} units",
            help="Units to order now for the forecast horizon",
        )
    with col2:
        st.metric(
            "Safety Stock",
            f"{int(round(decision['safety_stock'])):,} units",
            help="Buffer stock to absorb demand variability",
        )
    with col3:
        st.metric(
            "Reorder Point (ROP)",
            f"{int(round(decision['rop'])):,} units",
            help="Place a new order when inventory reaches this level",
        )
    with col4:
        st.metric(
            "Z-Score",
            f"{decision['z_score']:.3f}",
            help=f"Z-score for {sl_pct}% service level",
        )

    st.markdown("---")

    # Detailed breakdown
    st.markdown("#### Decision Parameters")
    col_a, col_b = st.columns(2)

    with col_a:
        params_df = pd.DataFrame({
            "Parameter": [
                "Service Level",
                "Lead Time",
                "Forecast Horizon",
                "Mean Weekly Demand (forecast)",
                "Estimated Current Inventory",
                "Total Forecast Demand",
            ],
            "Value": [
                f"{sl_pct}%",
                f"{lead_time} weeks",
                f"{horizon} weeks",
                f"{decision['mean_weekly_demand']:.1f} units/week",
                f"{decision['current_inventory_estimate']:,.1f} units",
                f"{decision['total_forecast_demand']:,.1f} units",
            ],
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Interpretation")
        st.markdown(decision["interpretation_text"])

        st.markdown("""
**How to read this:**
- **Safety stock** absorbs demand spikes and supplier delays.
- **ROP** is your trigger: when stock hits this level, place the order.
- **Order Qty** accounts for forecasted demand plus safety stock, minus current inventory.
- A higher service level increases safety stock but reduces stockout risk.
        """)


# ===========================================================================
# TAB 4 — DECOMPOSITION
# ===========================================================================

with tab4:
    st.title(f"Time Series Decomposition — {store_id} / {item_id}")

    with st.spinner("Running STL decomposition..."):
        try:
            stl_result = get_stl_decomposition_cached(store_id, item_id)
            from src.decomposition import seasonal_strength, trend_strength

            fs = seasonal_strength(stl_result.seasonal, stl_result.residual)
            ft = trend_strength(stl_result.trend, stl_result.residual)

            # STL 4-panel subplot
            dates_stl = stl_result.dates if stl_result.dates is not None else all_dates

            fig_stl = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
                vertical_spacing=0.06,
            )

            fig_stl.add_trace(go.Scatter(
                x=dates_stl, y=all_sales,
                mode="lines", name="Original",
                line=dict(color="#2c3e50", width=1.5),
            ), row=1, col=1)

            fig_stl.add_trace(go.Scatter(
                x=dates_stl, y=stl_result.trend,
                mode="lines", name="Trend",
                line=dict(color="#3498db", width=2),
            ), row=2, col=1)

            fig_stl.add_trace(go.Scatter(
                x=dates_stl, y=stl_result.seasonal,
                mode="lines", name="Seasonal",
                line=dict(color="#27ae60", width=1.5),
            ), row=3, col=1)

            fig_stl.add_trace(go.Scatter(
                x=dates_stl, y=stl_result.residual,
                mode="lines", name="Residual",
                line=dict(color="#e74c3c", width=1),
            ), row=4, col=1)

            fig_stl.update_layout(
                height=700,
                template="plotly_white",
                showlegend=False,
                title_text=f"STL Decomposition (robust) — {store_id} / {item_id}",
            )

            st.plotly_chart(fig_stl, use_container_width=True)

            # Strength metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Seasonal Strength (F_S)",
                    f"{fs * 100:.1f}%",
                    help="Hyndman & Athanasopoulos: F_S = max(0, 1 - Var(R)/Var(S+R)). Higher = stronger seasonality.",
                )
            with col2:
                st.metric(
                    "Trend Strength (F_T)",
                    f"{ft * 100:.1f}%",
                    help="F_T = max(0, 1 - Var(R)/Var(T+R)). Higher = stronger trend.",
                )

            st.markdown("---")
            st.markdown("#### Seasonal Heatmap")

            # Build seasonal heatmap: pivot week_of_year vs year
            series_df_full = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date").copy()
            iso = series_df_full["date"].dt.isocalendar()
            series_df_full["week"] = iso["week"].astype(int)
            series_df_full["year"] = iso["year"].astype(int)

            pivot = series_df_full.groupby(["year", "week"])["sales"].mean().reset_index()
            pivot_table = pivot.pivot(index="year", columns="week", values="sales")

            # Restrict to weeks 1-52 only
            valid_weeks = [w for w in range(1, 53) if w in pivot_table.columns]
            pivot_table = pivot_table[valid_weeks]

            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=[f"W{w}" for w in pivot_table.columns],
                y=[str(y) for y in pivot_table.index],
                colorscale="RdBu",
                colorbar=dict(title="Mean Sales"),
            ))
            fig_heat.update_layout(
                title=f"Seasonal Heatmap — Mean Sales by Week-of-Year and Year",
                xaxis_title="Week of Year",
                yaxis_title="Year",
                height=300,
                template="plotly_white",
            )

            st.plotly_chart(fig_heat, use_container_width=True)

        except Exception as e:
            st.error(f"STL decomposition failed: {e}")


# ===========================================================================
# TAB 5 — DIAGNOSTICS
# ===========================================================================

with tab5:
    st.title(f"Model Diagnostics — {store_id} / {item_id}")

    with st.spinner("Fitting SARIMA for residual diagnostics..."):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.stattools import acf
            from scipy.stats import shapiro, norm as sp_norm

            diag_model = SARIMAX(
                all_sales,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 0, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            diag_result = diag_model.fit(disp=False, maxiter=200)
            residuals = diag_result.resid
            n_resid = len(residuals)

            col_acf, col_hist = st.columns(2)

            # --- Left: ACF bar chart ---
            with col_acf:
                st.markdown("#### ACF of Residuals (lags 1–20)")
                acf_vals = acf(residuals, nlags=20, fft=True)
                lags = np.arange(1, 21)
                ci_bound = 1.96 / np.sqrt(n_resid)

                fig_acf = go.Figure()
                # Bars for ACF
                colors = ["#e74c3c" if abs(v) > ci_bound else "#3498db" for v in acf_vals[1:21]]
                for i, (lag, val, color) in enumerate(zip(lags, acf_vals[1:21], colors)):
                    fig_acf.add_trace(go.Bar(
                        x=[lag], y=[val],
                        marker_color=color,
                        showlegend=False,
                        name=f"lag {lag}",
                    ))

                # Reference lines
                fig_acf.add_hline(y=ci_bound, line_dash="dash", line_color="gray", opacity=0.7, annotation_text="+1.96/√n")
                fig_acf.add_hline(y=-ci_bound, line_dash="dash", line_color="gray", opacity=0.7, annotation_text="-1.96/√n")
                fig_acf.add_hline(y=0, line_color="black", line_width=1)

                fig_acf.update_layout(
                    xaxis_title="Lag",
                    yaxis_title="ACF",
                    height=350,
                    template="plotly_white",
                    bargap=0.1,
                )
                st.plotly_chart(fig_acf, use_container_width=True)

            # --- Right: Residual histogram with normal overlay ---
            with col_hist:
                st.markdown("#### Residual Distribution")
                resid_mean = float(np.mean(residuals))
                resid_std = float(np.std(residuals))

                x_range = np.linspace(resid_mean - 4 * resid_std, resid_mean + 4 * resid_std, 200)
                y_normal = sp_norm.pdf(x_range, resid_mean, resid_std)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals,
                    histnorm="probability density",
                    name="Residuals",
                    marker_color="#3498db",
                    opacity=0.7,
                    nbinsx=30,
                ))
                fig_hist.add_trace(go.Scatter(
                    x=x_range,
                    y=y_normal,
                    mode="lines",
                    name="Normal fit",
                    line=dict(color="#e74c3c", width=2),
                ))
                fig_hist.update_layout(
                    xaxis_title="Residual",
                    yaxis_title="Density",
                    height=350,
                    template="plotly_white",
                    legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")

            # --- Ljung-Box test table ---
            st.markdown("#### Ljung-Box Test (White Noise Check)")
            lb_result = acorr_ljungbox(residuals, lags=[5, 10, 20], return_df=True)
            lb_display = pd.DataFrame({
                "Lag": [5, 10, 20],
                "Q-statistic": lb_result["lb_stat"].values.round(3),
                "p-value": lb_result["lb_pvalue"].values.round(4),
            })
            st.dataframe(lb_display, use_container_width=True, hide_index=True)

            # --- Shapiro-Wilk normality test ---
            st.markdown("#### Shapiro-Wilk Normality Test")
            sw_stat, sw_p = shapiro(residuals[:min(len(residuals), 5000)])
            sw_df = pd.DataFrame({
                "Statistic": [round(sw_stat, 4)],
                "p-value": [round(sw_p, 4)],
            })
            st.dataframe(sw_df, use_container_width=True, hide_index=True)

            # --- Interpretation ---
            lb_p_lag10 = lb_result["lb_pvalue"].iloc[1]  # lag 10
            if lb_p_lag10 > 0.05:
                st.success("Residuals look white noise (Ljung-Box p > 0.05 at lag 10). SARIMA adequately captures serial structure.")
            else:
                st.warning(f"Residual autocorrelation detected (Ljung-Box p = {lb_p_lag10:.4f} at lag 10). Consider a more complex model or additional regressors.")

        except Exception as e:
            st.error(f"Diagnostics failed: {e}")


# ===========================================================================
# TAB 6 — PORTFOLIO
# ===========================================================================

with tab6:
    st.title("Portfolio Overview — All Stores & Items")

    with st.spinner("Computing portfolio analytics..."):
        try:
            abc_df = get_abc_classification()
            cv_df = get_demand_cv()

            col_abc, col_heat = st.columns(2)

            # --- Left: ABC classification table ---
            with col_abc:
                st.markdown("#### ABC Classification")
                abc_display = abc_df.copy()
                abc_display["total_sales"] = abc_display["total_sales"].apply(lambda x: f"{x:,.1f}")
                abc_display["cumulative_pct"] = abc_display["cumulative_pct"].apply(lambda x: f"{x:.1f}%")
                abc_display.columns = ["Store", "Item", "Total Sales", "Cumulative %", "Class"]
                st.dataframe(abc_display, use_container_width=True, hide_index=True)

            # --- Right: Demand heatmap ---
            with col_heat:
                st.markdown("#### Mean Weekly Sales Heatmap")
                heat_data = df.groupby(["store_id", "item_id"])["sales"].mean().reset_index()
                heat_pivot = heat_data.pivot(index="store_id", columns="item_id", values="sales")

                annotation_text = [[f"{heat_pivot.loc[s, i]:.1f}" for i in heat_pivot.columns] for s in heat_pivot.index]

                fig_demand_heat = go.Figure(data=go.Heatmap(
                    z=heat_pivot.values,
                    x=heat_pivot.columns.tolist(),
                    y=heat_pivot.index.tolist(),
                    colorscale="Viridis",
                    text=annotation_text,
                    texttemplate="%{text}",
                    colorbar=dict(title="Mean Weekly Sales"),
                ))
                fig_demand_heat.update_layout(
                    xaxis_title="Item",
                    yaxis_title="Store",
                    height=320,
                    template="plotly_white",
                )
                st.plotly_chart(fig_demand_heat, use_container_width=True)

            st.markdown("---")

            # --- CV table color-coded by demand class ---
            st.markdown("#### Demand Variability (Coefficient of Variation)")

            def color_demand_class(val):
                if val == "Smooth":
                    return "background-color: #d5e8d4; color: #1a5e20"
                elif val == "Erratic":
                    return "background-color: #fff3cd; color: #856404"
                else:
                    return "background-color: #f8d7da; color: #721c24"

            cv_display = cv_df.copy()
            cv_display.columns = ["Store", "Item", "Mean Sales", "Std Sales", "CV (%)", "Demand Class"]
            styled_cv = cv_display.style.applymap(color_demand_class, subset=["Demand Class"])
            st.dataframe(styled_cv, use_container_width=True, hide_index=True)

            st.markdown("---")

            # --- Rolling 8-week mean for all 15 series ---
            st.markdown("#### Rolling 8-Week Mean Sales — All Series")

            all_stores = df["store_id"].unique()
            all_items = df["item_id"].unique()
            colors_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            ]

            fig_rolling = go.Figure()
            color_idx = 0

            for sid in sorted(all_stores):
                for iid in sorted(all_items):
                    sub = df[(df["store_id"] == sid) & (df["item_id"] == iid)].sort_values("date")
                    if len(sub) < 8:
                        continue
                    rolling_mean = sub["sales"].rolling(8).mean().values
                    color = colors_palette[color_idx % len(colors_palette)]
                    fig_rolling.add_trace(go.Scatter(
                        x=sub["date"].values,
                        y=rolling_mean,
                        mode="lines",
                        name=f"{sid}/{iid}",
                        line=dict(color=color, width=1.5),
                        opacity=0.75,
                    ))
                    color_idx += 1

            fig_rolling.update_layout(
                title="Rolling 8-Week Mean Sales — All 15 Series",
                xaxis_title="Date",
                yaxis_title="Rolling Mean Sales",
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    x=1.01, y=1,
                    font=dict(size=10),
                ),
                height=500,
                template="plotly_white",
            )
            st.plotly_chart(fig_rolling, use_container_width=True)

        except Exception as e:
            st.error(f"Portfolio analytics failed: {e}")

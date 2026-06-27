"""
Stock Sense — Retail Demand Forecasting App
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
def get_backtest(store_id: str, item_id: str):
    """Cache backtest results by store/item."""
    from src.forecasting import run_backtest
    df = load_data()
    series = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")["sales"].values
    return run_backtest(series)


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
st.sidebar.caption("Model: SARIMA(1,1,1)(1,1,0,52) | ETS Auto")

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

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Benchmark", "🎯 Decision"])

# ===========================================================================
# TAB 1 — FORECAST
# ===========================================================================

with tab1:
    st.title(f"Demand Forecast — {store_id} / {item_id}")

    with st.spinner("Fitting SARIMA model..."):
        fc_sarima, lower_sarima, upper_sarima = get_sarima_forecast(store_id, item_id, horizon, n_train)

    with st.spinner("Fitting ETS model..."):
        fc_ets, lower_ets, upper_ets = get_ets_forecast(store_id, item_id, horizon, n_train)

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
    st.header("Backtest Comparison (last 12 weeks)")

    with st.spinner("Running backtest..."):
        bt = get_backtest(store_id, item_id)

    # Metrics table
    metrics_data = {
        "Model": ["SARIMA(1,1,1)(1,1,0,52)", "ETS (Auto)"],
        "MAE": [
            bt["sarima"]["MAE"] if bt["sarima"]["MAE"] is not None else "N/A",
            bt["ets"]["MAE"] if bt["ets"]["MAE"] is not None else "N/A",
        ],
        "MAPE (%)": [
            bt["sarima"]["MAPE"] if bt["sarima"]["MAPE"] is not None else "N/A",
            bt["ets"]["MAPE"] if bt["ets"]["MAPE"] is not None else "N/A",
        ],
    }
    metrics_df = pd.DataFrame(metrics_data)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Model Metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Highlight winner
        if bt["sarima"]["MAE"] is not None and bt["ets"]["MAE"] is not None:
            if bt["sarima"]["MAE"] < bt["ets"]["MAE"]:
                st.success("SARIMA wins on MAE for this series.")
            elif bt["ets"]["MAE"] < bt["sarima"]["MAE"]:
                st.success("ETS wins on MAE for this series.")
            else:
                st.info("Both models tie on MAE.")

    with col_right:
        # Backtest chart: actual vs predicted
        n_test = bt["n_test"]
        # Use last n_test dates from the series
        test_dates = all_dates[-n_test:]

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

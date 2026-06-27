# Stock Sense — Time-Series Forecasting as a Decision Tool

A retail demand forecasting application that benchmarks four time-series models, evaluates them with rigorous walk-forward cross-validation, and translates every forecast into a concrete, mathematically grounded inventory action.

The point of this project is not the model — it is the **decision**. A forecast that nobody acts on is useless. Stock Sense closes the loop: it produces probabilistic demand forecasts, exposes their uncertainty honestly via prediction intervals, evaluates those intervals with a proper scoring rule, and converts the result into safety-stock and reorder recommendations in plain business language.

---

## Screenshots

![Forecast chart with prediction interval](screenshots/01_forecast_chart.png)

![Benchmark backtest comparison](screenshots/02_benchmark_table.png)

![Decision callout with inventory metrics](screenshots/03_decision_callout.png)

---

## How It Works

### 1. Data

The application ships with a synthetic-but-realistic retail sales dataset (3 stores x 5 items, 156 weeks, 2021-01-04 to 2023-12-25). The generator (`data/generate_data.py`) superimposes four components on a log-normal base:

| Component | Implementation |
|---|---|
| Upward trend | Compound factor $(1 + 0.015)^{t/52}$ per year |
| Yearly seasonality | Von Mises kernel peaked in weeks 47-52 (holiday season), trough in weeks 1-8 |
| Promotional spikes | Bernoulli-sampled weeks (~19% rate) with a uniform multiplier $U[1.4, 1.8]$ |
| Noise | Lognormal multiplicative: $\varepsilon \sim \text{LogNormal}(0,\, 0.15)$ |

The dataset is stated as synthetic to avoid credential dependencies. It exhibits non-trivial seasonality, heteroskedasticity, trend, and causal promotional structure — the full set of features that differentiate the four forecasting approaches.

---

### 2. Forecasting Models

#### Model 1 — SARIMA

The workhorse statistical baseline, specified as $\text{SARIMA}(1,1,1)(1,1,0)_{52}$:

$$
(1 - \Phi_1 B^{52})(1 - \phi_1 B)(1 - B)(1 - B^{52})\, y_t = (1 + \theta_1 B)\, \varepsilon_t, \quad \varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)
$$

where $B$ is the backshift operator, $(1-B)$ is the non-seasonal difference operator, and $(1-B^{52})$ is the seasonal difference operator at period $s = 52$. Parameters are estimated by maximum likelihood via the Kalman filter (Durbin & Koopman 2002). Exact Gaussian prediction intervals follow from the state-space forecast covariance:

$$
\hat{y}_{t+h|t} \pm z_{\alpha/2} \cdot \sqrt{P_{t+h|t}}
$$

Fallback chain: $\text{ARIMA}(1,1,1)$ → naive mean $\pm 1.96\hat{\sigma}$.

#### Model 2 — AutoETS

An ETS (Error, Trend, Seasonality) state-space model selected by AIC minimization over all valid (error $\times$ trend $\times$ seasonal) combinations. The additive ETS(A,A,A) form (Hyndman et al. 2002):

$$
y_t = \ell_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t
$$

$$
\ell_t = \ell_{t-1} + b_{t-1} + \alpha\, \varepsilon_t, \qquad b_t = b_{t-1} + \beta\, \varepsilon_t, \qquad s_t = s_{t-m} + \gamma\, \varepsilon_t
$$

AutoETS (Nixtla `statsforecast`) selects the model minimizing $\text{AIC} = -2\ln\hat{L} + 2k$. On three annual cycles the optimizer typically selects ETS(A,N,N) — simple exponential smoothing — because 52 seasonal parameters are not identified with confidence. This is a correct statistical outcome, not a failure.

#### Model 3 — Theta Method

The winner of the M3 International Forecasting Competition (Assimakopoulos & Nikolopoulos 2000), later shown by Hyndman & Billah (2003) to be equivalent to SES with a linear drift component:

$$
\hat{y}_{T+h} = \hat{\ell}_T + \frac{b}{2} \cdot h
$$

where $\hat{\ell}_T$ is the SES level at the end of the training set (with optimized smoothing parameter $\alpha$) and $b$ is the OLS slope of the linear trend fitted to the full training series. The key insight: the Theta method gives the trend half the weight of a simple linear extrapolation, which is conservative and empirically well-calibrated on business time series. Prediction intervals are approximated as:

$$
\hat{y}_{T+h} \pm z_{0.025} \cdot \hat{\sigma}_\varepsilon \cdot \sqrt{h}
$$

where $\hat{\sigma}_\varepsilon$ is the SES in-sample residual standard deviation.

#### Model 4 — SARIMAX with Promotional Exogenous

The same SARIMA structure augmented with the `is_promo` indicator as a regression covariate:

$$
(1 - \Phi_1 B^{52})(1 - \phi_1 B)(1 - B)(1 - B^{52})\, y_t = \beta \cdot x_t + (1 + \theta_1 B)\, \varepsilon_t
$$

where $x_t \in \{0, 1\}$ is the promotional dummy and $\beta$ is estimated jointly with the ARMA parameters by MLE. The coefficient $\beta$ represents the average causal promotional lift in units per promotional week — a directly interpretable business quantity. Future promotional weeks in the forecast horizon are assumed zero (conservative: no planned promotions) unless explicitly provided.

#### Model 5 — Ensemble (Bates & Granger 1969)

A weighted combination of all four models using inverse-MAE weights from rolling-origin cross-validation:

$$
\hat{y}^{\text{ens}}_{T+h} = \sum_{m} w_m \cdot \hat{y}^{(m)}_{T+h}, \qquad w_m = \frac{1/\text{MAE}_m}{\sum_j 1/\text{MAE}_j}
$$

Bates & Granger showed that under general conditions, combining forecasts reduces expected squared error compared to the best individual model. The improvement is largest when models capture different features of the series, which is the case here: SARIMA captures linear dynamics, ETS captures level changes, Theta adds a conservative trend, and SARIMAX captures promotions.

---

### 3. Evaluation

#### Rolling-Origin Cross-Validation

Single-holdout evaluation is unreliable for short time series. The benchmark tab uses rolling-origin (walk-forward) cross-validation with $K = 3$ origins and $h = 8$ week test windows:

| Origin | Train window | Test window |
|---|---|---|
| 1 | weeks 1–132 | weeks 133–140 |
| 2 | weeks 1–140 | weeks 141–148 |
| 3 | weeks 1–148 | weeks 149–156 |

Error metrics are pooled across all $K \times h = 24$ forecast points.

#### Point Forecast Metrics

$$
\text{MAE} = \frac{1}{Kh} \sum_{k=1}^{K} \sum_{i=1}^{h} \left| y_{T_k+i} - \hat{y}_{T_k+i} \right|
$$

$$
\text{MAPE} = \frac{100}{Kh} \sum_{k=1}^{K} \sum_{i=1}^{h} \left| \frac{y_{T_k+i} - \hat{y}_{T_k+i}}{y_{T_k+i}} \right|
$$

#### Winkler Interval Score (Winkler 1972)

A proper scoring rule for prediction intervals — the only metric that simultaneously rewards narrow intervals and correct coverage. For a $(1-\alpha)$ interval $[L_t, U_t]$:

$$
W_t = (U_t - L_t) + \frac{2}{\alpha} \max(L_t - y_t,\, 0) + \frac{2}{\alpha} \max(y_t - U_t,\, 0)
$$

$\overline{W} = \frac{1}{n}\sum_t W_t$. Lower is better; a perfectly calibrated interval with zero coverage failures equals its own width. This penalizes overconfident intervals (too narrow, many violations) far more than conservative ones.

#### Diebold-Mariano Test (Harvey et al. 1997)

Statistical significance test for forecast accuracy differences. Let $d_t = e_{1,t}^2 - e_{2,t}^2$ be the MSE loss differential between two models. The Harvey-corrected test statistic is:

$$
\text{DM}^* = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \cdot \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}
$$

which follows a $t_{n-1}$ distribution under $H_0$ (equal predictive accuracy). The correction matters for small samples ($n = 24$ in our setting).

---

### 4. Decomposition

STL — Seasonal and Trend decomposition using LOESS (Cleveland et al. 1990) — separates a series into three additive components:

$$
y_t = T_t + S_t + R_t
$$

where $T_t$ is the LOESS-smoothed trend, $S_t$ is the periodic seasonal component (with period $m = 52$), and $R_t = y_t - T_t - S_t$ is the remainder. STL is fitted with `robust=True`, which downweights outliers (promotional spikes) in the LOESS smoother so they do not contaminate the seasonal estimate.

**Seasonal strength** (Hyndman & Athanasopoulos):

$$
F_S = \max\!\left(0,\; 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right) \in [0, 1]
$$

$F_S \approx 1$ means the seasonal component explains nearly all non-trend variation; $F_S \approx 0$ means the series is essentially a-seasonal.

**Trend strength:**

$$
F_T = \max\!\left(0,\; 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t + R_t)}\right) \in [0, 1]
$$

The seasonal heatmap (week-of-year $\times$ year) visualizes exactly which calendar weeks drive demand peaks, making the seasonal pattern directly actionable for procurement teams.

---

### 5. Residual Diagnostics

Model adequacy is checked on SARIMA in-sample residuals via:

**Ljung-Box portmanteau test** at lags 5, 10, 20:

$$
Q(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n - k} \overset{H_0}{\sim} \chi^2(m - p - q)
$$

Rejection ($p < 0.05$) indicates unexplained autocorrelation — the model has missed structure.

**ACF of residuals** with $\pm 1.96/\sqrt{n}$ significance bounds. Bars exceeding the bounds at specific lags indicate seasonal or lag-specific patterns the model has not captured.

**Shapiro-Wilk normality test** on standardized residuals. Non-normality invalidates the Gaussian prediction intervals; in practice, retail demand residuals are often right-skewed due to promotional outliers.

---

### 6. Portfolio Analytics

**ABC classification** segments the 15 store-item combinations by cumulative revenue contribution, following standard inventory management practice:

- **A items**: top 80% of total sales volume — highest priority, tightest safety stock
- **B items**: next 15% — moderate attention
- **C items**: bottom 5% — simple reorder rules sufficient

**Demand classification by coefficient of variation:**

$$
\text{CV} = \frac{\sigma_d}{\bar{d}} \times 100
$$

| Class | CV range | Implication |
|---|---|---|
| Smooth | CV < 20% | Standard forecasting methods apply |
| Erratic | 20% ≤ CV ≤ 50% | Wider safety stock buffers needed |
| Lumpy | CV > 50% | Consider intermittent demand models (Croston, TSB) |

---

### 7. Decision Layer

Given a point forecast $\{\hat{y}_{T+1}, \ldots, \hat{y}_{T+H}\}$ from the selected model, a target service level $\alpha$, and lead time $L$:

$$
z_\alpha = \Phi^{-1}(\alpha) \quad \text{(via \texttt{scipy.stats.norm.ppf})}
$$

$$
\text{SS} = z_\alpha \cdot \hat{\sigma}_d \cdot \sqrt{L}
$$

$$
\text{ROP} = \bar{d} \cdot L + \text{SS}
$$

$$
Q = \max\!\left(0,\; \sum_{i=1}^{H} \hat{y}_{T+i} + \text{SS} - I_0\right)
$$

where $\hat{\sigma}_d$ is estimated from trailing 52-week historical demand (not forecast variance, which may be zero for flat ETS models), $\bar{d}$ is mean forecast weekly demand, and $I_0 \approx 2\bar{d}_{\text{hist}}$ is the estimated on-hand inventory. All four quantities update in real time as the service-level and lead-time sliders move.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         app.py  (Streamlit)                          │
│                                                                      │
│  Sidebar: store / item / horizon / service_level / lead_time         │
│                                                                      │
│  Tab 1: Forecast        Tab 2: Benchmark      Tab 3: Decision        │
│  ├─ SARIMA              ├─ Rolling-origin CV  ├─ SS / ROP / Q        │
│  ├─ ETS                 ├─ MAE / MAPE         ├─ Z-score             │
│  ├─ Theta               ├─ Winkler score      └─ Interpretation      │
│  ├─ SARIMAX+promo       ├─ DM test p-value                           │
│  ├─ Ensemble            └─ Actual-vs-predicted                       │
│  └─ Plotly: PI shading                                               │
│                                                                      │
│  Tab 4: Decomposition   Tab 5: Diagnostics    Tab 6: Portfolio        │
│  ├─ STL 4-panel plot    ├─ Residual ACF       ├─ ABC classification  │
│  ├─ Seasonal heatmap    ├─ Ljung-Box table    ├─ Demand heatmap      │
│  ├─ F_S strength        ├─ Shapiro-Wilk       ├─ CV classification   │
│  └─ F_T strength        └─ Error histogram    └─ Rolling 8-wk chart  │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
          ┌─────────────────────────┼───────────────────────────┐
          ▼                         ▼                           ▼
┌──────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│ src/forecasting  │   │    src/models.py      │   │ src/evaluation.py  │
│                  │   │                       │   │                    │
│ fit_sarima()     │   │ theta_forecast()      │   │ rolling_origin_cv()│
│ fit_ets()        │   │ sarimax_promo_        │   │ winkler_score()    │
│ run_backtest()   │   │   forecast()          │   │ diebold_mariano()  │
│                  │   │ ensemble_forecast()   │   └────────────────────┘
└──────────────────┘   │ compute_ensemble_     │
                       │   weights()           │   ┌────────────────────┐
                       └──────────────────────┘   │ src/decomposition  │
                                                   │                    │
                       ┌──────────────────────┐   │ get_stl_           │
                       │   src/decision.py    │   │   decomposition()  │
                       │                      │   │ seasonal_strength()│
                       │ SS = z·σ·sqrt(L)     │   │ trend_strength()   │
                       │ ROP = d·L + SS       │   │ abc_classification()
                       │ Q = Σfc + SS - I0    │   │ demand_cv()        │
                       └──────────────────────┘   └────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────┐
                       │ data/sales_data.csv  │ <── data/generate_data.py
                       │ 2,340 rows           │     (synthetic, seed=42)
                       │ 3 stores x 5 items   │
                       │ 156 weeks            │
                       └──────────────────────┘
```

### Module responsibility

| Module | Responsibility |
|---|---|
| `src/forecasting.py` | SARIMA, ETS, original backtest integration |
| `src/models.py` | Theta, SARIMAX+promo, ensemble weighting (Bates & Granger) |
| `src/evaluation.py` | Rolling-origin CV, Winkler score, Diebold-Mariano test |
| `src/decomposition.py` | STL, $F_S$/$F_T$ metrics, ABC classification, demand CV |
| `src/decision.py` | Safety stock, ROP, order quantity (pure closed-form math) |

### Caching strategy

All forecasting and decomposition calls are wrapped with `@st.cache_data` keyed on `(store_id, item_id, horizon, n_train)`. SARIMA and SARIMAX fits take ~5-10 seconds each; without caching every sidebar interaction re-triggers MLE. The decision layer and Winkler computation are not cached — both are millisecond-scale pure arithmetic and should re-execute on every slider drag to give live feedback.

### Fallback chains

```
SARIMA(1,1,1)(1,1,0,52)     -->  ARIMA(1,1,1)              -->  Naive mean
SARIMAX(1,1,1)(1,1,0,52)    -->  SARIMA (no exog)           -->  Naive mean
AutoETS(season_length=52)    -->  HoltWinters(add,add)       -->  Naive mean
Theta                        -->  SES level only (slope=0)   -->  Naive mean
```

Every model has a three-level fallback. The app never crashes on any store-item combination.

---

## Run It

```bash
git clone https://github.com/BillKladis/Time-Series-Forecasting-as-a-Decision-Tool.git
cd Time-Series-Forecasting-as-a-Decision-Tool
pip install -r requirements.txt
# Copy .env and add your key (not required for forecasting — reserved for future LLM integration)
cp .env .env.local
streamlit run app.py
```

The dataset is bundled at `data/sales_data.csv`. To regenerate it:

```bash
python data/generate_data.py
```

To re-capture screenshots after changes:

```bash
python capture_screenshots.py
```

---

## Key References

- Assimakopoulos, V. & Nikolopoulos, K. (2000). The Theta model: a decomposition approach to forecasting. *International Journal of Forecasting*, 16(4), 521-530.
- Bates, J.M. & Granger, C.W.J. (1969). The combination of forecasts. *Operations Research Quarterly*, 20(4), 451-468.
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on LOESS. *Journal of Official Statistics*, 6(1), 3-73.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Durbin, J. & Koopman, S.J. (2002). A simple and efficient simulation smoother for state space time series analysis. *Biometrika*, 89(3), 603-616.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
- Hyndman, R.J. & Billah, B. (2003). Unmasking the Theta method. *International Journal of Forecasting*, 19(2), 287-290.
- Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting*, 18(3), 439-454.
- Winkler, R.L. (1972). A decision theoretic approach to interval estimation. *Journal of the American Statistical Association*, 67(337), 187-191.

---

## Design Choices and Limitations

**Synthetic data.** The dataset was generated rather than sourced from Kaggle or a real retailer to keep the project runnable without authentication dependencies. The generator reproduces the statistical structure of real retail series — multiplicative trend, yearly seasonality, promotional spikes, heteroskedastic noise — closely enough to produce meaningful model comparisons. Absolute numbers are not calibrated to any real category or geography.

**AutoETS collapses to SES on short series.** With only three complete annual cycles available, AutoETS consistently selects ETS(A,N,N) because 52 seasonal parameters cannot be identified from 156 observations with statistical confidence. This is the correct AIC-optimal choice given the data, not a bug. The decision layer compensates by using trailing historical variance rather than forecast variance for safety-stock calculation. A production deployment would retrain on five or more years of history.

**Rolling-origin CV uses three origins.** Three origins is the minimum for a meaningful CV estimate given the series length (156 weeks) and test horizon (8 weeks). More origins would require shortening the test window or accepting very short training sets; neither is desirable. The DM test on 24 pooled observations has limited power (wide confidence intervals), so p-values should be interpreted directionally rather than as sharp hypothesis tests.

**Inventory assumptions.** The current-inventory estimate $I_0 = 2\bar{d}_{\text{hist}}$ is a placeholder for a real ERP feed. The safety-stock formula assumes i.i.d. demand across weeks within the lead time, ignoring the autocorrelation visible in the residual ACF. A more rigorous treatment would propagate the SARIMA forecast error covariance through the lead-time window, which increases safety stock for positively autocorrelated demand. The Diagnostics tab exposes whether this assumption is materially violated for a given series.

**Promotional lift is average, not dynamic.** The SARIMAX promotional coefficient $\hat{\beta}$ is a single scalar estimated across all weeks in the training set. It does not distinguish between different promotion types, depths, or channels. In a production setting, promotional features would be richer (discount depth, display flag, ad spend) and might warrant a separate promotional response model.

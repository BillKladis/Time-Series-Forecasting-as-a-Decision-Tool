"""
Generate synthetic retail sales dataset for Stock Sense.
3 stores x 5 items = 15 combinations
3 years of weekly data (156 weeks, 2021-01-04 to 2023-12-25)
"""

import numpy as np
import pandas as pd
import os

def generate_sales_data(seed=42):
    rng = np.random.default_rng(seed)

    stores = ["Store_1", "Store_2", "Store_3"]
    items = ["Item_1", "Item_2", "Item_3", "Item_4", "Item_5"]

    # Weekly dates: 156 weeks starting 2021-01-04
    dates = pd.date_range(start="2021-01-04", periods=156, freq="W-MON")

    # Seasonality: week-of-year pattern (peak Dec/Nov, slow Jan-Feb)
    week_of_year = dates.isocalendar().week.values.astype(float)
    # Normalize to [0, 2*pi], peak around week 48 (Dec)
    # Use cosine shifted so peak is at week 48
    peak_week = 48
    seasonality = 1.0 + 0.35 * np.cos(2 * np.pi * (week_of_year - peak_week) / 52)

    # Trend: +1.5% per month compounding
    # 156 weeks / 52 weeks per year = 3 years = 36 months
    week_index = np.arange(156)
    monthly_rate = 0.015
    trend = (1 + monthly_rate) ** (week_index / 4.33)  # weeks to months

    records = []

    for store in stores:
        for item in items:
            # Base demand varies per store-item combination
            base_demand = rng.uniform(80, 300)

            # Store-level scaling factor
            store_scale = {"Store_1": 1.0, "Store_2": 0.8, "Store_3": 1.2}[store]
            # Item-level scaling factor
            item_scale = {"Item_1": 1.0, "Item_2": 0.6, "Item_3": 1.5, "Item_4": 0.9, "Item_5": 1.3}[item]

            effective_base = base_demand * store_scale * item_scale

            # Generate promotional weeks: 8-12 per year, 3 years = ~30 total
            promo_weeks = set()
            for year_offset in range(3):
                n_promos = rng.integers(8, 13)  # 8 to 12 inclusive
                year_start = year_offset * 52
                year_end = year_start + 52
                week_indices_in_year = rng.choice(range(year_start, min(year_end, 156)), size=n_promos, replace=False)
                promo_weeks.update(week_indices_in_year.tolist())

            is_promo = np.zeros(156, dtype=int)
            for w in promo_weeks:
                if w < 156:
                    is_promo[w] = 1

            # Promotional multiplier: 1.4 to 1.8
            promo_multipliers = rng.uniform(1.4, 1.8, size=156)

            # Build sales series
            sales = np.zeros(156)
            for w in range(156):
                mu = effective_base * seasonality[w] * trend[w]
                if is_promo[w]:
                    mu *= promo_multipliers[w]

                # Lognormal noise: cv ~ 0.15
                # For lognormal: sigma_log = sqrt(log(1 + cv^2))
                cv = 0.15
                sigma_log = np.sqrt(np.log(1 + cv**2))
                mu_log = np.log(mu) - 0.5 * sigma_log**2
                sales[w] = rng.lognormal(mean=mu_log, sigma=sigma_log)

            sales = np.round(sales, 1)

            for w in range(156):
                records.append({
                    "date": dates[w].strftime("%Y-%m-%d"),
                    "store_id": store,
                    "item_id": item,
                    "sales": sales[w],
                    "is_promo": is_promo[w],
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_sales_data(seed=42)
    output_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")
    print(df.head(10))
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store_id'].unique().tolist()}")
    print(f"Items: {df['item_id'].unique().tolist()}")
    print(f"Promo weeks: {df['is_promo'].sum()} out of {len(df)}")

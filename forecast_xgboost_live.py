import json
import numpy as np
import pandas as pd

from benchmark_common import HORIZONS, load_market_data
from run_model import load_xgboost, forecast_xgb

market, meta = load_market_data(refresh=True)
runtime = load_xgboost()

origin = len(market)
current = float(market.price_toman.iloc[-1])
data_date = str(market.date.iloc[-1].date())

rows = []

for h, label in HORIZONS.items():
    q10_log, q50_log, q90_log = forecast_xgb(market, origin, h)

    vals = sorted([
        float(np.exp(q10_log)),
        float(np.exp(q50_log)),
        float(np.exp(q90_log)),
    ])

    q10, q50, q90 = vals

    rows.append({
        "horizon": h,
        "label": label,
        "q10_toman": q10,
        "q50_toman": q50,
        "q90_toman": q90,
        "current_toman": current,
        "q10_return_pct": (q10/current - 1) * 100,
        "q50_return_pct": (q50/current - 1) * 100,
        "q90_return_pct": (q90/current - 1) * 100,
    })

df = pd.DataFrame(rows)
df.to_csv("latest_xgboost_forecast.csv", index=False)

print()
print("===== GOLD18 XGBOOST-QUANTILE =====")
print("Data date:", data_date)
print("Current Gold18:", f"{current:,.0f}", "toman")
print("Inputs: Gold18 + USD/IRR + XAU/USD")
print()

for _, r in df.iterrows():
    print(
        f"{r['label']}: "
        f"Q10={r['q10_toman']:,.0f} | "
        f"Q50={r['q50_toman']:,.0f} | "
        f"Q90={r['q90_toman']:,.0f} toman"
    )
    print(
        f"     Returns: "
        f"{r['q10_return_pct']:+.1f}% | "
        f"{r['q50_return_pct']:+.1f}% | "
        f"{r['q90_return_pct']:+.1f}%"
    )

print()
print("Saved: latest_xgboost_forecast.csv")

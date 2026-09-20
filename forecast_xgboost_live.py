from pathlib import Path
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
        "data_date": data_date,
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

# ---------- Mobile HTML report ----------
r3, r6, r12 = [r for _, r in df.iterrows()]

all_prices = [current]
for _, r in df.iterrows():
    all_prices += [r["q10_toman"], r["q50_toman"], r["q90_toman"]]

vmin = min(all_prices) * 0.92
vmax = max(all_prices) * 1.05

def xpos(v):
    return 8 + 84 * (float(v) - vmin) / (vmax - vmin)

def money(v):
    return f"{float(v)/1_000_000:.2f} M"

bars = []
for _, r in df.iterrows():
    x1 = xpos(r["q10_toman"])
    xm = xpos(r["q50_toman"])
    x2 = xpos(r["q90_toman"])
    xc = xpos(current)

    bars.append(f"""
    <div class="range-row">
      <div class="range-title">{r['label']}</div>
      <svg viewBox="0 0 100 24" preserveAspectRatio="none">
        <line x1="{x1:.2f}" y1="12" x2="{x2:.2f}" y2="12"
              stroke="#777" stroke-width="3"/>
        <circle cx="{x1:.2f}" cy="12" r="2.3" fill="#777"/>
        <circle cx="{xm:.2f}" cy="12" r="3.2" fill="#111"/>
        <circle cx="{x2:.2f}" cy="12" r="2.3" fill="#777"/>
        <line x1="{xc:.2f}" y1="3" x2="{xc:.2f}" y2="21"
              stroke="#999" stroke-width="0.8" stroke-dasharray="2,2"/>
      </svg>
      <div class="range-labels">
        <span>Q10 {money(r['q10_toman'])}</span>
        <strong>Q50 {money(r['q50_toman'])}</strong>
        <span>Q90 {money(r['q90_toman'])}</span>
      </div>
    </div>
    """)

cards = []
for _, r in df.iterrows():
    cards.append(f"""
    <section class="card">
      <h2>{r['label']}</h2>
      <div class="median">{r['q50_toman']:,.0f} تومان</div>
      <div class="return">Median return: {r['q50_return_pct']:+.1f}%</div>
      <div class="quantiles">
        <div><small>Q10</small><b>{r['q10_toman']:,.0f}</b><span>{r['q10_return_pct']:+.1f}%</span></div>
        <div><small>Q50</small><b>{r['q50_toman']:,.0f}</b><span>{r['q50_return_pct']:+.1f}%</span></div>
        <div><small>Q90</small><b>{r['q90_toman']:,.0f}</b><span>{r['q90_return_pct']:+.1f}%</span></div>
      </div>
    </section>
    """)

html = f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold18 Forecast</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f4f4f4;color:#171717;font-family:Arial,Tahoma,sans-serif}}
.wrap{{max-width:760px;margin:auto;padding:16px}}
.hero{{background:#111;color:white;border-radius:20px;padding:22px;margin-bottom:14px}}
.hero small{{opacity:.65}}
.hero h1{{font-size:21px;margin:8px 0 18px}}
.price{{font-size:34px;font-weight:800;direction:ltr;text-align:right}}
.meta{{margin-top:8px;font-size:13px;opacity:.72}}
.grid{{display:grid;gap:12px}}
.card{{background:white;border-radius:18px;padding:18px;box-shadow:0 2px 10px #0000000a}}
.card h2{{margin:0 0 8px;font-size:17px}}
.median{{font-size:27px;font-weight:800}}
.return{{font-size:14px;margin:5px 0 16px;color:#555}}
.quantiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;direction:ltr}}
.quantiles div{{background:#f5f5f5;border-radius:12px;padding:10px;text-align:center}}
.quantiles small,.quantiles span{{display:block;font-size:11px;color:#777}}
.quantiles b{{display:block;font-size:12px;margin:5px 0}}
.chart{{background:white;border-radius:18px;padding:18px;margin-top:12px}}
.chart h2{{margin-top:0;font-size:17px}}
.range-row{{margin:20px 0;direction:ltr}}
.range-title{{font-weight:700;margin-bottom:4px}}
.range-row svg{{width:100%;height:42px;overflow:visible}}
.range-labels{{display:flex;justify-content:space-between;font-size:11px;color:#666}}
.range-labels strong{{color:#111}}
.note{{font-size:12px;line-height:1.8;color:#666;margin-top:16px}}
.footer{{font-size:11px;color:#777;text-align:center;padding:20px}}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <small>XGBoost Quantile · Gold18 + USD/IRR + XAU/USD</small>
    <h1>پیش‌بینی طلای ۱۸ عیار</h1>
    <div class="price">{current:,.0f} تومان</div>
    <div class="meta">تاریخ آخرین داده: {data_date}</div>
  </header>

  <div class="grid">
    {''.join(cards)}
  </div>

  <section class="chart">
    <h2>بازه پیش‌بینی Q10–Q90</h2>
    {''.join(bars)}
    <div class="note">
      خط نقطه‌چین = قیمت فعلی. Q50 پیش‌بینی میانه مدل است.
      Q10 و Q90 مرزهای صدکی مدل هستند و تضمین قیمت آینده محسوب نمی‌شوند.
    </div>
  </section>

  <div class="footer">
    Model: XGBoost-Quantile · Inputs: Gold18 / USD-IRR / XAU-USD
  </div>
</div>
</body>
</html>
"""

Path("gold18_forecast_report.html").write_text(html, encoding="utf-8")
print("Saved: gold18_forecast_report.html")


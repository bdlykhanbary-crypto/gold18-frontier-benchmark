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
fa_labels = {
    "3M": "۳ ماه آینده",
    "6M": "۶ ماه آینده",
    "12M": "۱۲ ماه آینده",
}

all_prices = [current]
for _, r in df.iterrows():
    all_prices += [r["q10_toman"], r["q50_toman"], r["q90_toman"]]

vmin = min(all_prices) * 0.92
vmax = max(all_prices) * 1.05

def xpos(v):
    return 8 + 84 * (float(v) - vmin) / (vmax - vmin)

def money(v):
    return f"{float(v)/1_000_000:.2f} میلیون"

def pct(v):
    return f"{float(v):+.1f}٪"

cards = []
bars = []

for _, r in df.iterrows():
    label = fa_labels.get(r["label"], r["label"])

    cards.append(f"""
    <section class="card">
      <div class="period">{label}</div>

      <div class="main-label">پیش‌بینی اصلی مدل</div>
      <div class="main-price">{r['q50_toman']:,.0f} تومان</div>

      <div class="change">
        یعنی حدود <b>{pct(r['q50_return_pct'])}</b> نسبت به قیمت امروز
      </div>

      <div class="three">
        <div class="scenario low">
          <span>برآورد پایین</span>
          <strong>{r['q10_toman']:,.0f}</strong>
          <small>تومان</small>
          <em>{pct(r['q10_return_pct'])} نسبت به امروز</em>
        </div>

        <div class="scenario main">
          <span>پیش‌بینی اصلی</span>
          <strong>{r['q50_toman']:,.0f}</strong>
          <small>تومان</small>
          <em>{pct(r['q50_return_pct'])} نسبت به امروز</em>
        </div>

        <div class="scenario high">
          <span>برآورد بالا</span>
          <strong>{r['q90_toman']:,.0f}</strong>
          <small>تومان</small>
          <em>{pct(r['q90_return_pct'])} نسبت به امروز</em>
        </div>
      </div>

      <div class="plain-explain">
        <b>به زبان ساده:</b>
        مدل برای {label} عدد
        <strong>{r['q50_toman']/1_000_000:.2f} میلیون تومان</strong>
        را برآورد اصلی خود می‌داند.
        اما اگر شرایط ضعیف‌تر یا قوی‌تر از انتظار مدل باشد،
        قیمت می‌تواند به سمت اعداد پایین‌تر یا بالاتر حرکت کند.
      </div>
    </section>
    """)

    x1 = xpos(r["q10_toman"])
    xm = xpos(r["q50_toman"])
    x2 = xpos(r["q90_toman"])
    xc = xpos(current)

    bars.append(f"""
    <div class="range-row">
      <div class="range-title">{label}</div>
      <svg viewBox="0 0 100 25" preserveAspectRatio="none">
        <line x1="{x1:.2f}" y1="12" x2="{x2:.2f}" y2="12"
              stroke="#888" stroke-width="3"/>
        <circle cx="{x1:.2f}" cy="12" r="2.2" fill="#888"/>
        <circle cx="{xm:.2f}" cy="12" r="3.5" fill="#111"/>
        <circle cx="{x2:.2f}" cy="12" r="2.2" fill="#888"/>
        <line x1="{xc:.2f}" y1="2" x2="{xc:.2f}" y2="22"
              stroke="#999" stroke-width="0.8" stroke-dasharray="2,2"/>
      </svg>

      <div class="range-labels">
        <span>پایین<br><b>{money(r['q10_toman'])}</b></span>
        <span class="center">اصلی<br><b>{money(r['q50_toman'])}</b></span>
        <span>بالا<br><b>{money(r['q90_toman'])}</b></span>
      </div>
    </div>
    """)

direction_text = "در هر سه بازه زمانی، پیش‌بینی اصلی مدل بالاتر از قیمت امروز است."

html = f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>پیش‌بینی طلای ۱۸ عیار</title>

<style>
*{{box-sizing:border-box}}
body{{
  margin:0;
  background:#f3f3f3;
  color:#171717;
  font-family:Tahoma,Arial,sans-serif;
  line-height:1.75;
}}
.wrap{{max-width:760px;margin:auto;padding:14px}}

.hero{{
  background:#111;
  color:white;
  border-radius:22px;
  padding:22px;
  margin-bottom:14px;
}}
.hero .small{{font-size:12px;opacity:.7}}
.hero h1{{font-size:22px;margin:5px 0 18px}}
.current-label{{font-size:13px;opacity:.72}}
.price{{font-size:34px;font-weight:900;direction:ltr;text-align:right}}
.date{{font-size:12px;opacity:.65;margin-top:6px}}

.summary{{
  background:#fff;
  border-radius:18px;
  padding:18px;
  margin-bottom:13px;
}}
.summary h2{{font-size:18px;margin:0 0 10px}}
.summary p{{margin:6px 0;font-size:14px}}
.summary .direction{{
  background:#f1f1f1;
  border-radius:12px;
  padding:11px;
  margin-top:10px;
  font-weight:700;
}}

.help{{
  background:#fff8df;
  border-radius:18px;
  padding:17px;
  margin-bottom:14px;
}}
.help h2{{font-size:17px;margin:0 0 10px}}
.help p{{font-size:13px;margin:7px 0}}

.card{{
  background:white;
  border-radius:20px;
  padding:18px;
  margin-bottom:13px;
  box-shadow:0 2px 10px #0000000a;
}}
.period{{font-size:18px;font-weight:800}}
.main-label{{font-size:13px;color:#777;margin-top:10px}}
.main-price{{font-size:28px;font-weight:900}}
.change{{font-size:13px;color:#555;margin:5px 0 16px}}

.three{{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:7px;
  direction:rtl;
}}
.scenario{{
  background:#f5f5f5;
  border-radius:13px;
  padding:10px 5px;
  text-align:center;
}}
.scenario span{{display:block;font-size:11px;color:#666}}
.scenario strong{{display:block;font-size:12px;margin-top:5px;direction:ltr}}
.scenario small{{font-size:10px;color:#888}}
.scenario em{{display:block;font-size:10px;color:#777;font-style:normal;margin-top:5px}}
.scenario.main{{background:#eaeaea}}

.plain-explain{{
  margin-top:15px;
  background:#f7f7f7;
  padding:12px;
  border-radius:12px;
  font-size:13px;
}}

.chart{{
  background:white;
  border-radius:20px;
  padding:18px;
  margin-top:14px;
}}
.chart h2{{font-size:18px;margin:0 0 5px}}
.chart-intro{{font-size:12px;color:#666;margin-bottom:15px}}

.range-row{{margin:22px 0;direction:ltr}}
.range-title{{font-weight:800;direction:rtl;text-align:right}}
.range-row svg{{width:100%;height:42px;overflow:visible}}
.range-labels{{
  display:flex;
  justify-content:space-between;
  text-align:center;
  font-size:11px;
  color:#777;
  direction:rtl;
}}
.range-labels b{{color:#111}}
.range-labels .center{{font-weight:800}}

.warning{{
  background:#fff;
  border-radius:20px;
  padding:18px;
  margin-top:14px;
}}
.warning h2{{font-size:17px;margin:0 0 10px}}
.warning p{{font-size:13px;margin:7px 0}}
.warning .important{{
  background:#f2f2f2;
  border-radius:12px;
  padding:11px;
  font-weight:700;
}}

details{{
  background:#fff;
  border-radius:16px;
  padding:14px;
  margin-top:14px;
  font-size:12px;
}}
summary{{font-weight:700;cursor:pointer}}

.footer{{
  font-size:10px;
  color:#888;
  text-align:center;
  padding:22px 5px;
}}

@media(max-width:430px){{
  .price{{font-size:30px}}
  .scenario strong{{font-size:10px}}
  .three{{gap:5px}}
}}
</style>
</head>

<body>
<div class="wrap">

  <header class="hero">
    <div class="small">پیش‌بینی آماری قیمت طلای ۱۸ عیار</div>
    <h1>وضعیت احتمالی قیمت طلا در ماه‌های آینده</h1>

    <div class="current-label">قیمت فعلی مورد استفاده مدل</div>
    <div class="price">{current:,.0f} تومان</div>

    <div class="date">
      آخرین روز داده‌ای که مدل دیده است: {data_date}
    </div>
  </header>


  <section class="summary">
    <h2>خلاصه خیلی ساده</h2>

    <p>
      این برنامه قیمت طلای ۱۸ عیار، قیمت دلار و قیمت جهانی طلا را بررسی کرده
      و بر اساس رفتار گذشته بازار، برای آینده چند برآورد ارائه کرده است.
    </p>

    <div class="direction">
      {direction_text}
    </div>

    <p>
      عددی که با عنوان <b>«پیش‌بینی اصلی»</b> می‌بینی،
      مهم‌ترین عددی است که مدل برای آن بازه زمانی برآورد کرده است.
    </p>
  </section>


  <section class="help">
    <h2>این سه عدد یعنی چه؟</h2>

    <p>
      <b>برآورد پایین:</b>
      اگر بازار ضعیف‌تر از انتظار مدل حرکت کند، قیمت ممکن است به این سمت برود.
    </p>

    <p>
      <b>پیش‌بینی اصلی:</b>
      عدد مرکزی و مهم‌ترین پیش‌بینی مدل است.
    </p>

    <p>
      <b>برآورد بالا:</b>
      اگر بازار قوی‌تر از انتظار مدل حرکت کند، قیمت ممکن است به این سمت برود.
    </p>

    <p>
      این سه عدد به این معنی نیست که قیمت حتماً بین عدد پایین و بالا باقی می‌ماند.
    </p>
  </section>


  {''.join(cards)}


  <section class="chart">
    <h2>تصویر ساده پیش‌بینی‌ها</h2>

    <div class="chart-intro">
      نقطه سیاه = پیش‌بینی اصلی مدل<br>
      دو سر خط = برآورد پایین و بالای مدل<br>
      خط‌چین = قیمت امروز
    </div>

    {''.join(bars)}
  </section>


  <section class="warning">
    <h2>نکته مهم قبل از استفاده</h2>

    <p>
      این برنامه آینده را نمی‌داند؛ فقط از اطلاعات گذشته و شرایط فعلی بازار
      برای تخمین آینده استفاده می‌کند.
    </p>

    <p>
      ممکن است اتفاق‌هایی مثل تغییر شدید دلار، جنگ، تصمیم‌های سیاسی،
      تغییر قیمت جهانی طلا یا شوک‌های اقتصادی باعث شوند قیمت واقعی
      با این پیش‌بینی‌ها تفاوت زیادی داشته باشد.
    </p>

    <div class="important">
      بنابراین «پیش‌بینی اصلی» هدف قطعی قیمت نیست و
      «برآورد پایین» هم کف تضمینی قیمت نیست.
    </div>

    <p>
      این گزارش فقط برای فهم بهتر شرایط احتمالی آینده است و
      به‌تنهایی دستور خرید یا فروش محسوب نمی‌شود.
    </p>
  </section>


  <details>
    <summary>جزئیات فنی برای کسانی که می‌خواهند بدانند</summary>
    <p>
      مدل مورد استفاده XGBoost Quantile است.
      ورودی‌های آن قیمت طلای ۱۸ عیار ایران، دلار آزاد و اونس جهانی طلا هستند.
    </p>
    <p>
      در اصطلاح فنی، برآورد پایین Q10، پیش‌بینی اصلی Q50
      و برآورد بالا Q90 است.
    </p>
  </details>


  <div class="footer">
    مدل: XGBoost Quantile · داده‌ها: طلای ۱۸ عیار + دلار + اونس جهانی
  </div>

</div>
</body>
</html>
"""

Path("gold18_forecast_report.html").write_text(html, encoding="utf-8")
print("Saved: gold18_forecast_report.html")

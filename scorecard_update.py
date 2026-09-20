from pathlib import Path
import html
import numpy as np
import pandas as pd

from benchmark_common import load_market_data

ROOT = Path(".")
LATEST = ROOT / "latest_xgboost_forecast.csv"
HISTORY = ROOT / "forecast_history.csv"
EVALS = ROOT / "forecast_evaluations.csv"
REPORT = ROOT / "gold18_forecast_report.html"

if not LATEST.exists():
    raise SystemExit("latest_xgboost_forecast.csv پیدا نشد.")
if not REPORT.exists():
    raise SystemExit("gold18_forecast_report.html پیدا نشد.")

latest = pd.read_csv(LATEST)
required = {
    "data_date", "horizon", "label", "q10_toman", "q50_toman", "q90_toman",
    "current_toman", "q10_return_pct", "q50_return_pct", "q90_return_pct"
}
missing = required - set(latest.columns)
if missing:
    raise SystemExit(f"ستون‌های لازم در فایل پیش‌بینی نیست: {sorted(missing)}")

latest["data_date"] = latest["data_date"].astype(str)
latest["horizon"] = latest["horizon"].astype(int)

# ---------- Persist forecast history ----------
hist_cols = [
    "data_date", "horizon", "label",
    "q10_toman", "q50_toman", "q90_toman", "current_toman",
    "q10_return_pct", "q50_return_pct", "q90_return_pct"
]

if HISTORY.exists():
    hist = pd.read_csv(HISTORY)
else:
    hist = pd.DataFrame(columns=hist_cols)

hist = pd.concat([hist, latest[hist_cols]], ignore_index=True)
hist["data_date"] = hist["data_date"].astype(str)
hist["horizon"] = pd.to_numeric(hist["horizon"], errors="coerce").astype("Int64")
hist = (
    hist.dropna(subset=["data_date", "horizon"])
        .drop_duplicates(subset=["data_date", "horizon"], keep="last")
        .sort_values(["data_date", "horizon"])
        .reset_index(drop=True)
)
hist.to_csv(HISTORY, index=False)

# ---------- Evaluate matured forecasts ----------
market, _meta = load_market_data(refresh=True)
market = market.sort_values("date").reset_index(drop=True).copy()
market["date_key"] = pd.to_datetime(market["date"]).dt.strftime("%Y-%m-%d")
date_to_idx = {d: i for i, d in enumerate(market["date_key"])}

eval_cols = [
    "forecast_date", "horizon", "label", "current_toman",
    "q10_toman", "q50_toman", "q90_toman",
    "actual_date", "actual_toman",
    "ape_pct", "signed_error_pct",
    "direction_correct", "inside_q10_q90"
]

if EVALS.exists():
    ev = pd.read_csv(EVALS)
else:
    ev = pd.DataFrame(columns=eval_cols)

done = set()
if len(ev):
    for _, r in ev.iterrows():
        try:
            done.add((str(r["forecast_date"]), int(r["horizon"])))
        except Exception:
            pass

new_eval_rows = []

for _, r in hist.iterrows():
    forecast_date = str(r["data_date"])
    h = int(r["horizon"])
    key = (forecast_date, h)

    if key in done:
        continue
    if forecast_date not in date_to_idx:
        continue

    origin_idx = date_to_idx[forecast_date]
    target_idx = origin_idx + h

    if target_idx >= len(market):
        continue

    actual = float(market.loc[target_idx, "price_toman"])
    actual_date = str(market.loc[target_idx, "date_key"])
    current = float(r["current_toman"])
    q10 = float(r["q10_toman"])
    q50 = float(r["q50_toman"])
    q90 = float(r["q90_toman"])

    ape = abs(q50 - actual) / actual * 100.0
    signed = (q50 - actual) / actual * 100.0

    pred_dir = np.sign(q50 - current)
    real_dir = np.sign(actual - current)
    direction_correct = bool(pred_dir == real_dir)

    inside = bool(q10 <= actual <= q90)

    new_eval_rows.append({
        "forecast_date": forecast_date,
        "horizon": h,
        "label": str(r["label"]),
        "current_toman": current,
        "q10_toman": q10,
        "q50_toman": q50,
        "q90_toman": q90,
        "actual_date": actual_date,
        "actual_toman": actual,
        "ape_pct": ape,
        "signed_error_pct": signed,
        "direction_correct": direction_correct,
        "inside_q10_q90": inside,
    })

if new_eval_rows:
    ev = pd.concat([ev, pd.DataFrame(new_eval_rows)], ignore_index=True)

if len(ev):
    ev["forecast_date"] = ev["forecast_date"].astype(str)
    ev["horizon"] = pd.to_numeric(ev["horizon"], errors="coerce").astype("Int64")
    ev = (
        ev.drop_duplicates(subset=["forecast_date", "horizon"], keep="last")
          .sort_values(["forecast_date", "horizon"])
          .reset_index(drop=True)
    )

ev.to_csv(EVALS, index=False)

# ---------- Helpers ----------
FA_DIGITS = str.maketrans("0123456789.-+,%", "۰۱۲۳۴۵۶۷۸۹٫−+،٪")

def fa(v):
    return str(v).translate(FA_DIGITS)

def money(v):
    return fa(f"{float(v)/1_000_000:.2f}") + " میلیون تومان"

def pct(v):
    return fa(f"{float(v):.1f}") + "٪"

def horizon_fa(h):
    return {63: "۳ ماهه", 126: "۶ ماهه", 252: "۱۲ ماهه"}.get(int(h), fa(h))

def yesno(v):
    if isinstance(v, str):
        v = v.strip().lower() in ("true", "1", "yes")
    return "بله" if bool(v) else "خیر"

# ---------- Build scorecard section ----------
forecast_days = hist["data_date"].nunique()
pending = len(hist) - len(ev)

if len(ev) == 0:
    score_body = f"""
      <div class="score-empty">
        <b>هنوز هیچ پیش‌بینی به زمان ارزیابی نرسیده است.</b>
        <p>
          از همین حالا ربات پیش‌بینی‌هایش را برای همیشه ثبت می‌کند.
          وقتی ۶۳ جلسه معاملاتی از یک پیش‌بینی بگذرد، اولین نتیجه واقعی
          در کارنامه ۳ ماهه ظاهر می‌شود. بعد از ۱۲۶ و ۲۵۲ جلسه معاملاتی نیز
          کارنامه ۶ ماهه و ۱۲ ماهه ساخته می‌شود.
        </p>
      </div>

      <div class="score-mini-grid">
        <div><span>روزهای پیش‌بینی ثبت‌شده</span><strong>{fa(forecast_days)}</strong></div>
        <div><span>پیش‌بینی‌های در انتظار نتیجه</span><strong>{fa(pending)}</strong></div>
        <div><span>پیش‌بینی‌های ارزیابی‌شده</span><strong>۰</strong></div>
      </div>
    """
else:
    summary_cards = []
    for h in (63, 126, 252):
        g = ev[pd.to_numeric(ev["horizon"], errors="coerce") == h].copy()
        if len(g) == 0:
            summary_cards.append(f"""
              <div class="score-horizon">
                <h3>{horizon_fa(h)}</h3>
                <p>هنوز نمونه‌ای به موعد ارزیابی نرسیده است.</p>
              </div>
            """)
            continue

        ape = pd.to_numeric(g["ape_pct"], errors="coerce").mean()
        direction = g["direction_correct"].astype(str).str.lower().isin(["true","1"]).mean() * 100
        coverage = g["inside_q10_q90"].astype(str).str.lower().isin(["true","1"]).mean() * 100

        summary_cards.append(f"""
          <div class="score-horizon">
            <h3>{horizon_fa(h)}</h3>
            <div class="metric"><span>تعداد نتیجه واقعی</span><b>{fa(len(g))}</b></div>
            <div class="metric"><span>میانگین خطای عدد اصلی</span><b>{pct(ape)}</b></div>
            <div class="metric"><span>تشخیص درست جهت رشد/افت</span><b>{pct(direction)}</b></div>
            <div class="metric"><span>واقعیت داخل بازه پایین تا بالا</span><b>{pct(coverage)}</b></div>
          </div>
        """)

    recent_rows = []
    recent = ev.tail(6).iloc[::-1]
    for _, r in recent.iterrows():
        recent_rows.append(f"""
          <tr>
            <td>{html.escape(horizon_fa(int(r["horizon"])))}</td>
            <td>{fa(str(r["forecast_date"]))}</td>
            <td>{money(r["q50_toman"])}</td>
            <td>{money(r["actual_toman"])}</td>
            <td>{pct(r["ape_pct"])}</td>
          </tr>
        """)

    score_body = f"""
      <div class="score-mini-grid">
        <div><span>روزهای پیش‌بینی ثبت‌شده</span><strong>{fa(forecast_days)}</strong></div>
        <div><span>پیش‌بینی‌های ارزیابی‌شده</span><strong>{fa(len(ev))}</strong></div>
        <div><span>در انتظار نتیجه</span><strong>{fa(max(pending, 0))}</strong></div>
      </div>

      <div class="score-horizons">
        {''.join(summary_cards)}
      </div>

      <div class="score-table-wrap">
        <h3>آخرین پیش‌بینی‌هایی که نتیجه واقعی‌شان مشخص شده</h3>
        <table class="score-table">
          <thead>
            <tr>
              <th>بازه</th>
              <th>تاریخ پیش‌بینی</th>
              <th>پیش‌بینی اصلی</th>
              <th>قیمت واقعی</th>
              <th>خطا</th>
            </tr>
          </thead>
          <tbody>
            {''.join(recent_rows)}
          </tbody>
        </table>
      </div>
    """

score_section = f"""
<section class="scorecard">
  <h2>کارنامه واقعی ربات</h2>

  <div class="score-intro">
    <b>این بخش چه چیزی را نشان می‌دهد؟</b>
    <p>
      ربات هر بار که پیش‌بینی می‌کند، نتیجه را ذخیره می‌کند و بعداً وقتی
      زمان آن پیش‌بینی برسد، آن را با قیمت واقعی بازار مقایسه می‌کند.
      بنابراین این بخش نشان می‌دهد ربات در استفاده واقعی چقدر درست یا اشتباه پیش‌بینی کرده است.
    </p>
    <p>
      ارزیابی‌ها بر اساس جلسه معاملاتی انجام می‌شوند:
      ۳ ماهه = ۶۳ جلسه، ۶ ماهه = ۱۲۶ جلسه و ۱۲ ماهه = ۲۵۲ جلسه معاملاتی.
    </p>
  </div>

  {score_body}

  <details class="score-help">
    <summary>معیارهای کارنامه یعنی چه؟</summary>
    <p><b>میانگین خطای عدد اصلی:</b> به طور متوسط پیش‌بینی اصلی مدل چند درصد با قیمت واقعی فاصله داشته است. کمتر بهتر است.</p>
    <p><b>تشخیص درست جهت:</b> چند درصد مواقع مدل درست فهمیده که قیمت نسبت به روز پیش‌بینی بالا می‌رود یا پایین.</p>
    <p><b>واقعیت داخل بازه:</b> چند درصد مواقع قیمت واقعی بین «برآورد پایین» و «برآورد بالا» قرار گرفته است.</p>
  </details>
</section>
"""

score_css = """
<style>
.scorecard{
  background:#fff;
  border-radius:20px;
  padding:18px;
  margin-top:14px;
  margin-bottom:14px;
  box-shadow:0 2px 10px #0000000a;
}
.scorecard h2{font-size:19px;margin:0 0 10px}
.score-intro{
  background:#f4f7fa;
  border-radius:13px;
  padding:13px;
  font-size:13px;
  margin-bottom:13px;
}
.score-intro p{margin:6px 0}
.score-empty{
  background:#fff8df;
  border-radius:13px;
  padding:13px;
  font-size:13px;
}
.score-empty p{margin:6px 0 0}
.score-mini-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:7px;
  margin-top:12px;
}
.score-mini-grid div{
  background:#f5f5f5;
  border-radius:12px;
  padding:10px 6px;
  text-align:center;
}
.score-mini-grid span{display:block;font-size:10px;color:#666}
.score-mini-grid strong{display:block;font-size:18px;margin-top:3px}
.score-horizons{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:8px;
  margin-top:12px;
}
.score-horizon{
  background:#f7f7f7;
  border-radius:13px;
  padding:11px;
}
.score-horizon h3{margin:0 0 8px;font-size:14px}
.metric{
  display:flex;
  justify-content:space-between;
  gap:6px;
  font-size:10px;
  border-top:1px solid #e7e7e7;
  padding:6px 0;
}
.metric:first-of-type{border-top:none}
.score-table-wrap{overflow-x:auto;margin-top:14px}
.score-table-wrap h3{font-size:14px}
.score-table{
  width:100%;
  border-collapse:collapse;
  min-width:620px;
  font-size:11px;
}
.score-table th,.score-table td{
  padding:8px;
  border-bottom:1px solid #e8e8e8;
  text-align:center;
}
.score-table th{background:#f4f4f4}
.score-help{
  margin-top:13px;
  background:#fafafa;
  border-radius:12px;
  padding:11px;
  font-size:12px;
}
@media(max-width:540px){
  .score-horizons{grid-template-columns:1fr}
  .score-mini-grid{grid-template-columns:1fr 1fr 1fr}
  .score-mini-grid strong{font-size:15px}
}
</style>
"""

report = REPORT.read_text(encoding="utf-8")

if "</head>" not in report:
    raise SystemExit("تگ </head> در گزارش پیدا نشد.")
report = report.replace("</head>", score_css + "\n</head>", 1)

target = '<section class="warning">'
if target in report:
    report = report.replace(target, score_section + "\n" + target, 1)
elif "<details>" in report:
    report = report.replace("<details>", score_section + "\n<details>", 1)
else:
    report = report.replace("</div>\n</body>", score_section + "\n</div>\n</body>", 1)

REPORT.write_text(report, encoding="utf-8")

print(f"Forecast history rows: {len(hist)}")
print(f"Evaluated rows: {len(ev)}")
print("Scorecard added to HTML.")

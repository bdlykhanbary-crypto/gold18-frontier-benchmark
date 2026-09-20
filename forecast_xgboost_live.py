from pathlib import Path
import numpy as np
import pandas as pd

from benchmark_common import HORIZONS, load_market_data
from run_model import load_xgboost, forecast_xgb


def fa_num(s):
    trans = str.maketrans("0123456789.,-%+", "۰۱۲۳۴۵۶۷۸۹٫٬−٪+")
    return str(s).translate(trans)


def fmt_toman(v):
    return fa_num(f"{float(v):,.0f}") + " تومان"


def fmt_million(v):
    return fa_num(f"{float(v)/1_000_000:.2f}") + " میلیون"


def fmt_pct(v):
    return fa_num(f"{float(v):+.1f}") + "٪"


def svg_chart(hist_prices, current_price, q10, q50, q90, horizon_label):
    hist = np.asarray(hist_prices, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) > 252:
        hist = hist[-252:]

    W, H = 1000, 430
    L, R, T, B = 82, 40, 34, 66
    split_x = 660
    end_x = W - R

    values = np.concatenate([hist, [current_price, q10, q50, q90]])
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    pad = max((ymax - ymin) * 0.12, current_price * 0.025)
    ymin -= pad
    ymax += pad

    def y(v):
        return T + (ymax - float(v)) / (ymax - ymin) * (H - T - B)

    hx = np.linspace(L, split_x, len(hist))
    hist_pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(hx, hist))

    n = 48
    t = np.linspace(0, 1, n)
    fx = split_x + (end_x - split_x) * t
    center = current_price + (q50 - current_price) * (t ** 0.92)
    low = current_price + (q10 - current_price) * (t ** 1.04)
    high = current_price + (q90 - current_price) * (t ** 0.86)
    low = np.minimum(low, center)
    high = np.maximum(high, center)

    center_pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(fx, center))
    low_pts = [(float(x), y(v)) for x, v in zip(fx, low)]
    high_pts = [(float(x), y(v)) for x, v in zip(fx, high)]
    band_pts = " ".join(
        [f"{x:.1f},{yy:.1f}" for x, yy in high_pts]
        + [f"{x:.1f},{yy:.1f}" for x, yy in reversed(low_pts)]
    )

    grid = []
    labels = []
    for i in range(5):
        val = ymin + (ymax - ymin) * i / 4
        yy = y(val)
        grid.append(
            f'<line x1="{L}" y1="{yy:.1f}" x2="{end_x}" y2="{yy:.1f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{L-12}" y="{yy+5:.1f}" text-anchor="end" font-size="20" fill="#777">{val/1_000_000:.1f}</text>'
        )

    return f'''
    <svg viewBox="0 0 {W} {H}" class="scenario-svg" role="img" aria-label="نمودار تاریخچه و بازه پیش‌بینی {horizon_label}">
      <rect x="0" y="0" width="{W}" height="{H}" rx="20" fill="#ffffff"/>
      {''.join(grid)}
      {''.join(labels)}
      <text x="{L}" y="24" font-size="19" fill="#777">میلیون تومان</text>
      <polygon points="{band_pts}" fill="#dbe9f6" opacity="0.8"/>
      <polyline points="{hist_pts}" fill="none" stroke="#2f6fb0" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
      <polyline points="{center_pts}" fill="none" stroke="#1f7a4d" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
      <line x1="{split_x}" y1="{T}" x2="{split_x}" y2="{H-B}" stroke="#777" stroke-width="2" stroke-dasharray="8,7"/>
      <circle cx="{split_x}" cy="{y(current_price):.1f}" r="7" fill="#111"/>
      <circle cx="{end_x}" cy="{y(q50):.1f}" r="7" fill="#1f7a4d"/>
      <g font-size="18" font-family="Arial, sans-serif">
        <text x="{end_x-5}" y="{y(q90)-10:.1f}" text-anchor="end" fill="#666">بالا {q90/1_000_000:.2f}M</text>
        <text x="{end_x-5}" y="{y(q50)-10:.1f}" text-anchor="end" fill="#111" font-weight="700">اصلی {q50/1_000_000:.2f}M</text>
        <text x="{end_x-5}" y="{y(q10)+25:.1f}" text-anchor="end" fill="#666">پایین {q10/1_000_000:.2f}M</text>
      </g>
    </svg>
    '''



def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0,31,59,90,120,151,181,212,243,273,304,334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        - 80
        + gd
        + g_d_m[gm - 1]
    )
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_date_string(gregorian_iso):
    gy, gm, gd = map(int, gregorian_iso.split("-"))
    jy, jm, jd = gregorian_to_jalali(gy, gm, gd)
    return f"{jy:04d}/{jm:02d}/{jd:02d}"


market, meta = load_market_data(refresh=True)
load_xgboost()

origin = len(market)
current = float(market.price_toman.iloc[-1])
data_date = str(market.date.iloc[-1].date())
jalali_date = jalali_date_string(data_date)

rows = []
for h, label in HORIZONS.items():
    q10_log, q50_log, q90_log = forecast_xgb(market, origin, h)
    vals = sorted([float(np.exp(q10_log)), float(np.exp(q50_log)), float(np.exp(q90_log))])
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

hist_prices = market.price_toman.tail(252).to_numpy(float)
fa_labels = {"3M": "۳ ماه آینده", "6M": "۶ ماه آینده", "12M": "۱۲ ماه آینده"}

cards = []
for _, r in df.iterrows():
    label = fa_labels.get(r["label"], r["label"])
    chart = svg_chart(hist_prices, current, float(r["q10_toman"]), float(r["q50_toman"]), float(r["q90_toman"]), label)
    cards.append(f'''
    <section class="card">
      <div class="period">{label}</div>
      <div class="main-label">پیش‌بینی اصلی مدل</div>
      <div class="main-price">{fmt_toman(r["q50_toman"])}</div>
      <div class="change">یعنی حدود <b>{fmt_pct(r["q50_return_pct"])}</b> نسبت به قیمت امروز</div>
      <div class="three">
        <div class="scenario"><span>برآورد پایین</span><strong>{fmt_million(r["q10_toman"])}</strong><em>{fmt_pct(r["q10_return_pct"])} نسبت به امروز</em></div>
        <div class="scenario main"><span>پیش‌بینی اصلی</span><strong>{fmt_million(r["q50_toman"])}</strong><em>{fmt_pct(r["q50_return_pct"])} نسبت به امروز</em></div>
        <div class="scenario"><span>برآورد بالا</span><strong>{fmt_million(r["q90_toman"])}</strong><em>{fmt_pct(r["q90_return_pct"])} نسبت به امروز</em></div>
      </div>
      <div class="plain-explain"><b>به زبان ساده:</b> مدل برای {label} عدد <strong>{fmt_million(r["q50_toman"])}</strong> را برآورد اصلی خود می‌داند. عدد پایین و بالا نشان می‌دهند اگر بازار ضعیف‌تر یا قوی‌تر از انتظار مدل حرکت کند، چه محدوده‌ای ممکن است دیده شود.</div>
      <div class="chart-title">نمودار ساده همین سناریو</div>
      <div class="chart-wrap">{chart}</div>
      <div class="phase-labels">
        <div class="phase-history">گذشته</div>
        <div class="phase-forecast">پیش‌بینی</div>
      </div>
      <div class="chart-help"><b>چطور نمودار را بخوانم؟</b><br>خط آبی = قیمت واقعی گذشته<br>خط‌چین = امروز و شروع پیش‌بینی<br>خط سبز = مسیر تصویریِ پیش‌بینی اصلی تا انتهای این بازه<br>ناحیه آبی کم‌رنگ = فاصله بین برآورد پایین و برآورد بالا</div>
      <div class="chart-warning">توجه: مسیر داخل قسمت پیش‌بینی، مسیر دقیق روزبه‌روز XGBoost نیست. مدل فقط قیمت انتهای بازه را پیش‌بینی می‌کند؛ این مسیر صرفاً برای فهم ساده‌تر همان برآورد پایین، اصلی و بالا رسم شده است.</div>
    </section>
    ''')

html = f'''<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>پیش‌بینی طلای ۱۸ عیار</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f3f3;color:#171717;font-family:Tahoma,Arial,sans-serif;line-height:1.8}}.wrap{{max-width:780px;margin:auto;padding:14px}}.hero{{background:#111;color:#fff;border-radius:22px;padding:22px;margin-bottom:14px}}.hero .small{{font-size:12px;opacity:.72}}.hero h1{{font-size:22px;margin:6px 0 18px}}.current-label{{font-size:13px;opacity:.72}}.price{{font-size:34px;font-weight:900;direction:ltr;text-align:right}}.date{{font-size:12px;opacity:.68;margin-top:6px}}.summary,.help,.warning,details{{background:#fff;border-radius:18px;padding:18px;margin-bottom:14px}}.summary h2,.help h2,.warning h2{{font-size:18px;margin:0 0 10px}}.summary p,.help p,.warning p{{font-size:13px;margin:7px 0}}.direction{{background:#f1f1f1;border-radius:12px;padding:11px;margin-top:10px;font-weight:700}}.help{{background:#fff8df}}.card{{background:#fff;border-radius:20px;padding:18px;margin-bottom:14px;box-shadow:0 2px 10px #0000000a}}.period{{font-size:19px;font-weight:900}}.main-label{{font-size:13px;color:#777;margin-top:10px}}.main-price{{font-size:28px;font-weight:900}}.change{{font-size:13px;color:#555;margin:5px 0 16px}}.three{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}}.scenario{{background:#f5f5f5;border-radius:13px;padding:10px 5px;text-align:center}}.scenario.main{{background:#e8f1eb}}.scenario span{{display:block;font-size:11px;color:#666}}.scenario strong{{display:block;font-size:12px;margin-top:5px}}.scenario em{{display:block;font-size:10px;color:#777;font-style:normal;margin-top:5px}}.plain-explain{{margin-top:15px;background:#f7f7f7;padding:12px;border-radius:12px;font-size:13px}}.chart-title{{font-size:16px;font-weight:900;margin-top:20px;margin-bottom:8px}}.chart-wrap{{width:100%;overflow:hidden;border:1px solid #ededed;border-radius:15px;background:#fff}}.scenario-svg{{width:100%;height:auto;display:block}}
.phase-labels{{
  display:grid;
  grid-template-columns:66% 34%;
  direction:ltr;
  width:100%;
  margin:7px 0 14px;
  padding:0 6px;
  color:#555;
  font-size:13px;
  font-weight:800;
  line-height:2;
}}
.phase-history{{
  direction:rtl;
  text-align:center;
  white-space:nowrap;
}}
.phase-forecast{{
  direction:rtl;
  text-align:center;
  white-space:nowrap;
}}
.chart-help{{margin-top:10px;background:#f4f7fa;border-radius:12px;padding:12px;font-size:12px}}.chart-warning{{margin-top:8px;background:#fff8df;border-radius:12px;padding:11px;font-size:11px;color:#555}}.important{{background:#f2f2f2;border-radius:12px;padding:11px;font-weight:700}}details{{font-size:12px}}summary{{font-weight:800;cursor:pointer}}.footer{{font-size:10px;color:#888;text-align:center;padding:22px 5px}}@media(max-width:430px){{.price{{font-size:29px}}.three{{gap:5px}}.scenario strong{{font-size:10px}}}}
</style></head><body><div class="wrap">
<header class="hero"><div class="small">پیش‌بینی آماری قیمت طلای ۱۸ عیار</div><h1>وضعیت احتمالی قیمت طلا در ماه‌های آینده</h1><div class="current-label">قیمت فعلی مورد استفاده مدل</div><div class="price">{fmt_toman(current)}</div><div class="date">آخرین روز داده‌ای که مدل دیده است: {fa_num(data_date)} میلادی | {fa_num(jalali_date)} شمسی</div></header>
<section class="summary"><h2>خلاصه خیلی ساده</h2><p>این برنامه قیمت طلای ۱۸ عیار، دلار و طلای جهانی را بررسی می‌کند و با استفاده از رفتار گذشته بازار، قیمت احتمالی آینده را برآورد می‌کند.</p><div class="direction">در هر سه بازه زمانی، پیش‌بینی اصلی فعلی مدل بالاتر از قیمت امروز است.</div><p>عدد <b>«پیش‌بینی اصلی»</b> مهم‌ترین عدد مدل برای پایان آن بازه زمانی است.</p></section>
<section class="help"><h2>این سه عدد یعنی چه؟</h2><p><b>برآورد پایین:</b> اگر بازار ضعیف‌تر از انتظار مدل حرکت کند.</p><p><b>پیش‌بینی اصلی:</b> عدد مرکزی و مهم‌ترین پیش‌بینی مدل.</p><p><b>برآورد بالا:</b> اگر بازار قوی‌تر از انتظار مدل حرکت کند.</p><p>این اعداد تضمین نمی‌کنند که قیمت حتماً بین برآورد پایین و بالا بماند.</p></section>
{''.join(cards)}
<section class="warning"><h2>نکته مهم قبل از استفاده</h2><p>این برنامه آینده را نمی‌داند. جنگ، تغییر شدید دلار، سیاست اقتصادی، تغییر قیمت جهانی طلا یا شوک‌های دیگر می‌توانند نتیجه واقعی را عوض کنند.</p><div class="important">پیش‌بینی اصلی هدف قطعی قیمت نیست و برآورد پایین نیز کف تضمینی قیمت نیست.</div><p>این گزارش به‌تنهایی دستور خرید یا فروش محسوب نمی‌شود.</p></section>
<details><summary>جزئیات فنی</summary><p>مدل اصلی: XGBoost Quantile. ورودی‌ها: طلای ۱۸ عیار ایران، دلار آزاد و اونس جهانی.</p><p>«برآورد پایین» همان Q10، «پیش‌بینی اصلی» Q50 و «برآورد بالا» Q90 است.</p></details>
<div class="footer">مدل: XGBoost Quantile · داده‌ها: طلای ۱۸ عیار + دلار + اونس جهانی</div>
</div></body></html>'''

Path("gold18_forecast_report.html").write_text(html, encoding="utf-8")
print("Saved: latest_xgboost_forecast.csv")
print("Saved: gold18_forecast_report.html")

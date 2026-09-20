from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HORIZONS = {63: "3M", 126: "6M", 252: "12M"}
MAX_H = 252
FOUNDATION_CONTEXT = 512
XGB_LAGS = 63
QUANTILES = [0.1, 0.5, 0.9]

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "gold18_ohlc.csv"
TGJU_URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/{slug}"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Origin": "https://www.tgju.org",
    "Referer": "https://www.tgju.org/",
}
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(s):
    return str(s).translate(PERSIAN_DIGITS).translate(ARABIC_DIGITS)


def strip_html(x):
    return re.sub(r"<.*?>", "", str(x)).strip() if x is not None else ""


def to_num(x):
    s = normalize_digits(strip_html(x)).replace(",", "").replace("٬", "").replace("−", "-")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def tgju_params(n=5000, n_columns=8):
    p = [("lang", "fa"), ("order_dir", "asc"), ("draw", "2")]
    for i in range(n_columns):
        p += [
            (f"columns[{i}][data]", str(i)),
            (f"columns[{i}][name]", ""),
            (f"columns[{i}][searchable]", "true"),
            (f"columns[{i}][orderable]", "true"),
            (f"columns[{i}][search][value]", ""),
            (f"columns[{i}][search][regex]", "false"),
        ]
    p += [
        ("start", "0"),
        ("length", str(n)),
        ("search", ""),
        ("order_col", ""),
        ("order_dir", ""),
        ("from", ""),
        ("to", ""),
        ("convert_to_ad", "1"),
        ("_", str(int(time.time() * 1000))),
    ]
    return p


def fetch_tgju_gold() -> pd.DataFrame:
    r = requests.get(
        TGJU_URL.format(slug="geram18"),
        params=tgju_params(),
        headers=HEADERS,
        timeout=35,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError("TGJU returned no rows")
    df = pd.DataFrame(
        [x[:8] for x in rows],
        columns=[
            "open",
            "low",
            "high",
            "close",
            "change_amount",
            "change_percent",
            "gregorian_date",
            "jalali_date",
        ],
    )
    for c in ["open", "low", "high", "close"]:
        df[c] = df[c].map(to_num)
    df["date"] = pd.to_datetime(
        df["gregorian_date"].map(lambda x: normalize_digits(strip_html(x))), errors="coerce"
    )
    return (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)[["date", "open", "low", "high", "close"]]
    )


def load_gold(refresh=True):
    local = pd.read_csv(DATA_PATH)
    local["date"] = pd.to_datetime(local["date"])
    local = local[["date", "open", "low", "high", "close"]].copy()
    source = "bundled verified history"
    note = None
    if refresh:
        try:
            live = fetch_tgju_gold()
            # Guard against a partial/broken live response.
            if len(live) >= int(0.95 * len(local)) and live["date"].max() >= local["date"].max():
                local = live
                source = "TGJU live history"
            else:
                note = f"TGJU response rejected by sanity check: rows={len(live)}, last={live['date'].max()}"
        except Exception as e:
            note = f"TGJU refresh failed: {type(e).__name__}: {e}"
    gold = local.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    gold["price_toman"] = gold["close"].astype(float) / 10.0
    gold["log_price"] = np.log(gold["price_toman"])
    return gold, source, note


def benchmark_origins(n: int):
    origins = []
    o = n - MAX_H
    floor = max(FOUNDATION_CONTEXT, XGB_LAGS + MAX_H + 128)
    while o >= floor:
        origins.append(o)
        o -= MAX_H
    origins = sorted(origins)
    if len(origins) < 5:
        raise RuntimeError(f"Too few non-overlapping origins: {len(origins)}")
    return origins


def to_row(gold, model, origin, h, q10_log, q50_log, q90_log):
    actual_log = float(gold.log_price.iloc[origin + h - 1])
    current_log = float(gold.log_price.iloc[origin - 1])
    vals = sorted([float(q10_log), float(q50_log), float(q90_log)])
    q10_log, q50_log, q90_log = vals
    return {
        "model": model,
        "origin_index": int(origin),
        "origin_date": str(gold.date.iloc[origin - 1].date()),
        "horizon": int(h),
        "label": HORIZONS[h],
        "current_price": float(math.exp(current_log)),
        "actual_price": float(math.exp(actual_log)),
        "actual_log": actual_log,
        "q10_log": q10_log,
        "q50_log": q50_log,
        "q90_log": q90_log,
        "q10_price": float(math.exp(q10_log)),
        "q50_price": float(math.exp(q50_log)),
        "q90_price": float(math.exp(q90_log)),
    }


def pinball(y, qhat, q):
    e = y - qhat
    return max(q * e, (q - 1) * e)


def summarize_results(df):
    rows = []
    if df is None or len(df) == 0:
        return pd.DataFrame()
    for (model, h), g in df.groupby(["model", "horizon"]):
        qloss = np.mean(
            [
                (
                    pinball(r.actual_log, r.q10_log, 0.1)
                    + pinball(r.actual_log, r.q50_log, 0.5)
                    + pinball(r.actual_log, r.q90_log, 0.9)
                )
                / 3
                for r in g.itertuples()
            ]
        )
        mape = np.mean(np.abs(g["q50_price"] / g["actual_price"] - 1)) * 100
        cov = np.mean(
            (g["actual_price"] >= g["q10_price"])
            & (g["actual_price"] <= g["q90_price"])
        ) * 100
        diracc = np.mean(
            np.sign(g["q50_price"] / g["current_price"] - 1)
            == np.sign(g["actual_price"] / g["current_price"] - 1)
        ) * 100
        rows.append(
            {
                "model": model,
                "horizon": int(h),
                "label": HORIZONS[int(h)],
                "n": int(len(g)),
                "mean_pinball": float(qloss),
                "median_mape_pct": float(mape),
                "coverage80_pct": float(cov),
                "direction_accuracy_pct": float(diracc),
            }
        )
    out = pd.DataFrame(rows)
    out["rank_pinball"] = out.groupby("horizon")["mean_pinball"].rank(method="min")
    return out.sort_values(["horizon", "mean_pinball"])


def save_status(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")

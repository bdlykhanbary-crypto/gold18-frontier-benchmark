from __future__ import annotations

import html
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from benchmark_common import HEADERS, normalize_digits, to_num

PROFILE_URL = "https://www.tgju.org/profile/{slug}"


def _fetch_current(slug: str):
    headers = dict(HEADERS)
    headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": PROFILE_URL.format(slug=slug),
    })

    r = requests.get(
        PROFILE_URL.format(slug=slug),
        params={"_": str(int(time.time() * 1000))},
        headers=headers,
        timeout=35,
    )
    r.raise_for_status()

    text = html.unescape(re.sub(r"<[^>]+>", " ", r.text))
    text = normalize_digits(text)
    text = re.sub(r"\s+", " ", text)

    # TGJU profile pages expose the real current quote under "نرخ فعلی".
    m = re.search(r"نرخ فعلی\s*:*[\s|]*([0-9][0-9,٬.]*)", text)
    if not m:
        raise RuntimeError(f"Could not parse live TGJU quote for {slug}")

    value = to_num(m.group(1))
    if value is None or not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid live TGJU quote for {slug}: {value}")

    mt = re.search(
        r"زمان ثبت آخرین نرخ\s*[:|]?\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)",
        text,
    )
    quote_time = mt.group(1) if mt else None
    return float(value), quote_time


def overlay_live_snapshot(market: pd.DataFrame, meta: dict):
    """Append/replace today's row with the actual current TGJU profile quotes.

    Daily historical data remain intact. Only the final live snapshot used by
    the production forecast is overlaid. If current quotes cannot be fetched,
    the function raises instead of silently publishing a stale report.
    """
    gold_irr, gold_time = _fetch_current("geram18")
    usd_irr, usd_time = _fetch_current("price_dollar_rl")
    xau_usd, xau_time = _fetch_current("ons")

    now_tehran = datetime.now(ZoneInfo("Asia/Tehran"))
    today = pd.Timestamp(now_tehran.date())

    out = market.copy()

    row = {
        "date": today,
        "gold_close_irr": gold_irr,
        "usd_close_irr": usd_irr,
        "xau_close_usd": xau_usd,
        "price_toman": gold_irr / 10.0,
        "log_gold": np.log(gold_irr / 10.0),
        "log_usd": np.log(usd_irr),
        "log_xau": np.log(xau_usd),
        "log_price": np.log(gold_irr / 10.0),
    }

    if len(out) and pd.Timestamp(out["date"].iloc[-1]).normalize() == today:
        idx = out.index[-1]
        for k, v in row.items():
            out.loc[idx, k] = v
    else:
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    out = (
        out.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    meta = dict(meta or {})
    meta["live_snapshot"] = {
        "date_tehran": now_tehran.strftime("%Y-%m-%d"),
        "gold18_irr": gold_irr,
        "gold18_toman": gold_irr / 10.0,
        "usd_irr": usd_irr,
        "xau_usd": xau_usd,
        "gold_quote_time": gold_time,
        "usd_quote_time": usd_time,
        "xau_quote_time": xau_time,
        "source": "TGJU current profile quotes",
    }
    meta["aligned_to"] = str(today.date())

    print(
        "LIVE_SNAPSHOT",
        f"gold18_toman={gold_irr/10.0:.0f}",
        f"usd_irr={usd_irr:.0f}",
        f"xau_usd={xau_usd:.2f}",
        f"date={today.date()}",
        flush=True,
    )

    return out, meta

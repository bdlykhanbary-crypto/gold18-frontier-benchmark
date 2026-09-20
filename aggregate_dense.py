from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_common import HORIZONS, summarize_results

ROOT = Path(__file__).resolve().parent
ART = ROOT / "downloaded_artifacts"
OUT = ROOT / "final_results"
OUT.mkdir(exist_ok=True)

statuses = []
records = []
forecasts = []

for p in ART.rglob("status.json"):
    try:
        statuses.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        statuses.append({"model": str(p), "status": "STATUS_READ_FAILED", "error": repr(e)})
for p in ART.rglob("walkforward_records.csv"):
    try:
        records.append(pd.read_csv(p))
    except Exception:
        pass
for p in ART.rglob("current_forecast.csv"):
    try:
        forecasts.append(pd.read_csv(p))
    except Exception:
        pass

status_df = pd.DataFrame(statuses)
status_df.to_csv(OUT / "model_status.csv", index=False)

rec = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
cf = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
if len(rec):
    rec.to_csv(OUT / "walkforward_records_all.csv", index=False)
if len(cf):
    cf.to_csv(OUT / "current_forecasts_all.csv", index=False)

summary = summarize_results(rec)
if len(summary):
    # Scientific point-control: current price as forecast for each origin.
    naive_rows = []
    for (model, h), g in rec.groupby(["model", "horizon"]):
        naive_mape = np.mean(np.abs(g["current_price"] / g["actual_price"] - 1)) * 100
        naive_rows.append({"model": model, "horizon": h, "naive_mape_pct": naive_mape})
    naive = pd.DataFrame(naive_rows)
    summary = summary.merge(naive, on=["model", "horizon"], how="left")
    summary["mape_vs_naive_delta_pctpt"] = summary["median_mape_pct"] - summary["naive_mape_pct"]
    summary.to_csv(OUT / "leaderboard_by_horizon.csv", index=False)

    overall = (
        summary.groupby("model", as_index=False)
        .agg(
            completed_horizons=("horizon", "nunique"),
            avg_pinball=("mean_pinball", "mean"),
            avg_rank=("rank_pinball", "mean"),
            avg_mape_pct=("median_mape_pct", "mean"),
            avg_coverage80_pct=("coverage80_pct", "mean"),
            avg_direction_accuracy_pct=("direction_accuracy_pct", "mean"),
            avg_mape_vs_naive_delta_pctpt=("mape_vs_naive_delta_pctpt", "mean"),
        )
        .sort_values(["completed_horizons", "avg_pinball"], ascending=[False, True])
    )
    # Only a model with all three horizons can win.
    eligible = overall[overall["completed_horizons"] == len(HORIZONS)].copy()
    winner = eligible.iloc[0]["model"] if len(eligible) else None
    overall["winner"] = overall["model"].eq(winner)
    overall.to_csv(OUT / "leaderboard_overall.csv", index=False)
else:
    overall = pd.DataFrame()
    winner = None

# HTML report
status_cols = [c for c in ["model", "status", "data_source", "data_to", "backtest_origins", "elapsed_seconds", "error"] if c in status_df.columns]
status_html = status_df[status_cols].fillna("").to_html(index=False, escape=True) if len(status_df) else "<p>No status files found.</p>"
leader_html = summary.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6f}") if len(summary) else "<p>No successful benchmark records.</p>"
overall_html = overall.to_html(index=False, escape=True, float_format=lambda x: f"{x:.6f}") if len(overall) else "<p>No overall ranking.</p>"
forecast_html = cf.to_html(index=False, escape=True, float_format=lambda x: f"{x:,.3f}") if len(cf) else "<p>No current forecasts.</p>"

report = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Gold 18K Frontier Benchmark</title><style>
body{{font-family:Arial,Tahoma,sans-serif;max-width:1200px;margin:20px auto;padding:0 12px;line-height:1.55;background:#f6f7f9;color:#111}}
.card{{background:white;border:1px solid #ddd;border-radius:12px;padding:14px;margin:14px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px;text-align:center}}th{{background:#f0f2f5}}.winner{{font-size:22px;font-weight:bold}}</style></head><body>
<h1>Iran 18K Gold — Multivariate Frontier Benchmark v2</h1>
<div class='card'><div class='winner'>Winner by pre-declared mean pinball rule: {html.escape(str(winner))}</div>
<p>No custom ensemble. Models are ranked separately. Only models completing all 3M/6M/12M horizons are eligible to win.</p></div>
<div class='card'><h2>Model execution status</h2>{status_html}</div>
<div class='card'><h2>Overall leaderboard</h2>{overall_html}</div>
<div class='card'><h2>Leaderboard by horizon</h2>{leader_html}</div>
<div class='card'><h2>Current forecasts</h2>{forecast_html}</div>
<div class='card'><h2>Method</h2><p>Foundation models: official pretrained checkpoints, zero-shot, 512-point Gold18 log-price target with USD/IRR and XAU/USD as past-only covariates, and dense walk-forward origins spaced every 21 trading sessions; forecast windows overlap. Factor alignment is causal (backward as-of on the Gold18 calendar); no future USD/XAU values are supplied. Primary metric: mean pinball loss over Q10/Q50/Q90 on Gold18 log price. XGBoost is a separate official quantile-regression challenger using 63 raw return lags from the same three series; models are never blended.</p></div>
</body></html>"""
(OUT / "report.html").write_text(report, encoding="utf-8")

with zipfile.ZipFile(ROOT / "gold18_frontier_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

print("WINNER:", winner)
print("RESULT ZIP:", ROOT / "gold18_frontier_results.zip")

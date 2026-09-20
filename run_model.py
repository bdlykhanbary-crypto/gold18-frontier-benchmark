from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_common import (
    FOUNDATION_CONTEXT,
    HORIZONS,
    MAX_H,
    QUANTILES,
    XGB_LAGS,
    benchmark_origins,
    load_market_data,
    save_status,
    summarize_results,
    to_row,
)

MODEL_NAMES = {
    "timesfm": "TimesFM-3",
    "chronos": "Chronos-2",
    "moirai": "Moirai-2.0",
    "xgboost": "XGBoost-Quantile",
}


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def contexts_from_frame(market, origin):
    lo = max(0, origin - FOUNDATION_CONTEXT)
    g = market.log_gold.iloc[lo:origin].to_numpy(np.float32)
    u = market.log_usd.iloc[lo:origin].to_numpy(np.float32)
    x = market.log_xau.iloc[lo:origin].to_numpy(np.float32)
    if min(len(g), len(u), len(x)) < 64:
        raise RuntimeError("Insufficient aligned multivariate context")
    return g, u, x


def forecast_timesfm(gold_context, usd_context, xau_context, horizon):
    # Official TimesFM-3 past-only covariate interface.
    out = list(
        _STATE["model"].predict_batch(
            contexts=[np.asarray(gold_context, dtype=np.float32)],
            horizon=horizon,
            past_only_covariates=[
                np.vstack([
                    np.asarray(usd_context, dtype=np.float32),
                    np.asarray(xau_context, dtype=np.float32),
                ])
            ],
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )[0]
    q = np.asarray(out.quantiles)
    # Univariate target -> (horizon, 9). Defensive support for (1,horizon,9).
    if q.ndim == 3:
        q = q[0]
    if q.ndim != 2 or q.shape[0] != horizon or q.shape[1] < 9:
        raise RuntimeError(f"Unexpected TimesFM quantile shape: {q.shape}")
    return q[:, 0], q[:, 4], q[:, 8]


def load_timesfm():
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    cfg = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=1,
        device="cpu",
    )
    _STATE["model"] = TimesFM3Evaluator(cfg)
    return {
        "checkpoint": "google/timesfm-3.0-pytorch",
        "torch": torch.__version__,
        "timesfm": pkg_version("timesfm"),
        "device": "cpu",
        "mode": "official past_only_covariates; USD/IRR + XAU/USD",
    }


def forecast_chronos(gold_context, usd_context, xau_context, horizon):
    # Official Chronos-2 dictionary input for past-only covariates.
    inp = {
        "target": np.asarray(gold_context, dtype=np.float32),
        "past_covariates": {
            "usd_irr": np.asarray(usd_context, dtype=np.float32),
            "xau_usd": np.asarray(xau_context, dtype=np.float32),
        },
    }
    qlist, _ = _STATE["model"].predict_quantiles(
        [inp],
        prediction_length=horizon,
        quantile_levels=QUANTILES,
        batch_size=1,
        context_length=FOUNDATION_CONTEXT,
        limit_prediction_length=False,
    )
    q = np.asarray(qlist[0].detach().cpu())
    if q.ndim == 3:
        q = q[0]
    if q.shape != (horizon, 3):
        raise RuntimeError(f"Unexpected Chronos-2 quantile shape: {q.shape}")
    return q[:, 0], q[:, 1], q[:, 2]


def load_chronos():
    import torch
    from chronos import Chronos2Pipeline

    _STATE["model"] = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return {
        "checkpoint": "amazon/chronos-2",
        "torch": torch.__version__,
        "chronos-forecasting": pkg_version("chronos-forecasting"),
        "device": "cpu",
        "mode": "official past_covariates; USD/IRR + XAU/USD",
    }


def forecast_moirai(gold_context, usd_context, xau_context, horizon):
    # Moirai-2 official forward path for past-only multivariate covariates.
    # We intentionally bypass the repository's convenience predict() wrapper because
    # its current padding helper hard-codes width=1 and fails for >1 past covariate.
    import torch

    gold = np.asarray(gold_context, dtype=np.float32)
    past_cov = np.column_stack([
        np.asarray(usd_context, dtype=np.float32),
        np.asarray(xau_context, dtype=np.float32),
    ]).astype(np.float32, copy=False)

    if gold.shape[0] != FOUNDATION_CONTEXT or past_cov.shape != (FOUNDATION_CONTEXT, 2):
        raise RuntimeError(
            f"Moirai-2 context shape mismatch: gold={gold.shape}, covariates={past_cov.shape}; "
            f"expected ({FOUNDATION_CONTEXT},) and ({FOUNDATION_CONTEXT}, 2)"
        )

    past_target = torch.from_numpy(gold.reshape(1, FOUNDATION_CONTEXT, 1))
    past_cov_t = torch.from_numpy(past_cov.reshape(1, FOUNDATION_CONTEXT, 2))

    with torch.no_grad():
        pred = _STATE["model"](
            past_target=past_target,
            past_observed_target=torch.ones_like(past_target, dtype=torch.bool),
            past_is_pad=torch.zeros((1, FOUNDATION_CONTEXT), dtype=torch.bool),
            past_feat_dynamic_real=past_cov_t,
            past_observed_feat_dynamic_real=torch.ones_like(past_cov_t, dtype=torch.bool),
        ).detach().cpu().numpy()

    q = pred[0]
    levels = list(map(float, _STATE["model"].module.quantile_levels))
    i10, i50, i90 = levels.index(0.1), levels.index(0.5), levels.index(0.9)
    if q.ndim != 2 or q.shape[1] < horizon:
        raise RuntimeError(f"Unexpected Moirai-2 prediction shape: {q.shape}")
    return q[i10, :horizon], q[i50, :horizon], q[i90, :horizon]


def load_moirai():
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=MAX_H,
        context_length=FOUNDATION_CONTEXT,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=2,
    ).to("cpu")
    model.eval()
    _STATE["model"] = model
    return {
        "checkpoint": "Salesforce/moirai-2.0-R-small",
        "torch": torch.__version__,
        "uni2ts": pkg_version("uni2ts"),
        "device": "cpu",
        "mode": "official past_feat_dynamic_real; USD/IRR + XAU/USD",
    }


def xgb_dataset_until(market, origin, h):
    arr = market[["log_gold", "log_usd", "log_xau"]].to_numpy(float)[:origin]
    r = np.diff(arr, axis=0)
    X, y = [], []
    for t in range(XGB_LAGS + 1, origin - h + 1):
        hist = r[t - 1 - XGB_LAGS : t - 1, :]
        if hist.shape != (XGB_LAGS, 3):
            continue
        X.append(hist.T.reshape(-1))
        y.append(arr[t + h - 1, 0] - arr[t - 1, 0])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def forecast_xgb(market, origin, h):
    import xgboost as xgb

    X, y = xgb_dataset_until(market, origin, h)
    if len(X) < 128:
        raise RuntimeError(f"Only {len(X)} XGBoost training samples")
    params = {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "quantile_alpha": np.array(QUANTILES),
        "learning_rate": 0.04,
        "max_depth": 5,
        "seed": 0,
        "nthread": max(1, os.cpu_count() or 1),
    }
    booster = xgb.train(
        params,
        xgb.QuantileDMatrix(X, y),
        num_boost_round=32,
        verbose_eval=False,
    )
    arr = market[["log_gold", "log_usd", "log_xau"]].to_numpy(float)[:origin]
    r = np.diff(arr, axis=0)
    feat = r[-XGB_LAGS:, :].T.reshape(1, -1).astype(np.float32)
    pred = np.asarray(booster.inplace_predict(feat)).reshape(-1)
    base = float(arr[-1, 0])
    vals = sorted([float(base + pred[0]), float(base + pred[1]), float(base + pred[2])])
    return vals


def load_xgboost():
    import xgboost as xgb
    return {
        "xgboost": xgb.__version__,
        "device": "cpu",
        "mode": "official reg:quantileerror; 63 raw return lags for Gold18 + USD/IRR + XAU/USD",
    }


_STATE = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    slug = args.model
    model_name = MODEL_NAMES[slug]
    outdir = Path(args.out) / slug
    outdir.mkdir(parents=True, exist_ok=True)
    status_path = outdir / "status.json"
    t0 = time.time()

    status = {
        "model": model_name,
        "slug": slug,
        "benchmark_version": "v2-multivariate",
        "status": "STARTED",
        "python": sys.version,
        "platform": platform.platform(),
        "started_utc": pd.Timestamp.utcnow().isoformat(),
    }
    save_status(status_path, **status)

    try:
        market, data_meta = load_market_data(refresh=True)
        origins = benchmark_origins(len(market))
        status.update(
            {
                "data_mode": "Gold18 target + USD/IRR + XAU/USD past-only covariates",
                "data_meta": data_meta,
                "rows": int(len(market)),
                "data_from": str(market.date.min().date()),
                "data_to": str(market.date.max().date()),
                "last_close_toman": float(market.price_toman.iloc[-1]),
                "backtest_origins": len(origins),
            }
        )

        if slug == "timesfm":
            status["runtime"] = load_timesfm()
            fn = forecast_timesfm
        elif slug == "chronos":
            status["runtime"] = load_chronos()
            fn = forecast_chronos
        elif slug == "moirai":
            status["runtime"] = load_moirai()
            fn = forecast_moirai
        else:
            status["runtime"] = load_xgboost()
            fn = None

        records = []
        for i, origin in enumerate(origins, 1):
            print(f"[{model_name}] origin {i}/{len(origins)}: {market.date.iloc[origin-1].date()}", flush=True)
            if slug == "xgboost":
                for h in HORIZONS:
                    a, b, c = forecast_xgb(market, origin, h)
                    records.append(to_row(market, model_name, origin, h, a, b, c))
            else:
                g, u, x = contexts_from_frame(market, origin)
                q10, q50, q90 = fn(g, u, x, MAX_H)
                for h in HORIZONS:
                    records.append(
                        to_row(market, model_name, origin, h, q10[h - 1], q50[h - 1], q90[h - 1])
                    )

        rec = pd.DataFrame(records)
        rec.to_csv(outdir / "walkforward_records.csv", index=False)
        summarize_results(rec).to_csv(outdir / "summary_by_horizon.csv", index=False)

        current_rows = []
        latest_origin = len(market)
        if slug == "xgboost":
            for h, label in HORIZONS.items():
                a, b, c = forecast_xgb(market, latest_origin, h)
                current_rows.append({
                    "model": model_name, "horizon": h, "label": label,
                    "q10_toman": float(np.exp(a)), "q50_toman": float(np.exp(b)), "q90_toman": float(np.exp(c)),
                })
        else:
            g, u, x = contexts_from_frame(market, latest_origin)
            q10, q50, q90 = fn(g, u, x, MAX_H)
            for h, label in HORIZONS.items():
                vals = sorted([float(q10[h - 1]), float(q50[h - 1]), float(q90[h - 1])])
                current_rows.append({
                    "model": model_name, "horizon": h, "label": label,
                    "q10_toman": float(np.exp(vals[0])), "q50_toman": float(np.exp(vals[1])), "q90_toman": float(np.exp(vals[2])),
                })

        cf = pd.DataFrame(current_rows)
        cf["current_toman"] = float(market.price_toman.iloc[-1])
        for c in ["q10_toman", "q50_toman", "q90_toman"]:
            cf[c.replace("_toman", "_return_pct")] = (cf[c] / cf["current_toman"] - 1) * 100
        cf.to_csv(outdir / "current_forecast.csv", index=False)

        status.update({
            "status": "SUCCESS",
            "record_count": int(len(rec)),
            "elapsed_seconds": round(time.time() - t0, 2),
            "finished_utc": pd.Timestamp.utcnow().isoformat(),
        })
        save_status(status_path, **status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        status.update({
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - t0, 2),
            "finished_utc": pd.Timestamp.utcnow().isoformat(),
        })
        save_status(status_path, **status)
        print(status["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
LSF-KR Ticket Forecast - Analysis Script

Reads all CSV snapshots from ./snapshots/ and produces data.json
consumed by index.html.

Workflow:
    1. Drop new CSV into ./snapshots/
    2. Run:  python3 analyze.py
    3. Commit data.json + the new snapshot; push to main
    4. GitHub Pages updates automatically

CSV schema expected (Lotte Cinema export):
    Date, Start Time, Screen, # of Tickets Sold, # of Seats, Occupancy Rate

Filename must contain a timestamp token like  ..._YYYYMMDD_HHMMSS_Full.csv
"""

from __future__ import annotations
import json
import math
import os
import re
import sys
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
OUTPUT_FILE  = Path(__file__).parent / "data.json"

# Event schedule (used to extrapolate total seat capacity even if future
# showings haven't appeared in snapshots yet).
# Lotte Cinema 월드타워 3관, 8 showings/day, 158 seats/showing.
# Opening day (4/15) is partial (5 showings only, afternoon start).
EVENT_START_DATE        = "2026-04-15"
EVENT_END_DATE          = "2026-05-26"
SHOWINGS_PER_DAY        = 8
SEATS_PER_SHOWING       = 158
OPENING_DAY_SHOWINGS    = 5       # 4/15 is a partial day

BOOTSTRAP_ITERATIONS = 1000
BLOCK_SIZE           = 2          # moving block bootstrap
DIRICHLET_ALPHA      = 10.0       # concentration for ensemble weight MC
SPECIAL_JUMP         = 80         # single-snapshot delta threshold (tickets)
SPECIAL_OCC          = 0.95       # occupancy threshold for "special" tag
BENCH_MATURITY_DAYS  = 5          # days-since-open for prior-benchmark set
BENCH_MIN_OCC        = 0.50       # min final occupancy to qualify as benchmark

# Phase-2 (later-batch) projection.
# Showings beyond what the CSV currently lists are assumed to convert at the
# same rate as Phase 1 (currently-listed showings). Multiplier widens P5-P95.
PHASE2_CONV_LOW_MULT  = 0.65      # P5  conversion = phase1_conv * this
PHASE2_CONV_HIGH_MULT = 1.50      # P95 conversion = phase1_conv * this

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_snapshot_time(path: Path) -> datetime:
    m = TIMESTAMP_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def load_snapshots(directory: Path) -> list[dict]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {directory}")
    snaps = []
    for f in files:
        ts = parse_snapshot_time(f)
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            "# of Tickets Sold": "sold",
            "# of Seats":        "seats",
            "Start Time":        "start_time",
            "Occupancy Rate":    "occupancy_str",
        })
        df["sold"]  = df["sold"].astype(int)
        df["seats"] = df["seats"].astype(int)
        df["start_time"] = df["start_time"].astype(int)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["showing_id"] = (
            df["Date"] + "_"
            + df["start_time"].astype(str).str.zfill(4) + "_"
            + df["Screen"]
        )
        snaps.append({"ts": ts, "df": df})
    snaps.sort(key=lambda x: x["ts"])
    return snaps


# ---------------------------------------------------------------------------
# Growth models
# ---------------------------------------------------------------------------
def logistic(t, K, r, t0):
    # S(t) = K / (1 + exp(-r(t-t0)))
    return K / (1.0 + np.exp(-r * (t - t0)))


def gompertz(t, K, b, c):
    # S(t) = K * exp(-b * exp(-c*t))
    return K * np.exp(-b * np.exp(-c * t))


def bass(t, K, p, q):
    # S(t) = K * (1 - exp(-(p+q)t)) / (1 + (q/p)*exp(-(p+q)t))
    pq = p + q
    return K * (1.0 - np.exp(-pq * t)) / (1.0 + (q / p) * np.exp(-pq * t))


def _fit(model_fn, t, y, x0, bounds):
    def residual(params):
        return model_fn(t, *params) - y
    try:
        res = least_squares(residual, x0=x0, bounds=bounds, max_nfev=2000)
        if not res.success:
            return None
        yhat = model_fn(t, *res.x)
        sse  = float(np.sum((yhat - y) ** 2))
        k    = len(res.x)
        n    = len(y)
        # AICc: small-sample corrected AIC assuming Gaussian residuals
        if n - k - 1 > 0:
            aicc = n * math.log(sse / n + 1e-9) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        else:
            aicc = n * math.log(sse / n + 1e-9) + 2 * k
        return {"params": res.x.tolist(), "sse": sse, "aicc": aicc, "yhat": yhat.tolist()}
    except Exception as e:
        return None


def fit_logistic(t, y, K_hard):
    # r in (0, 2], t0 in [-50, 400]
    return _fit(
        logistic, t, y,
        x0=[min(y[-1] * 3.0, K_hard), 0.05, t[-1] * 2.0],
        bounds=([y[-1] * 1.01, 1e-4, -100.0], [K_hard, 2.0, 2000.0]),
    )


def fit_gompertz(t, y, K_hard):
    return _fit(
        gompertz, t, y,
        x0=[min(y[-1] * 3.0, K_hard), 3.0, 0.02],
        bounds=([y[-1] * 1.01, 1e-3, 1e-5], [K_hard, 30.0, 2.0]),
    )


def fit_bass(t, y, K_hard):
    return _fit(
        bass, t, y,
        x0=[min(y[-1] * 3.0, K_hard), 0.01, 0.05],
        bounds=([y[-1] * 1.01, 1e-5, 1e-4], [K_hard, 0.5, 2.0]),
    )


MODELS = {
    "logistic": (logistic, fit_logistic),
    "gompertz": (gompertz, fit_gompertz),
    "bass":     (bass,     fit_bass),
}


def aicc_weights(fits: dict) -> dict:
    valid = {k: v for k, v in fits.items() if v is not None}
    if not valid:
        return {}
    min_aicc = min(v["aicc"] for v in valid.values())
    raws = {k: math.exp(-0.5 * (v["aicc"] - min_aicc)) for k, v in valid.items()}
    s = sum(raws.values())
    return {k: raws[k] / s for k in raws}


# ---------------------------------------------------------------------------
# Bootstrap + Monte Carlo on weights
# ---------------------------------------------------------------------------
def moving_block_bootstrap_residuals(residuals: np.ndarray, block_size: int, n: int, rng) -> np.ndarray:
    # tile blocks then trim to length n
    blocks = []
    while sum(len(b) for b in blocks) < n:
        start = rng.integers(0, max(1, len(residuals) - block_size + 1))
        blocks.append(residuals[start:start + block_size])
    out = np.concatenate(blocks)[:n]
    return out


def ensemble_forecast(fits: dict, weights: dict, t_grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t_grid, dtype=float)
    for name, fit in fits.items():
        if fit is None or name not in weights:
            continue
        model_fn = MODELS[name][0]
        out += weights[name] * model_fn(t_grid, *fit["params"])
    return out


def run_bootstrap(t, y, K_hard, t_grid, base_fits, base_weights, iterations, rng):
    samples = np.zeros((iterations, len(t_grid)))
    weight_samples = []
    # Concentrated Dirichlet on ensemble weights
    alpha_vec = np.array([DIRICHLET_ALPHA * base_weights.get(k, 1e-3)
                          for k in MODELS.keys()])
    for it in range(iterations):
        # 1) MC on weights
        w_draw = rng.dirichlet(alpha_vec + 1e-6)
        w_draw = {name: float(w) for name, w in zip(MODELS.keys(), w_draw)}
        # 2) Resample residuals per model and refit (lighter: perturb yhat)
        sim_fits = {}
        for name, base in base_fits.items():
            if base is None:
                sim_fits[name] = None
                continue
            yhat = np.array(base["yhat"])
            resid = y - yhat
            sim_resid = moving_block_bootstrap_residuals(resid, BLOCK_SIZE, len(y), rng)
            y_sim = np.maximum(yhat + sim_resid, 0)
            # enforce monotonic non-decreasing (sales only grows)
            y_sim = np.maximum.accumulate(y_sim)
            fitter = MODELS[name][1]
            sim_fit = fitter(t, y_sim, K_hard)
            sim_fits[name] = sim_fit
        # 3) Forecast on grid
        samples[it] = ensemble_forecast(sim_fits, w_draw, t_grid)
        weight_samples.append(w_draw)
    return samples, weight_samples


# ---------------------------------------------------------------------------
# Per-showing forecast
# ---------------------------------------------------------------------------
def per_showing_forecast(
    snaps: list[dict],
    aggregate_p50: float,
    aggregate_p5: float,
    aggregate_p95: float,
) -> list[dict]:
    """
    Two-stage per-showing estimate, reconciled to the aggregate ensemble forecast.

      Stage 1: compute raw "potential" for each showing
                  potential_i = max(bench_occ, current_occ) - current_occ
               where bench_occ is the dow x start_time median of current occ
               over non-special showings with occ >= 15%.

      Stage 2: distribute the aggregate remaining forecast
                  R = aggregate_p50 - current_total
               across showings proportional to potential_i.
               This guarantees sum(final_sold) == aggregate_p50 exactly
               (top-down / bottom-up reconciliation, Hyndman style).

      P5 / P95 per-showing use the same proportional allocation on R_p5, R_p95.
    """
    latest = snaps[-1]["df"].copy()
    prev   = snaps[-2]["df"].copy() if len(snaps) >= 2 else None

    latest["dow"] = pd.to_datetime(latest["Date"]).dt.day_name()
    latest["occ"] = latest["sold"] / latest["seats"]

    # Tag "special" showings: near-full OR big single-snapshot jump
    special_mask = latest["occ"] >= SPECIAL_OCC
    if prev is not None:
        merged = latest.merge(
            prev[["showing_id", "sold"]].rename(columns={"sold": "sold_prev"}),
            on="showing_id", how="left"
        )
        merged["sold_prev"] = merged["sold_prev"].fillna(0)
        jump = merged["sold"] - merged["sold_prev"]
        special_mask = special_mask | (jump >= SPECIAL_JUMP)

    latest["special"] = special_mask.values

    # Benchmark from "non-special, meaningfully filling" showings
    bench_set = latest[(~latest["special"]) & (latest["occ"] >= 0.15)]
    if len(bench_set) < 5:
        bench_set = latest[~latest["special"]]
    bench = (
        bench_set.groupby(["dow", "start_time"])["occ"]
        .median().rename("bench_occ").reset_index()
    )
    latest = latest.merge(bench, on=["dow", "start_time"], how="left")
    latest["bench_occ"] = latest["bench_occ"].fillna(latest["occ"])

    # Raw potential per showing (0 for special - they are frozen)
    latest["potential_occ"] = np.where(
        latest["special"],
        0.0,
        np.maximum(latest["bench_occ"] - latest["occ"], 0.0),
    )
    # Add a small baseline so every non-special showing can grow a bit
    latest.loc[~latest["special"], "potential_occ"] += 0.01

    latest["potential_seats"] = latest["potential_occ"] * latest["seats"]
    total_potential = float(latest["potential_seats"].sum())

    current_total = int(latest["sold"].sum())
    remaining_p50 = max(aggregate_p50 - current_total, 0.0)
    remaining_p5  = max(aggregate_p5  - current_total, 0.0)
    remaining_p95 = max(aggregate_p95 - current_total, 0.0)

    def allocate(total_extra):
        if total_potential <= 0:
            return np.zeros(len(latest))
        share = latest["potential_seats"].values / total_potential
        add = share * total_extra
        # enforce each showing's hard cap (158 seats)
        room = (latest["seats"].values - latest["sold"].values).astype(float)
        return np.minimum(add, room)

    add50 = allocate(remaining_p50)
    add5  = allocate(remaining_p5)
    add95 = allocate(remaining_p95)

    latest["final_sold_est"] = (latest["sold"].values + add50).round().astype(int)
    latest["final_sold_p5"]  = (latest["sold"].values + add5 ).round().astype(int)
    latest["final_sold_p95"] = (latest["sold"].values + add95).round().astype(int)

    latest["final_occ_est"]  = latest["final_sold_est"] / latest["seats"]
    latest["p5_occ"]         = latest["final_sold_p5"]  / latest["seats"]
    latest["p95_occ"]        = latest["final_sold_p95"] / latest["seats"]

    cols = ["Date", "dow", "start_time", "Screen", "sold", "seats",
            "occ", "special", "final_occ_est", "final_sold_est",
            "p5_occ", "p95_occ"]
    return latest[cols].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    snaps = load_snapshots(SNAPSHOT_DIR)
    print(f"[info] loaded {len(snaps)} snapshots: "
          f"{snaps[0]['ts']} … {snaps[-1]['ts']}", file=sys.stderr)

    # ---- Compute schedule-based total capacity ----
    ev_start = pd.Timestamp(EVENT_START_DATE)
    ev_end   = pd.Timestamp(EVENT_END_DATE)
    event_days = (ev_end - ev_start).days + 1
    full_days  = max(event_days - 1, 0)      # first day is partial
    scheduled_showings = OPENING_DAY_SHOWINGS + full_days * SHOWINGS_PER_DAY
    scheduled_seats    = scheduled_showings * SEATS_PER_SHOWING

    # Hard cap K: the larger of "seats currently listed in CSV" and
    # "seats that will exist by event end" — important because late-run
    # showings may not yet be for sale.
    csv_seats  = int(snaps[-1]["df"]["seats"].sum())
    K_hard = max(csv_seats, scheduled_seats)
    print(f"[info] event period: {EVENT_START_DATE} ~ {EVENT_END_DATE} "
          f"({event_days} days, {scheduled_showings} showings, "
          f"{scheduled_seats} seats scheduled)", file=sys.stderr)
    print(f"[info] CSV currently lists {csv_seats} seats, K_hard = {K_hard}",
          file=sys.stderr)

    # Aggregate time series
    t0 = snaps[0]["ts"]
    t_obs = np.array([(s["ts"] - t0).total_seconds() / 3600.0 for s in snaps])
    y_obs = np.array([int(s["df"]["sold"].sum()) for s in snaps], dtype=float)
    print(f"[info] y_obs = {y_obs.tolist()}", file=sys.stderr)

    # Fit all three models (need at least 3 points for 3-param fits)
    fits = {}
    if len(y_obs) >= 3:
        for name, (_, fitter) in MODELS.items():
            fits[name] = fitter(t_obs, y_obs, float(K_hard))
    else:
        # Degenerate: just use current as prediction
        fits = {k: None for k in MODELS}

    weights = aicc_weights(fits)
    print(f"[info] ensemble weights: {weights}", file=sys.stderr)

    # Forecast grid: from t0 out to +720h (30 days) or until event ends
    last_show = pd.to_datetime(snaps[-1]["df"]["Date"]).max()
    t0_midnight = pd.Timestamp(t0.date())
    event_end_h = (last_show - t0_midnight).total_seconds() / 3600.0 + 24
    horizon = max(event_end_h, t_obs[-1] + 72)
    t_grid = np.linspace(0, horizon, 200)

    central = ensemble_forecast(fits, weights, t_grid) if weights else np.full_like(t_grid, y_obs[-1])

    # Bootstrap bands
    if len(y_obs) >= 4 and weights:
        samples, _ = run_bootstrap(
            t_obs, y_obs, float(K_hard), t_grid,
            fits, weights, BOOTSTRAP_ITERATIONS, RNG,
        )
        p5  = np.percentile(samples, 5,  axis=0)
        p25 = np.percentile(samples, 25, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p75 = np.percentile(samples, 75, axis=0)
        p95 = np.percentile(samples, 95, axis=0)
        final_samples = samples[:, -1]
        final_p5, final_p50, final_p95 = (
            float(np.percentile(final_samples, 5)),
            float(np.percentile(final_samples, 50)),
            float(np.percentile(final_samples, 95)),
        )
    else:
        p5 = p25 = p50 = p75 = p95 = central.copy()
        final_p5 = final_p50 = final_p95 = float(central[-1])

    # ---- Phase-2 projection ----
    # CSV currently lists 'csv_seats' = Phase 1 bookable seats.
    # Scheduled - CSV = Phase 2 seats (not yet for sale).
    # Phase-1 conversion rate (central) -> applied to Phase-2 seat pool.
    phase1_seats  = csv_seats
    phase2_seats  = max(scheduled_seats - csv_seats, 0)
    phase1_conv   = final_p50 / phase1_seats if phase1_seats > 0 else 0.0
    # P5/P95 conversion widened by multipliers (demand uncertainty for a batch
    # that hasn't opened yet is materially larger than within-batch noise).
    phase1_conv_low  = (final_p5  / phase1_seats if phase1_seats > 0 else 0.0) * PHASE2_CONV_LOW_MULT
    phase1_conv_high = (final_p95 / phase1_seats if phase1_seats > 0 else 0.0) * PHASE2_CONV_HIGH_MULT
    phase2_p5_sales  = phase2_seats * phase1_conv_low
    phase2_p50_sales = phase2_seats * phase1_conv
    phase2_p95_sales = phase2_seats * phase1_conv_high
    combined_p5  = final_p5  + phase2_p5_sales
    combined_p50 = final_p50 + phase2_p50_sales
    combined_p95 = final_p95 + phase2_p95_sales

    print(f"[phase2] phase1 seats={phase1_seats}, phase2 seats={phase2_seats}",
          file=sys.stderr)
    print(f"[phase2] phase1 conv: p5={phase1_conv_low:.3f} p50={phase1_conv:.3f} "
          f"p95={phase1_conv_high:.3f}", file=sys.stderr)
    print(f"[phase2] phase2 sales: p5={phase2_p5_sales:.0f} "
          f"p50={phase2_p50_sales:.0f} p95={phase2_p95_sales:.0f}",
          file=sys.stderr)
    print(f"[phase2] combined total: p5={combined_p5:.0f} "
          f"p50={combined_p50:.0f} p95={combined_p95:.0f}", file=sys.stderr)

    # Per-showing, reconciled to aggregate (Phase 1 only — Phase 2 showings
    # aren't listed yet, so we can't assign per-showing yet)
    showings = per_showing_forecast(snaps, final_p50, final_p5, final_p95)

    # Reconciliation note: compare top-down vs bottom-up
    bottom_up = sum(s["final_sold_est"] for s in showings)

    # Heatmap matrix (latest occupancy and forecast occupancy)
    df_latest = snaps[-1]["df"].copy()
    df_latest["dow"] = pd.to_datetime(df_latest["Date"]).dt.day_name()
    df_latest["occ"] = df_latest["sold"] / df_latest["seats"]
    heatmap = []
    for s in showings:
        heatmap.append({
            "date":       s["Date"],
            "dow":        s["dow"],
            "start_time": int(s["start_time"]),
            "current_occ":  float(s["occ"]),
            "forecast_occ": float(s["final_occ_est"]),
            "special":      bool(s["special"]),
        })

    # Model summary
    model_summary = []
    for name, fit in fits.items():
        if fit is None:
            model_summary.append({"name": name, "status": "failed"})
            continue
        # predicted asymptote = params[0] for all three (K is first param)
        model_summary.append({
            "name":      name,
            "status":    "ok",
            "params":    fit["params"],
            "aicc":      fit["aicc"],
            "sse":       fit["sse"],
            "weight":    weights.get(name, 0.0),
            "predicted_K": fit["params"][0],
        })

    # Snapshot summary
    snap_summary = []
    for s in snaps:
        snap_summary.append({
            "timestamp": s["ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "total_sold": int(s["df"]["sold"].sum()),
            "total_seats": int(s["df"]["seats"].sum()),
            "rows": int(len(s["df"])),
        })

    # Speed panel: snapshot-to-snapshot velocity
    velocities = []
    for i in range(1, len(snaps)):
        hrs = max((snaps[i]["ts"] - snaps[i-1]["ts"]).total_seconds() / 3600.0, 1e-6)
        d = int(snaps[i]["df"]["sold"].sum()) - int(snaps[i-1]["df"]["sold"].sum())
        velocities.append({
            "from": snaps[i-1]["ts"].strftime("%Y-%m-%d %H:%M"),
            "to":   snaps[i]["ts"].strftime("%Y-%m-%d %H:%M"),
            "delta": d,
            "hours": hrs,
            "rate_per_hour": d / hrs,
        })

    # Days elapsed / remaining based on latest snapshot date
    now_date = pd.Timestamp(snaps[-1]["ts"].date())
    days_elapsed   = max((now_date - ev_start).days + 1, 0)
    days_remaining = max((ev_end - now_date).days, 0)

    data = {
        "meta": {
            "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "show_name":       "LE SSERAFIM VR Concert : Invitation",
            "venue":           "롯데시네마 월드타워 3관",
            "event_start":     EVENT_START_DATE,
            "event_end":       EVENT_END_DATE,
            "event_days":      event_days,
            "days_elapsed":    days_elapsed,
            "days_remaining":  days_remaining,
            "total_showings_scheduled": scheduled_showings,
            "total_seats":     K_hard,
            "csv_listed_seats": csv_seats,
            "num_snapshots":   len(snaps),
            "first_snapshot":  snaps[0]["ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "last_snapshot":   snaps[-1]["ts"].strftime("%Y-%m-%d %H:%M:%S"),
        },
        "snapshots":          snap_summary,
        "velocities":         velocities,
        "aggregate_forecast": {
            "t_grid_hours": t_grid.tolist(),
            "central":      central.tolist(),
            "p5":           p5.tolist(),
            "p25":          p25.tolist(),
            "p50":          p50.tolist(),
            "p75":          p75.tolist(),
            "p95":          p95.tolist(),
            "observed_t":   t_obs.tolist(),
            "observed_y":   y_obs.tolist(),
            "final": {
                "p5":  final_p5,
                "p50": final_p50,
                "p95": final_p95,
                "bottom_up": int(bottom_up),
            },
        },
        "phase_breakdown": {
            "phase1_seats":   phase1_seats,
            "phase2_seats":   phase2_seats,
            "phase1_conversion": {
                "p5":  phase1_conv_low,
                "p50": phase1_conv,
                "p95": phase1_conv_high,
            },
            "phase1_forecast": {
                "p5":  float(final_p5),
                "p50": float(final_p50),
                "p95": float(final_p95),
            },
            "phase2_forecast": {
                "p5":  float(phase2_p5_sales),
                "p50": float(phase2_p50_sales),
                "p95": float(phase2_p95_sales),
            },
            "combined_forecast": {
                "p5":  float(combined_p5),
                "p50": float(combined_p50),
                "p95": float(combined_p95),
            },
        },
        "models":    model_summary,
        "showings":  showings,
        "heatmap":   heatmap,
    }

    # default=str so datetimes etc. serialize
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[done] wrote {OUTPUT_FILE}", file=sys.stderr)
    print(f"[summary] current sold = {int(y_obs[-1])}, "
          f"forecast P50 = {int(final_p50)} "
          f"(P5 {int(final_p5)} ~ P95 {int(final_p95)}), "
          f"bottom-up sum = {bottom_up}", file=sys.stderr)


if __name__ == "__main__":
    main()

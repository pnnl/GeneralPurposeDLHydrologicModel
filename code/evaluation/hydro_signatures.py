"""Hydrologic signatures and KGE parts versus observations.

Used for the three products (gauged forecast, PUB estimate, mixture mean).
Definitions follow Gupta et al. (2009) for KGE and Yilmaz et al. (2008)
for FDC diagnostics. Seasonal metrics use the calendar year when dates
are supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8
MIN_N = 30
PEAK_FRAC = 0.02


def _clean(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0)
    return obs[m], pred[m]


def kge_parts(obs, pred):
    if len(obs) < MIN_N or np.std(obs) < EPS or np.std(pred) < EPS:
        return dict(kge=np.nan, r=np.nan, alpha=np.nan, beta=np.nan)
    r = float(np.corrcoef(obs, pred)[0, 1])
    if not np.isfinite(r):
        r = 0.0
    alpha = float(np.std(pred) / (np.std(obs) + EPS))
    beta = float(np.mean(pred) / (np.mean(obs) + EPS))
    kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return dict(kge=kge, r=r, alpha=alpha, beta=beta)


def hydro_signatures(obs, pred, dates=None):
    """Per-basin agreement with observed hydrograph and signatures.

    Returns a dict of scalars. Relative biases are (sim - obs) / obs.
    """
    obs, pred = _clean(obs, pred)
    out = dict(n_valid=int(len(obs)))
    out.update(kge_parts(obs, pred))
    if len(obs) < MIN_N:
        return out

    o_mean = float(np.mean(obs))
    p_mean = float(np.mean(pred))
    ss_tot = float(np.sum((obs - o_mean) ** 2))
    out["nse"] = 1.0 - float(np.sum((pred - obs) ** 2) / (ss_tot + EPS))
    lo = np.log(np.maximum(obs, 0.01))
    lp = np.log(np.maximum(pred, 0.01))
    out["log_nse"] = 1.0 - float(
        np.sum((lp - lo) ** 2) / (np.sum((lo - lo.mean()) ** 2) + EPS)
    )
    out["pbias"] = 100.0 * (p_mean - o_mean) / (o_mean + EPS)
    out["rmse"] = float(np.sqrt(np.mean((pred - obs) ** 2)))
    out["nrmse"] = out["rmse"] / (o_mean + EPS)
    out["mean_obs"] = o_mean
    out["mean_pred"] = p_mean

    # High / peak flows (top 2%)
    n_hi = max(1, int(round(PEAK_FRAC * len(obs))))
    hi = np.argsort(obs)[-n_hi:]
    o_hi, p_hi = obs[hi].sum(), pred[hi].sum()
    out["fhv"] = 100.0 * (p_hi - o_hi) / (o_hi + EPS)
    out["peak_bias"] = (pred[hi].mean() - obs[hi].mean()) / (obs[hi].mean() + EPS)
    out["peak_timing_days"] = float(abs(int(np.argmax(pred)) - int(np.argmax(obs))))

    # Low flows (lowest 30% of observed, log space; Yilmaz FLV)
    n_lo = max(1, int(0.30 * len(obs)))
    lo_idx = np.argsort(obs)[:n_lo]
    log_p = np.log(np.maximum(pred[lo_idx], 1e-3))
    log_o = np.log(np.maximum(obs[lo_idx], 1e-3))
    out["flv"] = -100.0 * (
        (log_p - log_p.min()).sum() - (log_o - log_o.min()).sum()
    ) / ((log_o - log_o.min()).sum() + EPS)

    # FDC percentiles and mid-slope (Yilmaz FMS, 0.2-0.7 exceedance)
    qo = np.quantile(obs, [0.05, 0.20, 0.50, 0.70, 0.95])
    qp = np.quantile(pred, [0.05, 0.20, 0.50, 0.70, 0.95])
    out["q5_bias"] = (qp[0] - qo[0]) / (qo[0] + EPS)
    out["q50_bias"] = (qp[2] - qo[2]) / (qo[2] + EPS)
    out["q95_bias"] = (qp[4] - qo[4]) / (qo[4] + EPS)
    sl_o = (np.log(max(qo[1], EPS)) - np.log(max(qo[3], EPS)))
    sl_p = (np.log(max(qp[1], EPS)) - np.log(max(qp[3], EPS)))
    out["fms"] = 100.0 * (sl_p - sl_o) / (abs(sl_o) + EPS)

    # Baseflow-ish: 7-day minimum / mean
    if len(obs) >= 7:
        k = np.ones(7) / 7.0
        o7 = np.convolve(obs, k, mode="valid")
        p7 = np.convolve(pred, k, mode="valid")
        out["bfi_obs"] = float(np.min(o7) / (o_mean + EPS))
        out["bfi_pred"] = float(np.min(p7) / (p_mean + EPS))
        out["bfi_bias"] = out["bfi_pred"] - out["bfi_obs"]
    else:
        out["bfi_obs"] = out["bfi_pred"] = out["bfi_bias"] = np.nan

    # High-pulse days (above 9 x median obs)
    thr = 9.0 * np.median(obs)
    out["high_pulse_obs"] = float(np.mean(obs > thr))
    out["high_pulse_pred"] = float(np.mean(pred > thr))
    out["high_pulse_bias"] = out["high_pulse_pred"] - out["high_pulse_obs"]

    out["com_obs"] = out["com_pred"] = out["com_bias_days"] = np.nan
    out["monthly_r"] = np.nan
    if dates is not None:
        d = pd.to_datetime(np.asarray(dates))
        if len(d) == len(obs) and len(obs) >= 60:
            doy = d.dayofyear.to_numpy()
            w = np.maximum(obs, 0)
            out["com_obs"] = float(np.sum(doy * w) / (w.sum() + EPS))
            w2 = np.maximum(pred, 0)
            out["com_pred"] = float(np.sum(doy * w2) / (w2.sum() + EPS))
            out["com_bias_days"] = out["com_pred"] - out["com_obs"]
            g = pd.DataFrame({"m": d.month, "o": obs, "p": pred}).groupby("m").mean()
            if len(g) >= 6 and g["o"].std() > EPS and g["p"].std() > EPS:
                out["monthly_r"] = float(np.corrcoef(g["o"], g["p"])[0, 1])
    return out


def ensemble_pit(obs, ens):
    """Empirical PIT: fraction of members <= observation."""
    obs = np.asarray(obs, dtype=np.float64)
    ens = np.asarray(ens, dtype=np.float64)
    return np.mean(ens <= obs[:, None], axis=1)


def ensemble_crps(obs, ens):
    """Standard ensemble CRPS, one value per day.

    CRPS = E|X - y| - 0.5 E|X - X'|, with X, X' i.i.d. from the
    discrete ensemble (Gneiting and Raftery 2007).
    """
    obs = np.asarray(obs, dtype=np.float64)
    ens = np.asarray(ens, dtype=np.float64)
    n = ens.shape[1]
    term1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    order = np.sort(ens, axis=1)
    coef = (2.0 * np.arange(1, n + 1) - n - 1.0)
    term2 = (2.0 / (n * n)) * np.sum(order * coef, axis=1)
    return term1 - 0.5 * term2


def ensemble_prob_metrics(obs, ens, peak_frac=PEAK_FRAC):
    """PIT coverage, CRPS, and peak-day bracketing from N members."""
    obs = np.asarray(obs, dtype=np.float64)
    ens = np.asarray(ens, dtype=np.float64)
    m = np.isfinite(obs) & np.all(np.isfinite(ens), axis=1) & (obs >= 0)
    obs, ens = obs[m], ens[m]
    out = dict(cov90=np.nan, cov95=np.nan, cov99=np.nan,
               crps=np.nan, crps_norm=np.nan,
               peak_cov95=np.nan, peak_cov99=np.nan, n_ens=int(len(obs)))
    if len(obs) < MIN_N:
        return out
    pit = ensemble_pit(obs, ens)
    crps = ensemble_crps(obs, ens)
    q90 = np.percentile(ens, 90, axis=1)
    q95 = np.percentile(ens, 95, axis=1)
    q99 = np.percentile(ens, 99, axis=1)
    out["cov90"] = float(np.mean(obs <= q90))
    out["cov95"] = float(np.mean(obs <= q95))
    out["cov99"] = float(np.mean(obs <= q99))
    out["crps"] = float(np.mean(crps))
    out["crps_norm"] = float(np.mean(crps) / (np.mean(obs) + EPS))
    n_hi = max(1, int(round(peak_frac * len(obs))))
    top = np.argsort(obs)[-n_hi:]
    out["peak_cov95"] = float(np.mean(obs[top] <= q95[top]))
    out["peak_cov99"] = float(np.mean(obs[top] <= q99[top]))
    out["median_pit"] = float(np.median(pit))
    return out


def summarize(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=[np.number])
    out = {"n_basins": int(len(df))}
    for c in num.columns:
        if c in ("comid", "node", "n_valid"):
            continue
        out[f"median_{c}"] = float(np.nanmedian(num[c]))
    if "kge" in df.columns:
        out["kge_gt_0"] = int((df["kge"] > 0).sum())
        out["kge_gt_0.5"] = int((df["kge"] > 0.5).sum())
        out["pct_kge_gt_0.5"] = float((df["kge"] > 0.5).mean())
    return out

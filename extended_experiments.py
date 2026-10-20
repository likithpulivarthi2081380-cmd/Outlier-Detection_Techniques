#!/usr/bin/env python3
"""
Extended experiments for the IEEE Access journal version.

Usage
-----
    python extended_experiments.py --csv train.csv --out tables --n-dev 100000 --seeds 5
    python extended_experiments.py --synthetic --out tables      # smoke test, no dataset needed

What it produces (LaTeX tables in --out, picked up automatically by access.tex):
    ext_detectors.tex   flag rate, runtime, plausibility precision/recall for 7 detectors x 2 feature spaces
    ext_injection.tex   AUROC / AP / precision@k / recall@k on data with INJECTED anomalies (true labels)
    ext_downstream.tex  downstream regression (RMSE, MAE, median AE, RMSLE) mean +- std over repeated splits,
                        evaluated on ONE common held-out set for every detector
    ext_fulllof.tex     LOF on the full dataset (kd-tree) with wall-clock time
Nothing here is invented: every number in the manuscript tables comes from running this file.
"""
import argparse, os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
FEATS = ["trip_duration", "trip_distance_km", "speed_kmph"]


# ----------------------------------------------------------------- data
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0  # same constant as the original notebook so counts reproduce
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_real(path):
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    df["trip_distance_km"] = haversine_km(df.pickup_latitude, df.pickup_longitude,
                                          df.dropoff_latitude, df.dropoff_longitude)
    df["speed_kmph"] = df.trip_distance_km / (df.trip_duration / 3600.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)   # removes duration == 0 rows
    df["pickup_hour"] = df.pickup_datetime.dt.hour
    df["pickup_dow"] = df.pickup_datetime.dt.dayofweek
    return df.reset_index(drop=True)


def make_synthetic(n=60000, seed=0):
    """Schema-compatible fake data (log-normal durations + a few gross errors). Smoke-test only."""
    rng = np.random.default_rng(seed)
    dist = rng.lognormal(0.7, 0.8, n)
    speed = np.clip(rng.normal(14, 5, n), 3, 45)
    dur = dist / speed * 3600
    k = rng.choice(n, int(0.01 * n), replace=False)
    dur[k[: len(k) // 3]] = rng.uniform(1, 9, len(k) // 3)
    dist[k[len(k) // 3: 2 * len(k) // 3]] = 0.0
    dur[k[2 * len(k) // 3:]] *= rng.uniform(30, 400, len(k) - 2 * len(k) // 3)
    df = pd.DataFrame({"trip_duration": dur, "trip_distance_km": dist,
                       "passenger_count": rng.integers(1, 5, n),
                       "pickup_hour": rng.integers(0, 24, n), "pickup_dow": rng.integers(0, 7, n)})
    df["speed_kmph"] = df.trip_distance_km / (df.trip_duration / 3600.0)
    return df


def plausibility_mask(df):
    """Rule-based reference set (same three rules as the conference paper)."""
    return ((df.speed_kmph > 120) | (df.trip_duration < 10) | (df.trip_distance_km < 0.01)).to_numpy()


# ----------------------------------------------------------------- feature spaces
def space(df, kind):
    X = df[FEATS].to_numpy(float)
    if kind == "raw":
        return X
    if kind == "log-robust":
        return RobustScaler().fit_transform(np.log1p(X))
    raise ValueError(kind)


# ----------------------------------------------------------------- detectors: each returns an anomaly SCORE (higher = more anomalous)
def sc_zscore(X, fit=None):
    f = X if fit is None else fit
    mu, sd = f.mean(0), f.std(0) + 1e-12
    return np.abs((X - mu) / sd).max(1)


def sc_iqr(X, fit=None):
    f = X if fit is None else fit
    q1, q3 = np.percentile(f, [25, 75], axis=0)
    iqr = q3 - q1 + 1e-12
    d = np.maximum((q1 - X) / iqr, (X - q3) / iqr)       # >1.5 means outside Tukey fence
    return d.max(1)


def sc_mad(X, fit=None):
    f = X if fit is None else fit
    med = np.median(f, 0)
    mad = np.median(np.abs(f - med), 0) * 1.4826 + 1e-12
    return np.abs((X - med) / mad).max(1)                # modified z-score, Leys et al.


def sc_mcd(X, fit=None, seed=0):
    f = X if fit is None else fit
    sub = f if len(f) <= 50000 else f[np.random.default_rng(seed).choice(len(f), 50000, replace=False)]
    m = MinCovDet(random_state=seed).fit(sub)
    return m.mahalanobis(X)


def sc_iforest(X, fit=None, seed=42):
    f = X if fit is None else fit
    return -IsolationForest(random_state=seed, n_jobs=-1).fit(f).score_samples(X)


def sc_lof(X, fit=None, k=20):
    # algorithm='kd_tree' makes full-data LOF feasible in 3-D; novelty=False so fit_predict semantics
    return -LocalOutlierFactor(n_neighbors=k, algorithm="kd_tree", n_jobs=-1).fit(X).negative_outlier_factor_


SCORERS = {"Z-score": sc_zscore, "IQR": sc_iqr, "MAD": sc_mad, "MCD": sc_mcd,
           "Isolation Forest": sc_iforest, "LOF": sc_lof}
# Native thresholds for the three classical rules; contamination-based rank rule for the rest
NATIVE_T = {"Z-score": 3.0, "IQR": 1.5, "MAD": 3.5}


def flag(name, score, contamination=0.05):
    if name in NATIVE_T:
        return score > NATIVE_T[name]
    if name == "MCD":
        return score > chi2.ppf(0.975, df=3)                 # classical chi-square cutoff
    return score > np.quantile(score, 1 - contamination)     # IF / LOF: fixed 5 % (strict, same rule as sklearn's predict)


# ----------------------------------------------------------------- latex helpers
def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


def _esc(v):
    return v.replace("\\_", "_").replace("_", "\\_").replace("&", "\\&").replace("#", "\\#") if isinstance(v, str) else v


def tex_table(df, caption, label, fmt, wide=False):
    cols = "l" + "c" * (df.shape[1] - 1)
    env = "table*" if wide else "table"
    s = ["\\begin{%s}[!t]" % env, "\\caption{%s}" % caption, "\\label{%s}" % label, "\\centering",
         "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize", "\\begin{tabular}{%s}" % cols, "\\hline",
         " & ".join(df.columns) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        s.append(" & ".join(_esc(v) if isinstance(v, str) else fmt(v) for v in r.values) + " \\\\")
    s += ["\\hline", "\\end{tabular}", "\\end{%s}" % env]
    return "\n".join(s)


# ----------------------------------------------------------------- experiments
def exp_detectors(dev, out):
    ref = plausibility_mask(dev)
    rows = []
    for kind in ["raw", "log-robust"]:
        X = space(dev, kind)
        for name, fn in SCORERS.items():
            t0 = time.perf_counter(); s = fn(X); rt = time.perf_counter() - t0
            m = flag(name, s)
            tp = int((m & ref).sum())
            rows.append({"Space": kind, "Detector": name, "Flagged (\\%)": 100 * m.mean(), "Time (s)": rt,
                         "Prec. (\\%)": 100 * tp / max(m.sum(), 1), "Recall (\\%)": 100 * tp / max(ref.sum(), 1)})
    df = pd.DataFrame(rows)
    f = lambda v: f"{v:.2f}"
    w(f"{out}/ext_detectors.tex", tex_table(
        df, "Extended detector comparison on the shared development sample. Precision and recall are measured "
            "against the rule-based plausibility set ($n=%d$ records) and are proxies, not ground truth." % ref.sum(),
        "tab:ext_detectors", f))
    print(df.to_string(index=False))


def inject(df, rate, rng):
    """Inject four documented anomaly types into a plausible subset; returns df, labels, type array."""
    d = df.copy()
    for c in FEATS:
        d[c] = d[c].astype(float)      # real data stores duration as int64
    n = len(d); k = int(rate * n)
    idx = rng.choice(n, k, replace=False)
    t = np.array(["none"] * n, dtype=object)
    q = np.array_split(idx, 4)
    d.loc[d.index[q[0]], "trip_duration"] = rng.uniform(1, 9, len(q[0]))                       # too-short
    d.loc[d.index[q[1]], "trip_duration"] *= rng.uniform(20, 200, len(q[1]))                   # stuck meter
    d.loc[d.index[q[2]], "trip_distance_km"] *= rng.uniform(15, 80, len(q[2]))                 # GPS jump
    d.loc[d.index[q[3]], "trip_distance_km"] = 0.0                                             # zero-distance
    for i, nm in zip(q, ["short", "long", "jump", "zero"]):
        t[i] = nm
    d["speed_kmph"] = d.trip_distance_km / (d.trip_duration / 3600.0)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.zeros(n, int); y[idx] = 1
    return d, y, t


def exp_injection(dev, out, seeds, rate=0.01):
    clean = dev[~plausibility_mask(dev)].reset_index(drop=True)
    recs = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        d, y, _ = inject(clean, rate, rng)
        for kind in ["raw", "log-robust"]:
            X = space(d, kind)
            for name, fn in SCORERS.items():
                s = fn(X)
                k = int(y.sum()); top = np.argsort(-s)[:k]
                p = y[top].mean()
                recs.append({"Space": kind, "Detector": name, "AUROC": roc_auc_score(y, s),
                             "AP": average_precision_score(y, s), "P@k": p, "seed": sd})
    r = pd.DataFrame(recs)
    g = r.groupby(["Space", "Detector"]).agg(["mean", "std"]).drop(columns="seed").reset_index()
    g.columns = ["Space", "Detector"] + [f"{a} {b}" for a, b in g.columns[2:]]
    disp = pd.DataFrame({"Space": g["Space"], "Detector": g["Detector"]})
    for m in ["AUROC", "AP", "P@k"]:
        disp[m] = [f"{a:.3f}$\\pm${b:.3f}" for a, b in zip(g[f"{m} mean"], g[f"{m} std"])]
    w(f"{out}/ext_injection.tex", tex_table(
        disp, "Detection quality with injected anomalies (%.0f\\%% of records; four anomaly types; mean$\\pm$std over %d seeds). "
              "Here ground truth is known by construction." % (rate * 100, seeds), "tab:ext_injection", lambda v: str(v), wide=True))
    print(disp.to_string(index=False))


def clean_pass(name, X_tr, kind, tr_df):
    """Fit on train only, return keep-mask for train rows."""
    s = SCORERS[name](X_tr)
    return ~flag(name, s)


def metrics(y, p):
    p = np.clip(p, 1, None)
    return dict(RMSE=mean_squared_error(y, p) ** 0.5, MAE=mean_absolute_error(y, p),
                MedAE=median_absolute_error(y, p), RMSLE=np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


def exp_downstream(dev, out, seeds):
    feats = ["trip_distance_km", "passenger_count", "pickup_hour", "pickup_dow"]
    recs = []
    for sd in range(seeds):
        tr, te = train_test_split(dev, test_size=0.2, random_state=42 + sd)
        te_ref = te[~plausibility_mask(te)]                        # common 'rule-clean' evaluation set
        for kind in ["raw", "log-robust"]:
            Xtr = space(tr, kind)
            for name in ["None"] + list(SCORERS):
                keep = np.ones(len(tr), bool) if name == "None" else clean_pass(name, Xtr, kind, tr)
                lr = LinearRegression().fit(tr.loc[keep, feats], tr.loc[keep, "trip_duration"])
                for ev, dd in [("full", te), ("rule-clean", te_ref)]:
                    m = metrics(dd["trip_duration"].to_numpy(), lr.predict(dd[feats]))
                    recs.append({"Space": kind, "Detector": name, "Eval": ev, "kept": keep.mean(), **m})
    r = pd.DataFrame(recs)
    g = r.groupby(["Eval", "Space", "Detector"]).agg(["mean", "std"]).reset_index()
    rows = []
    for _, row in g.iterrows():
        rows.append({"Eval": row[("Eval", "")], "Space": row[("Space", "")], "Detector": row[("Detector", "")],
                     "Kept (\\%)": f"{100 * row[('kept', 'mean')]:.1f}",
                     **{m: f"{row[(m, 'mean')]:.1f}$\\pm${row[(m, 'std')]:.1f}" if m != "RMSLE"
                        else f"{row[(m, 'mean')]:.3f}$\\pm${row[(m, 'std')]:.3f}" for m in ["RMSE", "MAE", "MedAE", "RMSLE"]}})
    d = pd.DataFrame(rows)
    w(f"{out}/ext_downstream.tex", tex_table(
        d, "Downstream linear regression with every detector evaluated on the same held-out sets "
           "(full test partition, and test partition after rule-based filtering); mean$\\pm$std over %d splits. "
           "Errors in seconds." % seeds, "tab:ext_downstream", lambda v: str(v), wide=True))
    print(d.to_string(index=False))


def exp_full_lof(df, out, kinds=("raw", "log-robust")):
    ref = plausibility_mask(df); rows = []
    for kind in kinds:
        X = space(df, kind)
        t0 = time.perf_counter(); s = sc_lof(X); rt = time.perf_counter() - t0
        m = flag("LOF", s)
        rows.append({"Rows": f"{len(df):,}", "Space": kind, "Flagged (\\%)": f"{100 * m.mean():.2f}",
                     "Time (s)": f"{rt:.1f}", "Prec. (\\%)": f"{100 * (m & ref).sum() / max(m.sum(), 1):.2f}",
                     "Recall (\\%)": f"{100 * (m & ref).sum() / max(ref.sum(), 1):.2f}"})
    d = pd.DataFrame(rows)
    w(f"{out}/ext_fulllof.tex", tex_table(d, "LOF on the complete dataset using a kd-tree neighbor search. Precision and recall are against the "
                                          "rule-based reference set computed on the full data.", "tab:ext_fulllof", lambda v: str(v)))
    print(d.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="tables")
    ap.add_argument("--n-dev", type=int, default=100000); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--skip-full-lof", action="store_true")
    a = ap.parse_args()
    full = make_synthetic() if a.synthetic else load_real(a.csv)
    dev = full.sample(min(a.n_dev, len(full)), random_state=42).reset_index(drop=True)
    print(f"full={len(full):,}  dev={len(dev):,}  rule-implausible in dev={plausibility_mask(dev).sum()}")
    exp_detectors(dev, a.out)
    exp_injection(dev, a.out, a.seeds)
    exp_downstream(dev, a.out, a.seeds)
    if not a.skip_full_lof:
        exp_full_lof(full, a.out)

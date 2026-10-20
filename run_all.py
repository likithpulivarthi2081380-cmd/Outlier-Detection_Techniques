#!/usr/bin/env python3
"""
ONE-COMMAND RUN for the journal version.

    python run_all.py --csv data/train.csv --out tables            # everything (needs ~10-30 min)
    python run_all.py --csv data/train.csv --out tables --quick    # skips full-data runs (~5 min)
    python run_all.py --synthetic --out /tmp/t --quick             # smoke test only (fake data!)

Writes LaTeX tables into --out (access.tex includes them automatically) and prints a
CHECK block comparing the reproduced conference numbers with the values printed in the paper.
"""
import argparse, os, platform, sys, time, itertools
import numpy as np, pandas as pd
import extended_experiments as E
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PAPER = {"Z-score": 3602, "IQR": 12525, "Isolation Forest": 4999, "LOF": 5000}   # dev-sample counts in the paper


def jac(a, b):
    a, b = set(np.flatnonzero(a)), set(np.flatnonzero(b))
    return len(a & b) / max(len(a | b), 1)


def native_flags(dev):
    """The exact four detectors of the conference paper, raw features, same settings."""
    X = E.space(dev, "raw"); out = {}; rt = {}
    for n in ["Z-score", "IQR", "Isolation Forest", "LOF"]:
        t0 = time.perf_counter(); s = E.SCORERS[n](X); rt[n] = time.perf_counter() - t0
        out[n] = E.flag(n, s)
    return out, rt


def exp_setup(full_rows_raw, full, dev, out):
    import sklearn, scipy
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        ram = float("nan")
    rows = [("Rows in train.csv", f"{full_rows_raw:,}"),
            ("Rows after removing non-finite speed (zero duration)", f"{len(full):,}"),
            ("Rows removed", f"{full_rows_raw - len(full):,}"),
            ("Development sample", f"{len(dev):,}"),
            ("Rule-violating trips in sample ($|R|$)", f"{E.plausibility_mask(dev).sum()}"),
            ("CPU / cores", f"{platform.processor() or platform.machine()} / {os.cpu_count()}"),
            ("RAM (GiB)", f"{ram:.1f}"),
            ("Python", platform.python_version()),
            ("pandas / NumPy / SciPy", f"{pd.__version__} / {np.__version__} / {scipy.__version__}"),
            ("scikit-learn", sklearn.__version__)]
    df = pd.DataFrame(rows, columns=["Item", "Value"])
    E.w(f"{out}/ext_setup.tex", E.tex_table(df, "Data counts and computing environment.", "tab:ext_setup", str))
    print(df.to_string(index=False))


def exp_core_check(dev, out):
    fl, rt = native_flags(dev); ref = E.plausibility_mask(dev)
    print("\n=== CHECK: reproduced vs paper (development sample) ===")
    for n in PAPER:
        got = int(fl[n].sum()); flag = "OK " if got == PAPER[n] else "DIFF"
        hits = int((fl[n] & ref).sum())
        print(f"{flag} {n:17s} flagged={got:6d} (paper {PAPER[n]})  hits_in_R={hits:4d}  "
              f"precision={100*hits/max(got,1):.3f}%  recall={100*hits/max(ref.sum(),1):.1f}%  runtime={rt[n]:.3f}s")
    names = list(fl)
    J = pd.DataFrame([[jac(fl[a], fl[b]) for b in names] for a in names], index=names, columns=names)
    print("\nJaccard (paper: Z-IF 0.697, Z-IQR 0.288, IQR-IF 0.379, LOF-others 0.102-0.120):\n", J.round(3))
    return fl


def exp_fences(dev, out):
    X = dev[E.FEATS].to_numpy(float)
    q1, q3 = np.percentile(X, [25, 75], axis=0); iqr = q3 - q1
    mu, sd = X.mean(0), X.std(0)
    df = pd.DataFrame({"Feature": ["Duration (s)", "Distance (km)", "Speed (km/h)"],
                       "IQR lower": q1 - 1.5 * iqr, "IQR upper": q3 + 1.5 * iqr,
                       "Z lower": mu - 3 * sd, "Z upper": mu + 3 * sd})
    E.w(f"{out}/ext_fences.tex", E.tex_table(df, "Decision boundaries of the two classical rules on the development sample. "
        "A negative lower boundary means no positive value can be flagged as too small.", "tab:ext_fences", lambda v: f"{v:.2f}"))
    print("\n", df.round(2).to_string(index=False))


def exp_stability(dev, out):
    X = E.space(dev, "raw")
    seeds = [42, 7, 123]
    sets = {s: E.flag("Isolation Forest", E.sc_iforest(X, seed=s)) for s in seeds}
    pairs = list(itertools.combinations(seeds, 2))
    jif = [jac(sets[a], sets[b]) for a, b in pairs]
    ks = [10, 20, 50]
    lk = {k: E.flag("LOF", E.sc_lof(X, k=k)) for k in ks}
    rows = [{"Comparison": f"Isolation Forest, seed {a} vs {b}", "Flagged A": int(sets[a].sum()), "Flagged B": int(sets[b].sum()),
             "Jaccard": f"{jac(sets[a], sets[b]):.3f}"} for a, b in pairs]
    rows += [{"Comparison": f"LOF, k={a} vs k={b}", "Flagged A": int(lk[a].sum()), "Flagged B": int(lk[b].sum()),
              "Jaccard": f"{jac(lk[a], lk[b]):.3f}"} for a, b in itertools.combinations(ks, 2)]
    df = pd.DataFrame(rows)
    E.w(f"{out}/ext_stability.tex", E.tex_table(df, "Stability of the flagged sets (membership overlap, not just counts) across random seeds and neighborhood sizes.",
                                               "tab:ext_stability", str))
    print("\n", df.to_string(index=False), f"\nmean IF seed Jaccard = {np.mean(jif):.3f}")


def exp_twostage(dev, out):
    ref = E.plausibility_mask(dev); Xr = E.space(dev, "raw"); Xl = E.space(dev, "log-robust")
    rows = []
    def add(name, m):
        rows.append({"Pipeline": name, "Flagged (\\%)": f"{100*m.mean():.2f}", "Prec. (\\%)": f"{100*(m&ref).sum()/max(m.sum(),1):.2f}",
                     "Recall (\\%)": f"{100*(m&ref).sum()/max(ref.sum(),1):.2f}"})
    s1 = E.flag("IQR", E.sc_iqr(Xr)); add("IQR only", s1)
    rest = ~s1; m2 = np.zeros(len(dev), bool)
    sc = E.sc_lof(Xl[rest]); m2[np.flatnonzero(rest)] = E.flag("LOF", sc)
    add("LOF only (log-robust)", E.flag("LOF", E.sc_lof(Xl)))
    add("IQR, then LOF on remainder", s1 | m2)
    s1b = E.flag("Isolation Forest", E.sc_iforest(Xr)); rest = ~s1b; m3 = np.zeros(len(dev), bool)
    m3[np.flatnonzero(rest)] = E.flag("LOF", E.sc_lof(Xl[rest]))
    add("Isolation Forest only", s1b); add("Isolation Forest, then LOF on remainder", s1b | m3)
    df = pd.DataFrame(rows)
    E.w(f"{out}/ext_twostage.tex", E.tex_table(df, "Two-stage pipelines against the plausibility reference set (development sample).", "tab:ext_twostage", str))
    print("\n", df.to_string(index=False))


def exp_original_protocol(dev, out, fl):
    """Reproduces conference Table VIII exactly: the whole sample is cleaned FIRST, split AFTER (so every detector is
    scored on a different, filtered test set, and detectors saw the test rows). Predictors as listed in the paper."""
    y = dev["trip_duration"]; feats = ["trip_distance_km", "passenger_count", "pickup_hour", "pickup_dow"]
    def r(mask):
        a, b, c, d = train_test_split(dev.loc[mask, feats], y[mask], test_size=0.2, random_state=42)
        return mean_squared_error(d, LinearRegression().fit(a, c).predict(b)) ** 0.5
    rows = [("No removal (baseline)", r(np.ones(len(dev), bool)))] + [(f"{n} removed", r(~fl[n])) for n in fl]
    df = pd.DataFrame(rows, columns=["Condition", "RMSE (s)"])
    E.w(f"{out}/ext_protocol.tex", E.tex_table(df, "Regression RMSE under the conference-version protocol (outliers removed from the whole sample before the split, "
        "so each row is scored on a different filtered test set). These values are not comparable across rows.", "tab:ext_protocol", lambda v: f"{v:.3f}"))
    paper = [3199.692, 382.135, 289.069, 333.596, 3036.696]
    print("\n=== CHECK: conference Table VIII (clean-before-split protocol) ===")
    for (n, v), pv in zip(rows, paper):
        print(f"{'OK  ' if abs(v-pv) < 0.01 else 'DIFF'} {n:28s} got {v:10.3f}   paper {pv:10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="tables")
    ap.add_argument("--n-dev", type=int, default=100000); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        full = E.make_synthetic(); raw_rows = len(full)
    else:
        raw_rows = len(pd.read_csv(a.csv, usecols=["id"])); full = E.load_real(a.csv)
    dev = full.sample(min(a.n_dev, len(full)), random_state=42).reset_index(drop=True)
    exp_setup(raw_rows, full, dev, a.out)
    fl = exp_core_check(dev, a.out)
    exp_fences(dev, a.out); exp_stability(dev, a.out); exp_twostage(dev, a.out); exp_original_protocol(dev, a.out, fl)
    E.exp_detectors(dev, a.out); E.exp_injection(dev, a.out, a.seeds); E.exp_downstream(dev, a.out, a.seeds)
    if not a.quick:
        E.exp_full_lof(full, a.out)
        X = full[E.FEATS]; rows = []
        for n in ["Z-score", "IQR", "Isolation Forest"]:
            t0 = time.perf_counter(); m = E.flag(n, E.SCORERS[n](E.space(full, "raw"))); rows.append((n, len(full), 100*m.mean(), time.perf_counter()-t0))
        print("\n=== Full-data Table V (LOF 300k row is in the notebook) ===")
        for r in rows: print(f"{r[0]:17s} rows={r[1]:,} flagged={r[2]:.3f}%  runtime={r[3]:.2f}s   (paper: Z 2.922/0.160, IQR 12.514/0.200, IF 5.000/18.552)")
    print("\nDONE. Copy the", a.out, "folder next to access.tex and run: latexmk -g -pdf access.tex")

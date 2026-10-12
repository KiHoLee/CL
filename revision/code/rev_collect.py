# =========================================================
# rev_collect.py — summarize data/*.json (one file per run) into
# data/rev_summary.csv and print the tables used in the manuscript and
# the response letter. Usage: python rev_collect.py [data_dir]
# =========================================================
import csv, glob, json, os, sys
import numpy as np

from rev_lib import DATA, EVAL_SNRS

d = sys.argv[1] if len(sys.argv) > 1 else DATA
runs = {}
for f in sorted(glob.glob(os.path.join(d, "*_s*.json"))):
    R = json.load(open(f))
    name = R["cfg"]["name"]
    runs.setdefault(name, []).append(R)

rows = []
for name, Rs in runs.items():
    for R in Rs:
        for U, ev in R["eval"].items():
            for s in EVAL_SNRS:
                e = ev[str(s)]
                rows.append(dict(name=name, seed=R["cfg"]["seed"],
                                 scheme=R["cfg"]["scheme"], U_train=R["cfg"]["U"],
                                 U_eval=int(U), K=R["cfg"]["K"],
                                 fading=R["cfg"]["fading"], snr=s,
                                 cos=e["cos"], mse=e["mse"],
                                 worst=e["cos_worst_user"],
                                 frame_std=e["cos_frame_std"],
                                 n_frames=e["n_frames"],
                                 probe=e.get("probe_acc", ""),
                                 train_s=R["train_s"], params=R["params"]))
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "rev_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {os.path.join(d, 'rev_summary.csv')}\n")


def agg(name, U=None, snr=10, key="cos"):
    vals = []
    for R in runs.get(name, []):
        U_ = str(U) if U is not None else str(R["cfg"]["U"])
        if U_ in R["eval"]:
            vals.append(R["eval"][U_][str(snr)][key])
    return (np.mean(vals), np.std(vals, ddof=0), len(vals)) if vals else (np.nan, np.nan, 0)


# ---- per-user CosSim table: every run, every SNR (mean over seeds) ----
print("Per-user CosSim (mean over seeds; n = seeds) at SNR 0..30")
print(f"{'run':28s} {'U':>3s} " + " ".join(f"{s:>7d}" for s in EVAL_SNRS) + "   n")
for name, Rs in runs.items():
    Us = sorted({int(u) for R in Rs for u in R["eval"]})
    for U in Us:
        m = [agg(name, U, s)[0] for s in EVAL_SNRS]
        n = agg(name, U, 10)[2]
        print(f"{name:28s} {U:3d} " + " ".join(f"{v:7.4f}" for v in m) + f"   {n}")
print()

# ---- seed spread at 10 and 20 dB ----
print("Seed spread (mean +- std) at 10 / 20 dB")
for name, Rs in runs.items():
    if len(Rs) < 2:
        continue
    Us = sorted({int(u) for R in Rs for u in R["eval"]})
    for U in Us:
        a, b = agg(name, U, 10), agg(name, U, 20)
        print(f"{name:28s} U={U}: {a[0]:.4f}+-{a[1]:.4f} / {b[0]:.4f}+-{b[1]:.4f} (n={a[2]})")
print()

# ---- probe accuracies ----
print("Probe accuracy at 5 / 10 / 20 dB (runs with --probe)")
for name, Rs in runs.items():
    for R in Rs:
        for U, ev in R["eval"].items():
            if "probe_acc" in ev["10"]:
                print(f"{name:28s} s{R['cfg']['seed']} U={U}: "
                      + " ".join(f"{ev[str(s)]['probe_acc']:.3f}" for s in (5, 10, 20))
                      + f"  clean={R.get('probe_clean_acc', float('nan')):.3f}")
print()

# ---- robustness blocks ----
print("Robustness (CosSim at 10 / 20 dB)")
for name, Rs in runs.items():
    for R in Rs:
        if "robust" not in R:
            continue
        print(f"{name} s{R['cfg']['seed']}:")
        for k, v in R["robust"].items():
            if isinstance(v["10"], dict):
                print(f"   {k:14s} mean {v['10']['cos']:.4f}/{v['20']['cos']:.4f}"
                      f"  worst {v['10']['worst']:.4f}/{v['20']['worst']:.4f}")
            else:
                print(f"   {k:14s} {v['10']:.4f}/{v['20']:.4f}")
print()

# ---- mask statistics, latency, train time ----
for name, Rs in runs.items():
    for R in Rs:
        ms = R.get("mask_stats")
        extra = f" lat={R['latency_ms']:.3f}ms" if "latency_ms" in R else ""
        if ms:
            print(f"{name:28s} s{R['cfg']['seed']} mask|corr| max {ms['max_abs_offdiag']:.4f} "
                  f"mean {ms['mean_abs_offdiag']:.4f} params {R['params']} "
                  f"train {R['train_s']/60:.1f} min{extra}")

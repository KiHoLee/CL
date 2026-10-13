# =========================================================
# rev_replot.py — regenerate the result figures of the revised letter
# from data/*.json (never from a live run) and print the numbers quoted
# in the text. Geometry rule (writing standard 9.3): one canvas and one
# 8:6 axes box for every result figure, no tight bounding box. The save
# helper asserts that no axis label is clipped and that no curve point
# lies inside the legend box (standard 12.9).
#
#   Fig. 2  fig_cl/fig2_snr_rev.pdf   per-user CosSim vs SNR at U=4
#   Fig. 3  fig_cl/fig3_load_rev.pdf  aggregate fidelity per block vs U at 10 dB
#
# Usage: python rev_replot.py [data_dir]
# =========================================================
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "fig_cl")
OLD = os.path.join(HERE, "fig_cl")        # ToDMA results of the submission
SNRS = [0, 5, 10, 15, 20, 25, 30]

plt.rcParams.update({"font.size": 20, "axes.linewidth": 1.2})
FIGSIZE = (7.5, 5.55)
AX_RECT = [0.155, 0.14, 0.76, 0.7804]       # 6.0 x 4.5 in axes box (8:6)
MS = 9

# one style per scheme, every figure (standard 9.1)
STYLE = {
    "prop":   dict(label="Proposed", color="#d62728", marker="v", ls="-"),
    "agn":    dict(label="Proposed, one model for all $U$", color="#ff7f0e", marker="o", ls="-"),
    "ods":    dict(label="ODS", color="#1a1a1a", marker="o", ls="--"),
    "static": dict(label="Static orthogonal ($K$ blocks)", color="#7f7f7f", marker="s", ls="--"),
    "fixed6": dict(label="Fixed six-way partition", color="#bdbdbd", marker="D", ls="--"),
    "rand":   dict(label="Random-projection mask", color="#984ea3", marker="s", ls="-."),
    "joint":  dict(label="Training without MAML", color="#8c564b", marker="x", ls=(0, (5, 2))),
    "todma":  dict(label="ToDMA $24\\times128$", color="#4393c3", marker="^", ls=":"),
}
ORDER = ["prop", "agn", "ods", "static", "fixed6", "rand", "joint", "todma"]


def load(name):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, f"{name}_s[0-9].json")))]
    assert runs, name
    return runs


def curve(name, U=None, key="cos"):
    """mean and std over seeds of the per-user CosSim at every SNR."""
    runs = load(name)
    vals = []
    for R in runs:
        U_ = str(U if U is not None else R["cfg"]["U"])
        vals.append([R["eval"][U_][str(s)][key] for s in SNRS])
    v = np.array(vals)
    return v.mean(0), v.std(0), len(runs)


def one_panel():
    fig = plt.figure(figsize=FIGSIZE)
    return fig, fig.add_axes(AX_RECT)


def save(fig, ax, path, legend=None):
    """Assert label visibility and legend clearance on the rendered figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    for lab in (ax.xaxis.label, ax.yaxis.label):
        bb = lab.get_window_extent(r)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, f"clipped label in {path}"
    if legend is not None:
        lb = legend.get_window_extent(r).expanded(1.0, 1.0)
        pad = MS / 2 * fig.dpi / 72 + 2
        lb = matplotlib.transforms.Bbox.from_extents(lb.x0 - pad, lb.y0 - pad, lb.x1 + pad, lb.y1 + pad)
        assert lb.x0 >= 0 and lb.y0 >= 0 and lb.x1 <= W and lb.y1 <= H, f"legend leaves canvas in {path}"
        for line in ax.get_lines():
            xy = ax.transData.transform(np.column_stack(line.get_data()))
            inside = [(x, y) for x, y in xy if lb.x0 <= x <= lb.x1 and lb.y0 <= y <= lb.y1]
            assert not inside, f"legend covers a curve point of '{line.get_label()}' in {path}"
        from matplotlib.container import BarContainer
        for c in getattr(ax, "containers", []):
            if not isinstance(c, BarContainer):
                continue
            for patch in c:
                bb = patch.get_window_extent(r)
                assert not (bb.x1 > lb.x0 and bb.x0 < lb.x1 and bb.y1 > lb.y0 and bb.y0 < lb.y1), f"legend covers a bar in {path}"
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".pdf", ".png"), dpi=120)
    plt.close(fig)
    print("saved", path)


# ---------------- numbers for the text ----------------
def report():
    print("\n=== per-user CosSim (mean over seeds) at 0/10/20/30 dB ===")
    for name in sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(os.path.join(DATA, "*_s*.json"))}):
        runs = load(name)
        for U in sorted({u for R in runs for u in R["eval"]}, key=int):
            m, s, n = curve(name, int(U))
            print(f"{name:26s} U={U}: " + " ".join(f"{m[i]:.4f}+-{s[i]:.4f}" for i in (0, 2, 4, 6)) + f" (n={n})")


# ---------------- Fig. 2: SNR sweep at U=4 ----------------
def fig2():
    fig, ax = one_panel()
    series = [("prop", "shared_U4", None), ("agn", "shared_agn", 4),
              ("ods", "ortho_U4", None), ("rand", "shared_U4_randmask", None),
              ("joint", "shared_U4_joint", None)]
    for key, name, U in series:
        try:
            m, s, n = curve(name, U)
        except AssertionError:
            print("  (missing)", name); continue
        st = STYLE[key]
        ax.errorbar(SNRS, m, yerr=s if n > 1 else None, label=st["label"], color=st["color"],
                    marker=st["marker"], linestyle=st["ls"], linewidth=2, markersize=MS,
                    capsize=3 if n > 1 else 0)
    try:
        R = json.load(open(os.path.join(OLD, "cl_results.json")))
        t = [R["todma_T24_L128"][str(s)]["cos"] for s in SNRS]
        st = STYLE["todma"]
        ax.plot(SNRS, t, label=st["label"], color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=2, markersize=MS)
    except FileNotFoundError:
        pass
    ax.set_xlabel("SNR (dB)", fontsize=22)
    ax.set_ylabel("Per-user CosSim", fontsize=22)
    ax.set_ylim([0.45, 1.0])
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    rank = {STYLE[k]["label"]: i for i, k in enumerate(ORDER)}
    hl = sorted(zip(handles, labels), key=lambda x: rank.get(x[1], 99))
    leg = ax.legend([h for h, _ in hl], [l for _, l in hl], fontsize=16.5, loc="lower right")
    save(fig, ax, os.path.join(FIG, "fig2_snr_rev.pdf"), leg)


# ---------------- Fig. 3: aggregate fidelity S(U) vs U at 10 dB ----------------
def fig3(snr=10):
    """Aggregate fidelity per block, U times the per-user CosSim; static
    allocation saturates beyond U=K since it serves at most K users."""
    fig, ax = one_panel()
    i = SNRS.index(snr)
    Us = list(range(1, 7))

    def series(key, name, perU):
        m, s = [], []
        for U in Us:
            try:
                mm, ss, n = curve(name.format(U=U), U if not perU else None)
            except AssertionError:
                mm, ss = np.full(7, np.nan), np.zeros(7)
            m.append(U * mm[i]); s.append(U * ss[i])
        st = STYLE[key]
        ax.errorbar(Us, m, yerr=s if any(s) else None, label=st["label"], color=st["color"],
                    marker=st["marker"], linestyle=st["ls"], linewidth=2, markersize=MS, capsize=3)

    series("prop", "shared_U{U}", True)
    series("agn", "shared_agn", False)
    series("ods", "ortho_U{U}", True)
    try:
        m4, _, _ = curve("ortho_U4")
        st = STYLE["static"]
        ax.plot(Us, [min(U, 4) * m4[i] for U in Us], label=st["label"], color=st["color"],
                marker=st["marker"], linestyle=st["ls"], linewidth=2, markersize=MS)
    except AssertionError:
        pass
    try:
        R = json.load(open(os.path.join(OLD, "cl_results.json")))
        RT = json.load(open(os.path.join(OLD, "cl_results_todma_u.json")))
        tod = [U * (RT[f"todma_U{U}_T24_L128"][str(snr)]["cos"] if U != 4
                    else R["todma_T24_L128"][str(snr)]["cos"]) for U in Us]
        st = STYLE["todma"]
        ax.plot(Us, tod, label=st["label"], color=st["color"], marker=st["marker"],
                linestyle=st["ls"], linewidth=2, markersize=MS)
    except FileNotFoundError:
        pass
    ax.set_xlabel("Number of users $U$", fontsize=22)
    ax.set_ylabel("Aggregate fidelity $S(U)$", fontsize=22)
    ax.set_xticks(Us)
    ax.set_ylim([0, 10.5])
    ax.set_yticks(np.arange(0, 6.1, 1))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    rank = {STYLE[k]["label"]: j for j, k in enumerate(ORDER)}
    hl = sorted(zip(handles, labels), key=lambda x: rank.get(x[1], 99))
    leg = ax.legend([h for h, _ in hl], [l for _, l in hl], fontsize=16.5, loc="upper left")
    save(fig, ax, os.path.join(FIG, "fig3_load_rev.pdf"), leg)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    report()
    fig2()
    fig3()

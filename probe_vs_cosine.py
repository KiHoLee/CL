# probe_vs_cosine.py - supplementary analysis for the letter.
#
# Scatter of downstream probe accuracy (AG News topic classification,
# linear probe trained on clean training-pool embeddings) against the
# cosine similarity of the recovered embeddings, across schemes and
# SNRs, with the Pearson correlation. Shows that the cosine metric used
# in the letter is consistent with downstream perception.
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.2})

with open("fig_cl/cl_results.json") as f:
    RJ = json.load(f)
with open("fig_cl/cl_results_maml.json") as f:
    RM = json.load(f)

SNRS = ["0", "5", "10", "15", "20", "25", "30"]

# (label, cos-source, acc-source, marker, color)
def sweep_pairs(entry, todma=False):
    cos, acc = [], []
    for s in SNRS:
        a = entry.get("probe_acc", {}).get(s)
        if a is None:
            continue
        c = entry[s]["cos"] if todma else entry["snr"][s]["cos"]
        cos.append(c)
        acc.append(a)
    return cos, acc


SCHEMES = [
    ("Proposed (MAML)", RM["mamlP_U4_K4"], False, "v", "#d62728"),
    ("Joint training [5]", RJ["prop_U4_K4"], False, "x", "#8c564b"),
    ("Random-projection mask (MAML)", RM["mamlR_U4_K4"], False, "s",
     "#984ea3"),
    ("Conventional orthogonal (MAML)", RM["mamlB_U1_K1"], False, "o",
     "#1a1a1a"),
    ("Conventional orthogonal (joint)", RJ["baseline_U1_K1"], False, "P",
     "#7f7f7f"),
    ("Random-projection mask (joint)", RJ["randmask_U4_K4"], False, "D",
     "#c994c7"),
    ("ToDMA 24x128", RJ["todma_T24_L128"], True, "^", "#4393c3"),
]

all_cos, all_acc = [], []
fig = plt.figure(figsize=(7.0, 5.4))
ax = fig.add_axes([0.12, 0.12, 0.83, 0.83])
for lab, entry, todma, mk, col in SCHEMES:
    cos, acc = sweep_pairs(entry, todma)
    ax.scatter(cos, acc, marker=mk, s=70, color=col, label=lab,
               zorder=3, alpha=0.9)
    all_cos += cos
    all_acc += acc

all_cos = np.array(all_cos)
all_acc = np.array(all_acc)
r = np.corrcoef(all_cos, all_acc)[0, 1]
b, a = np.polyfit(all_cos, all_acc, 1)
xg = np.linspace(all_cos.min(), all_cos.max(), 50)
ax.plot(xg, b * xg + a, color="#888888", linewidth=1.5, linestyle="--",
        zorder=2, label=f"Linear fit (Pearson $r$={r:.3f})")
clean = RM.get("probe_clean_acc", RJ.get("probe_clean_acc"))
ax.axhline(clean, color="#bbbbbb", linewidth=1.2, linestyle=":",
           zorder=1)
ax.text(all_cos.min(), clean + 0.004,
        f"Clean-embedding reference ({clean:.3f})", fontsize=11,
        color="#888888")
ax.set_xlabel("Cosine similarity of recovered embeddings", fontsize=15)
ax.set_ylabel("Downstream probe accuracy", fontsize=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10.5, loc="lower right")
fig.savefig("fig_cl/probe_vs_cosine.png", dpi=150)
fig.savefig("fig_cl/probe_vs_cosine.pdf", dpi=200)
print(f"Saved probe_vs_cosine.(png|pdf)  Pearson r = {r:.4f} "
      f"over {len(all_cos)} scheme-SNR points")

# Markdown table for the repository README
print("\n| Scheme | CosSim 5 dB | Acc 5 dB | CosSim 20 dB | Acc 20 dB |")
print("|---|---|---|---|---|")
for lab, entry, todma, _, _ in SCHEMES:
    def get(s):
        c = entry[s]["cos"] if todma else entry["snr"][s]["cos"]
        return c, entry.get("probe_acc", {}).get(s, float("nan"))
    c5, a5 = get("5")
    c20, a20 = get("20")
    print(f"| {lab} | {c5:.3f} | {a5:.3f} | {c20:.3f} | {a20:.3f} |")


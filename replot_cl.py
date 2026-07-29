# Plot CL-letter figures from fig_cl/cl_results*.json
#
# Geometry rule (paper_requirement): every result figure uses the same
# fixed canvas and the same 8:6 axes box, and no tight bounding box is
# applied at save time. Scheme names avoid the banned word "baseline".
#
# Figure set (single large graph per figure):
#   Fig. 2 (cl_fig_mux.pdf)   : one SNR sweep merging the load sweep
#                               (conventional + proposed U=1..4) and the
#                               matched-budget comparison (random mask,
#                               ToDMA x2) at U=4.
#   Fig. 3 (cl_fig_agg.pdf)   : aggregate fidelity bars for U=1..6 with
#                               per-user CosSim and the fully loaded
#                               orthogonal reference.
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.linewidth': 1.2})

with open("fig_cl/cl_results.json") as f:
    R = json.load(f)
# MAML-trained results (reported default protocol) overlay the joint
# runs: every reported transceiver key is remapped to its MAML twin.
with open("fig_cl/cl_results_maml.json") as f:
    R.update(json.load(f))
try:
    with open("fig_cl/cl_results_maml2.json") as f:
        R.update(json.load(f))
except FileNotFoundError:
    pass
KEYMAP = {
    "baseline_U1_K1": "mamlB_U1_K1",
    "prop_U1_K4": "mamlP_U1_K4", "prop_U2_K4": "mamlP_U2_K4",
    "prop_U3_K4": "mamlP_U3_K4", "prop_U4_K4": "mamlP_U4_K4",
    "prop_U5_K4": "mamlP_U5_K4", "prop_U6_K4": "mamlP_U6_K4",
    "randmask_U4_K4": "mamlR_U4_K4",
    "ksweep_U4_K1": "mamlK_U4_K1", "ksweep_U4_K2": "mamlK_U4_K2",
    "ksweep_U4_K8": "mamlK_U4_K8",
}
# Preserve the joint-trained runs before remapping: they appear in the
# figures as the joint-training ablation (same architecture, no MAML).
JOINT = {k: R[k] for k in list(KEYMAP.keys()) if k in R}
for old, new in KEYMAP.items():
    if new in R:
        R[old] = R[new]

SNRS = [0, 5, 10, 15, 20, 25, 30]

# Single-graph geometry shared by both result figures: canvas
# 7.5 x 5.55 in, axes box 6.0 x 4.5 in (exactly 8:6).
FIGSIZE = (7.5, 5.55)
AX_RECT = [0.105, 0.115, 0.77, 0.7804]


def one_panel():
    fig = plt.figure(figsize=FIGSIZE)
    return fig, fig.add_axes(AX_RECT)


def cos_curve(key):
    return [R[key]["snr"][str(s)]["cos"] for s in SNRS]


# ============================================================
# Fig 2: merged SNR sweep (load sweep + matched-budget comparison)
# ============================================================
fig, ax = one_panel()

curves = [
    ("prop_U1_K4", "Proposed $U$=1", "s", "-", "#1a9641", 2),
    ("prop_U2_K4", "Proposed $U$=2", "^", "-", "#2166ac", 2),
    ("prop_U3_K4", "Proposed $U$=3", "D", "-", "#d95f02", 2),
    ("prop_U4_K4", "Proposed $U$=4", "v", "-", "#d62728", 2.5),
    ("JOINT:prop_U4_K4", "Training w/o MAML [5]", "x",
     (0, (5, 2)), "#8c564b", 2),
    ("baseline_U1_K1", "Conventional orthogonal", "o", "--", "#1a1a1a", 2),
    ("randmask_U4_K4", "Random-projection mask", "s", "-.", "#984ea3", 2),
    ("todma_T24_L128", "ToDMA $24\\times128$", "^", ":",
     "#4393c3", 2),
    ("todma_T16_L192", "ToDMA $16\\times192$", "D", ":",
     "#92c5de", 2),
]
for key, lab, mk, ls, col, lw in curves:
    if key.startswith("todma"):
        vals = [R[key][str(s)]["cos"] for s in SNRS]
    elif key.startswith("JOINT:"):
        vals = [JOINT[key[6:]]["snr"][str(s)]["cos"] for s in SNRS]
    else:
        vals = cos_curve(key)
    ax.plot(SNRS, vals, marker=mk, linestyle=ls, color=col,
            linewidth=lw, markersize=8, label=lab)
ax.set_xlabel("SNR (dB)", fontsize=17)
ax.set_ylabel("Cosine Similarity", fontsize=17)
ax.set_ylim([0.45, 1.0])
ax.legend(fontsize=12.5, loc="lower right", ncol=1)
ax.grid(True, alpha=0.3)
fig.savefig("fig_cl/cl_fig_mux.pdf", dpi=200)
fig.savefig("fig_cl/cl_fig_mux.png", dpi=150)
plt.close(fig)
print("Saved cl_fig_mux.pdf")

# ============================================================
# Fig 3: aggregate fidelity across load (bars + per-user line)
# ============================================================
fig, ax2 = one_panel()

from matplotlib.patches import Patch

snr_show = "20"
conv_cos = R["baseline_U1_K1"]["snr"][snr_show]["cos"]
per_user = [R[f"prop_U{U}_K4"]["snr"][snr_show]["cos"] for U in range(1, 7)]
joint_pu = [JOINT[f"prop_U{U}_K4"]["snr"][snr_show]["cos"]
            for U in range(1, 7)]
thr = [U * c for U, c in zip(range(1, 7), per_user)]
thr_j = [U * c for U, c in zip(range(1, 7), joint_pu)]
xs = np.arange(1, 7)
colors = ["#1a9641", "#2166ac", "#d95f02", "#d62728",
          "#984ea3", "#666666"]
with open("fig_cl/cl_results_todma_u.json") as f:
    RT = json.load(f)
todma_pu = [RT[f"todma_U{U}_T24_L128"]["20"]["cos"] if U != 4
            else R["todma_T24_L128"]["20"]["cos"] for U in range(1, 7)]
thr_t = [U * c for U, c in zip(range(1, 7), todma_pu)]
ax2.bar([0], [conv_cos], width=0.55, color="#1a1a1a", alpha=0.85)
ax2.text(0, conv_cos + 0.08, f"{conv_cos:.2f}", ha="center",
         va="bottom", fontsize=12)
ax2.bar(xs - 0.27, thr, width=0.26, color=colors, alpha=0.9)
ax2.bar(xs, thr_j, width=0.26, color=colors, alpha=0.4,
        hatch="//", edgecolor="#555555", linewidth=0.5)
ax2.bar(xs + 0.27, thr_t, width=0.26, color="#4393c3", alpha=0.75,
        hatch="..", edgecolor="#1f5f8b", linewidth=0.5)
for x, val in zip(xs, thr):
    ax2.text(x - 0.27, val + 0.08, f"{val:.2f}", ha="center",
             va="bottom", fontsize=11)
# Fully loaded orthogonal aggregate (4 blocks x 768 uses = same budget)
ax2.axhline(4 * conv_cos, linestyle="-.", color="#555555", linewidth=2)
ax2.text(-0.45, 4 * conv_cos + 0.13, "Fully loaded orthogonal",
         fontsize=12.5, color="#555555")
ax2.set_xticks([0] + list(xs))
ax2.set_xticklabels(["Conv.\n$U$=1"] + [f"Multi.\n$U$={U}"
                    for U in range(1, 7)], fontsize=12)
ax2.set_ylabel(r"Aggregate fidelity ($U \!\cdot\! \mathrm{CosSim}$)",
               fontsize=16)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim([0, max(thr) * 1.22])
ax2r = ax2.twinx()
h_per, = ax2r.plot([0] + list(xs), [conv_cos] + per_user, marker="o",
                   color="#1a1a1a", linewidth=2, markersize=7,
                   linestyle="--", label="Per-user CosSim")
ax2r.set_ylabel("Per-user CosSim", fontsize=16)
ax2r.set_ylim([0.80, 1.0])
ax2r.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
handles = [Patch(facecolor="#888888", label="Proposed"),
           Patch(facecolor="#cccccc", hatch="//", edgecolor="#555555",
                 label="Training w/o MAML [5]"),
           Patch(facecolor="#4393c3", alpha=0.75, hatch="..",
                 edgecolor="#1f5f8b", label="ToDMA $24\\times128$"),
           h_per]
ax2r.legend(handles=handles, fontsize=10.5, loc="center left",
            bbox_to_anchor=(0.02, 0.44))
fig.savefig("fig_cl/cl_fig_agg.pdf", dpi=200)
fig.savefig("fig_cl/cl_fig_agg.png", dpi=150)
plt.close(fig)
print("Saved cl_fig_agg.pdf")

# ============================================================
# Print the numbers quoted in the letter
# ============================================================
print("\n===== NUMBERS FOR TEXT (MAML default, held-out) =====")
print("conv per-user@20:", round(conv_cos, 3),
      " fully loaded aggregate:", round(4 * conv_cos, 2))
for U in range(1, 7):
    v = R[f"prop_U{U}_K4"]["snr"]["20"]["cos"]
    print(f"U={U}: per-user {v:.3f} aggregate {U*v:.2f}")
print("overload ratio:",
      round(6 * R["prop_U6_K4"]["snr"]["20"]["cos"] / (4 * conv_cos), 2))
print("randmask@20:", round(R["randmask_U4_K4"]["snr"]["20"]["cos"], 3))
for s in ["0", "5", "15", "20", "30"]:
    print(f"todma24@{s}: {R['todma_T24_L128'][s]['cos']:.3f} "
          f"prop@{s}: {R['prop_U4_K4']['snr'][s]['cos']:.3f}")
for k in ["ksweep_U4_K1", "ksweep_U4_K2", "prop_U4_K4", "ksweep_U4_K8"]:
    print(k, "0dB:", round(R[k]["snr"]["0"]["cos"], 3),
          "20dB:", round(R[k]["snr"]["20"]["cos"], 3))
if "mamlD_U4_K4" in R:
    print("distil(MAML)@20:", round(R["mamlD_U4_K4"]["snr"]["20"]["cos"], 3))




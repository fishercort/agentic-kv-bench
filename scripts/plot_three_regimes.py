"""Render the three-regimes figure for the blog post from the committed results.

Two panels sharing the memory-pressure axis:
  A) miss reduction vs LRU: retirement (lifecycle) with leave-one-out spread, and
     Continuum (gap-aware protection), which HURTS.
  B) the economic policy's cost reduction vs retired-cache, with leave-one-out
     error bars that cross zero (not sign-robust).
Three regimes shaded. Numbers come from the committed results JSONs; the
leave-one-out points come from results/robustness-and-replication.json.

    uv run --with matplotlib python scripts/plot_three_regimes.py <out.png>
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, RED, ORANGE, GRAY = "#0072B2", "#D55E00", "#E69F00", "#7f7f7f"
CHURN, BAND, SLACK = "#f0f0f0", "#fde8dc", "#eef3f7"

root = pathlib.Path(__file__).resolve().parent.parent
sweep = json.loads((root / "results/retired-cache-pressure-sweep.json").read_text())
cont = json.loads((root / "results/ladder-06-continuum.json").read_text())

# retirement full-corpus saved-miss % (kept cells), and Continuum saved-miss %.
ret = [(r["cap_over_T"], r["saved_pct"] * 100)
       for r in sweep["sweep"] if not r.get("excluded_noise_floor")]
rx, ry = zip(*ret, strict=True)
con = [(r["cap_over_T"], r["saved_pct"] * 100)
       for r in cont["results"]["across_pressure_gap_60000"]["table"]]
cx, cy = zip(*con, strict=True)

# leave-one-out points (mean, min, max) from robustness-and-replication.json.
rob = json.loads((root / "results/robustness-and-replication.json").read_text())
rloo = rob["retirement_leave_one_out"]
ret_loo = [(0.49, rloo["cap_over_T_0.49"]), (0.53, rloo["cap_over_T_0.53"])]
eloo = rob["economic_leave_one_out_summary"]
econ_full = {0.45: 18.2, 0.49: 15.1}  # ladder-05 panel2 full-corpus, for the line
econ_loo = [(0.45, eloo["cap_over_T_0.45"]), (0.49, eloo["cap_over_T_0.49"])]

B_CHURN, B_SLACK = 0.42, 0.54
fig, (axA, axB) = plt.subplots(2, 1, figsize=(8, 7.6), sharex=True,
                               gridspec_kw={"hspace": 0.14})
for ax in (axA, axB):
    ax.axvspan(0.30, B_CHURN, color=CHURN, zorder=0)
    ax.axvspan(B_CHURN, B_SLACK, color=BAND, zorder=0)
    ax.axvspan(B_SLACK, 0.58, color=SLACK, zorder=0)
    ax.axhline(0, color=GRAY, lw=0.9, zorder=1)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0.30, 0.58)

# Panel A: lifecycle helps (sign-robust), protection hurts. vs LRU, miss reduction.
axA.plot(rx, ry, "-o", color=BLUE, lw=2, ms=6, zorder=3,
         label="retirement (lifecycle), full sample")
for x, cell in ret_loo:
    m, lo, hi = cell["mean"] * 100, cell["min"] * 100, cell["max"] * 100
    axA.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="#004c73",
                 ms=7, capsize=5, lw=1.6, zorder=4,
                 label="retirement, leave-one-out" if x == 0.49 else None)
axA.plot(cx, cy, "-s", color=RED, lw=2, ms=5, zorder=3, label="Continuum (protection)")
axA.set_ylabel("miss reduction vs LRU  (%)", fontsize=10.5)
axA.set_ylim(-70, 40)
axA.legend(loc="lower left", frameon=False, fontsize=8.5)
axA.annotate("protection HURTS", xy=(0.53, -61.9), xytext=(0.435, -55),
             fontsize=9, color=RED)

# Panel B: economic, cost reduction vs retired-cache, NOT sign-robust.
axB.plot(list(econ_full), list(econ_full.values()), "-o", color=ORANGE, lw=2, ms=6,
         zorder=3, label="economic, full sample")
for x, cell in econ_loo:
    m, lo, hi = cell["mean"] * 100, cell["min"] * 100, cell["max"] * 100
    lo_clip = max(lo, -24)
    axB.errorbar([x], [m], yerr=[[m - lo_clip], [hi - m]], fmt="D", color="#8a5a00",
                 ms=7, capsize=5, lw=1.6, zorder=4,
                 label="economic, leave-one-out" if x == 0.45 else None)
    if lo < -24:
        axB.annotate(f"one drop: {lo:.0f}%", xy=(x, -24), xytext=(x + 0.004, -19),
                     fontsize=8.5, color="#8a5a00",
                     arrowprops=dict(arrowstyle="->", color="#8a5a00", lw=1))
axB.set_ylabel("cost reduction vs\nretired-cache  (%)", fontsize=10.5)
axB.set_ylim(-26, 30)
axB.set_xlabel("memory pressure  (fraction of working set the cache holds)  ->  more slack",
               fontsize=11)
axB.legend(loc="lower left", frameon=False, fontsize=8.5)

for x, name in [(0.36, "CHURN"), (0.48, "CRITICAL BAND"), (0.56, "SLACK")]:
    axA.text(x, 42, name, ha="center", fontsize=9.5, color=GRAY, fontweight="bold")
fig.suptitle("What helps in the band, and how much to trust it", fontsize=13, y=0.965)

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "three-regimes.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

"""Render the three-regimes figure for the blog post from the committed results.

Reads results/retired-cache-pressure-sweep.json (retirement's saved-miss
fraction vs pressure) and results/ladder-05-economic.json (the economic policy's
marginal cost reduction and its leave-one-out spread), and draws a two-panel
figure sharing the pressure axis, with the three regimes shaded. Run:

    uv run --with matplotlib python scripts/plot_three_regimes.py <out.png>

All numbers come from the JSON artifacts, so the figure cannot drift from the
committed results.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette.
BLUE, VERMILLION, GRAY = "#0072B2", "#D55E00", "#7f7f7f"
CHURN, BAND, SLACK = "#f0f0f0", "#fde8dc", "#eef3f7"

root = pathlib.Path(__file__).resolve().parent.parent
sweep = json.loads((root / "results/retired-cache-pressure-sweep.json").read_text())
econ = json.loads((root / "results/ladder-05-economic.json").read_text())

# retirement: saved-miss fraction vs cap/T, kept (above noise floor) cells only.
ret = [(r["cap_over_T"], r["saved_pct"] * 100)
       for r in sweep["sweep"] if not r.get("excluded_noise_floor")]
rx, ry = zip(*ret, strict=True)

# economic: full-corpus marginal, and leave-one-out mean/range at the band cells.
p2 = econ["results"]["panel2_per_kind_cost"]["table_percent_of_oracle"]
ex = [r["cap_over_T"] for r in p2]
ey = [r["econ_vs_rc_scored_cost_reduction"] * 100 for r in p2]
loo = econ["robustness_verification"]["leave_one_out_seeds"]
loo_pts = []
for key, cell in loo.items():
    if isinstance(cell, dict) and "mean" in cell:
        x = float(key.replace("cap_over_T_", ""))
        loo_pts.append((x, cell["mean"] * 100, cell["min"] * 100, cell["max"] * 100))

# regime boundaries in cap/T (fraction of the working set the cache holds).
B_CHURN, B_SLACK = 0.42, 0.54

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

for ax in (ax1, ax2):
    ax.axvspan(0.30, B_CHURN, color=CHURN, zorder=0)
    ax.axvspan(B_CHURN, B_SLACK, color=BAND, zorder=0)
    ax.axvspan(B_SLACK, 0.58, color=SLACK, zorder=0)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# top: retirement benefit (robust inverted-U)
ax1.plot(rx, ry, "-o", color=BLUE, lw=2, ms=6, zorder=3)
ax1.set_ylabel("retirement:\n% of misses saved", fontsize=11)
ax1.annotate("peaks near critical pressure", xy=(0.53, 27.3), xytext=(0.435, 24),
             fontsize=9.5, color=BLUE)
ax1.set_ylim(-3, 32)

# bottom: economic marginal + leave-one-out spread (the honesty layer)
ax2.plot(ex, ey, "-o", color=VERMILLION, lw=2, ms=6, zorder=3, label="full corpus")
first = True
for x, mean, lo, hi in loo_pts:
    lo_clip = max(lo, -24)
    ax2.errorbar([x], [mean], yerr=[[mean - lo_clip], [hi - mean]], fmt="D",
                 color="#7a2f00", ms=7, capsize=5, lw=1.6, zorder=4,
                 label="leave-one-out mean and range" if first else None)
    first = False
    if lo < -24:
        ax2.annotate(f"one drop\nreaches {lo:.0f}%", xy=(x, -24), xytext=(x + 0.004, -18),
                     fontsize=8.5, color="#7a2f00",
                     arrowprops=dict(arrowstyle="->", color="#7a2f00", lw=1))
ax2.set_ylabel("economic policy:\n% cost reduction vs\nretired-cache", fontsize=11)
ax2.set_ylim(-26, 30)
ax2.set_xlabel("memory pressure  (fraction of working set the cache holds)  ->  more slack",
               fontsize=11)
ax2.legend(loc="lower left", frameon=False, fontsize=9)

# regime labels across the top
for x, name in [(0.36, "CHURN"), (0.48, "CRITICAL BAND"), (0.56, "SLACK")]:
    ax1.text(x, 33.6, name, ha="center", fontsize=9.5, color=GRAY, fontweight="bold")
ax1.text(0.36, -1.7, "recency wins", ha="center", fontsize=8.5, color=GRAY, style="italic")
ax1.text(0.48, -1.7, "lifecycle + pricing win", ha="center", fontsize=8.5,
         color=GRAY, style="italic")

ax1.set_xlim(0.30, 0.58)
fig.suptitle("Where agentic KV eviction policies actually help", fontsize=13, y=0.965)

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "three-regimes.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

"""Figure: closed-loop safety vs. velocity latency. Usage: python plot_latency.py results_dir"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = Path(sys.argv[1])
df = pd.read_csv(out / "latency.csv")
STYLE = {"naive": ("#d62828", "assume standing until velocity arrives"),
         "robust": ("#2a9d8f", "average over walking-speed hypotheses until then")}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, name in zip(axes, ["oncoming", "crossing"]):
    for mode, (col, lab) in STYLE.items():
        g = df[(df.scenario == name) & (df.planner == mode)].groupby("latency_s").min_dist
        ax.plot(g.median().index, g.median().values, "-", color=col, lw=1.2, alpha=0.6)
        ax.plot(g.quantile(0.1).index, g.quantile(0.1).values, "o-", color=col, lw=2,
                label=lab + " (10th pct)")
    ax.axhline(0.45, color="black", ls=":", lw=1)
    ax.text(0.5, 0.48, "intimate zone 0.45 m", fontsize=8)
    ax.set_title(f"{name} walker (1 m/s)")
    ax.set_xlabel("time before the tracker reports velocity (s)")
    ax.set_ylim(0, 2.0)
axes[0].set_ylabel("closest distance to the person (m)\nthick: 10th percentile, thin: median")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Closed-loop approach, re-planning every 0.5 s (80 runs per point)")
fig.tight_layout()
fig.savefig(out / "latency.png", dpi=150)

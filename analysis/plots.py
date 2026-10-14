"""Figures and summary for the approach-pose noise study.

Usage: python plots.py results_dir
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from approach import SCENARIOS, V_ROBOT, Person, choose, grid, social_cost  # noqa: E402

COLOR = {"standing": "#4C72B0", "crossing": "#DD8452", "oncoming": "#55A868"}
XLABEL = {"position_m": "position noise σ (m)", "heading_deg": "heading noise σ (deg)",
          "speed_mps": "speed noise σ (m/s)"}


def costmaps(out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, p) in zip(axes, SCENARIOS.items()):
        g = choose(p, *grid(p))
        gx, gy = np.meshgrid(np.arange(-1, 8.55, 0.05), np.arange(-4, 4.55, 0.05))
        c = social_cost(gx, gy, p)
        lo = float(social_cost(np.array([[g[0]]]), np.array([[g[1]]]), p)[0, 0])
        im = ax.contourf(gx, gy, np.clip(c - lo, 0, 12), levels=np.linspace(0, 12, 25), cmap="viridis_r")
        T = np.hypot(*g) / V_ROBOT
        hx, hy = p.at(np.linspace(0, T, 50))
        ax.plot(hx, hy, color="white", lw=2)
        ax.plot(p.x, p.y, "o", color="white", ms=6, mec="black", label="person now")
        ax.plot(hx[-1], hy[-1], "o", color="black", ms=9, mec="white", label="person at robot arrival")
        ax.plot([0, g[0]], [0, g[1]], "--", color="red", lw=1.5)
        ax.plot(0, 0, "s", color="red", ms=8, label="robot start")
        ax.plot(*g, "*", color="red", ms=16, mec="black", label="chosen goal")
        ax.set_title(f"{name}: arrival after {T:.1f} s")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 8.5)
        ax.set_ylim(-4, 4.5)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=axes, shrink=0.85, label="cost above the best goal (clipped at 12)")
    fig.savefig(out / "costmaps.png", dpi=150, bbox_inches="tight")


def sweeps(df, out):
    sw = df[df.noise_type != "static_assumption"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey="row")
    for j, kind in enumerate(XLABEL):
        d = sw[sw.noise_type == kind]
        for name in SCENARIOS:
            s = d[d.scenario == name].groupby("level")
            axes[0, j].plot(s.violation.mean().index, 100 * s.violation.mean().values, "o-",
                            color=COLOR[name], label=name)
            med = s.min_dist.median()
            q1, q3 = s.min_dist.quantile(0.1), s.min_dist.quantile(0.9)
            axes[1, j].plot(med.index, med.values, "o-", color=COLOR[name], label=name)
            axes[1, j].fill_between(med.index, q1.values, q3.values, color=COLOR[name], alpha=0.15)
        axes[1, j].axhline(0.45, color="black", ls=":", lw=1)
        axes[1, j].set_xlabel(XLABEL[kind])
    axes[1, 0].text(axes[1, 0].get_xlim()[0], 0.47, " intimate zone 0.45 m", fontsize=8, va="bottom")
    axes[0, 0].set_ylabel("runs ending or passing\ninside 0.45 m (%)")
    axes[1, 0].set_ylabel("closest distance to person (m)\nmedian, 10–90% band")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Approach-pose selection under tracker noise (150 trials per point)")
    fig.tight_layout()
    fig.savefig(out / "noise_sweeps.png", dpi=150)


def summary(df, out):
    rows = []
    for name in SCENARIOS:
        d = df[df.scenario == name]
        base = d[(d.noise_type == "position_m") & (d.level == 0)].iloc[0]
        row = {"scenario": name, "closest_m_noise_free": base.min_dist}
        for kind, level in [("position_m", 0.3), ("heading_deg", 35), ("speed_mps", 0.35)]:
            x = d[(d.noise_type == kind) & (d.level == level)]
            row[f"viol_%_{kind}={level}"] = 100 * x.violation.mean()
        st = d[d.noise_type == "static_assumption"].iloc[0]
        row["static_assumption_closest_m"] = st.min_dist
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(s.to_markdown(index=False, floatfmt=".2f"))
    print(s.to_string(index=False))


def static_failure(out):
    """Oncoming walker: goal chosen with velocity vs. with velocity ignored."""
    p = SCENARIOS["oncoming"]
    static = Person(p.x, p.y, p.heading, 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for est, col, lab in [(p, "#2a9d8f", "tracker with velocity"), (static, "#d62828", "tracker without velocity")]:
        g = choose(est, *grid(p))
        T = np.hypot(*g) / V_ROBOT
        ts = np.linspace(0, 1, 200)
        px, py = ts * g[0], ts * g[1]
        qx, qy = p.at(ts * T)
        dist = np.hypot(px - qx, py - qy)
        k = int(np.argmin(dist))
        ax.plot([0, g[0]], [0, g[1]], "--", color=col, lw=2)
        ax.plot(*g, "*", color=col, ms=16, mec="black", label=f"{lab}: closest {dist[k]:.2f} m")
        ax.plot([px[k], qx[k]], [py[k], qy[k]], color=col, lw=1)
    T = 6.0
    hx, hy = p.at(np.linspace(0, T, 50))
    ax.plot(hx, hy, color="black", lw=2, label="person's true path (1 m/s)")
    ax.plot(p.x, p.y, "o", color="black", ms=8)
    ax.plot(0, 0, "s", color="gray", ms=9, label="robot start")
    ax.set_aspect("equal")
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 2.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Oncoming walker: ignoring velocity puts the robot in the person's path")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "static_failure.png", dpi=150)


def main(d):
    out = Path(d)
    df = pd.read_csv(out / "results.csv")
    costmaps(out)
    sweeps(df, out)
    summary(df, out)
    static_failure(out)


if __name__ == "__main__":
    main(sys.argv[1])

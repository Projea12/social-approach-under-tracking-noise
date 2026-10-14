"""Sweep tracker noise and record how the chosen approach pose degrades.

Usage: python run_sweep.py out_dir [trials]
Writes results.csv: scenario, noise_type, level, trial, regret, end_dist, min_dist, violation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from approach import SCENARIOS, Person, choose, evaluate, grid, social_cost

# Noise levels, chosen to span good to poor 2D people trackers.
SWEEPS = {
    "position_m": [0.0, 0.1, 0.2, 0.3, 0.5],
    "heading_deg": [0, 10, 20, 35, 60],
    "speed_mps": [0.0, 0.1, 0.2, 0.35, 0.5],
}


def perturb(p, kind, level, rng):
    x, y, h, v = p.x, p.y, p.heading, p.speed
    if kind == "position_m":
        x += rng.normal(0, level)
        y += rng.normal(0, level)
    elif kind == "heading_deg":
        h += np.deg2rad(rng.normal(0, level))
    elif kind == "speed_mps":
        v = max(0.0, v + rng.normal(0, level))
    return Person(x, y, h, v)


def main(out, trials):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, true_p in SCENARIOS.items():
        gx, gy = grid(true_p)
        best = choose(true_p, gx, gy)
        best_cost = float(social_cost(np.array([[best[0]]]), np.array([[best[1]]]), true_p)[0, 0])
        rng = np.random.default_rng(7)

        # Noise sweeps.
        for kind, levels in SWEEPS.items():
            for level in levels:
                n = 1 if level == 0 else trials
                for t in range(n):
                    est = perturb(true_p, kind, level, rng)
                    g = choose(est, gx, gy)
                    rows.append({"scenario": name, "noise_type": kind, "level": level, "trial": t,
                                 **evaluate(g, true_p, gx, gy, best_cost)})

        # Common tracker failure: velocity not estimated, every person treated as standing.
        static = Person(true_p.x, true_p.y, true_p.heading, 0.0)
        g = choose(static, gx, gy)
        rows.append({"scenario": name, "noise_type": "static_assumption", "level": 1, "trial": 0,
                     **evaluate(g, true_p, gx, gy, best_cost)})
        print(name, "done; noise-free goal", np.round(best, 2))

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)

"""A fix for the no-velocity failure: plan over plausible walking speeds.

When the tracker gives position and body orientation but no velocity, the naive planner
assumes the person is standing. The robust planners instead consider a set of speed
hypotheses along the person's facing direction and choose a goal that is good across all
of them:

  minimax - minimise the worst-case cost over the hypotheses
  mean    - minimise the average cost over the hypotheses

Usage: python robust.py out_dir
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from approach import SCENARIOS, Person, V_ROBOT, choose, evaluate, grid, social_cost

SPEEDS = [0.0, 0.4, 0.8, 1.2, 1.5]   # m/s; covers standing to brisk walking

# Extra scenario: a person who really is standing but faces sideways, to check the robust
# planners do not become overly cautious when there is no motion at all.
EXTRA = {"standing_sideways": Person(4.0, 0.0, np.pi / 2, 0.0)}


def choose_robust(est, gx, gy, mode):
    costs = np.stack([social_cost(gx, gy, Person(est.x, est.y, est.heading, s)) for s in SPEEDS])
    c = costs.max(axis=0) if mode == "minimax" else costs.mean(axis=0)
    i = np.unravel_index(np.argmin(c), c.shape)
    return gx[i], gy[i]


def main(out):
    out = Path(out)
    rows = []
    for name, true_p in {**SCENARIOS, **EXTRA}.items():
        gx, gy = grid(true_p)
        oracle = choose(true_p, gx, gy)
        best = float(social_cost(np.array([[oracle[0]]]), np.array([[oracle[1]]]), true_p)[0, 0])
        no_vel = Person(true_p.x, true_p.y, true_p.heading, 0.0)   # what the tracker reports
        goals = {
            "oracle (true velocity)": oracle,
            "naive (assume standing)": choose(no_vel, gx, gy),
            "robust minimax": choose_robust(no_vel, gx, gy, "minimax"),
            "robust mean": choose_robust(no_vel, gx, gy, "mean"),
        }
        for planner, g in goals.items():
            r = evaluate(g, true_p, gx, gy, best)
            rows.append({"scenario": name, "planner": planner, "goal_x": g[0], "goal_y": g[1],
                         "arrival_s": float(np.hypot(*g) / V_ROBOT), **r})
    df = pd.DataFrame(rows)
    df["in_conversation_range"] = df.end_dist.between(0.8, 2.0)
    df.to_csv(out / "robust.csv", index=False)
    cols = ["scenario", "planner", "min_dist", "end_dist", "arrival_s", "violation", "in_conversation_range"]
    print(df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])

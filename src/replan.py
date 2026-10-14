"""Closed-loop approach with re-planning as the velocity estimate arrives.

The robot re-chooses its goal every REPLAN_S seconds from its current position. A simple
tracker reports noisy position and body orientation at every step. Velocity becomes available
only after VEL_WINDOW_S seconds of observation, as a finite difference of noisy positions over
that window.

Before velocity is available:
  naive  - treats the person as standing
  robust - averages the cost over plausible walking speeds (see robust.py)
After it is available, both use the estimated velocity.

Usage: python replan.py out_dir [trials]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from approach import D_INTIMATE, SCENARIOS, V_ROBOT, Person, social_cost
from robust import EXTRA, SPEEDS

DT = 0.1             # simulation step (s)
REPLAN_S = 0.5       # re-plan period (s)
VEL_WINDOW_S = 1.0   # observation time needed before velocity is reported (s)
POS_SIGMA = 0.10     # tracker position noise (m)
HEAD_SIGMA = np.deg2rad(10)  # body-orientation noise
T_MAX = 15.0


def local_grid(px, py, step=0.12, half=5.0):
    xs = np.arange(px - half, px + half, step)
    ys = np.arange(py - half, py + half, step)
    return np.meshgrid(xs, ys)


def pick(robot, est_xy, heading, speed, mode):
    gx, gy = local_grid(*est_xy)
    if speed is None:   # velocity not yet available
        speeds = SPEEDS if mode == "robust" else [0.0]
    else:
        speeds = [speed]
    c = np.mean([social_cost(gx, gy, Person(est_xy[0], est_xy[1], heading, s), robot=robot)
                 for s in speeds], axis=0)
    i = np.unravel_index(np.argmin(c), c.shape)
    return np.array([gx[i], gy[i]])


def run(true_p, mode, rng, latency=VEL_WINDOW_S):
    robot = np.zeros(2)
    hist = []             # noisy position observations (t, x, y)
    goal = None
    min_d, t = np.inf, 0.0
    steps_per_replan = int(round(REPLAN_S / DT))
    k = 0
    while t <= T_MAX:
        hx, hy = true_p.at(t)
        d = float(np.hypot(robot[0] - hx, robot[1] - hy))
        min_d = min(min_d, d)
        obs = np.array([hx, hy]) + rng.normal(0, POS_SIGMA, 2)
        hist.append((t, *obs))
        if k % steps_per_replan == 0:
            heading = true_p.heading + rng.normal(0, HEAD_SIGMA)
            speed = None
            if t >= latency - 1e-9:
                win = min(latency, VEL_WINDOW_S)
                old = [h for h in hist if h[0] <= t - win + 1e-9][-1]
                v = (obs - np.array(old[1:])) / (t - old[0])
                speed = float(np.hypot(*v))
                if speed > 0.3:           # moving: trust the displacement direction
                    heading = float(np.arctan2(v[1], v[0]))
            goal = pick(robot, obs, heading, speed, mode)
        step = goal - robot
        dist = np.hypot(*step)
        if dist < 0.05 and speed is not None:
            return {"min_dist": min_d, "end_dist": d, "arrival_s": t}
        robot = robot + (step / dist * min(V_ROBOT * DT, dist) if dist > 1e-9 else 0)
        t += DT
        k += 1
    return {"min_dist": min_d, "end_dist": d, "arrival_s": np.nan}


def main(out, trials):
    out = Path(out)
    rows = []
    for name, p in {**SCENARIOS, **EXTRA}.items():
        for mode in ["naive", "robust"]:
            rng = np.random.default_rng(11)
            for i in range(trials):
                rows.append({"scenario": name, "planner": mode, "trial": i, **run(p, mode, rng)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df["violation"] = df.min_dist < D_INTIMATE
    df["in_conversation_range"] = df.end_dist.between(0.8, 2.0)
    df.to_csv(out / "replan.csv", index=False)
    s = df.groupby(["scenario", "planner"]).agg(
        violation_pct=("violation", lambda x: 100 * x.mean()),
        in_range_pct=("in_conversation_range", lambda x: 100 * x.mean()),
        median_closest_m=("min_dist", "median"),
        p10_closest_m=("min_dist", lambda x: x.quantile(0.1)),
        median_arrival_s=("arrival_s", "median"),
    ).round(2)
    s.to_csv(out / "replan_summary.csv")
    (out / "replan_summary.md").write_text(s.reset_index().to_markdown(index=False, floatfmt=".2f"))
    print(s.to_string())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)

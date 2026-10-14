"""How long can velocity be missing before closed-loop approach becomes unsafe?

Velocity is reported only after `latency` seconds of tracking (finite difference over the last
min(latency, 1 s) of noisy positions). Re-planning continues every 0.5 s throughout.
Usage: python latency_sweep.py out_dir [trials]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from approach import D_INTIMATE, SCENARIOS
from replan import run

LATENCIES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def main(out, trials):
    out = Path(out)
    rows = []
    for name in ["oncoming", "crossing"]:
        p = SCENARIOS[name]
        for mode in ["naive", "robust"]:
            for lat in LATENCIES:
                rng = np.random.default_rng(23)
                for i in range(trials):
                    rows.append({"scenario": name, "planner": mode, "latency_s": lat, "trial": i,
                                 **run(p, mode, rng, latency=lat)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df["violation"] = df.min_dist < D_INTIMATE
    df["in_conversation_range"] = df.end_dist.between(0.8, 2.0)
    df.to_csv(out / "latency.csv", index=False)
    s = df.groupby(["scenario", "planner", "latency_s"]).agg(
        violation_pct=("violation", lambda x: 100 * x.mean()),
        in_range_pct=("in_conversation_range", lambda x: 100 * x.mean()),
        p10_closest_m=("min_dist", lambda x: x.quantile(0.1))).round(2)
    print(s.to_string())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)

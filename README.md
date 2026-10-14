# Social approach-pose selection under people-tracking errors

A robot needs to approach a person, standing or walking, to start an interaction. It chooses a
goal point that is socially acceptable *where the person will be when the robot arrives*. This
repository re-implements that idea in a simplified form, then asks one question:

**The method assumes a tracked person state (position, heading, speed). How much do realistic
tracker errors change the chosen pose, and when do they make the robot end up too close?**

The approach-pose idea comes from Ngo and Nguyen, *Spatiotemporal Motion Profiles for Cost-Based
Optimal Approaching Pose Estimation*, IEEE/SICE SII 2024. This is **not** their code or their exact
cost function. It is a small, transparent version built to study the perception side of the
problem.

![Costmaps](results/costmaps.png)

## Model

Each candidate goal point g gets a cost. The person is predicted at constant velocity to the time
T at which the robot, moving straight at 0.8 m/s, would reach g. The cost has five terms:

| Term | Meaning |
|---|---|
| Personal space | Asymmetric Gaussian around the predicted person: larger ahead, smaller behind (Kirby, 2009). Its size grows with walking speed |
| Distance | Deviation from a 1.2 m conversation distance (Hall's social-zone boundary) |
| Visibility | Penalty for arriving outside the person's forward view, so they see the robot coming |
| Path intrusion | The worst personal-space intrusion along the robot's path, with the person moving at the same time. This is the spatiotemporal part |
| Time | Arrival time |

The goal is the minimum over a 5 cm to 8 cm grid. The figure above shows the resulting cost
fields:

- **Standing person:** a ring in front.
- **Person walking across the robot's view:** a band beside their path.
- **Person walking toward the robot:** a pocket diagonally ahead of them.

### One modelling choice to flag

Kirby's personal-space model was designed for robots *passing* people. It makes the space ahead
of a 1 m/s walker about 2 m deep. With it, the robot chose to approach walkers **from behind**,
which is socially wrong. I reduced the speed scaling for the approach task (`0.5 + 0.5·v` instead
of `max(2v, 0.5)`) and raised the visibility weight. The robot then approaches from the front-side.
How a passing model should change for engagement seems like an open question in itself.

## Experiment

Three scenarios: standing, crossing (1 m/s across the robot's view) and oncoming (1 m/s toward
the robot). Tracker error is added to the person state the planner sees. The chosen goal is then
scored against the person's **true** motion.

- Position noise σ = 0 to 0.5 m; heading noise σ = 0 to 60°; speed noise σ = 0 to 0.5 m/s.
  This gives 150 trials per setting and 5,409 goal selections in total.
- A common tracker failure was also tested: no velocity estimate, so every person is treated as
  standing.
- **Violation:** at any point on the approach, the robot comes within 0.45 m of the person
  (Hall's intimate zone).

![Noise sweeps](results/noise_sweeps.png)

## Findings

1. **The cost-based selection is robust to ordinary tracker noise.** Up to 0.3 m position noise,
   35° heading noise and 0.2 m/s speed noise, there were no violations in any scenario. Even at
   0.5 m position noise, violations stayed at 3.3% or below.
2. **Heading errors make the robot miss walkers.** For the crossing person, 60° of heading noise
   raises the median closest distance from 1.6 m to 2.4 m. The robot goes where the person was
   heading, not where they went, so the approach fails safely but socially: it never reaches
   conversation distance.
3. **Missing velocity is the dangerous failure.** Treat the oncoming walker as standing, and the
   robot drives toward where they *were*. The person walks into its path and the closest distance
   falls from 1.65 m to **0.35 m**, inside the intimate zone. For the crossing walker the same
   failure leaves the robot 3.3 m from where the person ends up.

![Static assumption failure](results/static_failure.png)

The practical reading: for spatiotemporal approach planning, *whether the tracker estimates
velocity at all* matters far more than how accurate its position is.

## Questions this raises

- In a real system, the motion profiles come from a people tracker. Should the personal-space cost
  grow with the tracker's uncertainty, for example by inflating it with the velocity covariance?
- How quickly should the goal be re-planned as the person's motion estimate converges? The
  failure in finding 3 disappears if the velocity estimate arrives before the robot commits.
- For older adults, whose walking is slower and less steady, is constant-velocity prediction
  enough, or does the motion profile need a richer gait model?

## Limitations

- The robot moves in a straight line at constant speed. There is no full kinodynamic planner and
  no costmap navigation stack; only the goal choice is studied.
- People move at constant velocity, one person at a time.
- Tracker noise is independent Gaussian noise, which is optimistic; real trackers drift and swap
  identities.
- The cost weights are hand-set, not fitted to human-preference data.

## Reproduce

Python 3.10+ with numpy, pandas, matplotlib and tabulate.

```bash
python src/run_sweep.py results 150   # about 4 minutes
python analysis/plots.py results
```

`results/results.csv` has one row per selection: scenario, noise type, level, trial, regret
(true cost above the best goal), end distance, closest distance and violation.

## Layout

```
src/approach.py        cost model, goal selection, evaluation against true motion, scenarios
src/run_sweep.py       noise sweeps and the no-velocity case
analysis/plots.py      costmaps, sweep figure, failure figure, summary table
results/               raw CSV, summary, figures
```

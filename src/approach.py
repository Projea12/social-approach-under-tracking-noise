"""Socially aware approach-pose selection, and how tracking errors degrade it.

A robot must approach a (possibly walking) person to start an interaction. It picks a goal
point by minimising a spatiotemporal social cost evaluated where the person is predicted to
be when the robot arrives. This is a simplified re-implementation of the idea in
Ngo and Nguyen, "Spatiotemporal Motion Profiles for Cost-Based Optimal Approaching Pose
Estimation" (IEEE/SICE SII 2024). It is not their code or their exact cost function.

The question studied here: the method assumes a tracked person state (position, heading,
speed). What happens to the chosen pose when that state comes from an imperfect tracker?
"""
from dataclasses import dataclass

import numpy as np

V_ROBOT = 0.8          # robot speed limit (m/s); straight-line travel approximates kinodynamics
D_SOCIAL = 1.2         # preferred conversation distance (m), Hall's social/personal boundary
D_INTIMATE = 0.45      # Hall's intimate zone: ending inside it counts as a violation
PATH_STEPS = 24        # samples along the robot's approach path

# Weights of the cost terms.
W_SPACE, W_DIST, W_VIS, W_PATH, W_TIME = 4.0, 1.0, 4.0, 6.0, 0.15


@dataclass(frozen=True)
class Person:
    x: float
    y: float
    heading: float  # rad
    speed: float    # m/s

    def at(self, t):
        """Constant-velocity prediction at time t (arrays allowed)."""
        return (self.x + self.speed * np.cos(self.heading) * t,
                self.y + self.speed * np.sin(self.heading) * t)


def personal_space(dx, dy, heading, speed):
    """Asymmetric Gaussian personal space (Kirby 2009), larger ahead of a walking person.

    dx, dy: offset of the query point from the person. Returns a value in [0, 1].
    """
    sig_front = 0.5 + 0.5 * speed      # approach setting; Kirby (passing) uses max(2v, 0.5)
    sig_side = 2.0 / 3.0 * sig_front
    sig_rear = 0.5 * sig_front
    rel = np.arctan2(dy, dx) - heading
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    sig_long = np.where(np.abs(rel) <= np.pi / 2, sig_front, sig_rear)
    c, s = np.cos(heading), np.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return np.exp(-(lon**2 / (2 * sig_long**2) + lat**2 / (2 * sig_side**2)))


def social_cost(gx, gy, person, robot=(0.0, 0.0)):
    """Total cost of goal points (gx, gy) for a robot starting at `robot`.

    Terms, all evaluated against the person's predicted motion:
      space - personal-space intrusion at the goal, at arrival time
      dist  - deviation from the preferred conversation distance
      vis   - goal outside the person's forward field of view (they should see the robot come)
      path  - worst personal-space intrusion along the straight approach path, over time
      time  - arrival time
    """
    rx, ry = robot
    T = np.hypot(gx - rx, gy - ry) / V_ROBOT
    hx, hy = person.at(T)
    dx, dy = gx - hx, gy - hy
    d = np.hypot(dx, dy)

    space = personal_space(dx, dy, person.heading, person.speed)
    dist = ((d - D_SOCIAL) / 0.6) ** 2
    rel = np.arctan2(dy, dx) - person.heading
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    vis = (np.abs(rel) / np.pi) ** 2

    # Spatiotemporal path term: sample the robot's motion and the person's motion together.
    path = np.zeros_like(gx)
    for k in range(1, PATH_STEPS):
        f = k / PATH_STEPS
        px, py = rx + f * (gx - rx), ry + f * (gy - ry)
        qx, qy = person.at(f * T)
        path = np.maximum(path, personal_space(px - qx, py - qy, person.heading, person.speed))

    return W_SPACE * space + W_DIST * dist + W_VIS * vis + W_PATH * path + W_TIME * T


def grid(person, half=4.0, step=0.08):
    """Candidate goals in a window around the person's position (window is large enough
    to contain any reachable socially acceptable goal for the scenarios used here)."""
    cx, cy = person.x, person.y
    xs = np.arange(cx - half - 2.0, cx + half + 2.0, step)
    ys = np.arange(cy - half - 2.0, cy + half + 2.0, step)
    return np.meshgrid(xs, ys)


def choose(person, gx, gy):
    """Goal minimising the cost for this person state."""
    c = social_cost(gx, gy, person)
    i = np.unravel_index(np.argmin(c), c.shape)
    return gx[i], gy[i]


def evaluate(goal, true_person, gx_ref, gy_ref, best_true_cost):
    """Score a goal chosen from an estimate against the true person motion."""
    g = np.array([[goal[0]]]), np.array([[goal[1]]])
    true_cost = float(social_cost(*g, true_person)[0, 0])
    T = np.hypot(*goal) / V_ROBOT
    hx, hy = true_person.at(T)
    end_dist = float(np.hypot(goal[0] - hx, goal[1] - hy))
    # Closest approach to the true person along the way (including the end).
    ts = np.linspace(0, 1, 200)
    px, py = ts * goal[0], ts * goal[1]
    qx, qy = true_person.at(ts * T)
    min_dist = float(np.min(np.hypot(px - qx, py - qy)))
    return {
        "regret": true_cost - best_true_cost,
        "end_dist": end_dist,
        "min_dist": min_dist,
        "violation": min_dist < D_INTIMATE,
    }


SCENARIOS = {
    "standing": Person(4.0, 1.0, np.pi + 0.3, 0.0),     # person standing, roughly facing the robot
    "crossing": Person(4.0, -2.0, np.pi / 2, 1.0),      # person walking across the robot's view
    "oncoming": Person(6.0, 0.5, np.pi, 1.0),           # person walking toward the robot
}

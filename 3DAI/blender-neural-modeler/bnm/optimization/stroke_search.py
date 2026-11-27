"""Brush-parameter optimisation by candidate evaluation (Blender brushes are not differentiable).

Optimises stroke parameters (position offset, radius, strength, drag) instead of raw vertices, so
every accepted change is still an interpretable brush stroke.  Coordinate descent with
snapshot/restore: apply candidate -> measure loss -> undo, keep the best, finally apply it.
"""
import itertools
import numpy as np


def evaluate(S, apply_fn, params, loss_fn):
    S.ckpt.snapshot("search")
    try:
        apply_fn(params)
        loss = float(loss_fn())
    finally:
        S.ckpt.undo(1)
        S.eng.resync()
    return loss


def coordinate_descent(S, apply_fn, loss_fn, init, steps, bounds=None, rounds=3, shrink=0.5, log_name="stroke_search"):
    """init/steps: dict param -> value/step.  Returns (best_params, best_loss, history)."""
    best = dict(init)
    base_loss = float(loss_fn())
    best_loss = evaluate(S, apply_fn, best, loss_fn)
    hist = [{"params": dict(best), "loss": best_loss, "baseline": base_loss}]
    step = dict(steps)
    for _ in range(rounds):
        improved = False
        for k in list(step):
            for sgn in (+1, -1):
                cand = dict(best)
                cand[k] = cand[k] + sgn * step[k]
                if bounds and k in bounds:
                    lo, hi = bounds[k]
                    cand[k] = float(np.clip(cand[k], lo, hi))
                l = evaluate(S, apply_fn, cand, loss_fn)
                hist.append({"params": dict(cand), "loss": l})
                if l < best_loss - 1e-9:
                    best, best_loss, improved = cand, l, True
        if not improved:
            step = {k: v * shrink for k, v in step.items()}
    accepted = best_loss < base_loss
    if accepted:
        apply_fn(best)
    S.log.record({"op": "stroke_search", "name": log_name, "baseline": base_loss, "best": best_loss,
                  "params": best, "evaluations": len(hist), "applied": accepted})
    return best, best_loss, hist


def grid_search(S, apply_fn, loss_fn, grid, log_name="grid_search"):
    keys = list(grid)
    base = float(loss_fn())
    best, best_loss = None, base
    hist = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        l = evaluate(S, apply_fn, p, loss_fn)
        hist.append({"params": p, "loss": l})
        if l < best_loss:
            best, best_loss = p, l
    if best is not None:
        apply_fn(best)
    S.log.record({"op": "grid_search", "name": log_name, "baseline": base, "best": best_loss, "params": best})
    return best, best_loss, hist


def grab_to_target(S, point, target_point, radius_range=(0.04, 0.2), measure=None):
    """Example: find Grab radius/strength that moves `measure()` (a landmark) closest to target."""
    point = np.asarray(point, float)
    tgt = np.asarray(target_point, float)
    measure = measure or (lambda: S.st.co[np.argmin(np.linalg.norm(S.st.co - point, axis=1))])

    def apply(p):
        S.stroke("GRAB", [point, point + (tgt - point) * p["gain"]], p["radius"], 1.0)

    def loss():
        return float(np.linalg.norm(measure() - tgt))
    return coordinate_descent(S, apply, loss, {"gain": 1.0, "radius": float(np.mean(radius_range))},
                              {"gain": 0.25, "radius": (radius_range[1] - radius_range[0]) / 4},
                              {"gain": (0.2, 2.0), "radius": radius_range}, log_name="grab_to_target")

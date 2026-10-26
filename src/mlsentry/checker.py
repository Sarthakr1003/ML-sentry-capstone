"""
ML Sentry: threshold checks.

Compares one evaluation result with the limits from .env and lists every
metric that is out of bounds. Also gives a simple rule-based severity,
used as a hint for the agent and as the fallback when no LLM is reachable.
"""

SEVERITIES = ["minor", "moderate", "critical"]


def check(metrics: dict, limits: dict) -> tuple:
    """
    Return (breaches, missing).
      breaches: one dict per metric outside its limit
      missing:  watched metrics this run did not log at all
    """
    breaches, missing = [], []
    for name, bounds in limits.items():
        if name not in metrics:
            missing.append(name)
            continue
        value = metrics[name]
        for side, limit in bounds.items():
            too_high = side == "max" and value > limit
            too_low = side == "min" and value < limit
            if too_high or too_low:
                breaches.append({
                    "metric": name,
                    "value": round(value, 4),
                    "limit": limit,
                    "side": side,
                    # How far past the limit, relative to the limit (0.5 = 50% worse than allowed)
                    "excess": round(abs(value - limit) / (abs(limit) or 1.0), 4),
                })
    return breaches, missing


def rule_severity(breaches: list, streak: int) -> str:
    """
    Simple, explainable rules:
      worst metric under 25% past its limit  -> minor
      25% to 100% past its limit             -> moderate
      more than double the allowed distance  -> critical
    A problem that has lasted 3 or more runs in a row goes up one level.
    """
    worst = max(b["excess"] for b in breaches)
    level = 0 if worst < 0.25 else 1 if worst < 1.0 else 2
    if streak >= 3:
        level = min(level + 1, 2)
    return SEVERITIES[level]

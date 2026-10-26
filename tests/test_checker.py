from mlsentry.checker import check, rule_severity

LIMITS = {"rmse": {"max": 0.6}, "r2": {"min": 0.7}}


def test_healthy_run_has_no_breaches():
    breaches, missing = check({"rmse": 0.5, "r2": 0.8}, LIMITS)
    assert breaches == [] and missing == []


def test_max_and_min_limits_are_both_checked():
    breaches, _ = check({"rmse": 0.9, "r2": 0.6}, LIMITS)
    assert [(b["metric"], b["side"]) for b in breaches] == [("rmse", "max"), ("r2", "min")]
    assert breaches[0]["excess"] == 0.5  # 0.9 is 50% past 0.6


def test_value_exactly_on_the_limit_is_ok():
    breaches, _ = check({"rmse": 0.6, "r2": 0.7}, LIMITS)
    assert breaches == []


def test_missing_metric_is_reported():
    breaches, missing = check({"rmse": 0.5}, LIMITS)
    assert breaches == [] and missing == ["r2"]


def test_rule_severity_levels():
    assert rule_severity([{"excess": 0.1}], streak=1) == "minor"
    assert rule_severity([{"excess": 0.5}], streak=1) == "moderate"
    assert rule_severity([{"excess": 2.0}], streak=1) == "critical"


def test_long_streak_raises_severity_one_level():
    assert rule_severity([{"excess": 0.1}], streak=3) == "moderate"
    assert rule_severity([{"excess": 2.0}], streak=5) == "critical"  # already the top level

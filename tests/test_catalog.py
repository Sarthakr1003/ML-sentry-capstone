"""Checks of the metric catalog the setup wizard uses."""

from mlsentry.metrics_catalog import CATALOG, MODEL_TYPES, known_metric, parse_threshold, suggest_threshold


def test_every_model_type_has_metrics_with_a_direction():
    assert set(CATALOG) == set(MODEL_TYPES)
    for metrics in CATALOG.values():
        assert metrics
        assert all(m.direction in ("max", "min") for m in metrics)


def test_logged_names_are_recognised():
    assert known_metric("classification", "acc").name == "accuracy"
    assert known_metric("classification", "val_accuracy").name == "accuracy"
    assert known_metric("classification", "F1_Score").name == "f1"
    assert known_metric("regression", "test_rmse").name == "rmse"
    assert known_metric("regression", "train_time") is None
    assert known_metric("rag", "answer_relevance").name == "answer_relevancy"


def test_errors_alert_above_and_scores_below():
    assert known_metric("regression", "rmse").direction == "max"
    assert known_metric("regression", "r2").direction == "min"
    assert known_metric("classification", "accuracy").direction == "min"


def test_suggested_threshold_is_a_little_worse_than_now():
    assert suggest_threshold("max", 0.48, None) == 0.6        # lower is better: 25% above
    assert suggest_threshold("min", 0.9, 0.8) == 0.765        # higher is better: 15% below
    assert suggest_threshold("min", None, 0.8) == 0.8         # no value yet: catalog default


def test_threshold_input_is_checked():
    assert parse_threshold(" 0.6 ") == (0.6, None)
    for bad in ("", "abc", "0,6", "nan"):
        value, reason = parse_threshold(bad)
        assert value is None and reason
    assert parse_threshold("1.5", 0, 1)[0] is None             # accuracy cannot be 150%
    assert parse_threshold("-1", 0, 1)[0] is None
    assert parse_threshold("-0.2", None, 1) == (-0.2, None)    # R2 can be negative

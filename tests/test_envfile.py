"""Checks that the setup wizard edits .env without losing anything."""

from dotenv import dotenv_values

from mlsentry.envfile import EnvFile

ORIGINAL = """# ML Sentry settings
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
METRICS=rmse,r2,max_psi
RMSE_MAX=0.60
R2_MIN=0.70
MAX_PSI_MAX=0.20

# Demo only
DEMO_BATCH_SIZE=500
"""


def test_values_are_read_and_changed_in_place(tmp_path):
    path = tmp_path / ".env"
    path.write_text(ORIGINAL)
    env = EnvFile(path)
    assert env.get("RMSE_MAX") == "0.60"
    env.set("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    env.set("MODEL_TYPE", "regression")
    env.save()

    text = path.read_text()
    assert text.startswith("# ML Sentry settings\nMLFLOW_TRACKING_URI=http://127.0.0.1:8080\n")
    assert "# Demo only\nDEMO_BATCH_SIZE=500" in text      # comments and other settings kept
    assert text.rstrip().endswith("MODEL_TYPE=regression")  # new settings added at the end


def test_metrics_and_limits_are_replaced(tmp_path):
    path = tmp_path / ".env"
    path.write_text(ORIGINAL)
    env = EnvFile(path)
    env.set_metrics({"rmse": {"max": 0.5}, "mae": {"max": 0.4}})
    env.save()

    values = dotenv_values(path)
    assert values["METRICS"] == "rmse,mae"
    assert values["RMSE_MAX"] == "0.5"
    assert values["MAE_MAX"] == "0.4"
    assert "R2_MIN" not in values and "MAX_PSI_MAX" not in values   # no longer watched
    assert values["DEMO_BATCH_SIZE"] == "500"


def test_new_file_is_created(tmp_path):
    path = tmp_path / "state" / ".env"
    env = EnvFile(path)
    assert env.get("METRICS") == ""
    env.set_metrics({"accuracy": {"min": 0.85}})
    env.save()
    assert dotenv_values(path) == {"METRICS": "accuracy", "ACCURACY_MIN": "0.85"}

"""The setup questionnaire, answered by a script instead of a person (no terminal, no network)."""

from types import SimpleNamespace

import pytest
from dotenv import dotenv_values

from mlsentry import setup_cli


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """Replace the terminal questions with a list of answers given in order."""
    answers = []

    def fake_ask(question):
        if not answers:
            raise AssertionError("the questionnaire asked more questions than expected")
        answer = answers.pop(0)
        if answer is None:              # None = the user pressed Ctrl+C
            raise setup_cli.Cancelled
        return answer

    question = lambda *a, **k: None    # the question objects are never shown
    monkeypatch.setattr(setup_cli, "questionary", SimpleNamespace(
        text=question, password=question, select=question, checkbox=question, confirm=question,
        print=lambda *a, **k: None))
    monkeypatch.setattr(setup_cli, "_ask", fake_ask)
    monkeypatch.setattr(setup_cli.checks, "in_docker", lambda: False)
    monkeypatch.setattr(setup_cli.checks, "check_mlflow", lambda url: (True, "Connected", ["house-prices"]))
    monkeypatch.setattr(setup_cli.checks, "logged_metrics",
                        lambda url, exp: {"rmse": 0.48, "r2": 0.83, "loss": 0.2})
    path = tmp_path / ".env"
    path.write_text("# my notes\nDEMO_BATCH_SIZE=500\nGROQ_API_KEY=your-groq-key-here\n")
    return answers, path


def test_first_time_setup_writes_env(scripted):
    answers, path = scripted
    answers += [
        "",                      # Groq key: skipped
        "regression",            # model type
        "8080",                  # MLflow port
        "house-prices",          # experiment
        ["rmse", "r2", "loss"],  # metrics (loss is not a known metric)
        "0.6", "0.7",            # limits for rmse and r2
        "max", "0.3",            # loss: lower is better, limit 0.3
        [],                      # no Slack / Telegram
        False,                   # no advanced settings
        "Yes, save",
    ]
    assert setup_cli.first_time_setup(setup_cli.EnvFile(path)) is True
    assert answers == []

    values = dotenv_values(path)
    assert values["MODEL_TYPE"] == "regression"
    assert values["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:8080"
    assert values["DOCKER_MLFLOW_TRACKING_URI"] == "http://host.docker.internal:8080"
    assert values["MLFLOW_EXPERIMENT"] == "house-prices"
    assert values["METRICS"] == "rmse,r2,loss"
    assert (values["RMSE_MAX"], values["R2_MIN"], values["LOSS_MAX"]) == ("0.6", "0.7", "0.3")
    assert values["ALERT_CHANNEL"] == "console"
    assert values["GROQ_API_KEY"] == ""
    assert values["POLL_INTERVAL_SECONDS"] == "10"            # default filled in
    assert values["DEMO_BATCH_SIZE"] == "500"                 # other settings kept
    assert path.read_text().startswith("# my notes")


def test_ctrl_c_saves_nothing(scripted):
    answers, path = scripted
    before = path.read_text()
    answers += ["", "regression", None]                       # Ctrl+C at the MLflow question
    with pytest.raises(setup_cli.Cancelled):
        setup_cli.first_time_setup(setup_cli.EnvFile(path))
    assert path.read_text() == before


def test_menu_changes_one_threshold_only(scripted):
    answers, path = scripted
    path.write_text("MODEL_TYPE=regression\nMLFLOW_EXPERIMENT=house-prices\n"
                    "MLFLOW_TRACKING_URI=http://127.0.0.1:5000\nMETRICS=rmse,r2\nRMSE_MAX=0.6\nR2_MIN=0.7\n")
    answers += ["metrics", ["rmse", "r2"], "0.55", "0.7", "save"]
    assert setup_cli.menu(setup_cli.EnvFile(path)) is True
    values = dotenv_values(path)
    assert (values["RMSE_MAX"], values["R2_MIN"], values["MODEL_TYPE"]) == ("0.55", "0.7", "regression")


def test_no_terminal_gives_a_clear_message(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    assert setup_cli.run(None) == (setup_cli.NOT_SAVED, False)
    assert "needs a terminal" in capsys.readouterr().out

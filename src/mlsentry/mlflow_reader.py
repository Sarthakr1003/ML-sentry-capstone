"""
ML Sentry: reads evaluation results from the user's own MLflow.

Talks to MLflow's REST API over HTTP, so ML Sentry needs neither the user's
model nor the mlflow Python package. Each finished MLflow run = one
evaluation result (a set of metric values).
"""

import requests

TIMEOUT = 15  # seconds to wait for MLflow before giving up on this poll


class MLflowError(Exception):
    """MLflow could not be reached, or answered with an error."""


class MLflowReader:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.base = tracking_uri.rstrip("/") + "/api/2.0/mlflow"
        self.experiment_name = experiment_name
        self._experiment_id = None

    def _call(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = requests.request(method, self.base + path, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as error:
            raise MLflowError(f"Cannot reach MLflow at {self.base}: {error}") from error
        if response.status_code != 200:
            raise MLflowError(f"MLflow answered {response.status_code} for {path}: {response.text[:200]}")
        return response.json()

    def experiment_id(self) -> str:
        """Find the experiment's id from its name (asked once, then remembered)."""
        if self._experiment_id is None:
            try:
                data = self._call("GET", "/experiments/get-by-name",
                                  params={"experiment_name": self.experiment_name})
            except MLflowError as error:
                if "RESOURCE_DOES_NOT_EXIST" in str(error):
                    raise MLflowError(
                        f"MLflow has no experiment named '{self.experiment_name}' yet "
                        "(check MLFLOW_EXPERIMENT in .env, or log a first run)"
                    ) from error
                raise
            self._experiment_id = data["experiment"]["experiment_id"]
        return self._experiment_id

    def finished_runs(self, since_ms: int = 0) -> list:
        """
        Return finished runs that started at or after since_ms (milliseconds since 1970),
        oldest first, as simple dicts: {run_id, run_name, start_time, metrics}.
        """
        body = {
            "experiment_ids": [self.experiment_id()],
            "filter": f"attributes.status = 'FINISHED' and attributes.start_time >= {int(since_ms)}",
            "order_by": ["attributes.start_time ASC"],
            "max_results": 500,
        }
        runs = []
        while True:
            data = self._call("POST", "/runs/search", json=body)
            runs.extend(_simple(run) for run in data.get("runs", []))
            token = data.get("next_page_token")
            if not token:
                return runs
            body["page_token"] = token

    def recent_runs(self, count: int = 20) -> list:
        """The newest finished runs, newest first (used by the setup wizard to see what is logged)."""
        body = {
            "experiment_ids": [self.experiment_id()],
            "filter": "attributes.status = 'FINISHED'",
            "order_by": ["attributes.start_time DESC"],
            "max_results": count,
        }
        return [_simple(run) for run in self._call("POST", "/runs/search", json=body).get("runs", [])]

    def experiments(self) -> list:
        """Names of all active experiments in this MLflow (used by the setup wizard)."""
        data = self._call("POST", "/experiments/search", json={"max_results": 1000})
        return [e["name"] for e in data.get("experiments", []) if e.get("lifecycle_stage", "active") == "active"]


def _simple(run: dict) -> dict:
    """Turn one MLflow run into {run_id, run_name, start_time, metrics}."""
    info = run["info"]
    return {
        "run_id": info["run_id"],
        "run_name": info.get("run_name", info["run_id"][:8]),
        "start_time": int(info["start_time"]),
        "metrics": {m["key"]: float(m["value"]) for m in run.get("data", {}).get("metrics", [])},
    }

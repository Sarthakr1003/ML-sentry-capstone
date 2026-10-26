"""
ML Sentry: metric history in SQLite.

Keeps a small record of every evaluation run ML Sentry has looked at, its
metric values, whether it passed, and every incident (alert) raised.
Rows are tiny, so the file stays small even if ML Sentry runs for months.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    run_name   TEXT,
    start_time INTEGER,          -- when the run started in MLflow (ms since 1970)
    checked_at TEXT,             -- when ML Sentry looked at it
    status     TEXT              -- ok, breach, or history (imported at first start, not checked)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT,
    name   TEXT,
    value  REAL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS incidents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    created_at TEXT,
    kind       TEXT,             -- breach or recovered
    severity   TEXT,             -- minor, moderate, critical (empty for recovered)
    engine     TEXT,             -- which brain judged it: groq, ollama or rules
    breaches   TEXT,             -- JSON list of the metrics that were out of bounds
    message    TEXT,
    sent_to    TEXT              -- channels the alert reached
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def is_empty(self) -> bool:
        return self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

    def seen(self, run_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def last_start_time(self) -> int:
        row = self.db.execute("SELECT MAX(start_time) FROM runs").fetchone()
        return row[0] or 0

    def save_run(self, run: dict, status: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (run["run_id"], run["run_name"], run["start_time"], _now(), status),
            )
            self.db.executemany(
                "INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)",
                [(run["run_id"], name, value) for name, value in run["metrics"].items()],
            )

    def last_status(self) -> str:
        """Status of the most recent checked run: ok, breach, or '' if none yet."""
        row = self.db.execute(
            "SELECT status FROM runs WHERE status != 'history' ORDER BY start_time DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else ""

    def breach_streak(self) -> int:
        """How many of the latest checked runs in a row were breaches."""
        streak = 0
        for (status,) in self.db.execute(
            "SELECT status FROM runs WHERE status != 'history' ORDER BY start_time DESC"
        ):
            if status != "breach":
                break
            streak += 1
        return streak

    def history(self, metric: str, limit: int = 10) -> list:
        """The last `limit` values of one metric, oldest first."""
        rows = self.db.execute(
            """SELECT m.value FROM metrics m JOIN runs r ON r.run_id = m.run_id
               WHERE m.name = ? ORDER BY r.start_time DESC LIMIT ?""",
            (metric, limit),
        ).fetchall()
        return [round(value, 4) for (value,) in reversed(rows)]

    def save_incident(self, run_id: str, kind: str, severity: str, engine: str,
                      breaches: list, message: str, sent_to: list) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO incidents (run_id, created_at, kind, severity, engine, breaches, message, sent_to) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, _now(), kind, severity, engine, json.dumps(breaches), message, ",".join(sent_to)),
            )

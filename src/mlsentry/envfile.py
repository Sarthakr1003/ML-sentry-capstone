"""
ML Sentry: reading and writing the .env file for the setup wizard.

Changes only the settings it is told to change. Comments, empty lines and every
other setting (for example the DEMO_ ones) stay exactly as they were, so a user
who also edits .env by hand loses nothing.

The file is rewritten in place (not replaced), because inside Docker .env is a
single mounted file that cannot be swapped for a new one.
"""

from pathlib import Path


def _key_of(line: str) -> str:
    """The setting name on a line like 'RMSE_MAX=0.6', or '' for comments and empty lines."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return ""
    key = stripped.split("=", 1)[0].strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    return key


def _number(value) -> str:
    """0.6 -> '0.6', 5.0 -> '5': short, but never rounded."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


class EnvFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []

    def get(self, key: str, default: str = "") -> str:
        for line in self.lines:
            if _key_of(line) == key:
                value = line.split("=", 1)[1].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                return value
        return default

    def has(self, key: str) -> bool:
        return any(_key_of(line) == key for line in self.lines)

    def set(self, key: str, value) -> None:
        """Change a setting where it already is, or add it at the end."""
        new_line = f"{key}={value}"
        for i, line in enumerate(self.lines):
            if _key_of(line) == key:
                self.lines[i] = new_line
                return
        self.lines.append(new_line)

    def remove(self, key: str) -> None:
        self.lines = [line for line in self.lines if _key_of(line) != key]

    def set_metrics(self, limits: dict) -> None:
        """
        Write the watched metrics and their limits, e.g.
            {"rmse": {"max": 0.6}, "r2": {"min": 0.7}}
        becomes METRICS=rmse,r2  RMSE_MAX=0.6  R2_MIN=0.7.
        Limits of metrics that are no longer watched are removed.
        """
        old = [m.strip() for m in self.get("METRICS").split(",") if m.strip()]
        for metric in set(old) | set(limits):
            for side in ("min", "max"):
                if side not in limits.get(metric, {}):
                    self.remove(f"{metric.upper()}_{side.upper()}")
        self.set("METRICS", ",".join(limits))
        for metric, bounds in limits.items():
            for side, value in bounds.items():
                self.set(f"{metric.upper()}_{side.upper()}", _number(value))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.lines) + "\n")

# ML Sentry container: only the watcher itself, no model and no demo code.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install ML Sentry and its few dependencies (see pyproject.toml)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Run as a normal user, not root; state/ holds the SQLite history
RUN useradd --create-home sentry && mkdir -p /app/state && chown sentry /app/state
USER sentry

CMD ["python", "-m", "mlsentry"]

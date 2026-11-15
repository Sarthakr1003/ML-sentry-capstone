"""
ML Sentry: the metrics it knows for each type of model.

The setup wizard uses this to recommend metrics, to know which way is "better"
for each one, and to check that a threshold makes sense. ML Sentry itself still
watches whatever names are in METRICS; this list only helps the user choose.

  direction "max" -> lower is better, alert when the value goes ABOVE the limit (errors, drift)
  direction "min" -> higher is better, alert when the value goes BELOW the limit (scores)
"""

from dataclasses import dataclass, field

MODEL_TYPES = {
    "regression": "Regression (predicts a number, e.g. a house price)",
    "classification": "Classification (predicts a class, e.g. approve / reject)",
    "rag": "RAG (an LLM answering from documents, evaluated with RAGAS)",
}

# Teams often add these to a metric's name, e.g. val_accuracy or test_rmse
PREFIXES = ("val_", "valid_", "validation_", "test_", "eval_", "eval/", "test/", "val/")


@dataclass(frozen=True)
class Metric:
    name: str
    label: str                    # plain-language meaning, shown in the wizard
    direction: str                # "max" or "min" (see top of file)
    low: float | None = None      # smallest sensible threshold (None = no bound)
    high: float | None = None     # largest sensible threshold (None = no bound)
    default: float | None = None  # suggested threshold when the latest value is not known
    aliases: tuple = field(default=())


CATALOG = {
    "regression": [
        Metric("rmse", "typical size of a prediction error", "max", low=0,
               aliases=("root_mean_squared_error",)),
        Metric("mae", "average size of a prediction error", "max", low=0,
               aliases=("mean_absolute_error",)),
        Metric("mse", "average squared prediction error", "max", low=0,
               aliases=("mean_squared_error",)),
        Metric("r2", "share of the variation the model explains (1 = perfect)", "min", high=1,
               default=0.7, aliases=("r2_score", "r_squared", "rsquared")),
        Metric("rmse_gap", "error gap between groups (bias)", "max", low=0, default=0.2),
        Metric("max_psi", "drift of the most-changed input feature (PSI)", "max", low=0,
               default=0.2, aliases=("psi",)),
    ],
    "classification": [
        Metric("accuracy", "share of correct predictions", "min", 0, 1, 0.8, aliases=("acc",)),
        Metric("precision", "of the predicted positives, share that are right", "min", 0, 1, 0.8),
        Metric("recall", "of the real positives, share that are found", "min", 0, 1, 0.8),
        Metric("f1", "balance of precision and recall", "min", 0, 1, 0.8,
               aliases=("f1_score", "f1score")),
        Metric("roc_auc", "how well the model ranks positives above negatives", "min", 0, 1, 0.8,
               aliases=("auc", "rocauc", "roc_auc_score")),
        Metric("log_loss", "how wrong the predicted probabilities are", "max", low=0,
               aliases=("logloss",)),
        Metric("max_psi", "drift of the most-changed input feature (PSI)", "max", low=0,
               default=0.2, aliases=("psi",)),
    ],
    "rag": [
        Metric("faithfulness", "answers stick to the retrieved documents", "min", 0, 1, 0.8),
        Metric("answer_relevancy", "answers address the question", "min", 0, 1, 0.8,
               aliases=("answer_relevance",)),
        Metric("context_precision", "retrieved documents are relevant", "min", 0, 1, 0.8),
        Metric("context_recall", "the needed documents were retrieved", "min", 0, 1, 0.8),
        Metric("answer_correctness", "answers match the reference answers", "min", 0, 1, 0.8),
    ],
}


def _plain(name: str) -> str:
    """Lower-case a logged metric name and drop a common prefix like val_ or test_."""
    name = name.strip().lower()
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def known_metric(model_type: str, logged_name: str):
    """The catalog entry a logged name stands for (e.g. 'val_acc' -> accuracy), or None."""
    plain = _plain(logged_name)
    for metric in CATALOG[model_type]:
        if plain == metric.name or plain in metric.aliases:
            return metric
    return None


def suggest_threshold(direction: str, latest: float | None, default: float | None) -> float | None:
    """
    A starting point for the limit, a little worse than the model is doing now:
    25% above the latest value for "lower is better", 15% below it for "higher is better".
    Falls back to the catalog default when MLflow has no value yet.
    """
    if latest is None:
        return default
    suggestion = latest * 1.25 if direction == "max" else latest * 0.85
    return round(suggestion, 3)


def parse_threshold(text: str, low: float | None = None, high: float | None = None):
    """
    Turn what the user typed into a number.
    Returns (value, None) if it is valid, or (None, reason) if not.
    """
    text = (text or "").strip()
    if not text:
        return None, "Please type a number, e.g. 0.6"
    try:
        value = float(text)
    except ValueError:
        return None, f"'{text}' is not a number. Please type a number, e.g. 0.6"
    if value != value or value in (float("inf"), float("-inf")):  # NaN or infinity
        return None, "Please type a normal number, e.g. 0.6"
    if low is not None and value < low:
        return None, f"The limit cannot be below {low:g}"
    if high is not None and value > high:
        return None, f"The limit cannot be above {high:g}"
    return value, None

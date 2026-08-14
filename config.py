# ============================================================
# config.py — Central project configuration (single source of truth)
# ============================================================
from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────
RANDOM_STATE = 42

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "final_model.joblib"

# ─── Columns ──────────────────────────────────────────────
ID_COLUMNS = ["UDI", "Product ID"]

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

BINARY_TARGET = "Machine failure"
FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MULTI_CLASS_TARGET = "Failure Type"
CATEGORICAL_COL = "Type"

ENGINEERED_COLUMNS = ["Power [W]", "Temp Diff [K]", "Overstrain [min·Nm]"]

# ─── Best model hyperparameters (from GridSearchCV) ───────
BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
}

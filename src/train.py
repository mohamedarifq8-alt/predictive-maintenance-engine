# ============================================================
# train.py — Train and save the final model
# ============================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from data_loader import load_and_prepare
from features import engineer_features
from preprocessing import build_preprocessor


def build_features_targets(df: pd.DataFrame):
    """Assemble feature matrix X and binary target y."""
    df = engineer_features(df)
    feature_cols = config.SENSOR_COLUMNS + config.ENGINEERED_COLUMNS + [config.CATEGORICAL_COL]
    X = df[feature_cols]
    y = df[config.BINARY_TARGET]
    return X, y


def main():
    np.random.seed(config.RANDOM_STATE)

    # Load + build targets + engineer features
    df = load_and_prepare()
    X, y = build_features_targets(df)

    numeric_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            **config.BEST_PARAMS,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
        )),
    ])

    model.fit(X_train, y_train)

    # Save the full pipeline (preprocessor + classifier)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    print(f"Model saved to {config.MODEL_FILE}")


if __name__ == "__main__":
    main()

# ============================================================
# evaluate.py — Evaluate the trained model on the test set
# ============================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import config
from data_loader import load_and_prepare
from features import engineer_features
from train import build_features_targets


def main():
    np.random.seed(config.RANDOM_STATE)

    df = load_and_prepare()
    X, y = build_features_targets(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y,
    )

    import joblib
    model = joblib.load(config.MODEL_FILE)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("Evaluation on test set")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    main()

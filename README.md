# 🏭 Industrial Predictive Maintenance & Fault Diagnostic Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)](https://xgboost.ai)

An end-to-end machine learning system that **predicts imminent industrial equipment failures before they happen** and **diagnoses their failure type**, based on live sensor readings — to prevent unplanned downtime and reduce maintenance costs.

---

## 🎯 Problem & Solution

| | |
|---|---|
| **Problem** | Unexpected equipment failures stop production lines and cost companies millions |
| **Solution** | A predictive model that determines *when*, *which machine*, and *what type of failure* — before it occurs |

## 🧩 Task Definition

- **Binary Classification** — Will the machine fail? (`Machine failure` = 0/1)
- **Multi-class Classification** — What failure type? (TWF / HDF / PWF / OSF / RNF + co-occurring failures)

## 📊 Dataset

The [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 records of an industrial milling machine with 5 sensors:

| Sensor | Unit |
|--------|------|
| Air temperature | K |
| Process temperature | K |
| Rotational speed | rpm |
| Torque | Nm |
| Tool wear | min |

---

## 🛠️ Methodology (CRISP-DM)

```
[Problem Framing] → [EDA] → [Feature Engineering] → [Modeling & Tuning] → [SHAP Interpretation] → [Packaging]
```

## 🔬 Exploratory Data Analysis

### Sensor Distributions
![Sensor Distributions](assets/distributions.png)

### Healthy vs Failed
![Healthy vs Failed](assets/boxplot_comparison.png)

### Correlation Matrix
![Correlation](assets/correlation.png)

### Failure Rate by Product Quality
![Failure by Type](assets/failure_by_type.png)

### Key Findings

1. **Documented internal inconsistency** in the original data between failure-type flags and the `Machine failure` column — reconciled with a clear policy.
2. **Severe class imbalance** (3.39% failures) — handled with `scale_pos_weight` + stratified cross-validation.
3. **Tool wear** is the leading driver of failures; product quality `Type` clearly affects failure rate.
4. **Physics-based feature engineering** derived from the known failure-generation rules: `Power`, `Temp Diff`, `Overstrain`.

---

## 🧪 Modeling

Three models compared fairly via 5-fold stratified cross-validation, then tuned with `GridSearchCV`:

| Model | Role |
|-------|------|
| Random Forest | Baseline |
| XGBoost | Advanced (best) |
| LightGBM | Advanced alternative |

**Chosen metric: Recall** — because missing a real failure is far more costly than a false alarm in predictive maintenance.

## 📈 Results (Tuned XGBoost)

| Metric | Value |
|--------|-------|
| **Recall (failure detection)** | **88.2%** |
| Precision | 61.9% |
| F1-Score | 72.7% |
| Accuracy | 97.75% |
| ROC-AUC | **98.6%** |

> 💡 **Recall was deliberately prioritized over Precision**: the cost of a missed failure (production downtime) vastly outweighs the cost of a false alarm (routine check).

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### ROC Curve
![ROC Curve](assets/roc_curve.png)

### SHAP Feature Importance
![SHAP](assets/shap_summary.png)

SHAP confirms that **Tool Wear**, followed by **rotational speed drop** and **power/overstrain**, are the strongest signals the model captures — consistent with the real physics of the failures.

---

## ✨ Key Features

- ✅ Full **CRISP-DM** lifecycle (6 phases) in clean, documented Python.
- ✅ **Physics-based feature engineering** (Power, Temp Diff, Overstrain).
- ✅ **Imbalanced-data handling** (`scale_pos_weight`) + **stratified** validation.
- ✅ **Hyperparameter tuning** with `GridSearchCV`.
- ✅ **Model interpretability** with SHAP — engineer-friendly, not a black box.
- ✅ **Production-ready packaging**: modular `src/`, `config.py`, saved `.joblib` model.

## 🚀 Quick Start

```bash
git clone <repo-url>
cd predictive-maintenance-engine
pip install -r requirements.txt
python src/train.py       # train & save the model
python src/evaluate.py    # evaluate the saved model
```

## 📁 Repository Structure

```
predictive-maintenance-engine/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py                    # central configuration
├── data/
│   └── ai4i2020.csv
├── assets/                      # analysis figures
├── notebooks/                   # five phase notebooks
├── models/                      # saved model (.joblib)
└── src/
    ├── data_loader.py
    ├── features.py
    ├── preprocessing.py
    ├── train.py
    └── evaluate.py
```

## 📄 License

MIT License — see [LICENSE](LICENSE).

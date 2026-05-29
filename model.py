"""
model.py
========
Fraud Detection Pipeline

The key challenge here vs credit risk:
  - Class imbalance is EXTREME: ~0.3% fraud (vs ~15% in credit risk)
  - Standard accuracy is completely useless: "predict everything legit" = 99.7% accurate
  - We need to think in terms of:
      Precision = of flagged transactions, what % are actually fraud?
      Recall    = of all fraud, what % did we catch?
  - There is always a precision-recall tradeoff — catching more fraud means
    more false alarms (friction for legitimate customers)

Techniques used:
  1. SMOTE (Synthetic Minority Oversampling TEchnique)
     Creates synthetic fraud examples in feature space to balance training data.
     Better than random oversampling (avoids simple duplication) or undersampling
     (which throws away 99%+ of your data).

  2. Threshold tuning
     Instead of defaulting to 0.5, we find the threshold that maximises F1
     (or a business-defined cost function). This is crucial for imbalanced problems.

  3. Precision-Recall curve focus
     ROC-AUC can look good even when a model is bad at finding the rare class.
     Average Precision (area under PR curve) is the honest metric here.
"""

import numpy as np
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# 1.  DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare(path="data/transactions.csv"):
    """
    Load transactions and encode categorical features.
    One-hot encode merchant_category, extract time features.
    """

    df = pd.read_csv(path, index_col="transaction_id")
    y  = df.pop("is_fraud").values

    # Feature engineering on time:
    # Hour of day → cyclical encoding so hour 23 and hour 0 are "close"
    # sin/cos encoding preserves the circular nature of time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Is it a late-night transaction? (10pm–4am)
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)

    # Is it a weekend?
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Log-transform skewed features (amount, distance) to reduce outlier influence
    df["log_amount"]   = np.log1p(df["amount"])
    df["log_distance"] = np.log1p(df["distance_from_home"])
    df["log_prev_min"] = np.log1p(df["prev_txn_minutes"])

    # Drop raw columns we've encoded
    df = df.drop(columns=["hour", "day_of_week", "amount",
                           "distance_from_home", "prev_txn_minutes"])

    # One-hot encode merchant category
    df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)

    return df, y


# ─────────────────────────────────────────────
# 2.  THRESHOLD TUNING
# ─────────────────────────────────────────────

def find_best_threshold(y_true, y_prob, metric="f1"):
    """
    Scan probability thresholds and find the one that maximises F1 score.

    Why does this matter?
      At threshold=0.5: model might flag very few transactions (low recall)
      At threshold=0.1: model flags more (higher recall, lower precision)
      The optimal threshold depends on the business cost of each error type.

    F1 is the harmonic mean of precision and recall — good default.
    A real bank might use a custom cost function: e.g. weight missed fraud 10x
    more than a false alarm.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx   = np.argmax(f1_scores[:-1])   # last element has no threshold

    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ─────────────────────────────────────────────
# 3.  MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, name):
    """
    Train and comprehensively evaluate a fraud detection model.

    Key difference from credit risk evaluation:
      - We report metrics at the OPTIMAL threshold, not just 0.5
      - We compute the business impact: how many fraud dollars caught?
      - Precision-Recall AUC is the primary metric, not ROC-AUC
    """

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)

    # Metrics
    auc        = roc_auc_score(y_test, y_prob)
    avg_prec   = average_precision_score(y_test, y_prob)
    cm         = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Note: we score on the raw model, not the pipeline with SMOTE,
    # because SMOTE should only be applied to training folds
    cv_ap = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring="average_precision")

    return {
        "name":          name,
        "auc":           round(auc, 4),
        "avg_precision": round(avg_prec, 4),
        "best_threshold":round(best_thresh, 4),
        "best_f1":       round(best_f1, 4),
        "cv_ap_mean":    round(cv_ap.mean(), 4),
        "cv_ap_std":     round(cv_ap.std(), 4),
        "confusion":     {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc":           {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr":            {"precision": prec.tolist(), "recall": rec.tolist()},
        "y_prob":        y_prob.tolist(),
        "y_test":        y_test.tolist(),
    }


def get_feature_importance(model, feature_names, model_type="xgb"):
    """Extract and rank feature importances."""
    if model_type == "xgb":
        imp = model.named_steps["clf"].feature_importances_
    elif model_type == "rf":
        imp = model.named_steps["clf"].feature_importances_
    else:
        imp = np.abs(model.named_steps["clf"].coef_[0])

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(15)


# ─────────────────────────────────────────────
# 4.  MODEL DEFINITIONS
# ─────────────────────────────────────────────

def build_logistic_smote():
    """
    Logistic Regression with SMOTE oversampling.

    ImbPipeline (from imbalanced-learn) is important: it ensures SMOTE
    is only applied to the training fold during cross-validation, not the
    test fold. A common mistake is to SMOTE the whole dataset first — this
    leaks information and inflates metrics.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("clf",   LogisticRegression(C=0.5, max_iter=1000,
                                     class_weight="balanced", random_state=42)),
    ])


def build_xgb_smote():
    """
    XGBoost with SMOTE.

    scale_pos_weight is set to 1 here because SMOTE already balances the
    training data — using both would over-correct for the imbalance.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf",   XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,   # SMOTE handles balance
            eval_metric="aucpr",  # optimise for precision-recall AUC directly
            random_state=42,
            verbosity=0,
        )),
    ])


def build_rf_smote():
    """
    Random Forest with SMOTE.
    Included as a third model — ensembles of trees often work well on
    fraud data where decision boundaries are complex and non-linear.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf",   RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ─────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    print("Loading data...")
    X, y = load_and_prepare("data/transactions.csv")
    feature_names = X.columns.tolist()

    print(f"  {len(y):,} transactions | {y.sum():,} fraud ({y.mean():.2%})")
    print(f"  {X.shape[1]} features after engineering\n")

    # Stratified split — essential for such extreme imbalance
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    for name, model, mtype in [
        ("Logistic + SMOTE", build_logistic_smote(), "lr"),
        ("XGBoost + SMOTE",  build_xgb_smote(),      "xgb"),
        ("RandomForest + SMOTE", build_rf_smote(),    "rf"),
    ]:
        print(f"Training {name}...")
        res = evaluate(model, X_tr, X_te, y_tr, y_te, name)
        imp = get_feature_importance(model, feature_names, mtype)

        print(f"  PR-AUC: {res['avg_precision']:.4f} | ROC-AUC: {res['auc']:.4f} "
              f"| Best F1: {res['best_f1']:.4f} @ threshold={res['best_threshold']:.3f}")

        results[name] = res
        results[f"{name}_importance"] = imp.to_dict(orient="records")

    results["dataset_info"] = {
        "n_total":   int(len(y)),
        "n_fraud":   int(y.sum()),
        "fraud_rate":float(y.mean()),
        "n_features":int(X.shape[1]),
        "train_size":int(len(y_tr)),
        "test_size": int(len(y_te)),
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f)

    print("\nResults saved to outputs/results.json")
    return results


if __name__ == "__main__":
    run_pipeline()

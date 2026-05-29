# Fraud Detection: Imbalanced Classification with SMOTE

A machine learning project detecting credit card fraud — one of the most challenging
classification problems in practice due to extreme class imbalance (0.3% fraud rate)
and high asymmetry in the cost of errors.

---

## The Core Challenge

**0.3% fraud rate means standard accuracy is meaningless.**

A model that predicts "legitimate" for every transaction achieves 99.7% accuracy —
and catches zero fraud. This project is built around the correct metrics and techniques
for imbalanced classification.

| Error type | Business cost |
|---|---|
| **False Negative** (missed fraud) | Full transaction loss — the most expensive mistake |
| **False Positive** (false alarm) | Customer friction, analyst review time — less costly |

This asymmetry shapes every modelling decision: the metric, the threshold, the sampling strategy.

---

## Dataset

100,000 synthetic credit card transactions with realistic fraud patterns:

**Fraud signals built into the data:**
- Higher transaction amounts on average
- More late-night activity (10pm–4am)
- More foreign transactions (35% vs 4% for legitimate)
- More high-risk merchant categories (40% vs 3%)
- Rapid successive transactions (short time since last transaction)
- Unusual amount relative to cardholder's historical average
- More travel and "other" merchant categories

**300 fraudulent transactions (0.30%)** — matching real-world fraud rates.

---

## Techniques

### SMOTE (Synthetic Minority Oversampling TEchnique)
Creates synthetic fraud examples in feature space rather than simply duplicating
existing ones. Applied inside an `ImbLearn Pipeline` to ensure it's only applied
to training folds during cross-validation — a common mistake is applying SMOTE
before splitting, which leaks information.

### Feature Engineering
- **Cyclical time encoding** — sin/cos transform of hour and day-of-week, so hour 23
  and hour 0 are treated as adjacent (as they are in reality)
- **Log transforms** — amount, distance, and time-since-last-transaction are
  right-skewed; log transform reduces outlier influence
- **Derived flags** — `is_night`, `is_weekend` as binary features
- **Ratio features** — `amount_vs_avg` captures deviation from cardholder baseline

### Threshold Optimisation
Instead of defaulting to 0.5, we scan all thresholds and pick the one maximising F1.
Reported in the dashboard with an interactive slider to explore the precision-recall tradeoff.

### Three Models Compared
- **Logistic Regression + SMOTE** — interpretable baseline
- **XGBoost + SMOTE** — strongest performer (PR-AUC 0.98)
- **Random Forest + SMOTE** — tree ensemble comparison

---

## Results

| Model | PR-AUC | ROC-AUC | Best F1 | Optimal Threshold |
|---|---|---|---|---|
| Logistic + SMOTE | 0.9681 | 0.9997 | 0.9580 | 0.999 |
| **XGBoost + SMOTE** | **0.9800** | **0.9997** | **0.9748** | 0.650 |
| RandomForest + SMOTE | 0.9515 | 0.9991 | 0.9138 | 0.687 |

XGBoost leads on all metrics. The synthetic dataset has clean structure, so high
performance is expected — real fraud data has more noise (concept drift, adversarial
fraudsters) and PR-AUC of 0.30–0.60 is typical in production.

---

## Dashboard (5 pages)

| Page | Content |
|---|---|
| **Overview & Metrics** | Why accuracy fails, model KPIs, confusion matrices at optimal threshold |
| **Precision-Recall Analysis** | PR + ROC curves, interactive threshold slider |
| **Fraud Patterns** | EDA: amount distributions, hour-of-day, distance, merchant category rates |
| **Score Distribution** | Probability histograms by actual class, optimal threshold overlay |
| **Flag a Transaction** | Enter any transaction details, get live predictions from all 3 models |

---

## Running

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/model.py
streamlit run app.py
```

---

## Project Structure

```
fraud_detection/
├── app.py                   # 5-page Streamlit dashboard
├── src/
│   ├── generate_data.py     # Synthetic transaction generation
│   └── model.py             # Pipeline: SMOTE, training, evaluation, threshold tuning
├── data/
│   └── transactions.csv     # Generated dataset
├── outputs/
│   └── results.json         # Cached model results
└── README.md
```

---

## Skills Demonstrated

- **Imbalanced classification** — SMOTE, class weighting, stratified CV, ImbLearn pipelines
- **Correct metric selection** — PR-AUC over accuracy/ROC-AUC for rare event detection
- **Threshold tuning** — scanning thresholds to optimise F1 or custom cost functions
- **Feature engineering** — cyclical encoding, log transforms, ratio features
- **Model comparison** — 3 models with principled evaluation framework
- **Business framing** — cost asymmetry, false negative vs false positive trade-offs
- **Python** — scikit-learn, imbalanced-learn, XGBoost, Plotly, Streamlit

---

*Synthetic data. For educational/portfolio use only.*

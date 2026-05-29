"""
generate_data.py
================
Generates a realistic synthetic credit card transaction dataset for fraud detection.

Real-world fraud datasets (like the famous Kaggle credit card dataset) have:
  - Extreme class imbalance: fraud is typically 0.1% – 0.5% of transactions
  - Transactions spanning multiple time windows (day/night, weekend/weekday)
  - Fraud patterns: unusual amounts, foreign locations, late-night activity,
    rapid successive transactions, mismatched merchant categories

We replicate all of this synthetically so the models have genuine signal to learn.

Why is fraud detection HARDER than credit risk?
  - Even more extreme imbalance (0.3% vs ~15% in credit risk)
  - Fraudsters actively adapt to detection — patterns shift over time
  - Cost asymmetry: a missed fraud (false negative) costs the full transaction;
    a false alarm (false positive) costs customer friction + analyst time
"""

import numpy as np
import pandas as pd


def generate_transactions(n: int = 100_000, fraud_rate: float = 0.003, seed: int = 42) -> pd.DataFrame:
    """
    Generate n credit card transactions with realistic fraud patterns.

    Features
    --------
    transaction_id     : unique ID
    amount             : transaction amount ($)
    hour               : hour of day (0–23)
    day_of_week        : 0=Monday … 6=Sunday
    merchant_category  : type of merchant
    distance_from_home : km from cardholder's home address
    foreign_transaction: 1 if in a different country
    high_risk_merchant : 1 if casino / wire transfer / gift cards etc.
    prev_txn_minutes   : minutes since previous transaction on this card
    txn_count_24h      : how many transactions on this card in past 24h
    amount_vs_avg      : this amount / cardholder's historical average amount
    is_fraud           : TARGET — 1 = fraudulent transaction
    """

    rng = np.random.default_rng(seed)
    n_fraud  = int(n * fraud_rate)
    n_legit  = n - n_fraud

    # ── Legitimate transactions ───────────────────────────────────────────────

    # Amounts: log-normal, typical everyday spending ($5–$500)
    legit_amount = np.exp(rng.normal(3.8, 0.9, n_legit)).clip(1, 5000)

    # Hours: bimodal — lunch (12–14) and evening (18–21), quiet at night
    legit_hour = rng.choice(
        np.arange(24),
        size=n_legit,
        p=(lambda a: a/a.sum())(np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1,
                    0.5, 1.5, 2.5, 2.0, 2.0, 2.5,
                    3.5, 3.5, 2.5, 2.0, 2.5, 3.0,
                    4.0, 4.0, 3.5, 3.0, 2.0, 1.0], dtype=float)),
    )

    legit_dow     = rng.integers(0, 7, n_legit)
    legit_dist    = np.exp(rng.normal(2.5, 1.2, n_legit)).clip(0, 2000)  # mostly local
    legit_foreign = (rng.random(n_legit) < 0.04).astype(int)   # 4% foreign
    legit_highrisk= (rng.random(n_legit) < 0.03).astype(int)   # 3% high-risk merchant

    # Time since last transaction: mostly hours apart
    legit_prev_min = np.exp(rng.normal(5.0, 1.5, n_legit)).clip(1, 10000)

    # Transaction count in past 24h: usually 1–5
    legit_count24h = rng.integers(1, 8, n_legit)

    # Amount vs cardholder's average: usually close to 1.0
    legit_amt_ratio = rng.lognormal(0.0, 0.4, n_legit).clip(0.1, 10)

    legit_cat = rng.choice(
        ["grocery", "restaurant", "gas", "retail", "entertainment", "travel", "other"],
        size=n_legit,
        p=[0.25, 0.20, 0.15, 0.20, 0.08, 0.07, 0.05],
    )

    # ── Fraudulent transactions ───────────────────────────────────────────────
    # Fraud patterns baked in:
    #   - Higher amounts (fraudsters test limits or go big)
    #   - More late-night activity
    #   - More foreign, high-risk merchants
    #   - Rapid succession (multiple fraud attempts in a short window)
    #   - Unusual amount vs cardholder's history

    fraud_amount  = np.exp(rng.normal(5.0, 1.2, n_fraud)).clip(10, 8000)

    # Fraud is more uniformly distributed through the night
    fraud_hour = rng.choice(
        np.arange(24),
        size=n_fraud,
        p=(lambda a: a/a.sum())(np.array([3, 3, 3, 3, 2, 1,
                    1, 1, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 3,
                    3, 3, 3, 3, 3, 3], dtype=float)),
    )

    fraud_dow      = rng.integers(0, 7, n_fraud)
    fraud_dist     = np.exp(rng.normal(5.0, 1.5, n_fraud)).clip(0, 20000)  # further from home
    fraud_foreign  = (rng.random(n_fraud) < 0.35).astype(int)   # 35% foreign
    fraud_highrisk = (rng.random(n_fraud) < 0.40).astype(int)   # 40% high-risk merchant

    # Rapid succession: fraud often follows very quickly
    fraud_prev_min = np.exp(rng.normal(1.5, 1.5, n_fraud)).clip(0.1, 500)

    # Multiple fraud attempts close together
    fraud_count24h = rng.integers(3, 20, n_fraud)

    # Amount very different from cardholder's normal
    fraud_amt_ratio = rng.lognormal(1.0, 1.0, n_fraud).clip(0.5, 50)

    fraud_cat = rng.choice(
        ["grocery", "restaurant", "gas", "retail", "entertainment", "travel", "other"],
        size=n_fraud,
        p=[0.05, 0.05, 0.05, 0.15, 0.10, 0.30, 0.30],  # skewed to travel/other
    )

    # ── Combine and shuffle ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "amount":             np.concatenate([legit_amount,  fraud_amount]),
        "hour":               np.concatenate([legit_hour,    fraud_hour]),
        "day_of_week":        np.concatenate([legit_dow,     fraud_dow]),
        "merchant_category":  np.concatenate([legit_cat,     fraud_cat]),
        "distance_from_home": np.concatenate([legit_dist,    fraud_dist]).round(1),
        "foreign_transaction":np.concatenate([legit_foreign, fraud_foreign]),
        "high_risk_merchant": np.concatenate([legit_highrisk,fraud_highrisk]),
        "prev_txn_minutes":   np.concatenate([legit_prev_min,fraud_prev_min]).round(1),
        "txn_count_24h":      np.concatenate([legit_count24h,fraud_count24h]),
        "amount_vs_avg":      np.concatenate([legit_amt_ratio,fraud_amt_ratio]).round(3),
        "is_fraud":           np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int),
    })

    # Shuffle rows (so fraud isn't all at the end)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.index.name = "transaction_id"

    return df


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("data/transactions.csv")
    print(f"Generated {len(df):,} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}  ({df['is_fraud'].sum():,} fraudulent)")
    print(f"\nFeatures:\n{df.dtypes}")
    print(f"\nSample fraud transaction:\n{df[df['is_fraud']==1].head(2).to_string()}")

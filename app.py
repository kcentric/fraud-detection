"""
app.py — Fraud Detection Dashboard
Run: streamlit run app.py
     (run src/generate_data.py and src/model.py first)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_data import generate_transactions
from model import load_and_prepare, run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
  }
  h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

  .stat {
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
  }
  .stat-label { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
  .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 500; }
  .stat-sub   { color: #475569; font-size: 11px; margin-top: 4px; }

  .red    { color: #f87171; }
  .green  { color: #34d399; }
  .blue   { color: #60a5fa; }
  .amber  { color: #fbbf24; }
  .purple { color: #a78bfa; }

  .alert-fraud {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
  }
  .alert-legit {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-left: 4px solid #34d399;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
  }

  div[data-testid="stSidebar"] { background: #07070d; border-right: 1px solid #1e1e2e; }
</style>
""", unsafe_allow_html=True)

BG    = "#0a0a0f"
CARD  = "#12121a"
GRID  = "#1e1e2e"
FONT  = dict(color="#94a3b8", family="Space Grotesk")
MODEL_COLORS = {"Logistic + SMOTE": "#60a5fa", "XGBoost + SMOTE": "#34d399", "RandomForest + SMOTE": "#a78bfa"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fraud\nDetection")
    st.markdown("---")
    page = st.radio("", [
        "Overview & Metrics",
        "Precision-Recall Analysis",
        "Fraud Patterns",
        "Score Distribution",
        "Flag a Transaction",
    ], label_visibility="collapsed")

# ── Load results ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training models… (~60s)")
def load_results():
    if not os.path.exists("data/transactions.csv"):
        os.makedirs("data", exist_ok=True)
        df = generate_transactions()
        df.to_csv("data/transactions.csv")
    if not os.path.exists("outputs/results.json"):
        return run_pipeline()
    with open("outputs/results.json") as f:
        return json.load(f)

results = load_results()
info    = results["dataset_info"]
models  = ["Logistic + SMOTE", "XGBoost + SMOTE", "RandomForest + SMOTE"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Credit Card Fraud Detection")
st.markdown(
    f"**{info['n_total']:,} transactions** · "
    f"**{info['n_fraud']:,} fraudulent** · "
    f"**{info['fraud_rate']:.2%} fraud rate** · "
    f"3 models with SMOTE oversampling"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview & Metrics":

    st.markdown("### Why accuracy is useless here")
    st.markdown("""
    <div class="alert-fraud">
    <b>The naive baseline:</b> predict "legit" for every transaction → <b>99.70% accuracy</b>.
    But it catches <b>zero fraud</b>. This is why we ignore accuracy entirely and focus on
    Precision-Recall AUC, F1, and threshold-tuned confusion matrices.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Model Performance")

    # KPI cards per model
    for m in models:
        r = results[m]
        col1, col2, col3, col4, col5 = st.columns(5)
        color = MODEL_COLORS[m].replace("#", "")

        with col1:
            st.markdown(f"<div style='color:{MODEL_COLORS[m]};font-weight:700;padding:8px 0'>{m}</div>",
                        unsafe_allow_html=True)
        for col, label, val, sub in [
            (col2, "PR-AUC",     f"{r['avg_precision']:.4f}", "Primary metric"),
            (col3, "ROC-AUC",    f"{r['auc']:.4f}",           "Secondary"),
            (col4, "Best F1",    f"{r['best_f1']:.4f}",        f"@ threshold {r['best_threshold']:.3f}"),
            (col5, "CV PR-AUC",  f"{r['cv_ap_mean']:.4f}",    f"±{r['cv_ap_std']:.4f}"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat">
                  <div class="stat-label">{label}</div>
                  <div class="stat-value" style="color:{MODEL_COLORS[m]}">{val}</div>
                  <div class="stat-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### Confusion Matrices (at optimal threshold)")

    cols = st.columns(3)
    for col, m in zip(cols, models):
        with col:
            r  = results[m]
            cm = r["confusion"]
            tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
            total = tn + fp + fn + tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

            st.markdown(f"<div style='color:{MODEL_COLORS[m]};font-weight:600;margin-bottom:8px'>{m}</div>",
                        unsafe_allow_html=True)

            fig = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp]],
                text=[[f"{tn:,}<br><span style='font-size:10px'>{tn/total:.1%}</span>",
                       f"{fp:,}<br><span style='font-size:10px'>{fp/total:.1%}</span>"],
                      [f"{fn:,}<br><span style='font-size:10px'>{fn/total:.1%}</span>",
                       f"{tp:,}<br><span style='font-size:10px'>{tp/total:.1%}</span>"]],
                texttemplate="%{text}",
                colorscale=[[0,"#12121a"],[0.3,"#1e3a5f"],[1, MODEL_COLORS[m]]],
                showscale=False, xgap=4, ygap=4,
            ))
            fig.update_layout(
                xaxis=dict(ticktext=["Pred: Legit","Pred: Fraud"], tickvals=[0,1]),
                yaxis=dict(ticktext=["Actual: Legit","Actual: Fraud"], tickvals=[0,1], autorange="reversed"),
                plot_bgcolor=CARD, paper_bgcolor=CARD, font=FONT,
                margin=dict(l=10,r=10,t=10,b=10), height=230,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"Precision: **{precision:.1%}** · Recall: **{recall:.1%}**  \n"
                f"Missed fraud: **{fn}** · False alarms: **{fp}**"
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PR CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Precision-Recall Analysis":
    st.markdown("### Precision-Recall Curves")
    st.caption(
        "For imbalanced problems, PR curves are more informative than ROC curves. "
        "The baseline (dashed line) is the fraud rate — a random classifier's PR-AUC. "
        "Both curves show the tradeoff: higher recall → lower precision (more false alarms)."
    )

    col_pr, col_roc = st.columns(2)

    with col_pr:
        fig = go.Figure()
        fig.add_hline(y=info["fraud_rate"], line_dash="dot", line_color="#475569",
                      annotation_text=f"Baseline = {info['fraud_rate']:.2%}",
                      annotation_position="bottom right")
        for m in models:
            r = results[m]
            fig.add_trace(go.Scatter(
                x=r["pr"]["recall"], y=r["pr"]["precision"],
                mode="lines", name=f"{m} (AP={r['avg_precision']:.4f})",
                line=dict(color=MODEL_COLORS[m], width=2.5),
            ))
        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis=dict(title="Recall", gridcolor=GRID, range=[0,1]),
            yaxis=dict(title="Precision", gridcolor=GRID, range=[0,1]),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.35),
            margin=dict(l=10,r=10,t=40,b=10), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_roc:
        fig2 = go.Figure()
        fig2.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                       line=dict(dash="dot",color="#475569"))
        for m in models:
            r = results[m]
            fig2.add_trace(go.Scatter(
                x=r["roc"]["fpr"], y=r["roc"]["tpr"],
                mode="lines", name=f"{m} (AUC={r['auc']:.4f})",
                line=dict(color=MODEL_COLORS[m], width=2.5),
            ))
        fig2.update_layout(
            title="ROC Curves",
            xaxis=dict(title="False Positive Rate", gridcolor=GRID),
            yaxis=dict(title="True Positive Rate", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)", x=0.4, y=0.1),
            margin=dict(l=10,r=10,t=40,b=10), height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### The Threshold Tradeoff")
    st.caption("Move the threshold to see how precision and recall change for XGBoost.")

    threshold = st.slider("Decision Threshold", 0.01, 0.99, 0.5, 0.01)

    xgb_r  = results["XGBoost + SMOTE"]
    y_prob = np.array(xgb_r["y_prob"])
    y_test = np.array(xgb_r["y_test"])
    y_pred = (y_prob >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix as cm_fn
    tn2, fp2, fn2, tp2 = cm_fn(y_test, y_pred).ravel()
    prec2 = tp2 / (tp2 + fp2 + 1e-9)
    rec2  = tp2 / (tp2 + fn2 + 1e-9)
    f1_2  = 2 * prec2 * rec2 / (prec2 + rec2 + 1e-9)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, color in [
        (c1, "Threshold",  f"{threshold:.2f}", "blue"),
        (c2, "Precision",  f"{prec2:.1%}",     "green"),
        (c3, "Recall",     f"{rec2:.1%}",       "amber"),
        (c4, "F1 Score",   f"{f1_2:.4f}",       "purple"),
        (c5, "False Alarms", f"{fp2:,}",        "red"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat">
              <div class="stat-label">{label}</div>
              <div class="stat-value {color}">{val}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FRAUD PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fraud Patterns":
    st.markdown("### Fraud Patterns in the Data")
    st.caption("Exploratory analysis of what distinguishes fraudulent from legitimate transactions.")

    df_raw = pd.read_csv("data/transactions.csv", index_col="transaction_id")
    fraud  = df_raw[df_raw["is_fraud"] == 1]
    legit  = df_raw[df_raw["is_fraud"] == 0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Transaction Amount Distribution")
        fig = go.Figure()
        for subset, name, color in [(legit, "Legitimate", "#60a5fa"), (fraud, "Fraud", "#f87171")]:
            fig.add_trace(go.Histogram(
                x=np.log1p(subset["amount"]), nbinsx=60,
                name=name, opacity=0.7, marker_color=color,
                histnorm="probability density",
            ))
        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="log(1 + Amount)", gridcolor=GRID),
            yaxis=dict(title="Density", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=20,b=10), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Hour of Day")
        hours = np.arange(24)
        fig2 = go.Figure()
        for subset, name, color in [(legit, "Legitimate", "#60a5fa"), (fraud, "Fraud", "#f87171")]:
            counts = subset["hour"].value_counts().reindex(hours, fill_value=0)
            counts = counts / counts.sum()
            fig2.add_trace(go.Bar(x=hours, y=counts, name=name,
                                  marker_color=color, opacity=0.8))
        fig2.update_layout(
            barmode="overlay",
            xaxis=dict(title="Hour of Day", gridcolor=GRID, dtick=3),
            yaxis=dict(title="Proportion", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=20,b=10), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Distance from Home")
        fig3 = go.Figure()
        for subset, name, color in [(legit, "Legitimate", "#60a5fa"), (fraud, "Fraud", "#f87171")]:
            fig3.add_trace(go.Histogram(
                x=np.log1p(subset["distance_from_home"]), nbinsx=50,
                name=name, opacity=0.7, marker_color=color,
                histnorm="probability density",
            ))
        fig3.update_layout(
            barmode="overlay",
            xaxis=dict(title="log(1 + Distance km)", gridcolor=GRID),
            yaxis=dict(title="Density", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=20,b=10), height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Merchant Category")
        cats = df_raw["merchant_category"].unique()
        fraud_rates = {c: df_raw[df_raw["merchant_category"]==c]["is_fraud"].mean() for c in cats}
        fraud_rates = dict(sorted(fraud_rates.items(), key=lambda x: x[1], reverse=True))

        fig4 = go.Figure(go.Bar(
            x=list(fraud_rates.values()), y=list(fraud_rates.keys()),
            orientation="h",
            marker_color=[f"rgba(248,113,113,{v*50})" for v in fraud_rates.values()],
            marker_line_color="#f87171", marker_line_width=1,
        ))
        fig4.add_vline(x=info["fraud_rate"], line_dash="dash", line_color="#64748b",
                       annotation_text="Overall rate")
        fig4.update_layout(
            xaxis=dict(title="Fraud Rate", gridcolor=GRID, tickformat=".1%"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            margin=dict(l=10,r=10,t=20,b=10), height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Summary stats
    st.markdown("---")
    st.markdown("#### Key Differences (Fraud vs Legitimate)")
    summary = pd.DataFrame({
        "Feature": ["Avg Amount ($)", "Foreign (%)", "High-Risk Merchant (%)",
                    "Avg Distance (km)", "Avg Txns in 24h", "Avg Amount/Avg Ratio"],
        "Legitimate": [
            f"${legit['amount'].mean():.0f}",
            f"{legit['foreign_transaction'].mean():.1%}",
            f"{legit['high_risk_merchant'].mean():.1%}",
            f"{legit['distance_from_home'].mean():.0f}",
            f"{legit['txn_count_24h'].mean():.1f}",
            f"{legit['amount_vs_avg'].mean():.2f}x",
        ],
        "Fraud": [
            f"${fraud['amount'].mean():.0f}",
            f"{fraud['foreign_transaction'].mean():.1%}",
            f"{fraud['high_risk_merchant'].mean():.1%}",
            f"{fraud['distance_from_home'].mean():.0f}",
            f"{fraud['txn_count_24h'].mean():.1f}",
            f"{fraud['amount_vs_avg'].mean():.2f}x",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Score Distribution":
    st.markdown("### Predicted Fraud Probability Distributions")
    st.caption(
        "A well-calibrated model pushes legitimate transactions toward 0 and "
        "fraudulent transactions toward 1, with as little overlap as possible."
    )

    for m in models:
        r      = results[m]
        y_prob = np.array(r["y_prob"])
        y_test = np.array(r["y_test"])
        color  = MODEL_COLORS[m]

        fig = go.Figure()
        for label, lcolor, name in [(0, "#94a3b8", "Legitimate"), (1, color, "Fraud")]:
            mask = y_test == label
            fig.add_trace(go.Histogram(
                x=y_prob[mask], nbinsx=80, name=name,
                marker_color=lcolor, opacity=0.75,
                histnorm="probability density",
            ))

        fig.add_vline(x=r["best_threshold"], line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"Optimal threshold={r['best_threshold']:.3f}",
                      annotation_position="top left")

        fig.update_layout(
            title=m, barmode="overlay",
            xaxis=dict(title="P(fraud)", gridcolor=GRID, range=[0,1]),
            yaxis=dict(title="Density", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=40,b=10), height=260,
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: FLAG A TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Flag a Transaction":
    st.markdown("### Live Transaction Scorer")
    st.caption("Enter transaction details and see what each model predicts.")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_amount   = st.number_input("Transaction Amount ($)", 1.0, 10000.0, 250.0, 10.0)
        s_hour     = st.slider("Hour of Day", 0, 23, 14)
        s_dow      = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    with col2:
        s_dist     = st.number_input("Distance from Home (km)", 0.0, 20000.0, 5.0, 1.0)
        s_foreign  = st.toggle("Foreign Transaction", value=False)
        s_highrisk = st.toggle("High-Risk Merchant", value=False)
    with col3:
        s_prev_min = st.number_input("Minutes Since Last Transaction", 0.1, 10000.0, 120.0, 1.0)
        s_count24h = st.slider("Transactions in Last 24h", 1, 25, 3)
        s_amtratio = st.number_input("Amount vs Cardholder Avg (ratio)", 0.1, 50.0, 1.0, 0.1)
        s_merchant = st.selectbox("Merchant Category",
                                  ["grocery","restaurant","gas","retail","entertainment","travel","other"])

    if st.button("Analyse Transaction", type="primary"):
        with st.spinner("Scoring…"):
            X_full, y_full = load_and_prepare("data/transactions.csv")

            from model import (build_logistic_smote, build_xgb_smote, build_rf_smote,
                               load_and_prepare)
            from sklearn.model_selection import train_test_split as tts

            X_tr, X_te, y_tr, y_te = tts(X_full.values, y_full,
                                          test_size=0.2, random_state=42, stratify=y_full)

            trained = {}
            for name, builder in [("Logistic + SMOTE", build_logistic_smote),
                                   ("XGBoost + SMOTE",  build_xgb_smote),
                                   ("RandomForest + SMOTE", build_rf_smote)]:
                m = builder()
                m.fit(X_tr, y_tr)
                trained[name] = m

            # Build applicant row
            dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
            row = pd.DataFrame([{
                "amount": s_amount, "hour": s_hour,
                "day_of_week": dow_map[s_dow],
                "merchant_category": s_merchant,
                "distance_from_home": s_dist,
                "foreign_transaction": int(s_foreign),
                "high_risk_merchant":  int(s_highrisk),
                "prev_txn_minutes": s_prev_min,
                "txn_count_24h": s_count24h,
                "amount_vs_avg": s_amtratio,
                "is_fraud": 0,
            }])
            row.to_csv("/tmp/single_txn.csv", index_label="transaction_id")
            row_X, _ = load_and_prepare("/tmp/single_txn.csv")
            row_X = row_X.reindex(columns=X_full.columns, fill_value=0)

        st.markdown("---")
        cols = st.columns(3)
        for col, name in zip(cols, models):
            prob = trained[name].predict_proba(row_X.values)[0, 1]
            thresh = results[name]["best_threshold"]
            verdict = "🚨 FRAUD" if prob >= thresh else "✅ LEGITIMATE"
            color = "red" if prob >= thresh else "green"
            with col:
                st.markdown(f"""
                <div class="stat" style="border-color:{'#f87171' if prob>=thresh else '#34d399'}">
                  <div class="stat-label">{name}</div>
                  <div class="stat-value {color}">{prob:.1%}</div>
                  <div class="stat-sub">{verdict}</div>
                </div>""", unsafe_allow_html=True)

        avg_prob = np.mean([trained[n].predict_proba(row_X.values)[0,1] for n in models])
        avg_verdict = "🚨 HIGH FRAUD RISK" if avg_prob >= 0.5 else "✅ LIKELY LEGITIMATE"
        box_class = "alert-fraud" if avg_prob >= 0.5 else "alert-legit"
        st.markdown(f"""
        <div class="{box_class}" style="margin-top:16px">
          <b>Ensemble average: {avg_prob:.1%} probability of fraud — {avg_verdict}</b>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#1e293b;font-size:12px;font-family:JetBrains Mono'>"
    "SMOTE · Logistic Regression · XGBoost · Random Forest · Precision-Recall Optimisation"
    "</div>", unsafe_allow_html=True,
)

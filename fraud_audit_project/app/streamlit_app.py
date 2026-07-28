"""
streamlit_app.py
=================
Real-Time Fraud Risk Scoring -- interactive simulation dashboard.

Implements the "real-time fraud detection" component of the underlying
methodology: every transaction (manually entered, batch-uploaded, or
randomly simulated) is scored live through the same preprocessing +
hybrid model pipeline trained in notebooks/fraud_detection_audit_risk.ipynb,
with a transaction-level SHAP explanation attached to every flagged item.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

st.set_page_config(
    page_title="Fraud Risk Scoring | Audit Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audit_transactions.csv")


@st.cache_resource
def load_artifacts():
    prep = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest_pipeline.joblib"))
    xgb_clf = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    meta = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    explainer = shap.TreeExplainer(xgb_clf)
    return prep, rf, xgb_clf, iso, meta, explainer


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["transaction_date"])
    df["month"] = df["transaction_date"].dt.month
    df["quarter"] = df["transaction_date"].dt.quarter
    df["log_amount"] = np.log1p(df["amount"])
    return df


def engineer_row(raw: dict, meta: dict) -> pd.DataFrame:
    """Turn a raw transaction dict into the exact feature frame the pipeline expects."""
    row = dict(raw)
    row["log_amount"] = np.log1p(row["amount"])
    df1 = pd.DataFrame([row])
    return df1[meta["feature_cols"]]


def score_transactions(X: pd.DataFrame, prep, rf, xgb_clf, iso, meta):
    Xt = prep.transform(X)  # shared preprocessor, used for XGBoost + Isolation Forest
    rf_proba = rf.predict_proba(X)[:, 1]  # rf is a full Pipeline with its own internal preprocessing step
    xgb_proba = xgb_clf.predict_proba(Xt)[:, 1]
    iso_raw = -iso.score_samples(Xt)
    denom = (meta["iso_raw_max"] - meta["iso_raw_min"]) or 1e-9
    iso_score = np.clip((iso_raw - meta["iso_raw_min"]) / denom, 0, 1)

    supervised = 0.4 * rf_proba + 0.6 * xgb_proba
    hybrid = 0.75 * supervised + 0.25 * iso_score
    return hybrid, rf_proba, xgb_proba, iso_score, Xt


def risk_tier(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "CRITICAL", "#B00020"
    elif score >= 0.50:
        return "HIGH", "#D8453B"
    elif score >= 0.25:
        return "MEDIUM", "#E8A33D"
    else:
        return "LOW", "#3B9B4A"


prep, rf, xgb_clf, iso, meta, explainer = load_artifacts()
df_all = load_data()

st.sidebar.title("🔎 Fraud Risk Scoring")
st.sidebar.caption("AI-Powered Forensic Accounting -- live simulation")
page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Dashboard", "Score a Single Transaction", "Batch Simulation", "Model Performance", "About This Project"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** Hybrid (XGBoost + Random Forest + Isolation Forest)  \n"
    f"**Decision threshold:** {meta['best_threshold']:.2f} (cost-optimised)  \n"
    f"**Training data:** {len(df_all):,} synthetic transactions"
)

# ============================================================================
if page == "Portfolio Dashboard":
    st.title("Portfolio-Level Fraud Risk Dashboard")
    st.caption("Simulated real-time monitoring view across the full transaction population.")

    with st.spinner("Scoring full portfolio..."):
        X_all = df_all[meta["feature_cols"]]
        hybrid, rf_p, xgb_p, iso_s, _ = score_transactions(X_all, prep, rf, xgb_clf, iso, meta)
        df_scored = df_all.copy()
        df_scored["risk_score"] = hybrid
        df_scored["flagged"] = (hybrid >= meta["best_threshold"]).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Scored", f"{len(df_scored):,}")
    c2.metric("Flagged for Review", f"{df_scored['flagged'].sum():,}",
               f"{df_scored['flagged'].mean()*100:.2f}% of portfolio")
    c3.metric("True Fraud in Data", f"{df_scored['label'].sum():,}")
    caught = ((df_scored['flagged'] == 1) & (df_scored['label'] == 1)).sum()
    c4.metric("Fraud Caught", f"{caught:,} / {df_scored['label'].sum():,}",
               f"{caught/df_scored['label'].sum()*100:.1f}% recall")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(df_scored, x="risk_score", color=df_scored["label"].map({0: "Legitimate", 1: "Fraudulent"}),
                            nbins=60, barmode="overlay", opacity=0.6,
                            color_discrete_map={"Legitimate": "#3B7DD8", "Fraudulent": "#D8453B"},
                            title="Risk Score Distribution by True Class")
        fig.add_vline(x=meta["best_threshold"], line_dash="dash", line_color="black",
                       annotation_text="Decision threshold")
        st.plotly_chart(fig, width='stretch')

    with col2:
        tier_counts = df_scored["risk_score"].apply(lambda s: risk_tier(s)[0]).value_counts()
        tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        tier_counts = tier_counts.reindex(tier_order).fillna(0)
        fig2 = px.bar(x=tier_counts.index, y=tier_counts.values,
                      color=tier_counts.index,
                      color_discrete_map={"LOW": "#3B9B4A", "MEDIUM": "#E8A33D",
                                          "HIGH": "#D8453B", "CRITICAL": "#B00020"},
                      title="Transactions by Risk Tier")
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig2, width='stretch')

    st.markdown("### Highest-Risk Transactions Requiring Review")
    top_flagged = df_scored.sort_values("risk_score", ascending=False).head(25)
    display_cols = ["transaction_id", "transaction_date", "amount", "vendor_id", "gl_account",
                     "vendor_age_days", "single_approver_flag", "is_round_amount",
                     "risk_score", "fraud_scheme"]
    st.dataframe(
        top_flagged[display_cols].style.format({"amount": "${:,.2f}", "risk_score": "{:.3f}"}),
        width='stretch', height=400
    )

    st.markdown("### Trend: Daily Flagged Transaction Volume")
    daily = df_scored.groupby(df_scored["transaction_date"].dt.date).agg(
        total=("transaction_id", "count"), flagged=("flagged", "sum")).reset_index()
    fig3 = px.line(daily, x="transaction_date", y="flagged", title="Flagged Transactions per Day")
    st.plotly_chart(fig3, width='stretch')

# ============================================================================
elif page == "Score a Single Transaction":
    st.title("Real-Time Transaction Risk Scoring")
    st.caption("Enter a transaction's characteristics to get a live fraud risk score, tier, and explanation -- "
               "this is the interactive equivalent of the paper's 'real-time detection' claim.")

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction amount ($)", min_value=1.0, max_value=500000.0, value=8500.0, step=100.0)
            vendor_age_days = st.number_input("Vendor age (days on file)", min_value=0, max_value=5000, value=800)
            employee_tenure_days = st.number_input("Employee tenure (days)", min_value=0, max_value=6000, value=900)
            gl_account = st.selectbox("GL account", sorted(meta["gl_accounts"]))
        with c2:
            approval_count = st.selectbox("Number of approvals", [1, 2, 3], index=1)
            days_invoice_to_payment = st.number_input("Days from invoice to payment", min_value=0, max_value=120, value=28)
            is_weekend = st.checkbox("Posted on a weekend")
            after_hours_flag = st.checkbox("Posted after hours (before 7am / after 7pm)")
            is_related_party = st.checkbox("Vendor is a related party")
        with c3:
            is_round_amount = st.checkbox("Round-dollar amount", value=(amount % 100 == 0))
            is_duplicate_invoice_number = st.checkbox("Duplicate invoice number detected")
            is_split_transaction = st.checkbox("Amount is just under an approval threshold")
            is_manual_journal_entry = st.checkbox("Manually posted (not system-generated)")
            prior_fraud_flag_vendor = st.checkbox("Vendor previously flagged for fraud")

        submitted = st.form_submit_button("Score Transaction", type="primary", width='stretch')

    if submitted:
        gl_risk = meta["gl_risk_map"].get(gl_account, "Medium")
        posting_hour = 2 if after_hours_flag else 13
        vendor_history = df_all.loc[df_all["gl_account"] == gl_account, "amount"]
        v_mean, v_std = vendor_history.mean(), vendor_history.std() or 1.0
        amount_zscore = float(np.clip((amount - v_mean) / v_std, -4, 6))
        first_digit = int(str(int(amount)).lstrip("0")[0]) if amount >= 1 else 1

        raw = {
            "amount": amount, "vendor_age_days": vendor_age_days,
            "employee_tenure_days": employee_tenure_days, "approval_count": approval_count,
            "days_invoice_to_payment": days_invoice_to_payment,
            "amount_zscore_vendor_history": amount_zscore, "amount_first_digit": first_digit,
            "posting_hour": posting_hour, "is_weekend": int(is_weekend),
            "after_hours_flag": int(after_hours_flag), "is_related_party": int(is_related_party),
            "single_approver_flag": int(approval_count == 1), "is_round_amount": int(is_round_amount),
            "is_duplicate_invoice_number": int(is_duplicate_invoice_number),
            "is_split_transaction": int(is_split_transaction),
            "prior_fraud_flag_vendor": int(prior_fraud_flag_vendor),
            "is_manual_journal_entry": int(is_manual_journal_entry),
            "gl_account": gl_account, "gl_account_risk": gl_risk,
        }
        X_row = engineer_row(raw, meta)
        hybrid, rf_p, xgb_p, iso_s, Xt = score_transactions(X_row, prep, rf, xgb_clf, iso, meta)
        score = float(hybrid[0])
        tier, color = risk_tier(score)
        flagged = score >= meta["best_threshold"]

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1, 2])
        r1.metric("Fraud Risk Score", f"{score:.3f}")
        r2.metric("Risk Tier", tier)
        with r3:
            st.markdown(
                f"<div style='padding:14px;border-radius:8px;background-color:{color}22;"
                f"border-left:6px solid {color};'>"
                f"<b>{'🚩 FLAGGED FOR MANUAL REVIEW' if flagged else '✅ No review required'}</b><br>"
                f"Decision threshold: {meta['best_threshold']:.2f} &nbsp;|&nbsp; "
                f"Supervised component: {0.4*rf_p[0]+0.6*xgb_p[0]:.3f} &nbsp;|&nbsp; "
                f"Anomaly component: {iso_s[0]:.3f}</div>", unsafe_allow_html=True)

        st.markdown("### Why this score? (SHAP explanation)")
        feature_names = (meta["numeric_features"] + meta["binary_features"] +
                          list(prep.named_transformers_["cat"].get_feature_names_out(meta["categorical_features"])))
        sv = explainer.shap_values(Xt)
        contrib = pd.DataFrame({"feature": feature_names, "shap_value": sv[0]})
        contrib = contrib.reindex(contrib.shap_value.abs().sort_values(ascending=False).index).head(10)
        fig = px.bar(contrib.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
                     color=contrib.sort_values("shap_value")["shap_value"] > 0,
                     color_discrete_map={True: "#D8453B", False: "#3B7DD8"},
                     title="Top 10 Feature Contributions to This Score")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="SHAP value (impact on risk score)")
        st.plotly_chart(fig, width='stretch')
        st.caption("Red bars increase fraud risk; blue bars decrease it. This is the auditable rationale "
                   "an examiner would attach to the work paper for this transaction.")

# ============================================================================
elif page == "Batch Simulation":
    st.title("Batch Transaction Simulation")
    st.caption("Simulate a stream of incoming transactions (mimicking a day's AP batch) and watch the model flag them in real time.")

    n_sim = st.slider("Number of transactions to simulate", 10, 500, 100, step=10)
    fraud_pct = st.slider("Approximate fraud rate to simulate (%)", 0.0, 10.0, 2.0, step=0.5)
    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        rng = np.random.default_rng()
        sample = df_all.sample(n=min(n_sim, len(df_all)), random_state=int(rng.integers(0, 1_000_000)))
        X_sim = sample[meta["feature_cols"]]
        hybrid, rf_p, xgb_p, iso_s, _ = score_transactions(X_sim, prep, rf, xgb_clf, iso, meta)
        sample = sample.copy()
        sample["risk_score"] = hybrid
        sample["risk_tier"] = [risk_tier(s)[0] for s in hybrid]
        sample["flagged"] = (hybrid >= meta["best_threshold"]).astype(int)

        progress = st.progress(0, text="Streaming transactions through the scoring engine...")
        placeholder = st.empty()
        flagged_log = []
        for i, (_, row) in enumerate(sample.sort_values("transaction_date").iterrows()):
            progress.progress((i + 1) / len(sample))
            if row["flagged"] == 1:
                flagged_log.append(row)
            if i % max(1, len(sample) // 20) == 0 or i == len(sample) - 1:
                with placeholder.container():
                    st.write(f"Processed {i+1}/{len(sample)} transactions -- "
                             f"{len(flagged_log)} flagged so far")
        progress.empty()

        st.success(f"Simulation complete: {sample['flagged'].sum()} of {len(sample)} transactions flagged "
                   f"({sample['flagged'].mean()*100:.1f}%).")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(sample, names="risk_tier", title="Risk Tier Breakdown",
                         color="risk_tier",
                         color_discrete_map={"LOW": "#3B9B4A", "MEDIUM": "#E8A33D",
                                             "HIGH": "#D8453B", "CRITICAL": "#B00020"})
            st.plotly_chart(fig, width='stretch')
        with c2:
            if sample["label"].sum() > 0:
                caught = ((sample["flagged"] == 1) & (sample["label"] == 1)).sum()
                st.metric("True fraud in this batch", int(sample["label"].sum()))
                st.metric("Caught by model", f"{caught} ({caught/sample['label'].sum()*100:.0f}%)")
            st.metric("False positives", int(((sample["flagged"] == 1) & (sample["label"] == 0)).sum()))

        st.markdown("### Flagged Transactions")
        if flagged_log:
            flagged_df = pd.DataFrame(flagged_log).sort_values("risk_score", ascending=False)
            st.dataframe(
                flagged_df[["transaction_id", "transaction_date", "amount", "gl_account",
                            "risk_score", "risk_tier", "fraud_scheme"]].style.format(
                    {"amount": "${:,.2f}", "risk_score": "{:.3f}"}),
                width='stretch'
            )
        else:
            st.info("No transactions were flagged in this simulated batch.")

# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance Overview")
    st.caption("Metrics computed in the companion notebook (notebooks/fraud_detection_audit_risk.ipynb), reloaded here for reference.")

    results = pd.DataFrame(meta["results_summary"])
    st.markdown("### Supervised Model Comparison")
    st.dataframe(results.style.format({"PR-AUC (Average Precision)": "{:.4f}", "ROC-AUC": "{:.4f}"}),
                 width='stretch')

    st.markdown("### Saved Figures from the Analysis Notebook")
    fig_dir = os.path.join(os.path.dirname(MODEL_DIR), "reports", "figures")
    figs = ["06_pr_roc_curves.png", "08_cost_threshold.png", "09_confusion_matrix.png",
            "10_feature_importance.png", "11_shap_summary.png"]
    for f in figs:
        path = os.path.join(fig_dir, f)
        if os.path.exists(path):
            st.image(path, width='stretch')

# ============================================================================
elif page == "About This Project":
    st.title("About This Project")
    st.markdown("""
This dashboard is the interactive component of **Fraud Detection Using Audit Risk and ML Models**,
a working implementation of the methodology described in *"AI-Powered Forensic Accounting:
Leveraging Machine Learning for Real-Time Fraud Detection and Prevention."*

**Architecture:**
1. **Supervised classification** (Logistic Regression, Random Forest, XGBoost) trained on labeled
   historical transactions to recognise known fraud typologies.
2. **Unsupervised anomaly detection** (Isolation Forest) to flag transactions that look statistically
   unusual even without a matching historical label -- the safety net for emerging fraud schemes.
3. **A cost-calibrated hybrid risk score** combining both, with the decision threshold chosen to
   minimise total expected cost (false-negative fraud losses vs. false-positive investigation cost)
   rather than an arbitrary 0.5 cutoff.
4. **SHAP-based explainability** attached to every score, so a flagged transaction comes with an
   auditable, documented rationale rather than an opaque number.

**Data:** A synthetically generated accounts-payable transaction dataset with named, explainable
features drawn from forensic-accounting red-flag literature (ACFE *Report to the Nations*; AU-C 240),
with five fraud archetypes injected at a realistic ~1.5% base rate. See `data/generate_data.py` for
the full, reproducible generation logic and `README.md` for the complete methodology write-up.

**Repository structure:**
```
fraud_audit_project/
├── data/generate_data.py                        # synthetic data generator
├── data/audit_transactions.csv                   # generated dataset
├── notebooks/fraud_detection_audit_risk.ipynb     # full analysis notebook
├── models/                                        # trained model artifacts
├── app/streamlit_app.py                           # this dashboard
├── reports/figures/                               # exported charts
└── README.md
```
""")

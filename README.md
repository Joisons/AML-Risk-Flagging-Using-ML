# Fraud Detection Using Audit Risk and Machine Learning Models

A hybrid supervised-classification + anomaly-detection system for accounts-payable / journal-entry fraud, with cost-based decision thresholding and transaction-level explainability implemented as a working companion to the paper **"AI-Powered Forensic Accounting: Leveraging Machine Learning for Real-Time Fraud Detection and Prevention."**

---

## Why this project exists

A published paper describes a methodology. This repository *runs* it on real, executable code, against a full model-development pipeline, with an interactive simulation an auditor (or an adjudicator) can actually click through. Every claim below is backed by a number produced by the notebook in this repo, not an estimate.

## Methodology → implementation map

| Paper's methodology component | Where it's implemented |
|---|---|
| Feature engineering from documented audit red flags | `notebooks/fraud_detection_audit_risk.ipynb` §3–4 |
| Supervised classification for known fraud typologies | §6 — Logistic Regression, Random Forest, XGBoost |
| Anomaly detection for emerging / unlabeled fraud patterns | §7 — Isolation Forest |
| Composite real-time risk score | §8 — Hybrid ensemble |
| Cost-sensitive decisioning (audit economics, not just an ML metric) | §9 — cost-minimising threshold selection |
| Explainability for audit work-paper defensibility | §10 — global feature importance + per-transaction SHAP |
| "Real-time" detection | `app/streamlit_app.py` — live single-transaction and streaming-batch scoring |

## Results

Evaluated on a held-out 12,500-transaction test set (187 true fraud cases):

| Model | PR-AUC (Average Precision) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.785 | 0.986 |
| Random Forest | 0.707 | 0.985 |
| XGBoost | **0.845** | 0.988 |
| **Hybrid ensemble (XGB + RF + Isolation Forest)** | **0.847** | **0.987** |

At the cost-minimising decision threshold (0.27, selected by minimising expected loss given an illustrative $15,000 average fraud loss and $150 investigation cost per flagged item see §9 of the notebook):

- **96.8% of fraud caught** (181 of 187 cases)
- **Estimated 94.6% reduction in expected financial exposure** versus flagging nothing (\$152,700 vs. \$2,805,000)
- 30.2% precision meaning roughly 1 in 3 flagged transactions is a true positive, a deliberate and disclosed trade-off given that missed fraud costs an estimate of 100x more than an unnecessary review

**Why ROC-AUC is not the headline number:** all three models post ROC-AUC above 0.98, which looks reassuring but is a known artefact of severe class imbalance (98.5% legitimate) the false-positive *rate* stays small even when the absolute count of false positives is large enough to overwhelm a review team. PR-AUC does not benefit from that denominator effect and is reported first for that reason. This distinction and the decision to select a threshold by expected cost rather than a default 0.5 cutoff is itself part of what the notebook is trying to demonstrate: audit-usable ML requires audit-relevant evaluation, not just a high benchmark score.

## A note on the data

Real, labeled forensic-accounting engagement data is essentially never publicly releasable client confidentiality and active-engagement restrictions make that so by design. Public benchmark datasets such as the ULB Credit Card Fraud dataset and IEEE-CIS are anonymised, PCA-transformed *consumer payment* data (features named `V1` ... `V28`) with no auditable meaning. They're useful for benchmarking raw classifier performance, but cannot demonstrate an *audit-risk* methodology at all an auditor cannot write a work paper that says "`V17 < -3.4`."

This project instead uses a **synthetically generated accounts-payable transaction dataset** (`data/generate_data.py`, fully reproducible, seeded) with named, explainable features drawn directly from the forensic-accounting and fraud-examination literature:

- Vendor age, related-party status, approval-chain length core AU-C 240 fraud-risk factors
- Round-dollar amounts and Benford's-Law leading-digit deviation classical forensic-accounting screens
- Invoice-to-payment lag, duplicate invoice numbers, split transactions just under approval thresholds documented AP-fraud red flags from the ACFE *Report to the Nations*

Five archetypal fraud schemes (shell/ghost vendor, invoice splitting, duplicate payment, round-dollar kickback, fictitious expense reimbursement) are injected at a realistic estimate of 1.5% base rate, with **deliberate noise and class overlap** each red flag is applied probabilistically, amount ranges overlap the legitimate population, and a matching fraction of legitimate transactions are given elevated-risk characteristics purely by chance. Real fraud is not perfectly separable from legitimate activity; a dataset where it is would make for an artificially easy (and unconvincing) demonstration.

## Repository structure

```
fraud_audit_project/
├── data/
│   ├── generate_data.py              # synthetic data generator (reproducible, seeded)
│   └── audit_transactions.csv        # generated dataset (50,000 transactions)
├── notebooks/
│   └── fraud_detection_audit_risk.ipynb   # full analysis, executed with saved outputs
├── models/                           # trained model artifacts (joblib)
│   ├── preprocessor.joblib
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_model.joblib
│   ├── isolation_forest.joblib
│   └── metadata.joblib
├── app/
│   └── streamlit_app.py              # interactive real-time scoring dashboard
├── reports/figures/                  # 12 exported charts (PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd fraud_audit_project
python3 -m venv venv && source venv/bin/activate     # or your preferred env manager
pip install -r requirements.txt

# 2. (Optional) regenerate the dataset -- a copy is already committed
python data/generate_data.py --n 50000 --fraud-rate 0.015 --seed 42 --out data/audit_transactions.csv

# 3. Run the notebook (outputs are already baked in; re-run only if you want to reproduce)
jupyter notebook notebooks/fraud_detection_audit_risk.ipynb

# 4. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

The Streamlit app has five pages:

1. **Portfolio Dashboard** — scores the full transaction population, risk-tier breakdown, highest-risk transaction table
2. **Score a Single Transaction** — enter a transaction's characteristics and get a live risk score, tier, and a SHAP waterfall explaining exactly why
3. **Batch Simulation** — streams a simulated batch of incoming transactions through the model, mimicking real-time monitoring of a day's AP activity
4. **Model Performance** — the comparison table and key charts from the notebook, for reference without re-running it
5. **About This Project** — architecture and methodology summary

## Design choices worth calling out

- **Two-layer detection, not one.** Supervised models only recognise patterns present in labeled training history. The Isolation Forest layer exists specifically to catch novel schemes that have no prior label directly implementing the paper's "real-time detection of emerging fraud" claim rather than only its "classification of known fraud" claim.
- **Cost-based thresholding, not a default cutoff.** A 0.5 probability threshold is arbitrary and ignores that a missed fraud and a false alarm have very different costs. The threshold is instead selected to minimise total expected cost, using disclosed, adjustable loss/investigation-cost assumptions (§9).
- **SHAP at the transaction level, not just global feature importance.** A risk score an auditor cannot explain is one they cannot defend in a work paper or an audit-committee presentation. Every score in the Streamlit app comes with a ranked, signed breakdown of exactly which features pushed it up or down.
- **Identical preprocessing across training, evaluation, and the live app**, via a single serialized `ColumnTransformer` a common source of silent bugs in deployed fraud models is train/serve preprocessing skew, so this was a deliberate design choice.

## Limitations & future work

- Results demonstrate the methodology on synthetic data, not a validated real-world detection rate; production use requires calibration against an institution's actual transaction history and loss experience.
- A time-based (train-on-past, test-on-future) validation split, rather than random stratification, would better simulate deployment and would surface concept drift.
- The \$15,000 / \$150 cost assumptions in §9 are illustrative; a real deployment should use the institution's own figures.
- Free-text invoice descriptions (NLP), device/IP metadata, and vendor/employee relationship-graph features would likely improve recall specifically on the shell-vendor and related-party schemes.


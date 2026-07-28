"""
generate_data.py
=================
Synthetic audit-transaction (accounts-payable / journal-entry) data generator
for the "Fraud Detection Using Audit Risk and ML Models" project.

WHY SYNTHETIC DATA
-------------------
Real, labeled forensic-accounting engagement data is almost never publicly
releasable (client confidentiality, ongoing litigation, regulatory
restriction). Public "fraud" datasets such as the ULB Credit Card Fraud
dataset or IEEE-CIS are card-present/card-not-present *consumer payment*
data with PCA-anonymised features (V1-V28) -- useful for benchmarking raw
classification performance, but they contain no auditable, named,
explainable features and therefore cannot demonstrate an *audit-risk*
methodology at all (an auditor cannot work-paper "V17 < -3.4").

This generator instead produces transaction-level accounts-payable data with
NAMED, EXPLAINABLE features that mirror the red flags documented in the
forensic-accounting and fraud-examination literature (ACFE Report to the
Nations; AU-C 240 fraud-risk-factor guidance; Benford's Law analysis), and
injects five archetypal fraud schemes at a realistic, low base rate. This
keeps the notebook's feature-engineering and interpretability sections
grounded in genuine audit practice rather than an opaque benchmark dataset.

The five injected fraud archetypes:
  1. Shell / ghost vendor fraud
  2. Invoice-splitting to evade approval thresholds
  3. Duplicate-payment fraud
  4. Round-dollar kickback / collusion schemes
  5. Fictitious expense reimbursement

Usage:
    python generate_data.py --n 50000 --fraud-rate 0.015 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

GL_ACCOUNTS = [
    ("Office Supplies", 0.10, "Low"),
    ("IT Services", 0.15, "Medium"),
    ("Travel & Entertainment", 0.12, "Medium"),
    ("Consulting Fees", 0.10, "High"),
    ("Professional Services", 0.10, "High"),
    ("Marketing", 0.10, "Medium"),
    ("Facilities & Maintenance", 0.10, "Low"),
    ("Equipment & Capital", 0.08, "Medium"),
    ("Utilities", 0.08, "Low"),
    ("Miscellaneous / Other", 0.07, "High"),
]

def _sample_gl_account(rng, n):
    names = [g[0] for g in GL_ACCOUNTS]
    probs = np.array([g[1] for g in GL_ACCOUNTS])
    probs = probs / probs.sum()
    risk_map = {g[0]: g[2] for g in GL_ACCOUNTS}
    idx = rng.choice(len(names), size=n, p=probs)
    accounts = np.array(names)[idx]
    risk = np.array([risk_map[a] for a in accounts])
    return accounts, risk


def first_digit(x):
    """Return the leading (first significant) digit of each value, per Benford's Law convention."""
    x = np.abs(np.asarray(x, dtype=float))
    x = np.where(x <= 0, 1.0, x)
    exponent = np.floor(np.log10(x)).astype(int)
    leading = np.floor(x / (10.0 ** exponent)).astype(int)
    return np.clip(leading, 1, 9)


def generate_legitimate_transactions(rng, n, start_date, end_date, n_vendors, n_employees):
    days_range = (end_date - start_date).days
    tx_dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, days_range, size=n)]
    dow = np.array([d.weekday() for d in tx_dates])
    is_weekend = (dow >= 5).astype(int)
    hour = np.clip(rng.normal(13, 3, size=n), 0, 23).astype(int)
    after_hours = ((hour < 7) | (hour > 19)).astype(int)

    # Amounts: legitimate business spend, log-normal, realistic AP distribution
    amount = np.round(rng.lognormal(mean=6.2, sigma=1.15, size=n), 2)
    amount = np.clip(amount, 12.0, 250_000.0)

    vendor_id = rng.integers(1, n_vendors + 1, size=n)
    vendor_age_days = rng.integers(30, 3650, size=n)  # established vendors
    is_related_party = (rng.random(n) < 0.02).astype(int)

    employee_id = rng.integers(1, n_employees + 1, size=n)
    employee_tenure_days = rng.integers(60, 4000, size=n)

    gl_account, gl_risk = _sample_gl_account(rng, n)

    approval_count = rng.choice([1, 2, 3], size=n, p=[0.15, 0.55, 0.30])
    single_approver_flag = (approval_count == 1).astype(int)

    is_round_amount = ((amount % 100 == 0) | (amount % 1000 == 0)).astype(int)
    # legitimate round amounts happen sometimes too (rent, retainers) -> keep low base rate
    round_mask = rng.random(n) < 0.04
    amount = np.where(round_mask, np.round(amount / 100) * 100, amount)
    is_round_amount = ((amount % 100 == 0)).astype(int)

    days_invoice_to_payment = np.clip(rng.normal(28, 9, size=n), 1, 90).astype(int)
    is_duplicate_invoice_number = np.zeros(n, dtype=int)
    is_split_transaction = np.zeros(n, dtype=int)
    amount_zscore_vendor_history = np.clip(rng.normal(0, 1, size=n), -3, 3)
    prior_fraud_flag_vendor = np.zeros(n, dtype=int)
    is_manual_journal_entry = (rng.random(n) < 0.08).astype(int)

    label = np.zeros(n, dtype=int)
    scheme = np.array(["none"] * n, dtype=object)

    df = pd.DataFrame({
        "transaction_date": tx_dates,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "posting_hour": hour,
        "after_hours_flag": after_hours,
        "amount": amount,
        "vendor_id": vendor_id,
        "vendor_age_days": vendor_age_days,
        "is_related_party": is_related_party,
        "employee_id": employee_id,
        "employee_tenure_days": employee_tenure_days,
        "gl_account": gl_account,
        "gl_account_risk": gl_risk,
        "approval_count": approval_count,
        "single_approver_flag": single_approver_flag,
        "is_round_amount": is_round_amount,
        "days_invoice_to_payment": days_invoice_to_payment,
        "is_duplicate_invoice_number": is_duplicate_invoice_number,
        "is_split_transaction": is_split_transaction,
        "amount_zscore_vendor_history": amount_zscore_vendor_history,
        "prior_fraud_flag_vendor": prior_fraud_flag_vendor,
        "is_manual_journal_entry": is_manual_journal_entry,
        "fraud_scheme": scheme,
        "label": label,
    })
    return df


def inject_fraud(df, rng, fraud_rate, n_vendors):
    """
    Inject fraud with REALISTIC, NOISY, PARTIALLY-OVERLAPPING signal.

    Real fraud is not perfectly separable from legitimate activity -- that is
    precisely why it evades simple rule-based controls and requires a
    statistical/ML approach. Each red flag below is therefore applied
    *probabilistically* (not deterministically), amount ranges are drawn from
    distributions that overlap the legitimate population, and a fraction of
    perpetrators deliberately keep amounts small ("salami slicing") to stay
    under scrutiny. A matching fraction of legitimate transactions are also
    given elevated-risk characteristics by pure chance (new vendors, round
    invoices, single-approver postings), so that no single red flag is a
    perfect fraud indicator on its own -- consistent with real audit
    experience and with the "AI-powered forensic accounting" paper's premise
    that composite, model-based risk scoring outperforms single-rule flags.
    """
    n = len(df)
    n_fraud = int(n * fraud_rate)
    scheme_weights = {
        "shell_vendor": 0.24,
        "invoice_splitting": 0.22,
        "duplicate_payment": 0.18,
        "round_dollar_kickback": 0.20,
        "expense_reimbursement": 0.16,
    }
    schemes = rng.choice(list(scheme_weights.keys()), size=n_fraud,
                          p=list(scheme_weights.values()))
    fraud_idx = rng.choice(df.index, size=n_fraud, replace=False)
    ghost_vendor_pool = rng.integers(n_vendors + 1, n_vendors + 41, size=40)

    def maybe(p):
        return rng.random() < p

    for i, idx in zip(range(n_fraud), fraud_idx):
        s = schemes[i]
        df.at[idx, "fraud_scheme"] = s
        df.at[idx, "label"] = 1

        if s == "shell_vendor":
            if maybe(0.75):
                df.at[idx, "vendor_id"] = int(rng.choice(ghost_vendor_pool))
                df.at[idx, "vendor_age_days"] = int(rng.integers(1, 60))
            if maybe(0.30):
                df.at[idx, "is_related_party"] = 1
            if maybe(0.30):
                df.at[idx, "amount"] = round(float(rng.lognormal(6.6, 0.7)), 2)
            else:
                df.at[idx, "amount"] = round(float(rng.uniform(2500, 42000)), 2)
            if maybe(0.65):
                df.at[idx, "approval_count"] = 1
                df.at[idx, "single_approver_flag"] = 1
            if maybe(0.55):
                df.at[idx, "gl_account"] = rng.choice(["Consulting Fees", "Professional Services", "Miscellaneous / Other"])
                df.at[idx, "gl_account_risk"] = "High"
            if maybe(0.55):
                df.at[idx, "days_invoice_to_payment"] = int(rng.integers(1, 9))

        elif s == "invoice_splitting":
            threshold = 10000.0
            if maybe(0.80):
                df.at[idx, "amount"] = round(float(rng.uniform(threshold * 0.82, threshold * 0.995)), 2)
            else:
                df.at[idx, "amount"] = round(float(rng.uniform(threshold * 0.5, threshold * 0.82)), 2)
            if maybe(0.70):
                df.at[idx, "approval_count"] = 1
                df.at[idx, "single_approver_flag"] = 1
            df.at[idx, "is_split_transaction"] = 1 if maybe(0.85) else 0
            if maybe(0.5):
                df.at[idx, "gl_account"] = rng.choice(["Consulting Fees", "Equipment & Capital", "Professional Services"])
            if maybe(0.5):
                df.at[idx, "days_invoice_to_payment"] = int(rng.integers(1, 14))

        elif s == "duplicate_payment":
            df.at[idx, "is_duplicate_invoice_number"] = 1 if maybe(0.85) else 0
            if maybe(0.6):
                df.at[idx, "days_invoice_to_payment"] = int(rng.integers(0, 7))
            df.at[idx, "amount"] = round(float(rng.uniform(300, 26000)), 2)
            if maybe(0.5):
                df.at[idx, "is_manual_journal_entry"] = 1

        elif s == "round_dollar_kickback":
            if maybe(0.80):
                df.at[idx, "amount"] = float(rng.choice([2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000]))
                df.at[idx, "is_round_amount"] = 1
            else:
                df.at[idx, "amount"] = round(float(rng.uniform(1500, 26000)), 2)
            if maybe(0.60):
                df.at[idx, "approval_count"] = 1
                df.at[idx, "single_approver_flag"] = 1
            if maybe(0.5):
                df.at[idx, "gl_account"] = rng.choice(["Consulting Fees", "Professional Services", "Marketing"])
                df.at[idx, "gl_account_risk"] = "High"
            if maybe(0.30):
                df.at[idx, "is_weekend"] = 1
            if maybe(0.30):
                df.at[idx, "after_hours_flag"] = 1

        elif s == "expense_reimbursement":
            df.at[idx, "gl_account"] = "Travel & Entertainment"
            df.at[idx, "gl_account_risk"] = "Medium"
            df.at[idx, "amount"] = round(float(rng.uniform(150, 5500)), 2)
            if maybe(0.45):
                df.at[idx, "is_round_amount"] = 1
            if maybe(0.5):
                df.at[idx, "is_manual_journal_entry"] = 1

    legit_idx = df.index[df["label"] == 0]
    n_hard = int(len(legit_idx) * 0.035)
    hard_idx = rng.choice(legit_idx, size=n_hard, replace=False)
    for idx in hard_idx:
        choice = rng.choice(["new_vendor", "round_amt", "single_appr", "fast_pay", "weekend"])
        if choice == "new_vendor":
            df.at[idx, "vendor_age_days"] = int(rng.integers(5, 90))
        elif choice == "round_amt":
            df.at[idx, "amount"] = float(rng.choice([1000, 2500, 5000, 7500, 10000]))
            df.at[idx, "is_round_amount"] = 1
        elif choice == "single_appr":
            df.at[idx, "approval_count"] = 1
            df.at[idx, "single_approver_flag"] = 1
        elif choice == "fast_pay":
            df.at[idx, "days_invoice_to_payment"] = int(rng.integers(1, 6))
        elif choice == "weekend":
            df.at[idx, "is_weekend"] = 1

    return df


def add_benford_feature(df):
    df["amount_first_digit"] = first_digit(df["amount"].values)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--fraud-rate", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="audit_transactions.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    n_vendors = 900
    n_employees = 150

    df = generate_legitimate_transactions(rng, args.n, start_date, end_date, n_vendors, n_employees)
    df = inject_fraud(df, rng, args.fraud_rate, n_vendors)

    # Compute amount_zscore_vendor_history from the ACTUAL realised amounts per
    # vendor (rather than hand-authored ranges), so the feature is genuinely
    # noisy and overlaps between classes -- exactly as it would from real data.
    vendor_stats = df.groupby("vendor_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "_v_mean", "std": "_v_std"})
    vendor_stats["_v_std"] = vendor_stats["_v_std"].replace(0, np.nan).fillna(vendor_stats["_v_mean"] * 0.35 + 1)
    df = df.merge(vendor_stats, left_on="vendor_id", right_index=True, how="left")
    df["amount_zscore_vendor_history"] = np.clip(
        (df["amount"] - df["_v_mean"]) / df["_v_std"], -4, 6
    ).round(3)
    df = df.drop(columns=["_v_mean", "_v_std"])

    df = add_benford_feature(df)

    df.insert(0, "transaction_id", [f"TXN{100000+i}" for i in range(len(df))])
    df.insert(6, "invoice_number", [f"INV{rng.integers(10000,99999)}" for _ in range(len(df))])

    # create genuine duplicate invoice numbers for the duplicate_payment fraud rows
    dup_rows = df[df["fraud_scheme"] == "duplicate_payment"].index
    for idx in dup_rows:
        # reuse another random invoice number already in the data to simulate a true duplicate
        donor = df.index[df.index != idx]
        donor_idx = rng.choice(donor)
        df.at[idx, "invoice_number"] = df.at[donor_idx, "invoice_number"]

    df = df.sort_values("transaction_date").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df):,} transactions -> {args.out}")
    print(f"Fraud count: {df['label'].sum():,}  ({df['label'].mean()*100:.3f}% of transactions)")
    print(df["fraud_scheme"].value_counts())


if __name__ == "__main__":
    main()

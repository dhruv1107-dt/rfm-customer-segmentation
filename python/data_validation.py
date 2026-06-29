"""
RFM Segmentation — Data Validation (3 automated routines)
Author : Dhruv Tandel
Runs on every ingestion cycle to keep the 50K-row dataset analysis-ready.
"""
import pandas as pd, numpy as np

REQUIRED_COLS  = ["customer_id","order_id","order_date","order_value","order_status"]
VALID_STATUSES = {"completed","shipped","processing","cancelled","returned"}

# ROUTINE 1 — NULL DETECTION
def check_nulls(df):
    print("── ROUTINE 1: NULL DETECTION ──────────────────────")
    missing = df[REQUIRED_COLS].isnull().sum()
    missing = missing[missing>0]
    if missing.empty: print("  [PASS] No nulls in required columns.")
    else: [print(f"  [FAIL] {c}: {n} nulls ({n/len(df)*100:.1f}%)") for c,n in missing.items()]
    return missing.empty

# ROUTINE 2 — FORMAT VALIDATION
def validate_formats(df):
    print("── ROUTINE 2: FORMAT VALIDATION ───────────────────")
    passed = True
    if "order_date" in df.columns:
        try: pd.to_datetime(df["order_date"]); print("  [PASS] order_date — valid.")
        except: print("  [FAIL] order_date — unparseable."); passed=False
    if "order_value" in df.columns:
        neg=(pd.to_numeric(df["order_value"],errors="coerce")<0).sum()
        print(f"  {'[FAIL]' if neg else '[PASS]'} order_value — {neg} negative values.")
        if neg: passed=False
    if "order_status" in df.columns:
        bad=df[~df["order_status"].str.lower().isin(VALID_STATUSES)]["order_status"].unique()
        print(f"  {'[FAIL]' if len(bad) else '[PASS]'} order_status{' — invalid: '+str(bad) if len(bad) else ' — all valid.'}")
        if len(bad): passed=False
    return passed

# ROUTINE 3 — OUTLIER FLAGGING (IQR × 3 — lenient for financial data)
def flag_outliers(df, col="order_value", mult=3.0):
    print("── ROUTINE 3: OUTLIER FLAGGING (IQR × 3) ─────────")
    df=df.copy()
    q1,q3=df[col].quantile(0.25),df[col].quantile(0.75)
    iqr=q3-q1; lo,hi=q1-mult*iqr,q3+mult*iqr
    df["is_outlier"]=(df[col]<lo)|(df[col]>hi)
    n=df["is_outlier"].sum()
    print(f"  Bounds: [{lo:.2f}, {hi:.2f}]  |  Flagged: {n} ({n/len(df)*100:.2f}%)")
    return df

# MASTER RUNNER
def run_validation(filepath="data/orders.csv"):
    print("="*55+"
  RFM DATA VALIDATION — 3-ROUTINE CHECK
"+"="*55)
    df=pd.read_csv(filepath,parse_dates=["order_date"])
    print(f"  Rows: {len(df):,}
")
    null_ok=check_nulls(df)
    fmt_ok=validate_formats(df)
    df=flag_outliers(df)
    print(f"
── SUMMARY {'─'*36}")
    print(f"  Null check:   {'PASS' if null_ok else 'FAIL'}")
    print(f"  Format check: {'PASS' if fmt_ok else 'FAIL'}")
    print(f"  Outliers:     {df['is_outlier'].sum():,} rows flagged")
    print("="*55)
    return df

if __name__=="__main__":
    df=run_validation()
    clean=df[~df["is_outlier"]].drop(columns=["is_outlier"])
    clean.to_csv("data/orders_clean.csv",index=False)
    print(f"
[SAVE] {len(clean):,} clean rows → data/orders_clean.csv")

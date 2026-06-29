"""
RFM Analysis — Python Implementation
Author : Dhruv Tandel
Computes RFM scores from 50K-row orders dataset using Pandas.
"""
import pandas as pd, numpy as np
from datetime import date

ORDERS_FILE, RFM_OUT = "data/orders_clean.csv", "data/rfm_scores.csv"
SNAPSHOT = date.today()

def load_orders(f):
    df=pd.read_csv(f,parse_dates=["order_date"])
    df=df[~df["order_status"].isin(["cancelled","returned"])]
    print(f"[LOAD] {len(df):,} valid orders"); return df

def compute_rfm(df, snapshot=SNAPSHOT):
    snap=pd.Timestamp(snapshot)
    rfm=df.groupby("customer_id").agg(
        last_order_date=("order_date","max"), frequency=("order_id","count"), monetary=("order_value","sum")
    ).reset_index()
    rfm["recency_days"]=(snap-rfm["last_order_date"]).dt.days
    rfm["monetary"]=rfm["monetary"].round(2)
    return rfm

def ntile_score(series, n=5, ascending=True):
    return pd.qcut(series.rank(method="first",ascending=ascending), q=n, labels=range(1,n+1)).astype(int)

def score_rfm(rfm):
    rfm=rfm.copy()
    rfm["r_score"]=ntile_score(rfm["recency_days"], ascending=False)
    rfm["f_score"]=ntile_score(rfm["frequency"])
    rfm["m_score"]=ntile_score(rfm["monetary"])
    rfm["rfm_avg_score"]=((rfm.r_score+rfm.f_score+rfm.m_score)/3).round(2)
    rfm["rfm_cell"]=rfm.r_score.astype(str)+rfm.f_score.astype(str)+rfm.m_score.astype(str)
    return rfm

def assign_tier(row):
    r,f,m=row.r_score,row.f_score,row.m_score
    if r>=4 and f>=4 and m>=4: return "Champions"
    if f>=4 and m>=4:           return "Loyal Customers"
    if r>=3 and f<=2:           return "Potential Loyalists"
    if r<=2 and f>=3 and m>=3: return "At-Risk"
    return "Lost"

def run_rfm():
    print("="*55+"\n  RFM ANALYSIS PIPELINE\n"+"="*55)
    rfm=score_rfm(compute_rfm(load_orders(ORDERS_FILE)))
    rfm["rfm_tier"]=rfm.apply(assign_tier,axis=1)
    rfm.to_csv(RFM_OUT,index=False)
    summary=(rfm.groupby("rfm_tier").agg(customers=("customer_id","count"),
        avg_monetary=("monetary","mean"),total_revenue=("monetary","sum")).round(1).reset_index())
    summary["pct"]=(summary.total_revenue/summary.total_revenue.sum()*100).round(1)
    print(summary.sort_values("total_revenue",ascending=False).to_string(index=False))
    return rfm

if __name__=="__main__": run_rfm()

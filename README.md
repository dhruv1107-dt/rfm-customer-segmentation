# RFM + K-Means Customer Segmentation

**SQL (MySQL) · Python · Scikit-learn · Power BI** | Jun 2025 – Present

End-to-end customer segmentation on a **50,000-row e-commerce dataset**. An SQL scoring pipeline (CTEs, RANK, LAG, NTILE window functions) classifies customers into RFM tiers; K-Means clustering validated by the elbow method groups them into 5 actionable segments. Results feed a Power BI dashboard of 12 DAX measures.

---

## What is RFM?

| Dimension | Description |
|---|---|
| **Recency (R)** | Days since last purchase |
| **Frequency (F)** | Total number of transactions |
| **Monetary (M)** | Total amount spent |

Each customer scores 1–5 on each dimension (via NTILE). K-Means then clusters them into 5 segments.

---

## 5 Customer Segments

| Segment | RFM Profile | Action |
|---|---|---|
| Champions | High R, F, M | Reward & retain |
| Loyal Customers | High F & M | Upsell / loyalty programme |
| Potential Loyalists | Recent, moderate F & M | Nurture & engage |
| At-Risk | Low R, previously High F & M | Win-back campaigns |
| Lost | Low R, F, M | Re-engage or deprioritise |

---

## Tech Stack

| Layer | Tools |
|---|---|
| RFM Scoring | MySQL 8.0 — CTEs, RANK, LAG, NTILE window functions |
| Validation | Python — Pandas (3 automated routines) |
| Clustering | Python — Scikit-learn K-Means, elbow method |
| Visualisation | Power BI — 12 DAX measures |

---

## Repository Structure

```
rfm-customer-segmentation/
├── sql/
│   └── rfm_scoring.sql         # Full RFM pipeline (CTEs + window functions)
├── python/
│   ├── data_validation.py      # 3 automated validation routines
│   ├── rfm_analysis.py         # RFM score computation in Pandas
│   └── kmeans_clustering.py   # K-Means + elbow + segment labelling
└── README.md
```

---

## How to Run

```bash
# 1. Score in MySQL
mysql -u root -p < sql/rfm_scoring.sql

# 2. Python pipeline
pip install pandas numpy matplotlib scikit-learn mysql-connector-python
python python/data_validation.py
python python/rfm_analysis.py
python python/kmeans_clustering.py
```

---

## Key Results

- **50,000 customers** segmented into 5 actionable tiers
- Elbow method confirmed **K = 5** as optimal
- Champions (~12% of customers) drive ~48% of revenue
- At-Risk segment: ~8,300 customers with 90+ day recency

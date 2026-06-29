-- ============================================================
-- RFM + K-Means Customer Segmentation — SQL Scoring Pipeline
-- Author : Dhruv Tandel | 50,000-row e-commerce dataset
-- Techniques: CTEs · RANK · LAG · NTILE window functions
-- ============================================================
USE ecommerce_db;

-- STEP 1 — RAW RFM METRICS PER CUSTOMER
WITH rfm_raw AS (
    SELECT
        customer_id,
        DATEDIFF(CURDATE(), MAX(order_date))  AS recency_days,
        COUNT(order_id)                        AS frequency,
        ROUND(SUM(order_value), 2)             AS monetary
    FROM orders
    WHERE order_status NOT IN ('cancelled','returned')
    GROUP BY customer_id
),

-- STEP 2 — INTER-ORDER GAP (LAG) — early At-Risk signal
order_gaps AS (
    SELECT customer_id, order_date,
        DATEDIFF(order_date, LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)) AS days_between_orders
    FROM orders WHERE order_status NOT IN ('cancelled','returned')
),
avg_gap AS (
    SELECT customer_id, ROUND(AVG(days_between_orders),1) AS avg_order_gap_days
    FROM order_gaps WHERE days_between_orders IS NOT NULL GROUP BY customer_id
),

-- STEP 3 — NTILE SCORING 1-5 per dimension
rfm_scored AS (
    SELECT r.*, g.avg_order_gap_days,
        6 - NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,  -- lower days = higher score
        NTILE(5)     OVER (ORDER BY frequency ASC)      AS f_score,
        NTILE(5)     OVER (ORDER BY monetary ASC)       AS m_score
    FROM rfm_raw r LEFT JOIN avg_gap g USING (customer_id)
),

-- STEP 4 — COMPOSITE SCORE + TIER LABEL + RANK WITHIN TIER
rfm_composite AS (
    SELECT *,
        ROUND((r_score+f_score+m_score)/3.0, 2)  AS rfm_avg_score,
        CONCAT(r_score,f_score,m_score)            AS rfm_cell,
        CASE
            WHEN r_score>=4 AND f_score>=4 AND m_score>=4 THEN 'Champions'
            WHEN f_score>=4 AND m_score>=4                 THEN 'Loyal Customers'
            WHEN r_score>=3 AND f_score<=2                 THEN 'Potential Loyalists'
            WHEN r_score<=2 AND f_score>=3 AND m_score>=3  THEN 'At-Risk'
            ELSE 'Lost'
        END AS rfm_tier,
        RANK() OVER (
            PARTITION BY CASE
                WHEN r_score>=4 AND f_score>=4 AND m_score>=4 THEN 'Champions'
                WHEN f_score>=4 AND m_score>=4                 THEN 'Loyal Customers'
                WHEN r_score>=3 AND f_score<=2                 THEN 'Potential Loyalists'
                WHEN r_score<=2 AND f_score>=3 AND m_score>=3  THEN 'At-Risk'
                ELSE 'Lost' END
            ORDER BY monetary DESC
        ) AS rank_within_tier
    FROM rfm_scored
)

-- FINAL VIEW (connect Power BI here)
SELECT * FROM rfm_composite ORDER BY rfm_avg_score DESC;

-- TIER SUMMARY for Power BI KPI cards
SELECT rfm_tier,
    COUNT(*)                                           AS customer_count,
    ROUND(AVG(recency_days),1)                         AS avg_recency_days,
    ROUND(AVG(frequency),1)                            AS avg_frequency,
    ROUND(AVG(monetary),0)                             AS avg_monetary,
    ROUND(SUM(monetary),0)                             AS total_revenue,
    ROUND(SUM(monetary)*100.0/SUM(SUM(monetary)) OVER(),1) AS revenue_share_pct
FROM rfm_composite
GROUP BY rfm_tier ORDER BY total_revenue DESC;

-- Recency/Frequency/Monetary scoring (NTILE quintiles) mapped to named
-- segments. "as of" date = one day after the last transaction in the data.
WITH ref_date AS (
    SELECT MAX(invoice_date) + INTERVAL 1 DAY AS as_of FROM transactions
),
customer_rfm AS (
    SELECT
        t.customer_id,
        DATE_DIFF('day', MAX(t.invoice_date), (SELECT as_of FROM ref_date)) AS recency_days,
        COUNT(DISTINCT t.invoice)                                          AS frequency,
        SUM(t.revenue)                                                     AS monetary
    FROM transactions t
    GROUP BY t.customer_id
),
scored AS (
    SELECT
        *,
        -- higher score = better in all three: recent, frequent, big spender
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC)      AS m_score
    FROM customer_rfm
)
SELECT
    *,
    CASE
        WHEN r_score >= 4 AND f_score >= 4                  THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 4                  THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2                  THEN 'Hibernating'
        ELSE 'Needs Attention'
    END AS segment
FROM scored;

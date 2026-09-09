-- Repeat purchase rate by months since acquisition, cohort by cohort.
-- Feeds the cohort retention heatmap (customer/cohorts.py).
WITH first_order AS (
    SELECT customer_id, DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
    FROM transactions
    GROUP BY customer_id
),
orders_by_month AS (
    SELECT customer_id, DATE_TRUNC('month', invoice_date) AS order_month
    FROM transactions
    GROUP BY customer_id, DATE_TRUNC('month', invoice_date)
),
joined AS (
    SELECT
        f.customer_id,
        f.cohort_month,
        o.order_month,
        DATE_DIFF('month', f.cohort_month, o.order_month) AS months_since_acquisition
    FROM first_order f
    JOIN orders_by_month o ON f.customer_id = o.customer_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size
    FROM first_order
    GROUP BY cohort_month
)
SELECT
    j.cohort_month,
    j.months_since_acquisition,
    COUNT(DISTINCT j.customer_id)                                AS active_customers,
    cs.cohort_size,
    COUNT(DISTINCT j.customer_id)::DOUBLE / cs.cohort_size        AS retention_rate
FROM joined j
JOIN cohort_sizes cs ON j.cohort_month = cs.cohort_month
GROUP BY j.cohort_month, j.months_since_acquisition, cs.cohort_size
ORDER BY 1, 2;

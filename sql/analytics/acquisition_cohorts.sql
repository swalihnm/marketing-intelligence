-- Monthly acquisition cohorts: size and average first-order value.
-- Source: transactions (real, UCI Online Retail II — see
-- docs/01-data-sources-and-honesty.md).
WITH first_order AS (
    SELECT customer_id, MIN(invoice_date) AS first_order_date
    FROM transactions
    GROUP BY customer_id
),
first_order_value AS (
    SELECT t.customer_id, SUM(t.revenue) AS first_order_revenue
    FROM transactions t
    JOIN first_order f
      ON t.customer_id = f.customer_id
     AND t.invoice_date = f.first_order_date
    GROUP BY t.customer_id
)
SELECT
    DATE_TRUNC('month', f.first_order_date) AS cohort_month,
    COUNT(DISTINCT f.customer_id)           AS cohort_size,
    AVG(fov.first_order_revenue)            AS avg_first_order_value
FROM first_order f
JOIN first_order_value fov ON f.customer_id = fov.customer_id
GROUP BY 1
ORDER BY 1;

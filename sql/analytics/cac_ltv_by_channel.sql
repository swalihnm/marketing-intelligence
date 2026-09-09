-- LTV-to-CAC and average LTV per (synthetic) acquisition channel.
-- customer_channel_synthetic is written by customer/channel_synth.py — see
-- that module's docstring and docs/01-data-sources-and-honesty.md before
-- reading these numbers as anything other than illustrative.
WITH customer_ltv AS (
    SELECT customer_id, SUM(revenue) AS ltv
    FROM transactions
    GROUP BY customer_id
)
SELECT
    c.channel,
    COUNT(DISTINCT c.customer_id)         AS n_customers,
    AVG(l.ltv)                            AS avg_ltv,
    c.cac,
    AVG(l.ltv) / c.cac                    AS ltv_to_cac_ratio
FROM customer_channel_synthetic c
JOIN customer_ltv l ON c.customer_id = l.customer_id
GROUP BY c.channel, c.cac
ORDER BY ltv_to_cac_ratio DESC;

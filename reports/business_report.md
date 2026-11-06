# Business Report: Reducing Customer Churn

## Executive summary
Of 500 customers, 53 (10.6%) left. We built a model that flags at-risk customers with very high accuracy and found that **churn is a first-year problem**: 63% of customers in their first 12 months left, while **none** of the 416 customers who passed month 12 did. A retention campaign aimed at the customers the model flags is estimated to **earn about 13,300 more than it costs** on this customer base, and the model can be used today through a simple web app.

## Problem statement
Winning a new customer costs much more than keeping one. The company needs to know **who will leave, why, and what to do about it** before they go.

## Key findings

| Finding | Evidence |
|---|---|
| Churn happens only in the first year | 53 of 84 customers with tenure up to 12 months churned (63%). 0 of 416 others. |
| Contract type matters | Month-to-month 20.6%, one year 4.3%, two year 6.9% |
| High bills raise risk | Monthly charges above 100: 14.2% churn. 100 or below: 5.7%. |
| High bill and new is the worst mix | New customers above 100 a month: 79% churn. New customers at or below 100: 38%. |
| Payment method has a small effect | Bank transfer 6.9%, electronic check 11.0%, credit card 13.5% |
| Paperless billing and senior status barely matter | Churn is 10% to 11% either way |

## What the model does
- Catches **all 11 of 11** churners in the unseen test set (recall 100%).
- About 6 in 10 customers it flags really would have left (precision 61%). The rest get an offer they did not strictly need, which is cheap compared with losing a customer.
- Overall ROC-AUC 0.997 (cross-validated 0.993).

## Financial estimate

Scoring all 500 customers with out-of-fold predictions and contacting every flagged customer:

| Item | Value |
|---|---|
| Customers flagged | 81 |
| Churners reached | 53 of 53 |
| Annual revenue at risk (all churners) | 82,536 |
| Campaign cost | 11,444 |
| Revenue saved | 24,761 |
| **Net benefit** | **13,316** |

**Assumptions** (change them in `src/business.py`): an offer costs 10% of the customer's annual revenue, and 30% of contacted churners accept it and stay. Currency is as in the dataset. If real acceptance is lower, the campaign still breaks even at roughly a 14% save rate, so the downside is limited.

## Recommendations
1. **Invest in the first 12 months.** Add onboarding calls and check-ins at months 1, 3 and 6.
2. **Convert month-to-month customers to one-year plans** with a small discount, since contract length is the strongest lever the company controls.
3. **Review pricing for new customers paying over 100 a month.** They are the highest risk group.
4. **Use the model as a weekly watch-list.** Score new customers each week and send the "High" list to the retention team.
5. **Measure results.** Track how many contacted customers stay, then replace the two assumptions with real numbers.

## Risks and next steps
- The data is small and simulated, so real-world accuracy will be lower. Run a pilot on a live customer group before a full roll-out.
- Add richer data (support tickets, usage, complaints).
- Retrain every quarter and monitor the churn rate of customers the model called "Low risk".

# Data Dictionary: customer_churn.csv

500 rows, 9 columns, no missing values, no duplicates.

| Column | Type | Values / range | Description |
|---|---|---|---|
| CustomerID | text | C00001 to C00500 | Unique customer code. Not used for modeling. |
| Tenure | integer | 1 to 71 | Months the customer has been with the company |
| MonthlyCharges | integer | about 20 to 199 | Amount billed each month |
| TotalCharges | integer | up to 7,992 | Total amount billed so far |
| Contract | category | Month-to-month, One year, Two year | Contract type |
| PaymentMethod | category | Bank Transfer, Credit Card, Electronic Check | How the customer pays |
| PaperlessBilling | category | Yes, No | Whether bills are paperless |
| SeniorCitizen | integer | 0, 1 | 1 if the customer is a senior citizen |
| Churn | integer | 0, 1 | **Target.** 1 means the customer left (53 of 500, 10.6%) |

## Engineered features
| Feature | Formula | Reason |
|---|---|---|
| AvgMonthlySpend | TotalCharges / max(Tenure, 1) | Actual spend per month, safe when tenure is 0 |
| IsNewCustomer | 1 if Tenure <= 12 | Churn happens only in the first year |
| HighCharges | 1 if MonthlyCharges > 100 | High bills raise churn risk |

## Quality notes
- `TotalCharges` is not exactly `Tenure x MonthlyCharges`, so the data is simulated.
- Churn is imbalanced (89.4% stayed), so recall and AUC are used instead of accuracy alone.

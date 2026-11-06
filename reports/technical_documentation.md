# Technical Documentation

## 1. Objective
Predict whether a customer will churn (Churn = 1) from billing and contract information, and serve the prediction through a web app.

## 2. Methodology

| Step | What was done | Where |
|---|---|---|
| Data quality | Checked columns, missing values, duplicates, duplicate IDs, negative numbers. All passed. | `src/data_loader.py` |
| EDA | Churn balance, churn by contract / payment / billing / tenure, box plots, correlation | notebook, section 3 |
| Feature engineering | Added AvgMonthlySpend, IsNewCustomer, HighCharges | `src/features.py` |
| Preprocessing | StandardScaler on numbers, OneHotEncoder on categories, inside a Pipeline | `src/features.py` |
| Split | 80% train (400) / 20% test (100), stratified, random_state=42 | `src/train.py` |
| Model comparison | 5-fold stratified CV, ROC-AUC, on the training set only | `src/train.py` |
| Tuning | GridSearchCV over n_estimators (100, 200, 300), max_depth (3, 5, None), min_samples_leaf (1, 3, 5) | `src/train.py` |
| Evaluation | One final pass on the untouched test set | `src/evaluate.py` |
| Deployment | Flask app loads `model.joblib` and exposes `/predict` | `deployment/app.py` |

## 3. Algorithms
- **Logistic Regression:** a linear baseline that gives the probability of churn.
- **Random Forest (chosen):** many decision trees, each trained on a random sample, whose votes are averaged. It handles non-linear rules such as "new AND month-to-month", resists overfitting and gives feature importances.
- **Gradient Boosting:** trees built one after another, each fixing the previous errors.
- **Class imbalance:** only 10.6% of customers churn, so `class_weight="balanced"` makes mistakes on churners cost more.

## 4. Results

**Cross-validation (training data, ROC-AUC)**

| Model | Mean | Std |
|---|---|---|
| Logistic Regression | 0.9922 | 0.0046 |
| Random Forest | 0.9928 | 0.0038 |
| Gradient Boosting | 0.9940 | 0.0060 |

The three models are within noise of each other. Random Forest was chosen for robustness and explainability.

**Tuned Random Forest:** `max_depth=3, min_samples_leaf=5, n_estimators=300`, CV AUC 0.9934.

**Test set (100 customers, 11 churners)**

| Metric | Value |
|---|---|
| Accuracy | 0.930 |
| Precision (churn) | 0.611 |
| Recall (churn) | 1.000 |
| F1 | 0.759 |
| ROC-AUC | 0.997 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted stay | Predicted churn |
|---|---|---|
| Actual stay | 82 | 7 |
| Actual churn | 0 | 11 |

**Top features:** Tenure (0.37), IsNewCustomer (0.28), AvgMonthlySpend (0.20), Contract (month-to-month and one-year), MonthlyCharges.

Figures: `reports/figures/`.

## 5. Architecture

```
Browser form --JSON--> Flask /predict --> validate_input() --> add_features()
                                      --> Pipeline(scaler + encoder + Random Forest)
                                      --> {probability, prediction, risk level}
```

- `src/predict.py` validates every field (types, ranges, allowed categories) and raises a clear error. The API turns that into HTTP 400.
- The model and the list of feature names are saved together in `model.joblib`, so training and serving cannot drift apart.
- Endpoints: `GET /` (page), `GET /health`, `POST /predict`.

## 6. Testing

16 tests, all passing (`reports/test_results.txt`):

| File | Covers |
|---|---|
| `tests/test_data.py` | Data loads, quality check passes and fails correctly, feature engineering is safe (zero tenure, input not modified) |
| `tests/test_model.py` | Model file exists, metrics above the bar, high-risk and low-risk examples, probabilities in 0..1, six invalid inputs rejected, missing field rejected |
| `tests/test_api.py` | Health, home page, good prediction, bad input gives 400, non-JSON gives 400 |
| `tests/test_business.py` | Campaign calculation checked by hand |

Run: `python -m unittest discover -s tests`

## 7. Limitations and risks
- Small, simulated dataset. Scores are far higher than real-world churn models.
- Only 11 churners in the test set, so single-split metrics are noisy.
- No data on complaints, support calls or service quality.
- The model has not been tested on live data and needs retraining as customers change.
- The Flask development server is for demonstration. For production use `gunicorn` (see `deployment/Dockerfile`).

## 8. Reproducibility
Fixed `random_state=42`, pinned library versions in `requirements.txt`, and the full pipeline is run by `python -m src.train`.

"""Train, compare, tune and save the churn model.

Run from the project root:  python -m src.train
"""
import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import (CATEGORICAL, ENGINEERED, FIG_DIR, METRICS_PATH, MODEL_PATH,
                     NUMERIC, RANDOM_STATE, TARGET)
from .data_loader import load_data, validate_data
from .evaluate import evaluate_model, plot_confusion, plot_importance, plot_roc
from .features import add_features, build_preprocessor

FEATURES = NUMERIC + ENGINEERED + CATEGORICAL


def get_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    quality = validate_data(df)
    assert quality["passed"], f"Data quality check failed: {quality}"

    df = add_features(df)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for name, model in get_candidates().items():
        scores = cross_val_score(make_pipeline(model), X_train, y_train, cv=cv, scoring="roc_auc")
        comparison[name] = {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}
        print(f"{name:20s} CV AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Tune the Random Forest (robust and interpretable on small tabular data)
    grid = GridSearchCV(
        make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        {"model__n_estimators": [100, 200, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3, 5]},
        cv=cv, scoring="roc_auc", n_jobs=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    metrics = evaluate_model(best, X_test, y_test)
    metrics.update({
        "best_params": grid.best_params_,
        "cv_comparison": comparison,
        "best_cv_auc": float(grid.best_score_),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data_quality": quality,
    })
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:10s} {metrics[k]:.4f}")

    plot_confusion(best, X_test, y_test, FIG_DIR / "confusion_matrix.png")
    plot_roc(best, X_test, y_test, FIG_DIR / "roc_curve.png")
    plot_importance(best, FIG_DIR / "feature_importance.png")

    joblib.dump({"pipeline": best, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()

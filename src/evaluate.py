"""Model evaluation helpers and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def evaluate_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def plot_confusion(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Stay", "Churn"],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="#0B6E4F")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_importances(pipeline):
    names = pipeline.named_steps["prep"].get_feature_names_out()
    imp = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1]
    return [(str(names[i]).split("__", 1)[-1], float(imp[i])) for i in order]


def plot_importance(pipeline, path, top=10):
    items = get_importances(pipeline)[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([n for n, _ in items], [v for _, v in items], color="#0B6E4F")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

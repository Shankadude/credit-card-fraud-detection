import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

from .config import FIG_DIR

def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)

    # If model supports probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for models without predict_proba
        y_prob = model.decision_function(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(FIG_DIR / f"confusion_matrix_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {model_name} (AUC={roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(FIG_DIR / f"roc_curve_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve - {model_name} (PR-AUC={pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(FIG_DIR / f"pr_curve_{model_name}.png", dpi=200, bbox_inches="tight")
    plt.close()

    results = {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
    return results

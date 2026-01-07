from src.data_loader import load_data
from src.preprocessing import preprocess
from src.eda import (
    plot_class_distribution, plot_amount_distribution,
    plot_time_fraud, correlation_heatmap
)
from src.modeling import get_models, resample_data
from src.evaluation import evaluate_model
from src.tuning import tune_xgboost
from src.visualization import pca_plot, tsne_plot
from src.utils import save_model, save_report

def run():
    # 1) Load data
    df = load_data()
    print("Dataset loaded:", df.shape)

    # 2) EDA
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_fraud(df)
    correlation_heatmap(df)
    print("EDA plots saved in outputs/figures")

    # 3) Preprocess
    X_train, X_test, y_train, y_test = preprocess(df)

    # 4) Visualize PCA / TSNE
    pca_plot(X_train, y_train, name="pca_train")
    tsne_plot(X_train, y_train, sample_size=5000, name="tsne_train")
    print("PCA/t-SNE plots saved.")

    # 5) Resample training data
    X_res, y_res = resample_data(X_train, y_train, method="smote_under")
    print("Resampled training distribution:")
    print(y_res.value_counts())

    # 6) Train & evaluate models
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_res, y_res)

        res = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(res)

        # Save reports
        save_report(res["classification_report"], f"{name}_classification_report.txt")

        print(f"{name} ROC-AUC:", res["roc_auc"])
        print(f"{name} PR-AUC:", res["pr_auc"])

    # 7) Tune XGBoost (optional but recommended)
    print("\nTuning XGBoost...")
    best_xgb, best_params = tune_xgboost(X_res, y_res)
    tuned_result = evaluate_model(best_xgb, X_test, y_test, model_name="xgboost_tuned")
    print("Best XGBoost Params:", best_params)

    save_report(str(best_params), "xgboost_best_params.txt")
    save_report(tuned_result["classification_report"], "xgboost_tuned_report.txt")

    # 8) Save best model
    model_path = save_model(best_xgb, "best_xgboost_model.pkl")
    print("Best model saved to:", model_path)

if __name__ == "__main__":
    run()

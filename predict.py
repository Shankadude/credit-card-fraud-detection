import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.utils import load_model
from src.config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "best_xgboost_model.pkl"

def predict_single_transaction(model, row_df):
    """
    Predict fraud probability for a single transaction row.
    row_df must have same columns as training features.
    """
    prob = model.predict_proba(row_df)[:, 1][0]
    pred = 1 if prob >= 0.5 else 0
    return pred, prob

def run_demo():
    # Load model
    model = load_model(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # Load and preprocess dataset to get feature column structure
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    # Example: predict a random transaction from test set
    sample_row = X_test.sample(1, random_state=42)
    true_label = y_test.loc[sample_row.index].values[0]

    pred, prob = predict_single_transaction(model, sample_row)

    print("\n--- Real-Time Prediction Demo ---")
    print("True Label (0=Normal, 1=Fraud):", true_label)
    print("Predicted Label:", pred)
    print("Fraud Probability:", round(prob, 6))

    if pred == 1:
        print("Alert: Fraud transaction detected!")
    else:
        print("Transaction seems normal.")

if __name__ == "__main__":
    run_demo()

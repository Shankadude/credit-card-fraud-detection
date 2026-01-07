import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import RANDOM_STATE, TEST_SIZE

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Scale Amount
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df.drop(["Amount"], axis=1, inplace=True)

    # Features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

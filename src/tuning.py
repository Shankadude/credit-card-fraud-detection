from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from .config import RANDOM_STATE

def tune_xgboost(X_train, y_train):
    params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    base = XGBClassifier(
        scale_pos_weight=10,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

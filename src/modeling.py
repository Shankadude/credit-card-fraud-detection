from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import RANDOM_STATE

def get_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }
    return models

def resample_data(X_train, y_train, method="smote_under"):
    if method == "none":
        return X_train, y_train

    if method == "smote":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res

    if method == "under":
        under = RandomUnderSampler(random_state=RANDOM_STATE)
        X_res, y_res = under.fit_resample(X_train, y_train)
        return X_res, y_res

    if method == "smote_under":
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("under", RandomUnderSampler(random_state=RANDOM_STATE))
        ])
        X_res, y_res = pipe.fit_resample(X_train, y_train)
        return X_res, y_res

    raise ValueError("Invalid resampling method. Choose: none, smote, under, smote_under")

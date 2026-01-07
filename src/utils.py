import joblib
from .config import MODEL_DIR, REPORT_DIR

def save_model(model, name="best_model.pkl"):
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path

def save_report(text, filename="report.txt"):
    path = REPORT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def load_model(path):
    return joblib.load(path)
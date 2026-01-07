# Credit Card Fraud Detection (Machine Learning)

This project builds a fraud detection model using:
- Logistic Regression
- Random Forest
- XGBoost

Includes:
- SMOTE + undersampling for imbalance handling
- ROC-AUC and Precision-Recall evaluation
- Hyperparameter tuning
- PCA and t-SNE visualization
- Real-time prediction demo

## Dataset
Download the dataset from Kaggle:
Credit Card Fraud Detection Dataset (creditcard.csv)

Place it in:
data/creditcard.csv

## Setup
```bash
pip install -r requirements.txt
python main.py
python predict.py

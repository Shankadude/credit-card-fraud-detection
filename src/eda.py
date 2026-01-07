import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIG_DIR

def plot_class_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution (0=Normal, 1=Fraud)")
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_amount_distribution(df):
    plt.figure(figsize=(10,5))
    sns.kdeplot(df[df["Class"]==0]["Amount"], label="Normal", fill=True)
    sns.kdeplot(df[df["Class"]==1]["Amount"], label="Fraud", fill=True)
    plt.title("Transaction Amount Distribution")
    plt.legend()
    plt.savefig(FIG_DIR / "amount_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_time_fraud(df):
    plt.figure(figsize=(10,4))
    sns.histplot(df[df["Class"]==1]["Time"], bins=50)
    plt.title("Fraud Transactions Over Time")
    plt.savefig(FIG_DIR / "fraud_over_time.png", dpi=200, bbox_inches="tight")
    plt.close()

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.2)
    plt.title("Correlation Heatmap")
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

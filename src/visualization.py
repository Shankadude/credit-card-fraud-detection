import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .config import FIG_DIR
from .config import RANDOM_STATE

def pca_plot(X, y, name="pca_projection"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(7,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, s=2)
    plt.title("PCA Projection of Transactions")
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

def tsne_plot(X, y, sample_size=5000, name="tsne_projection"):
    # Sample for performance
    if X.shape[0] > sample_size:
        X = X.sample(sample_size, random_state=RANDOM_STATE)
        y = y.loc[X.index]

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(7,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, s=2)
    plt.title("t-SNE Projection of Transactions")
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

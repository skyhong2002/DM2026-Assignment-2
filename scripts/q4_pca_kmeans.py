from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report_assets" / "assignment2"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def save_markdown_table(df, path, float_digits=4):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.{float_digits}f}")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
    path.write_text("\n".join([header, sep, *rows]) + "\n", encoding="utf-8")


def load_scaled_data():
    df = pd.read_csv(ROOT / "mobile_price.csv")
    X = df.drop(columns=["price_range"])
    y = df["price_range"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y


def plot_scatter(points, labels, title, output_path, legend_title):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="viridis",
        s=24,
        alpha=0.78,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title=legend_title, loc="best", frameon=True)
    ax.add_artist(legend)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    X, X_scaled, y = load_scaled_data()

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    plot_scatter(
        X_pca,
        y,
        "PCA Projection Colored by Price Range",
        ASSET_DIR / "q4b_pca_class_scatter.png",
        "price_range",
    )

    kmeans_all = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    labels_all = kmeans_all.fit_predict(X_scaled)
    ari_all = adjusted_rand_score(y, labels_all)

    plot_scatter(
        X_pca,
        labels_all,
        "K-Means Clusters Using All Standardized Features",
        ASSET_DIR / "q4c_kmeans_all_features_scatter.png",
        "cluster",
    )

    kmeans_pca = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    ari_pca = adjusted_rand_score(y, labels_pca)

    plot_scatter(
        X_pca,
        labels_pca,
        "K-Means Clusters Using PCA 2D Features",
        ASSET_DIR / "q4d_kmeans_pca_features_scatter.png",
        "cluster",
    )

    summary_df = pd.DataFrame(
        [
            {
                "method": "KMeans on all standardized features",
                "n_features": X_scaled.shape[1],
                "adjusted_rand_score": ari_all,
            },
            {
                "method": "KMeans on 2D PCA features",
                "n_features": X_pca.shape[1],
                "adjusted_rand_score": ari_pca,
            },
        ]
    )
    pca_df = pd.DataFrame(
        [
            {"component": "PC1", "explained_variance_ratio": explained[0]},
            {"component": "PC2", "explained_variance_ratio": explained[1]},
            {"component": "PC1+PC2", "explained_variance_ratio": explained.sum()},
        ]
    )

    pd.DataFrame(X_scaled, columns=X.columns).describe().to_csv(ASSET_DIR / "q4a_standardized_feature_summary.csv")
    pca_df.to_csv(ASSET_DIR / "q4b_pca_explained_variance.csv", index=False)
    summary_df.to_csv(ASSET_DIR / "q4_kmeans_ari_summary.csv", index=False)

    save_markdown_table(pca_df, ASSET_DIR / "q4b_pca_explained_variance.md")
    save_markdown_table(summary_df, ASSET_DIR / "q4_kmeans_ari_summary.md")

    print("PCA explained variance:")
    print(pca_df)
    print("\nKMeans ARI summary:")
    print(summary_df)


if __name__ == "__main__":
    main()

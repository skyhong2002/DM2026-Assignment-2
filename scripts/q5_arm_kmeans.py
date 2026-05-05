from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


SEEDS = [0, 10, 42, 100, 999]
ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report_assets" / "assignment2"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.45
MIN_LIFT = 1.05
RULE_SCORE_WEIGHT = 4.0


def save_markdown_table(df, path, float_digits=4):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.{float_digits}f}")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
    path.write_text("\n".join([header, sep, *rows]) + "\n", encoding="utf-8")


def itemset_to_string(itemset):
    return ", ".join(sorted(itemset))


def discretize_343(df, feature_cols):
    binned = pd.DataFrame(index=df.index)
    thresholds = []
    for feature in feature_cols:
        min_value = df[feature].min()
        max_value = df[feature].max()
        width = max_value - min_value
        low_upper = min_value + 0.3 * width
        high_lower = min_value + 0.7 * width
        thresholds.append(
            {
                "feature": feature,
                "min": min_value,
                "low_upper": low_upper,
                "high_lower": high_lower,
                "max": max_value,
            }
        )
        binned[feature] = np.select(
            [df[feature] < low_upper, df[feature] < high_lower],
            [f"{feature}_low", f"{feature}_medium"],
            default=f"{feature}_high",
        )
    return binned, pd.DataFrame(thresholds)


def build_transactions(binned_df, y):
    transactions = []
    for idx, row in binned_df.iterrows():
        items = row.tolist()
        items.append(f"price_range_{int(y.loc[idx])}")
        transactions.append(items)
    return transactions


def mine_label_rules(transactions):
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(encoded, columns=encoder.columns_)

    itemsets = fpgrowth(transaction_df, min_support=MIN_SUPPORT, use_colnames=True, max_len=3)
    rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

    def is_label_item(item):
        return str(item).startswith("price_range_")

    keep = []
    for _, rule in rules.iterrows():
        consequents = set(rule["consequents"])
        antecedents = set(rule["antecedents"])
        keep.append(
            len(consequents) == 1
            and any(is_label_item(item) for item in consequents)
            and not any(is_label_item(item) for item in antecedents)
            and rule["support"] >= MIN_SUPPORT
            and rule["lift"] >= MIN_LIFT
        )
    label_rules = rules[keep].copy()
    label_rules["label"] = label_rules["consequents"].map(lambda x: int(next(iter(x)).split("_")[-1]))
    label_rules["rule_score"] = label_rules["support"] * label_rules["confidence"] * label_rules["lift"]
    label_rules = label_rules.sort_values(
        ["rule_score", "confidence", "lift", "support"], ascending=[False, False, False, False]
    ).reset_index(drop=True)
    return label_rules


def build_rule_score_features(binned_df, label_rules):
    rule_scores = np.zeros((len(binned_df), 4), dtype=float)
    sample_itemsets = binned_df.apply(lambda row: set(row.tolist()), axis=1).tolist()

    prepared_rules = [
        (set(row["antecedents"]), int(row["label"]), float(row["rule_score"]))
        for _, row in label_rules.iterrows()
    ]

    for i, items in enumerate(sample_itemsets):
        for antecedents, label, score in prepared_rules:
            if antecedents.issubset(items):
                rule_scores[i, label] += score

    score_df = pd.DataFrame(
        rule_scores,
        columns=[f"arm_score_price_{label}" for label in range(4)],
        index=binned_df.index,
    )
    return score_df


def map_clusters_to_labels(y_true, cluster_labels):
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    labels = np.unique(y_true)
    clusters = np.unique(cluster_labels)
    size = max(len(labels), len(clusters))
    contingency = np.zeros((size, size), dtype=int)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    cluster_to_idx = {cluster: i for i, cluster in enumerate(clusters)}
    for true_label, cluster in zip(y_true, cluster_labels):
        contingency[label_to_idx[true_label], cluster_to_idx[cluster]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    cluster_to_label = {}
    for row, col in zip(row_ind, col_ind):
        if row < len(labels) and col < len(clusters):
            cluster_to_label[clusters[col]] = labels[row]

    fallback = int(pd.Series(y_true).mode().iloc[0])
    return np.array([cluster_to_label.get(cluster, fallback) for cluster in cluster_labels])


def evaluate_clustering(y_true, cluster_labels):
    y_pred = map_clusters_to_labels(y_true, cluster_labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_kmeans_experiment(X, y, method_name):
    records = []
    for seed in SEEDS:
        labels = KMeans(n_clusters=4, random_state=seed, n_init=10).fit_predict(X)
        records.append({"method": method_name, "seed": seed, **evaluate_clustering(y, labels)})
    return records


def main():
    df = pd.read_csv(ROOT / "mobile_price.csv")
    feature_cols = [col for col in df.columns if col != "price_range"]
    X = df[feature_cols]
    y = df["price_range"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    binned_df, threshold_df = discretize_343(df, feature_cols)
    transactions = build_transactions(binned_df, y)
    label_rules = mine_label_rules(transactions)
    score_df = build_rule_score_features(binned_df, label_rules)
    score_scaled = StandardScaler().fit_transform(score_df)
    X_arm = np.hstack([X_scaled, RULE_SCORE_WEIGHT * score_scaled])

    records = []
    records.extend(run_kmeans_experiment(X_scaled, y, "Original KMeans"))
    records.extend(run_kmeans_experiment(X_arm, y, "ARM-guided KMeans"))
    result_df = pd.DataFrame(records)

    avg_df = (
        result_df.groupby("method", as_index=False)[
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        ]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )
    std_df = (
        result_df.groupby("method", as_index=False)[
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        ]
        .std()
        .sort_values("method")
        .reset_index(drop=True)
    )

    top_rules = label_rules.head(15).copy()
    top_rules["antecedents"] = top_rules["antecedents"].map(itemset_to_string)
    top_rules["consequents"] = top_rules["consequents"].map(itemset_to_string)
    top_rules = top_rules[
        ["antecedents", "consequents", "support", "confidence", "lift", "rule_score"]
    ]

    threshold_df.to_csv(ASSET_DIR / "q5_feature_thresholds.csv", index=False)
    label_rules.to_csv(ASSET_DIR / "q5_label_association_rules_raw.csv", index=False)
    top_rules.to_csv(ASSET_DIR / "q5_top_label_association_rules.csv", index=False)
    score_df.to_csv(ASSET_DIR / "q5_rule_score_features.csv", index=False)
    result_df.to_csv(ASSET_DIR / "q5_seed_metrics.csv", index=False)
    avg_df.to_csv(ASSET_DIR / "q5_average_metrics.csv", index=False)
    std_df.to_csv(ASSET_DIR / "q5_std_metrics.csv", index=False)

    save_markdown_table(top_rules, ASSET_DIR / "q5_top_label_association_rules.md")
    save_markdown_table(result_df, ASSET_DIR / "q5_seed_metrics.md")
    save_markdown_table(avg_df, ASSET_DIR / "q5_average_metrics.md")
    save_markdown_table(std_df, ASSET_DIR / "q5_std_metrics.md")

    print(f"Mined label rules: {len(label_rules)}")
    print("\nTop label rules:")
    print(top_rules)
    print("\nSeed metrics:")
    print(result_df)
    print("\nAverage metrics:")
    print(avg_df)
    print("\nStd metrics:")
    print(std_df)


if __name__ == "__main__":
    main()

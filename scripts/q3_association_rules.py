from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "report_assets" / "assignment2"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["ram", "int_memory", "px_width", "battery_power"]
MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.4
MIN_LIFT = 0.8


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


def categorize_value(value, min_value, max_value):
    width = max_value - min_value
    low_upper = min_value + 0.3 * width
    high_lower = min_value + 0.7 * width

    if value < low_upper:
        return "low"
    if value < high_lower:
        return "medium"
    return "high"


def build_transactions(df):
    thresholds = []
    transactions = []

    for feature in FEATURES:
        min_value = df[feature].min()
        max_value = df[feature].max()
        width = max_value - min_value
        thresholds.append(
            {
                "feature": feature,
                "min": min_value,
                "low_upper": min_value + 0.3 * width,
                "high_lower": min_value + 0.7 * width,
                "max": max_value,
            }
        )

    threshold_df = pd.DataFrame(thresholds)
    threshold_map = threshold_df.set_index("feature").to_dict("index")

    for _, row in df.iterrows():
        transaction = []
        for feature in FEATURES:
            bounds = threshold_map[feature]
            category = categorize_value(row[feature], bounds["min"], bounds["max"])
            transaction.append(f"{feature}_{category}")
        transactions.append(transaction)

    return transactions, threshold_df


def main():
    df = pd.read_csv(ROOT / "mobile_price.csv")
    filtered_df = df[df["price_range"] == 1].copy()
    transactions, threshold_df = build_transactions(filtered_df)

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(encoded, columns=encoder.columns_)

    frequent_itemsets = fpgrowth(transaction_df, min_support=MIN_SUPPORT, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].map(len)
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].map(itemset_to_string)
    frequent_itemsets = frequent_itemsets.sort_values(
        ["length", "support", "itemsets"], ascending=[True, False, True]
    ).reset_index(drop=True)

    raw_itemsets = fpgrowth(transaction_df, min_support=MIN_SUPPORT, use_colnames=True)
    rules = association_rules(raw_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[(rules["support"] >= MIN_SUPPORT) & (rules["confidence"] >= MIN_CONFIDENCE) & (rules["lift"] >= MIN_LIFT)]
    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].map(itemset_to_string)
    rules["consequents"] = rules["consequents"].map(itemset_to_string)
    rules = rules[
        ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    ].sort_values(["support", "confidence", "lift", "antecedents", "consequents"], ascending=[False, False, False, True, True])
    rules = rules.reset_index(drop=True)

    threshold_df.to_csv(ASSET_DIR / "q3_feature_thresholds.csv", index=False)
    frequent_itemsets.to_csv(ASSET_DIR / "q3_frequent_patterns.csv", index=False)
    rules.to_csv(ASSET_DIR / "q3_association_rules.csv", index=False)

    save_markdown_table(threshold_df, ASSET_DIR / "q3_feature_thresholds.md")
    save_markdown_table(frequent_itemsets, ASSET_DIR / "q3_frequent_patterns.md")
    save_markdown_table(rules, ASSET_DIR / "q3_association_rules.md")

    print(f"Filtered samples with price_range=1: {len(filtered_df)}")
    print("\nFeature thresholds:")
    print(threshold_df)
    print("\nFrequent itemsets:")
    print(frequent_itemsets)
    print("\nAssociation rules:")
    print(rules)


if __name__ == "__main__":
    main()

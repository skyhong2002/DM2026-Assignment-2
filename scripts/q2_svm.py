from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

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


def load_split():
    df = pd.read_csv(ROOT / "mobile_price.csv")
    X = df.drop(columns=["price_range"])
    y = df["price_range"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=0.6,
        shuffle=True,
        random_state=SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        shuffle=True,
        random_state=SEED,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(c_value):
    return make_pipeline(StandardScaler(), SVC(C=c_value))


def evaluate_split(model, split_name, X, y):
    y_pred = model.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
    }


def run_experiments():
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    print(f"train={X_train.shape}, validation={X_val.shape}, test={X_test.shape}")

    records = []
    for c_value in C_VALUES:
        model = build_model(c_value)
        model.fit(X_train, y_train)
        for split_name, X, y in [
            ("training", X_train, y_train),
            ("validation", X_val, y_val),
            ("testing", X_test, y_test),
        ]:
            records.append({"C": c_value, **evaluate_split(model, split_name, X, y)})
        print(f"finished C={c_value}")

    results_df = pd.DataFrame(records)
    return results_df


def plot_metric(results_df, metric, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for split_name in ["training", "validation", "testing"]:
        split_df = results_df[results_df["split"] == split_name]
        ax.plot(split_df["C"], split_df[metric], marker="o", linewidth=1.8, label=split_name.capitalize())
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"SVM {metric.replace('_', ' ').title()} across C")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    results_df = run_experiments()

    c1_df = results_df[results_df["C"] == 1.0].copy()
    wide_accuracy = results_df.pivot(index="C", columns="split", values="accuracy").reset_index()
    wide_f1 = results_df.pivot(index="C", columns="split", values="f1_macro").reset_index()

    results_df.to_csv(ASSET_DIR / "q2_svm_all_results.csv", index=False)
    c1_df.to_csv(ASSET_DIR / "q2a_svm_c1_results.csv", index=False)
    wide_accuracy.to_csv(ASSET_DIR / "q2b_svm_accuracy_by_c.csv", index=False)
    wide_f1.to_csv(ASSET_DIR / "q2b_svm_f1_by_c.csv", index=False)

    save_markdown_table(c1_df, ASSET_DIR / "q2a_svm_c1_results.md")
    save_markdown_table(wide_accuracy, ASSET_DIR / "q2b_svm_accuracy_by_c.md")
    save_markdown_table(wide_f1, ASSET_DIR / "q2b_svm_f1_by_c.md")

    plot_metric(results_df, "accuracy", ASSET_DIR / "q2b_svm_accuracy_by_c.png")
    plot_metric(results_df, "f1_macro", ASSET_DIR / "q2b_svm_f1_by_c.png")

    validation_df = results_df[results_df["split"] == "validation"].copy()
    best_row = validation_df.sort_values(["f1_macro", "accuracy", "C"], ascending=[False, False, True]).iloc[0]
    print("\nC=1.0 results:")
    print(c1_df)
    print("\nAccuracy by C:")
    print(wide_accuracy)
    print("\nF1 by C:")
    print(wide_f1)
    print(f"\nBest validation setting: C={best_row['C']} validation_f1={best_row['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

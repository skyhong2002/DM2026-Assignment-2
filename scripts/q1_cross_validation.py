from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.activations import sigmoid
from model.gradients import logloss_sigmoid_grad
from model.linear_model import LinearModel
from model.metrics import evaluate_binary_classifier, logloss


SEED = 40
LEARNING_RATES = [0.005, 0.01, 0.1, 0.5]
REG_LAMBDAS = [1.0, 2.0, 4.0, 8.0]
N_ITERATION = 10000

A1_ROOT = ROOT.parent / "DM2026-Assignment-1"
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


def data_preprocessing(df):
    df = df.copy()
    df["Species"] = df["Species"].astype(str).str.strip()
    df["Species"] = LabelEncoder().fit_transform(df["Species"])

    feature_cols = [c for c in df.columns if c not in ["Id", "Species"]]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    imputer = KNNImputer(n_neighbors=5)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df, feature_cols


def load_assignment1_q3_split():
    np.random.seed(SEED)
    df_raw = pd.read_csv(A1_ROOT / "data" / "NYCU_Iris.csv")
    df, feature_cols = data_preprocessing(df_raw)

    model_df = df.copy()
    for col in feature_cols:
        col_min = model_df[col].min()
        col_max = model_df[col].max()
        if col_max > col_min:
            model_df[col] = (model_df[col] - col_min) / (col_max - col_min)
        else:
            model_df[col] = 0.0

    X = model_df[feature_cols].values.astype(float)
    y = model_df["Species"].values.astype(int)

    return train_test_split(X, y, test_size=0.3, random_state=SEED)


def build_model(lr, reg_lambda):
    return LinearModel(
        dim=None,
        is_reg=False,
        loss_fn=logloss,
        grad_fn=logloss_sigmoid_grad,
        act_fn=sigmoid,
        lr=lr,
        reg_type="l2",
        reg_lambda=reg_lambda,
        n_iteration=N_ITERATION,
        val_ratio=0.0,
        random_state=SEED,
        verbose=False,
        plot_curve=False,
    )


def run_cv(X_train, y_train):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    records = []
    for lr in LEARNING_RATES:
        for reg_lambda in REG_LAMBDAS:
            model = build_model(lr=lr, reg_lambda=reg_lambda)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
            records.append(
                {
                    "learning_rate": lr,
                    "reg_lambda": reg_lambda,
                    "fold_1": scores[0],
                    "fold_2": scores[1],
                    "fold_3": scores[2],
                    "fold_4": scores[3],
                    "fold_5": scores[4],
                    "avg_accuracy": scores.mean(),
                    "std_accuracy": scores.std(),
                }
            )
            print(f"lr={lr:<5} lambda={reg_lambda:<4} avg_accuracy={scores.mean():.4f}")
    return pd.DataFrame(records)


def evaluate_top_two(cv_df, X_train, X_test, y_train, y_test):
    top_two = cv_df.sort_values(
        ["avg_accuracy", "std_accuracy", "learning_rate", "reg_lambda"],
        ascending=[False, True, True, True],
    ).head(2)

    records = []
    for _, row in top_two.iterrows():
        lr = float(row["learning_rate"])
        reg_lambda = float(row["reg_lambda"])
        model = build_model(lr=lr, reg_lambda=reg_lambda)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        title = f"Q1 Test Evaluation - lr={lr}, reg_lambda={reg_lambda}"
        metrics = evaluate_binary_classifier(y_test, y_pred, title=title)
        fig_path = ASSET_DIR / f"q1_test_confusion_lr_{str(lr).replace('.', 'p')}_lambda_{str(reg_lambda).replace('.', 'p')}.png"
        plt.gcf().savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close("all")
        records.append(
            {
                "learning_rate": lr,
                "reg_lambda": reg_lambda,
                "cv_avg_accuracy": float(row["avg_accuracy"]),
                **metrics,
                "confusion_matrix_path": str(fig_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(records)


def main():
    X_train, X_test, y_train, y_test = load_assignment1_q3_split()
    print(f"X_train={X_train.shape}, X_test={X_test.shape}")

    cv_df = run_cv(X_train, y_train)
    cv_table = cv_df.pivot(index="learning_rate", columns="reg_lambda", values="avg_accuracy")
    cv_table = cv_table.reset_index().rename_axis(None, axis=1)
    cv_table.columns = ["learning_rate", "lambda_1", "lambda_2", "lambda_4", "lambda_8"]

    eval_df = evaluate_top_two(cv_df, X_train, X_test, y_train, y_test)

    cv_df.to_csv(ASSET_DIR / "q1_cv_detailed_results.csv", index=False)
    cv_table.to_csv(ASSET_DIR / "q1_cv_4x4_table.csv", index=False)
    eval_df.to_csv(ASSET_DIR / "q1_top2_test_metrics.csv", index=False)

    save_markdown_table(cv_table, ASSET_DIR / "q1_cv_4x4_table.md")
    save_markdown_table(eval_df, ASSET_DIR / "q1_top2_test_metrics.md")

    print("\n4x4 CV table:")
    print(cv_table)
    print("\nTop-two test metrics:")
    print(eval_df)


if __name__ == "__main__":
    main()

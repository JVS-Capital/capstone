import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Bull vs Bear confusion matrix (Predicted vs Actual)"
    )
    p.add_argument("csv",
                   help="CSV with predicted and true semantic labels")
    p.add_argument("--pred-col", default="sem",
                   help="Column name for predicted labels (default: sem)")
    p.add_argument("--true-col", default="true_sem",
                   help="Column name for true labels (default: true_sem)")
    p.add_argument("--labels", nargs=2, default=["Bull", "Bear"],
                   help="Two labels in order: Positive Negative (default: Bull Bear)")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(Path(args.csv))

    # extract only Bull/Bear rows
    pos, neg = args.labels
    mask = df[args.true_col].isin(args.labels)
    y_true = df.loc[mask, args.true_col]
    y_pred = df.loc[mask, args.pred_col]

    # confusion_matrix with rows=predicted, cols=actual
    cm = confusion_matrix(y_pred, y_true, labels=[pos, neg])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Pred_{pos}", f"Pred_{neg}"],
        columns=[f"Actual_{pos}", f"Actual_{neg}"],
    )

    print("\nConfusion matrix:\n")
    print(cm_df.to_string())
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, labels=[pos, neg]))

if __name__ == "__main__":
    main()

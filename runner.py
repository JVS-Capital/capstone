import io
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for headless plotting
matplotlib.use("Agg")

# File paths
INPUT_FILE_PATH = "data.csv"
OUTPUT_CONFUSION_MATRIX_PATH = "confusion_matrix.png"
OUTPUT_STATES_DATA_PATH = "output_data.csv"


def load_csv_data(file_path: str) -> pd.DataFrame:
    raw = open(file_path, "rb").read()
    # Try ODS (zipped) first
    if raw[:2] == b"PK":
        try:
            return pd.read_excel(BytesIO(raw), engine="odf")
        except Exception:
            pass
    # Otherwise strip NULs and parse as CSV
    text = raw.replace(b"\x00", b"").decode("latin-1", errors="ignore")
    return pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")


def compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df.dropna(subset=["mid_price"], inplace=True)
    return df


def preprocess_time_series(df: pd.DataFrame):
    y = np.log(df["mid_price"].to_numpy() + 1).flatten()
    return y, len(y)


def initialize_model_params(y: np.ndarray):
    n = 3
    mu0 = np.array([
        np.percentile(y, 25),
        np.percentile(y, 75),
        np.mean(y),
    ])
    sigma0 = np.array([np.std(y) * 0.5] * n)
    P0 = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.3, 0.5],
    ])
    pi0 = np.array([0.33, 0.33, 0.34])
    return mu0, sigma0, P0, pi0, n


def forward_algorithm(pi0, n, T, P, mu, sigma, y):
    fwd = np.zeros((T, n))
    nxt = np.zeros((T, n))
    phi0 = norm.pdf((y[0] - mu) / sigma)
    fwd[0] = pi0 * phi0 / (pi0.dot(phi0))
    nxt[0] = P.dot(fwd[0])
    for t in range(1, T):
        phi = norm.pdf((y[t] - mu) / sigma)
        fwd[t] = nxt[t-1] * phi / (nxt[t-1].dot(phi))
        nxt[t] = P.dot(fwd[t])
    return fwd, nxt


def backward_algorithm(fwd, nxt, n, T, P, mu, sigma, y):
    bwd = np.zeros((T, n))
    bwd[-1] = fwd[-1]
    for t in range(T-2, -1, -1):
        ratio = np.divide(
            bwd[t+1],
            nxt[t],
            out=np.zeros(n),
            where=nxt[t] != 0
        )
        bwd[t] = fwd[t] * P.dot(ratio)
    return bwd


def classify_market_state(df: pd.DataFrame) -> pd.DataFrame:
    df["State"] = np.where(
        df["mid_price"].diff() > 0, 1,
        np.where(df["mid_price"].diff() < 0, -1, 0)
    )
    return df


def calculate_accuracy_and_plot(df: pd.DataFrame, y_true, y_pred):
    # 3×3 confusion matrix
    labels = [1, -1, 0]
    names = ["Bull", "Bear", "Sideways"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 1. Save & show heatmap
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix: Bull | Bear | Sideways")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Saved confusion matrix to {OUTPUT_CONFUSION_MATRIX_PATH}\n")

    # 2. Print numeric accuracies
    overall_acc = np.trace(cm) / cm.sum() * 100
    per_class_acc = np.diag(cm) / cm.sum(axis=1) * 100
    print(f"Overall accuracy: {overall_acc:.2f}%")
    for name, acc in zip(names, per_class_acc):
        print(f"{name:9s} accuracy: {acc:.2f}%")
    print()

    # 3. Print confusion matrix table
    df_cm = pd.DataFrame(cm,
                         index=[f"True {n}" for n in names],
                         columns=[f"Pred {n}" for n in names])
    print("Confusion Matrix (counts):")
    print(df_cm, "\n")

    # 4. Print TP, FN, FP, TN per class
    for i, name in enumerate(names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        print(f"{name} -> TP={TP}, FN={FN}, FP={FP}, TN={TN}")
    print()


def main():
    # Load & inspect
    df = load_csv_data(INPUT_FILE_PATH)
    print("Loaded columns:", df.columns.tolist(), "\n")

    # Ensure required columns
    for col in ("bid", "ask", "time"):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in input data")

    # HMM setup
    df = compute_mid_price(df)
    y, T = preprocess_time_series(df)
    mu0, sigma0, P0, pi0, n_states = initialize_model_params(y)
    fwd, nxt = forward_algorithm(pi0, n_states, T, P0, mu0, sigma0, y)
    bwd = backward_algorithm(fwd, nxt, n_states, T, P0, mu0, sigma0, y)

    # True vs. Predicted
    df = classify_market_state(df)
    df["Price_Change"]   = df["mid_price"].diff().shift(-1)
    df["Price_Movement"] = np.where(
        df["Price_Change"] > 0, 1,
        np.where(df["Price_Change"] < 0, -1, 0)
    )
    valid  = df.dropna(subset=["State", "Price_Movement"])
    y_true = valid["Price_Movement"].astype(int)
    y_pred = valid["State"].astype(int)

    # Accuracy & confusion
    calculate_accuracy_and_plot(df, y_true, y_pred)

    # Save state probabilities
    records = []
    for t in range(T):
        ts, price = df.iloc[t]["time"], df.iloc[t]["mid_price"]
        b, br, s = bwd[t]
        dom = "Sideways" if s > max(b, br) else ("Bull" if b > br else "Bear")
        records.append([ts, price, b, br, s, dom])
    states_df = pd.DataFrame(records, columns=[
        "Time", "Mid Price", "Bull", "Bear", "Sideways", "Dominant State"
    ])
    states_df.to_csv(OUTPUT_STATES_DATA_PATH, index=False)
    print(f"States data saved to {OUTPUT_STATES_DATA_PATH}")


if __name__ == "__main__":
    main()

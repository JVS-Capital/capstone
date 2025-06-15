import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

def parse_args():
    p = argparse.ArgumentParser(
        description="OHLCV → HMM regimes + semantic‐color plot + true labels"
    )
    p.add_argument("csv", help="Input CSV with ts,open,high,low,close,volume")
    p.add_argument("--states",     type=int, default=3, help="Number of HMM states")
    p.add_argument("--vol-window", type=int, default=7, help="Rolling window for volatility")
    p.add_argument("--iter",       type=int, default=25, help="HMM max iterations")
    p.add_argument("--out-csv",    help="Output CSV (defaults to <input>_hmm.csv)")
    p.add_argument("--out-plot",   help="Output PNG (defaults to <input>_hmm.png)")
    p.add_argument("--show",       action="store_true", help="Show plot interactively")
    return p.parse_args()

def load_and_featurize(path: Path, vol_window: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    df["mid"]     = (df["high"] + df["low"]) / 2
    df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1)).fillna(0)
    df["vol"]     = df["log_ret"].rolling(vol_window).std().fillna(0)
    return df

def fit_hmm(X: np.ndarray, n_states: int, n_iter: int) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42
    )
    model.fit(Xs)
    return model.predict(Xs)

def main():
    args     = parse_args()
    csv_path = Path(args.csv)

    # 1) load & featurize
    df = load_and_featurize(csv_path, args.vol_window)
    X  = df[["log_ret", "vol"]].values

    # 2) fit HMM & assign regimes
    regimes = fit_hmm(X, args.states, args.iter)
    df["regime"] = regimes

    # 3) semantic labels: Bull=highest mean return, Bear=lowest, Side=middle
    means      = df.groupby("regime")["log_ret"].mean()
    bull_state = means.idxmax()
    bear_state = means.idxmin()
    df["sem"] = df["regime"].map(
        lambda s: "Bull" if s == bull_state
                  else ("Bear" if s == bear_state else "Side")
    )
    color_map = {"Bull": "green", "Bear": "red", "Side": "blue"}
    df["col"] = df["sem"].map(color_map)

    # 4) derive true labels from log returns
    df["true_sem"] = df["log_ret"].apply(
        lambda r: "Bull" if r > 0 else ("Bear" if r < 0 else "Side")
    )

    # 5) save labeled CSV
    out_csv = args.out_csv or f"{csv_path.stem}_hmm.csv"
    df.reset_index().to_csv(out_csv, index=False)
    print(f"✓ regimes & true_sem saved → {out_csv}")

    # 6) plot with semantic colors
    out_png = args.out_plot or f"{csv_path.stem}_hmm.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    for label in ["Bull", "Side", "Bear"]:
        sub = df[df["sem"] == label]
        if not sub.empty:
            ax.scatter(
                sub.index, sub["close"],
                c=color_map[label],
                label=label,
                s=8,
                alpha=0.8
            )
    ax.set_title(f"{csv_path.name}   HMM {args.states}-state")
    ax.set_ylabel("Close")
    ax.legend(title="Regime")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"✓ plot saved → {out_png}")
    if args.show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()

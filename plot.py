

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

def getarg(flag, default):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx+1 < len(sys.argv):
            return sys.argv[idx+1]
    return default

def fit_hmm(X, n_states):
    cfg = dict(n_components=int(n_states),
               n_iter=200, tol=1e-4, min_covar=1e-3)
    # try full covariance
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = GaussianHMM(covariance_type="full", **cfg)
            model.fit(X)
        return model
    except Exception:
        model = GaussianHMM(covariance_type="diag", **cfg)
        model.fit(X)
        return model

def main(csv_file):
    path = Path(csv_file)
    df = pd.read_csv(path)

    # 1) parse timestamp → datetime
    df["dt"] = pd.to_datetime(df["ts"].astype(str).str.strip("'"),
                              errors="coerce")
    df.dropna(subset=["dt"], inplace=True)
    df.set_index("dt", inplace=True)
    df.sort_index(inplace=True)

    # 2) compute mid-price + log-returns
    df["mid"] = (df["high"] + df["low"]) / 2
    ret = np.log(df["mid"]).diff().dropna()

    # 3) build features: [ret, |ret|, (opt) vol-pct]
    feats = [ret.values.reshape(-1,1),
             np.abs(ret).values.reshape(-1,1)]
    if "--use-vol" in sys.argv:
        volr = df["volume"].pct_change().fillna(0).iloc[1:]
        feats.append(volr.values.reshape(-1,1))
    X = np.hstack(feats)

    # 4) fit + predict HMM
    n_states = int(getarg("--states", 3))
    model    = fit_hmm(X, n_states)
    states   = model.predict(X)

    # 5) align back to df
    df2 = df.iloc[1:].copy()
    df2["regime"] = states

    # 6) find true switch points
    switches = df2["regime"].diff().fillna(0).astype(bool)

    # 7) semantic mapping by mean returns
    df2["ret"] = np.log(df2["mid"]).diff()
    means = df2.groupby("regime")["ret"].mean()
    bear_s = means.idxmin()       # most negative
    bull_s = means.idxmax()       # most positive
    rem    = [s for s in means.index if s not in (bear_s, bull_s)]
    if len(rem)==1:
        side_s = rem[0]
    else:
        # pick one closest to zero if multiple remain
        side_s = min(rem, key=lambda s: abs(means[s]))
    sem_map   = {bear_s:"Bear", side_s:"Side", bull_s:"Bull"}
    color_map = {"Bear":"red", "Side":"violet", "Bull":"green"}
    df2["sem"]   = df2["regime"].map(sem_map)
    df2["color"] = df2["sem"].map(color_map)

    # 8) save output CSV
    out = path.with_name(path.stem + "_hmm_labeled.csv")
    df2.reset_index()[["dt","open","high","low","close","volume","regime","sem"]].rename(
        columns={"dt":"ts"}
    ).to_csv(out, index=False)
    print(f"✔ saved labeled data to {out}")
    print("Mean log-return per state:\n", means)
    print("Counts by semantic regime:\n", df2["sem"].value_counts())

    # 9) optional plot
    if "--plot" in sys.argv:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.scatter(df2.index, df2["mid"],
                   c=df2["color"], s=8, alpha=0.8)
        # one dashed line per regime switch
        for t in df2.index[switches]:
            ax.axvline(t, color="grey", linestyle="--", lw=1, alpha=0.6)

        ax.set_title(f"{path.name}   |   red=Bear, violet=Side, green=Bull")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mid-Price")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python plot.py <csv> [--states N] [--use-vol] [--plot]")
    else:
        main(sys.argv[1])

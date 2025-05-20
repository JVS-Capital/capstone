
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
COLMAP = {-1: "red", 0: "violet", 1: "green"}

# ---------------- CLI ----------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("csv", help="candle file produced by trades_to_candles.py")
cli.add_argument("--penalty", type=int, default=8, help="ruptures Pelt penalty")
cli.add_argument("--k", type=int, default=5, help="± bars around a break window")
cli.add_argument("--states", type=int, default=3, help="# HMM components")
cli.add_argument("--novol", action="store_true",
                 help="disable |return| feature (use only log-return)")
cli.add_argument("--out", help="output PNG path (auto if omitted)")
args = cli.parse_args()

# ---------------- LOAD DATA ---------------------------------------------
df = pd.read_csv(args.csv)
if "ts" not in df.columns:
    raise SystemExit("CSV must have a 'ts' column (ISO UTC)")

df["dt"] = pd.to_datetime(df["ts"])
df.set_index("dt", inplace=True)
df["mid"] = (df["high"] + df["low"]) / 2

# ---------------- STRUCTURAL BREAKS -------------------------------------
series = df["mid"].astype(float)
algo   = rpt.Pelt(model="rbf").fit(series.values)
break_pts = algo.predict(pen=args.penalty)[:-1]

df["break"] = 0
for ix in break_pts:
    df.iloc[max(0, ix-args.k): ix+args.k, df.columns.get_loc("break")] = 1

# ---------------- HMM FEATURES ------------------------------------------
ret = np.log(series).diff().dropna()
feat = ret.values.reshape(-1, 1)      # column 0

if not args.novol:
    feat = np.column_stack([feat, np.abs(ret.values)])   # column 1

# ---------------- HMM FIT -----------------------------------------------
hmm = GaussianHMM(n_components=args.states, covariance_type="full").fit(feat)
states = hmm.predict(feat)

if len(np.unique(states)) == 1:
    print("⚠️  HMM collapsed to one state – "
          "consider longer interval, more data, or additional features.")

# map states → Bear / Side / Bull by (mean, volatility) ranking
means = hmm.means_[:, 0]
vols  = np.sqrt(hmm.covars_[:, 0, 0])
rank  = sorted(zip(means, vols, range(args.states)))      # low→high mean
mapping = {rank[0][2]: -1,              # Bear   (red)
           rank[1][2]:  0,              # Side   (violet)
           rank[2][2]:  1}              # Bull   (green)

# align df length
df = df.iloc[1:]                        # drop first NaN diff
df["regime"] = [mapping[s] for s in states]

# ---------------- ROC-AUC -----------------------------------------------
chg  = (df["regime"].diff() != 0).astype(int)
prob = chg.rolling(args.k).mean().fillna(0)
auc  = roc_auc_score(df["break"], prob)
print(f"ROC-AUC vs. break windows : {auc:.3f}")

# ---------------- PLOT ---------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(df.index, df["mid"],
           c=df["regime"].map(COLMAP), s=6, alpha=0.85)

for b in break_pts:
    ax.axvline(df.index[b], color="grey", ls="--", lw=0.8, alpha=0.7)

ttl = f"{args.csv}  |  Bull/Side/Bear coloured  |  ROC AUC {auc:.3f}"
ax.set_title(ttl)
ax.set_ylabel("Mid-Price")
plt.tight_layout()

out_path = (args.out or
            Path(args.csv).with_suffix("").with_name(
                Path(args.csv).stem + "_hmm_plot.png"))
fig.savefig(out_path, dpi=150)
print(" Saved figure →", out_path)

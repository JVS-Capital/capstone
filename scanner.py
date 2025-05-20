
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ——— CLI ——————————————————————————————————————————————
p = argparse.ArgumentParser(description="HMM change-point scanner")
p.add_argument("csv", help="candle CSV (ts,open,high,low,close,volume)")
p.add_argument("--states", type=int, default=3, help="# HMM components")
p.add_argument("--k",      type=int, default=5, help="extend change ±K bars")
p.add_argument("--novol",  action="store_true",
               help="use only log-return, skip |return| feature")
args = p.parse_args()


path = Path(args.csv)
df   = pd.read_csv(path)
required = {"ts","open","high","low","close","volume"}
missing  = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns: {missing}")

def parse_ts(val: str) -> pd.Timestamp:
    s = str(val).strip().strip("'")
    if "/" in s:

        return pd.to_datetime(s, dayfirst=True, utc=True, errors="coerce")
    else:
       
        return pd.to_datetime(s, utc=True, errors="coerce")

df["dt"] = df["ts"].apply(parse_ts)
df = df.dropna(subset=["dt"]).set_index("dt")


df["mid"] = (df["high"] + df["low"]) / 2
ret       = np.log(df["mid"]).diff().dropna()

X = ret.values.reshape(-1,1)
if not args.novol:
    X = np.column_stack([X, np.abs(ret.values)])


def fit_hmm(X, n_states):
    kwargs = dict(n_components=n_states,
                  covariance_type="full",
                  n_iter=500,
                  tol=1e-3,
                  min_covar=1e-3,
                  verbose=False)
    try:
        # catch any LinAlg warnings as errors
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = GaussianHMM(**kwargs)
            model.fit(X)
        return model
    except Exception:
        # fallback to diagonal if full fails
        diag_kwargs = kwargs.copy()
        diag_kwargs["covariance_type"] = "diag"
        model = GaussianHMM(**diag_kwargs)
        model.fit(X)
        return model

hmm    = fit_hmm(X, args.states)
states = hmm.predict(X)

df2     = df.iloc[1:].copy()          
df2["regime"] = states


chg = df2["regime"].diff().fillna(0).astype(bool)


breaks = pd.Series(0, index=df2.index, dtype=int)
for t in df2.index[chg]:
    i = df2.index.get_loc(t)
    start, end = max(0, i-args.k), i+args.k+1
    breaks.iloc[start:end] = 1
df2["break"] = breaks.values


out = df2.reset_index().rename(columns={"dt":"timestamp", "ts":"ts_orig"})
# ensure ts column remains original string
out["ts"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
cols    = ["ts","open","high","low","close","volume","break","regime"]
save_to = path.with_name(path.stem + "_hmm_labeled.csv")
out[cols].to_csv(save_to, index=False)
print(f" wrote {save_to}")

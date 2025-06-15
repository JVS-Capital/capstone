import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Sharpe, Sortino, Max Drawdown & Calmar for a regime-based signal"
    )
    p.add_argument("csv", help="CSV with ts, price and signal columns")
    p.add_argument("--signal-col", default="sem",
                   help="Column name with semantic regimes (default: sem)")
    p.add_argument("--long", default="Bull",
                   help="Label in signal-col to go LONG (default: Bull)")
    p.add_argument("--short", default="Bear",
                   help="Label in signal-col to go SHORT (default: Bear)")
    p.add_argument("--price-col", default="close",
                   help="Price column name for returns (default: close)")
    p.add_argument("--ppy", type=float, default=252,
                   help="Periods per year for annualization (default: 252)")
    p.add_argument("--rf", type=float, default=0.0,
                   help="Per-period risk-free rate (default: 0.0)")
    return p.parse_args()

def compute_metrics(returns: pd.Series, ppy: float, rf: float):
    """
    Calculate Sharpe, Sortino, Max Drawdown, and Calmar Ratio.
    returns : per-period strategy returns
    ppy     : periods per year for annualization
    rf      : per-period risk-free rate
    """
    excess = returns - rf
    mu = excess.mean()
    sigma = excess.std(ddof=0)
    sharpe = (mu / sigma) * np.sqrt(ppy) if sigma > 0 else np.nan

    downside = excess[excess < 0]
    dd_std = downside.std(ddof=0)
    sortino = (mu / dd_std) * np.sqrt(ppy) if dd_std > 0 else np.nan

    # compute wealth curve
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (peak - wealth) / peak
    max_dd = drawdown.max()

    # annualized return (CAGR)
    periods = len(returns)
    if periods > 0 and wealth.iloc[-1] > 0:
        annual_return = wealth.iloc[-1] ** (ppy / periods) - 1
    else:
        annual_return = np.nan

    calmar = annual_return / max_dd if max_dd > 0 else np.nan

    return sharpe, sortino, max_dd, calmar

def main():
    args = parse_args()
    df = pd.read_csv(Path(args.csv), parse_dates=["ts"])
    df = df.sort_values("ts").set_index("ts")

    # compute simple returns
    price = df[args.price_col]
    ret = price.pct_change().fillna(0)

    # build signal: 1 for LONG, -1 for SHORT, 0 otherwise
    sig = pd.Series(0, index=df.index)
    sig[df[args.signal_col] == args.long]  = 1
    sig[df[args.signal_col] == args.short] = -1

    # shift signal to trade next bar
    sig = sig.shift(1).fillna(0)

    # strategy returns
    strat_ret = sig * ret

    # compute metrics
    sharpe, sortino, max_dd, calmar = compute_metrics(strat_ret, args.ppy, args.rf)

    # print results
    print("\nPerformance Summary:")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Sortino Ratio  : {sortino:.3f}")
    print(f"  Max Drawdown   : {max_dd:.2%}")
    print(f"  Calmar Ratio   : {calmar:.3f}\n")

if __name__ == "__main__":
    main()

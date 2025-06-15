from pathlib import Path
import argparse
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


parser = argparse.ArgumentParser(description="Trades to OHLCV converter")
parser.add_argument(
    "-i", "--interval",
    default="1s",
    help="pandas resample rule, e.g. 1s, 30s, 1min, 5min, 2H (default 5min)"
)
INTERVAL = parser.parse_args().interval.lower().replace(" ", "")


def parse_mixed(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.lstrip("'").str.strip()
    try:                                 # pandas ≥ 2.1
        return pd.to_datetime(txt, utc=True, format="mixed")
    except TypeError:                    # older pandas, two-pass
        dt = pd.to_datetime(txt, utc=True,
                            errors="coerce",
                            format="%Y-%m-%d %H:%M:%S.%f")
        miss = dt.isna()
        if miss.any():
            dt[miss] = pd.to_datetime(txt[miss], utc=True,
                                      errors="coerce",
                                      format="%Y-%m-%d %H:%M")
        return dt


def convert(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        return

    
    df["dt"] = parse_mixed(df["ts"])
    df = df.dropna(subset=["dt"])
    if df.empty:
        return

    
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"]  = pd.to_numeric(df["size"],  errors="coerce")
    df.dropna(subset=["price", "size"], inplace=True)
    if df.empty:
        return

    
    df.set_index("dt", inplace=True)
    agg = df.resample(INTERVAL, label="right", closed="right").apply({
        "price": ["first", "max", "min", "last"],
        "size" :  "sum",            
    }).dropna()

    if agg.empty:
        return

    
    agg.columns = ["open", "high", "low", "close", "volume"]
    agg.reset_index(inplace=True)
    agg["ts"] = agg["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = agg[["ts", "open", "high", "low", "close", "volume"]]

    suffix   = INTERVAL
    out_path = path.with_name(path.stem.replace("_trades", f"_{suffix}") + ".csv")
    out.to_csv(out_path, index=False)
    print(f"{path.name} → {out_path.name}  ({len(out)} bars)")


if __name__ == "__main__":
    trade_files = list(DATA_DIR.glob("*_trades.csv"))
    if not trade_files:
        print(" No *_trades.csv files found in", DATA_DIR)
    for f in trade_files:
        convert(f)

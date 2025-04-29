"""
OKX spot order-book streamer

 – REST pre-filter: 24 h quote-vol > VOLUME_TH
 – 5 m returns, pairwise < CORR
 – WebSocket 'books' snapshots per-pair CSV under ./data
"""

import os, json, csv, requests, websocket, pandas as pd, numpy as np
from datetime import datetime
import pytz


VOLUME_TH   = 2_000_000        # 24 h quote-volume in USDT
CORR_TH     = 0.20              # max corr for any pair
CANDLES     = 500               # bars for correlation calc
CAND_BAR    = "5m"              # timeframe
SNAP_VOL_TH = 1_000             # min quote-vol
HEARTBEAT   = 60                # seconds between “alive” prints
CSV_DIR     = "./data"


os.makedirs(CSV_DIR, exist_ok=True)


def high_volume_spot_pairs():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
    data = requests.get(url, timeout=10).json().get("data", [])
    return [d["instId"] for d in data
            if float(d["volCcy24h"]) > VOLUME_TH and d["instId"].endswith("USDT")]

def pct_returns(inst, limit=CANDLES, bar=CAND_BAR):
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst}&bar={bar}&limit={limit}"
    rows = requests.get(url, timeout=10).json().get("data", [])
    if not rows:
        return pd.Series(dtype=float)
    closes = np.array([float(r[4]) for r in rows[::-1]], dtype=float)
    return pd.Series(closes).pct_change().dropna()

def volume_and_corr_filter():
    vol_pairs = high_volume_spot_pairs()
    returns = {p: pct_returns(p) for p in vol_pairs}
    df = pd.DataFrame({k:v for k,v in returns.items() if not v.empty})
    if df.empty: return []
    corr = df.corr()
    return [p for p in corr.columns
            if all(abs(corr.loc[p,q]) < CORR_TH for q in corr.columns if q!=p)]

vol_pairs = high_volume_spot_pairs()
print(f" After volume filter: {len(vol_pairs)} pairs")

coins = volume_and_corr_filter()
print(f" After correlation filter: {len(coins)} pairs → {coins[:10]}")

if not coins:
    coins = ["BTC-USDT"]

print(" Subscribing to:", coins)

coins = volume_and_corr_filter() or ["BTC-USDT"]
print(" Subscribing to:", coins)


HDR = ['local_ts','exch_ts','inst','side','price','size','quote_vol','liq','cnt']
def csv_path(inst): return os.path.join(CSV_DIR, f"{inst.replace('-','_').lower()}_ob.csv")
def ensure_csv(inst):
    fn = csv_path(inst)
    if not os.path.exists(fn):
        with open(fn,"w",newline="") as f:
            csv.DictWriter(f,fieldnames=HDR).writeheader()
    return fn


last_print = 0
def parse_snap(data, inst):
    global last_print
    rows, local = [], datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    for snap in data:
        quote_vol = sum(float(p)*float(s) for p,s,_,_ in snap['bids']+snap['asks'])
        if quote_vol < SNAP_VOL_TH:
            continue
        exch = datetime.fromtimestamp(int(snap['ts'])/1e3, tz=pytz.utc)\
                       .strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        for key, side in (('asks','ask'), ('bids','bid')):
            for p,s,l,c in snap[key]:
                rows.append({'local_ts':local,'exch_ts':exch,'inst':inst,
                             'side':side,'price':p,'size':s,'quote_vol':float(p)*float(s),
                             'liq':l,'cnt':c})
    # heartbeat
    now = datetime.utcnow().timestamp()
    if rows and now - (last_print or 0) > HEARTBEAT:
        print(f" receiving {inst} snapshots ({len(rows)} rows written)")
        last_print = now
    return rows


def on_message(ws, message):
    """
    Handle every inbound WebSocket frame.

    • Skip control / heartbeat packets (no "data" field or an "event" key).
    • Ensure the channel is *books* before parsing.
    • Write filtered snapshots to CSV.
    """
    d = json.loads(message)


    if ("data" not in d) or d.get("event"):
        return

    #  Process only order-book updates
    if d.get("arg", {}).get("channel") != "books":
        return

    inst = d["arg"]["instId"]
    rows = parse_snap(d["data"], inst)

    if rows:
        fn = ensure_csv(inst)
        with open(fn, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=HDR).writerows(rows)


def on_open(ws):
    sub = {"op":"subscribe",
           "args":[{"channel":"books","instId":c} for c in coins]}
    ws.send(json.dumps(sub))
    print(" WebSocket opened & subscriptions sent.")

def on_error(ws, err):  print(" WS error:", err)
def on_close(ws, *_):   print(" WebSocket closed.")



if __name__ == "__main__":
    websocket.enableTrace(False)
    wsa = websocket.WebSocketApp(
        "wss://ws.okx.com:8443/ws/v5/public",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    wsa.run_forever()

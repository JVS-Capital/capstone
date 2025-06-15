
"""
Streams spot‑trade ticks from OKX and appends them to CSV.

* Works with older `websocket-client` (ping passed to `run_forever`).
* Keeps full ISO timestamp by prefixing apostrophe, so Excel shows date.
* `TRADE_PRINT_STEP` controls how often a heartbeat is printed.
"""

import csv, json, time, traceback, requests, websocket
from datetime import datetime
from pathlib import Path

#  PATHS 
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR  = BASE_DIR / "data"; CSV_DIR.mkdir(exist_ok=True)

# SETTINGS 
VOLUME_TH         = 80_000_000                # 24h volume filter (USDT)
EXCLUDED_PAIRS    = {"USDC-USDT"}
CHANNEL           = "trades"
MAX_BATCH         = 20                       # pairs per subscribe
PING_INTERVAL     = 20
PING_TIMEOUT      = 10
TRADE_PRINT_STEP  = 1000                     # heartbeat every N trades

HDR = ["ts", "price", "size", "side", "trade_id"]

# HELPERS 

def high_volume_pairs():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
    data = requests.get(url, timeout=10).json().get("data", [])
    return [d["instId"] for d in data
            if float(d.get("volCcy24h", 0)) > VOLUME_TH and d["instId"].endswith("USDT")]

def csv_path(inst):
    return CSV_DIR / f"{inst.replace('-', '_').lower()}_trades.csv"

def ensure_csv(inst):
    fn = csv_path(inst)
    if not fn.exists():
        with fn.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=HDR).writeheader()
        print("📄 Created", fn.name)
    return fn

#  STREAMER 
class TradeStreamer:
    def __init__(self, pairs):
        self.pairs   = pairs
        self.counter = {p: 0 for p in pairs}

    # callbacks 
    def on_open(self, ws):
        batches = [self.pairs[i:i+MAX_BATCH] for i in range(0, len(self.pairs), MAX_BATCH)]
        for b in batches:
            ws.send(json.dumps({
                "op": "subscribe",
                "args": [{"channel": CHANNEL, "instId": p} for p in b],
            }))
        # create files immediately
        for p in self.pairs:
            ensure_csv(p)
        print("🌐 WS opened – subscribed", len(self.pairs), "pairs")

    @staticmethod
    def _parse_trade(t):
        """Return (ts, price, size, side, tid) from OKX payload (dict or list)."""
        if isinstance(t, dict):
            return int(t['ts']), t['px'], t['sz'], t['side'], t.get('tradeId', '')
        if len(t) == 5:  # [tradeId, px, sz, side, ts]
            tid, px, sz, side, ts = t
        else:           # [ts, tradeId, px, sz, side, ...]
            ts, tid, px, sz, side = t[:5]
        return int(ts), px, sz, side, tid

    def on_message(self, ws, msg):
        d = json.loads(msg)
        if 'data' not in d:
            if d.get('event') == 'subscribe':
                ensure_csv(d['arg']['instId'])
            return
        if d['arg'].get('channel') != CHANNEL:
            return

        inst = d['arg']['instId']
        rows = []
        for raw in d['data']:
            try:
                ts, px, sz, side, tid = self._parse_trade(raw)
            except Exception as e:
                print('⚠️ parse error', e, raw)
                continue
            rows.append({
                'ts'      : "'" + datetime.utcfromtimestamp(ts/1e3).isoformat(sep=' ')[:-3],
                'price'   : px,
                'size'    : sz,
                'side'    : side,
                'trade_id': tid,
            })
        if not rows:
            return
        fn = ensure_csv(inst)
        with fn.open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=HDR).writerows(rows)
        self.counter[inst] += len(rows)
        if self.counter[inst] % TRADE_PRINT_STEP == 0:
            print(f"↳ {inst}: trades logged = {self.counter[inst]}")

    def on_error(self, ws, err):
        print('⚠️ WS error:', err)
    def on_close(self, ws, *_):
        print('🔒 WS closed')

    # main loop 
    def run(self):
        while True:
            try:
                app = websocket.WebSocketApp(
                    'wss://ws.okx.com:8443/ws/v5/public',
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                app.run_forever(ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)
            except Exception:
                traceback.print_exc()
            print(' Reconnecting in 5 s …')
            time.sleep(5)

#  MAIN 
if __name__ == '__main__':
    try:
        pairs = [p for p in high_volume_pairs() if p not in EXCLUDED_PAIRS]
    except Exception as e:
        print('Ticker fetch failed, defaulting to BTC-USDT:', e)
        pairs = []
    if not pairs:
        pairs = ['BTC-USDT']
    print('Pairs to stream:', pairs)
    TradeStreamer(pairs).run()

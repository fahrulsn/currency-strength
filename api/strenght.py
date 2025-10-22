# api/strength.py
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os, time, math, json
import requests
import pandas as pd
from typing import List, Dict, Tuple
from ta.trend import ema_indicator

# =======================
# DEFAULT CONFIG
# =======================
API_KEY = os.getenv("TWELVEDATA_API_KEY", "")   # set di Vercel → Project Settings → Environment Variables
INTERVAL_DEFAULT = "1h"
OUTPUTSIZE_DEFAULT = 260
RATE_LIMIT_PER_MIN = 8          # Twelve Data free ~8 req/min
SLEEP_BETWEEN_BATCH = 65        # aman antar batch
USE_EMA50_CONFLUENCE_DEFAULT = "1"  # "1" = True

PAIRS: List[str] = [
    "EUR/USD","GBP/USD","AUD/USD","NZD/USD",
    "USD/JPY","USD/CHF","USD/CAD",
    "EUR/GBP","EUR/JPY","EUR/CHF",
    "GBP/JPY","AUD/JPY","CAD/JPY","NZD/JPY",
    "AUD/CAD","AUD/CHF","AUD/NZD",
    "CAD/CHF","EUR/CAD","EUR/NZD",
    "GBP/CAD","GBP/CHF","GBP/NZD",
    "CHF/JPY","NZD/CHF"
]
CURRENCIES = ["USD","EUR","GBP","JPY","AUD","NZD","CAD","CHF"]

DXY_WEIGHTS = {
    "EUR/USD": 0.576,
    "USD/JPY": 0.136,
    "GBP/USD": 0.119,
    "USD/CAD": 0.091,
    "USD/CHF": 0.036,
}
DEFAULT_WEIGHT = 1.0
USD_DXY_BOOST_DEFAULT = 2.0  # 0 utk matikan

def fetch_timeseries(symbol: str, interval: str, outputsize: int, api_key: str) -> Dict:
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    )
    r = requests.get(url, timeout=25)
    try:
        return r.json()
    except Exception:
        return {"message": f"Non-JSON response: {r.text[:160]}..."}

def ema_status_from_values(values: list, use_confluence: bool) -> Tuple[float, float, float, bool]:
    if not isinstance(values, list) or len(values) < 200:
        raise ValueError("Insufficient values for EMA200")

    df = pd.DataFrame(values)
    df["close"] = df["close"].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    df["ema50"] = ema_indicator(df["close"], window=50)
    df["ema200"] = ema_indicator(df["close"], window=200)

    last_close = float(df["close"].iloc[-1])
    ema50 = float(df["ema50"].iloc[-1])
    ema200 = float(df["ema200"].iloc[-1])

    if math.isnan(ema200):
        raise ValueError("EMA200 NaN (data kurang panjang)")

    signal = (last_close > ema200) if not use_confluence else ((last_close > ema50) and (last_close > ema200))
    return last_close, ema50, ema200, signal

def get_weight(pair: str) -> float:
    return DXY_WEIGHTS.get(pair, DEFAULT_WEIGHT)

def score_pairs_weighted(pairs: List[str], interval: str, outputsize: int, api_key: str,
                         use_confluence: bool, obey_rate_limit: bool) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    scores: Dict[str, float] = {c: 0.0 for c in CURRENCIES}
    pair_rows, logs = [], []

    for i in range(0, len(pairs), RATE_LIMIT_PER_MIN):
        batch = pairs[i:i + RATE_LIMIT_PER_MIN]
        for pair in batch:
            base, quote = pair.split("/")
            try:
                data = fetch_timeseries(pair, interval, outputsize, api_key)

                if "message" in data and "API credits" in str(data["message"]).lower():
                    raise RuntimeError(data["message"])
                if "values" not in data:
                    raise ValueError(data.get("message", "No 'values' in response"))

                last_close, ema50, ema200, signal = ema_status_from_values(data["values"], use_confluence)
                w = get_weight(pair)

                if signal:
                    scores[base] += w
                    scores[quote] -= w
                    sig_txt = "BULL"
                else:
                    scores[base] -= w
                    scores[quote] += w
                    sig_txt = "BEAR"

                pair_rows.append({
                    "Pair": pair, "Base": base, "Quote": quote,
                    "Close": last_close, "EMA50": ema50, "EMA200": ema200,
                    "Signal": sig_txt, "Weight": w
                })
                logs.append(f"[OK] {pair}: close={last_close:.5f}, ema50={ema50:.5f}, ema200={ema200:.5f}, {sig_txt}, w={w}")

            except RuntimeError as e:
                logs.append(f"[RATE-LIMIT] {pair}: {e}")
            except Exception as e:
                logs.append(f"[FAIL] {pair}: {e}")

        if obey_rate_limit and (i + RATE_LIMIT_PER_MIN < len(pairs)):
            # Hindari timeout: tidur hanya jika user minta proses full
            time.sleep(SLEEP_BETWEEN_BATCH)

    df_pairs = pd.DataFrame(pair_rows)
    df_rank = (pd.DataFrame([{"Currency": k, "Score": v} for k, v in scores.items()])
               .sort_values("Score", ascending=False, ignore_index=True))
    return df_rank, df_pairs, logs

def get_dxy_signal(interval: str, outputsize: int, api_key: str) -> Tuple[float, float, float, int, str]:
    data = fetch_timeseries("DXY", interval, outputsize, api_key)
    if "values" not in data:
        raise ValueError(f"DXY fetch error: {data.get('message', 'No values')}")
    last_close, ema50, ema200, signal = ema_status_from_values(data["values"], use_confluence=False)
    boost_sign = 1 if last_close > ema200 else -1
    txt = "USD Boost + (DXY di atas EMA200)" if boost_sign > 0 else "USD Boost - (DXY di bawah EMA200)"
    return last_close, ema50, ema200, boost_sign, txt

def build_response(status: int, body: dict, headers: dict = None):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    base_headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*"
    }
    if headers:
        base_headers.update(headers)
    return status, base_headers, payload

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)
            api_key = q.get("apikey", [API_KEY])[0]
            if not api_key:
                status, headers, payload = build_response(400, {"error": "Missing API key (set TWELVEDATA_API_KEY or ?apikey=...)"})
                self._write(status, headers, payload); return

            interval = q.get("interval", [INTERVAL_DEFAULT])[0]
            outputsize = int(q.get("outputsize", [OUTPUTSIZE_DEFAULT])[0])
            use_confluence = q.get("useConfluence", [USE_EMA50_CONFLUENCE_DEFAULT])[0] == "1"
            # fast=1 → subset pairs agar tidak timeout; all=1 → proses semua + obey rate limit
            fast = q.get("fast", ["1"])[0] == "1"
            process_all = q.get("all", ["0"])[0] == "1"
            usd_boost = float(q.get("usdBoost", [str(USD_DXY_BOOST_DEFAULT)])[0])

            if fast and not process_all:
                # Subset "berbobot" (komponen DXY + beberapa JPY crosses) → cepat & informatif
                pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CAD", "USD/CHF", "AUD/USD", "NZD/USD", "EUR/JPY", "GBP/JPY"]
                obey_rate_limit = False
            else:
                pairs = PAIRS[:]
                obey_rate_limit = True  # jaga rate-limit dgn sleep antar batch

            df_rank, df_pairs, logs = score_pairs_weighted(pairs, interval, outputsize, api_key, use_confluence, obey_rate_limit)

            # DXY boost untuk USD
            dxy_info = "DXY disabled"
            try:
                dxy_close, dxy_ema50, dxy_ema200, usd_boost_sign, usd_boost_txt = get_dxy_signal(interval, outputsize, api_key)
                if usd_boost != 0:
                    df_rank.loc[df_rank["Currency"] == "USD", "Score"] += usd_boost * usd_boost_sign
                dxy_info = f"DXY Close={dxy_close:.4f}, EMA200={dxy_ema200:.4f} | {usd_boost_txt} (x{usd_boost})"
            except Exception as e:
                dxy_info = f"DXY fetch failed: {e}"

            # Normalisasi 0–100
            rank_norm = df_rank.copy()
            if rank_norm["Score"].nunique() > 1:
                mn, mx = rank_norm["Score"].min(), rank_norm["Score"].max()
                rank_norm["Strength%"] = (rank_norm["Score"] - mn) / (mx - mn) * 100.0
            else:
                rank_norm["Strength%"] = 50.0
            rank_norm = rank_norm.sort_values(["Strength%","Score"], ascending=False, ignore_index=True)

            body = {
                "meta": {
                    "pairs_count": len(pairs),
                    "interval": interval,
                    "outputsize": outputsize,
                    "useEMA50Confluence": use_confluence,
                    "mode": "fast" if fast and not process_all else "full"
                },
                "rank_raw": df_rank.to_dict(orient="records"),
                "rank_norm": rank_norm.to_dict(orient="records"),
                "pairs_detail": df_pairs.sort_values("Weight", ascending=False).reset_index(drop=True).to_dict(orient="records"),
                "dxy_info": dxy_info,
                "logs": logs[-200:]  # batasi
            }
            status, headers, payload = build_response(200, body)
            self._write(status, headers, payload)

        except Exception as e:
            status, headers, payload = build_response(500, {"error": str(e)})
            self._write(status, headers, payload)

    def _write(self, status: int, headers: dict, payload: bytes):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

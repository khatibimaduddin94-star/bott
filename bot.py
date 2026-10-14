"""
XAUUSD signal bot.

Every run (scheduled every 5 min):
  1. Pull the latest CLOSED M5 candles (+ H1 for trend context) from Twelve Data
  2. Run 5 strategies on the candles
  3. Score each side (BUY / SELL), pick the single best setup
  4. Send ONE signal with entry, SL, TP and reasons to Telegram
  5. Track the result of the open signal (win / loss) and keep stats

Env vars: TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional: MIN_SCORE (default 0.72), MIN_ATR (default 0.8), RR (default 1.5)
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

TD_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOL = "XAU/USD"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.72"))  # higher = fewer, pickier signals
MIN_ATR = float(os.getenv("MIN_ATR", "0.8"))       # skip dead markets (USD per candle)
RR = float(os.getenv("RR", "1.5"))                 # take-profit = RR x risk
SESSION_UTC = (7, 20)                              # London + New York hours only
COOLDOWN_MIN = 30
MAX_TRADE_HOURS = 8
STATE_FILE = "state.json"

WEIGHTS = {"trend_pullback": 1.0, "breakout": 0.95, "candle_at_level": 0.9,
           "mean_reversion": 0.85, "macd_momentum": 0.8}


# ───────────────────────── data ─────────────────────────
def fetch(interval, minutes, size=300):
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params=dict(symbol=SYMBOL, interval=interval, outputsize=size,
                    timezone="UTC", apikey=TD_KEY),
        timeout=20,
    )
    j = r.json()
    if "values" not in j:
        raise RuntimeError(f"Data error: {j.get('message', j)}")
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()[["open", "high", "low", "close"]].astype(float)
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]  # closed candles only


def add_indicators(df):
    d = df.copy()
    c, h, l = d.close, d.high, d.low
    for n in (20, 50, 200):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atr"] = atr

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macdh"] = macd - macd.ewm(span=9, adjust=False).mean()

    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = ma + 2 * sd, ma - 2 * sd

    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return d.dropna()


def htf_trend(h1):
    h1 = add_indicators(h1)
    last = h1.iloc[-1]
    if last.close > last.ema50 > last.ema200:
        return 1
    if last.close < last.ema50 < last.ema200:
        return -1
    return 0


def pivots(d, n=5, look=150):
    x = d.iloc[-look:]
    highs, lows = [], []
    for i in range(n, len(x) - n):
        w = x.iloc[i - n:i + n + 1]
        if x.high.iloc[i] == w.high.max():
            highs.append(x.high.iloc[i])
        if x.low.iloc[i] == w.low.min():
            lows.append(x.low.iloc[i])
    return highs, lows


# ───────────────────────── strategies ─────────────────────────
def strategies(d):
    out = []
    last, prev = d.iloc[-1], d.iloc[-2]
    atr = last.atr
    rng = last.high - last.low
    body = abs(last.close - last.open)

    def add(name, side, score, reason):
        out.append(dict(name=name, side=side, score=score * WEIGHTS[name], reason=reason))

    # 1) Trend pullback: trade with the trend after price dips to EMA20 and rejects
    if last.ema20 > last.ema50 > last.ema200:
        if (last.low <= last.ema20 + 0.25 * atr and last.close > last.ema20
                and last.close > last.open and last.close > prev.close and 40 <= last.rsi <= 65):
            add("trend_pullback", "BUY", 0.74,
                f"Uptrend (EMA20>50>200). Price pulled back to EMA20 and closed back above it "
                f"with a bullish candle. RSI {last.rsi:.0f} still has room.")
    if last.ema20 < last.ema50 < last.ema200:
        if (last.high >= last.ema20 - 0.25 * atr and last.close < last.ema20
                and last.close < last.open and last.close < prev.close and 35 <= last.rsi <= 60):
            add("trend_pullback", "SELL", 0.74,
                f"Downtrend (EMA20<50<200). Price rallied to EMA20 and got rejected with a "
                f"bearish close. RSI {last.rsi:.0f} still has room.")

    # 2) Breakout: close beyond the last 20-candle range with real momentum
    hh = d.high.iloc[-21:-1].max()
    ll = d.low.iloc[-21:-1].min()
    strong = rng >= 1.2 * atr and rng > 0 and body / rng >= 0.6 and last.adx >= 20
    if strong and last.close > hh and last.close - hh <= atr:
        add("breakout", "BUY", 0.68 + (0.04 if last.adx >= 25 else 0),
            f"Closed above the 20-candle high ({hh:.2f}) with a strong full-body candle. "
            f"ADX {last.adx:.0f} confirms momentum.")
    if strong and last.close < ll and ll - last.close <= atr:
        add("breakout", "SELL", 0.68 + (0.04 if last.adx >= 25 else 0),
            f"Closed below the 20-candle low ({ll:.2f}) with a strong full-body candle. "
            f"ADX {last.adx:.0f} confirms momentum.")

    # 3) Mean reversion: only in ranging markets (low ADX), fade a Bollinger overshoot
    if last.adx < 22:
        if prev.close < prev.bb_lo and last.close > last.bb_lo and last.close > last.open:
            add("mean_reversion", "BUY", 0.66,
                f"Range market (ADX {last.adx:.0f}). Price pushed below the lower Bollinger band "
                f"and closed back inside, RSI {prev.rsi:.0f} before the bounce.")
        if prev.close > prev.bb_up and last.close < last.bb_up and last.close < last.open:
            add("mean_reversion", "SELL", 0.66,
                f"Range market (ADX {last.adx:.0f}). Price pushed above the upper Bollinger band "
                f"and closed back inside, RSI {prev.rsi:.0f} before the drop.")

    # 4) MACD momentum shift, filtered by EMA50 and trend strength
    if prev.macdh <= 0 < last.macdh and last.close > last.ema50 and last.adx >= 18:
        add("macd_momentum", "BUY", 0.62,
            "MACD histogram flipped positive while price holds above EMA50.")
    if prev.macdh >= 0 > last.macdh and last.close < last.ema50 and last.adx >= 18:
        add("macd_momentum", "SELL", 0.62,
            "MACD histogram flipped negative while price holds below EMA50.")

    # 5) Candlestick reversal at a real support / resistance pivot
    highs, lows = pivots(d)
    lower_wick = min(last.open, last.close) - last.low
    upper_wick = last.high - max(last.open, last.close)
    bull_engulf = (prev.close < prev.open and last.close > last.open
                   and last.close >= prev.open and last.open <= prev.close)
    bear_engulf = (prev.close > prev.open and last.close < last.open
                   and last.close <= prev.open and last.open >= prev.close)
    bull_pin = rng > 0 and lower_wick >= 2 * max(body, 1e-9) and lower_wick >= 0.5 * rng \
        and last.close >= last.low + 0.66 * rng
    bear_pin = rng > 0 and upper_wick >= 2 * max(body, 1e-9) and upper_wick >= 0.5 * rng \
        and last.close <= last.high - 0.66 * rng
    sup = next((lv for lv in lows if abs(last.low - lv) <= 0.5 * atr), None)
    res = next((lv for lv in highs if abs(last.high - lv) <= 0.5 * atr), None)
    if sup is not None and (bull_engulf or bull_pin) and last.close > sup:
        add("candle_at_level", "BUY", 0.70,
            f"{'Bullish engulfing' if bull_engulf else 'Bullish pin bar'} at support "
            f"{sup:.2f} (previous swing low).")
    if res is not None and (bear_engulf or bear_pin) and last.close < res:
        add("candle_at_level", "SELL", 0.70,
            f"{'Bearish engulfing' if bear_engulf else 'Bearish pin bar'} at resistance "
            f"{res:.2f} (previous swing high).")
    return out


def evaluate(d, trend):
    found = strategies(d)
    result = {}
    for side, want in (("BUY", 1), ("SELL", -1)):
        s = sorted([x for x in found if x["side"] == side], key=lambda x: -x["score"])
        if not s:
            result[side] = (0.0, [])
            continue
        conf = s[0]["score"] + min(0.15, 0.07 * (len(s) - 1))  # extra agreement = bonus, not required
        if trend == want:
            conf += 0.08
        elif trend == -want:
            conf -= 0.15
        result[side] = (min(conf, 0.95), s)
    side = max(result, key=lambda k: result[k][0])
    other = "SELL" if side == "BUY" else "BUY"
    conf, reasons = result[side]
    if conf < MIN_SCORE or conf - result[other][0] < 0.15:
        return None
    return side, conf, reasons


def trade_plan(d, side):
    last, prev = d.iloc[-1], d.iloc[-2]
    atr, entry = last.atr, last.close
    if side == "BUY":
        risk = entry - (min(last.low, prev.low) - 0.3 * atr)
    else:
        risk = (max(last.high, prev.high) + 0.3 * atr) - entry
    risk = min(max(risk, 1.0 * atr), 2.0 * atr)
    sgn = 1 if side == "BUY" else -1
    return dict(entry=round(entry, 2), sl=round(entry - sgn * risk, 2),
                tp=round(entry + sgn * risk * RR, 2))


# ───────────────────────── state / telegram ─────────────────────────
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"open": None, "last_candle": None, "last_signal": None,
                "wins": 0, "losses": 0}


def save_state(state, original):
    new = json.dumps(state, sort_keys=True, indent=1)
    if new != original or not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            f.write(new)


def send(text):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text}, timeout=20)
    r.raise_for_status()


def winrate_line(state):
    w, l = state["wins"], state["losses"]
    return f"Record: {w}W / {l}L ({100 * w / (w + l):.0f}% win rate)" if w + l else "Record: no closed trades yet"


def check_open(state, d, now):
    o = state.get("open")
    if not o:
        return
    t0 = pd.Timestamp(o["time"])
    outcome = None
    for ts, c in d[d.index > t0].iterrows():
        if o["side"] == "BUY":
            if c.low <= o["sl"]:
                outcome = "LOSS"
            elif c.high >= o["tp"]:
                outcome = "WIN"
        else:
            if c.high >= o["sl"]:
                outcome = "LOSS"
            elif c.low <= o["tp"]:
                outcome = "WIN"
        if outcome:
            break  # if SL and TP are in the same candle, SL is assumed first (conservative)
    if outcome is None and now - t0.to_pydatetime() > timedelta(hours=MAX_TRADE_HOURS):
        outcome = "EXPIRED"
    if not outcome:
        return
    if outcome == "WIN":
        state["wins"] += 1
    elif outcome == "LOSS":
        state["losses"] += 1
    icon = {"WIN": "✅", "LOSS": "❌", "EXPIRED": "⏱"}[outcome]
    send(f"{icon} XAUUSD {o['side']} from {o['entry']} closed: {outcome}\n{winrate_line(state)}")
    state["open"] = None


# ───────────────────────── main ─────────────────────────
def main():
    if os.getenv("TEST_MODE", "").lower() == "true":
        send("✅ XAUUSD bot is connected. Signals will appear here.")
        return

    state = load_state()
    original = json.dumps(state, sort_keys=True, indent=1)
    now = datetime.now(timezone.utc)

    d = add_indicators(fetch("5min", 5))
    last_ts = d.index[-1]
    if now - last_ts.to_pydatetime() > timedelta(minutes=20):
        print("Market closed / stale data, skipping.")
        return

    check_open(state, d, now)
    if state["open"]:
        print("Trade still open, no new signal.")
        save_state(state, original)
        return

    if str(last_ts) == state.get("last_candle"):
        print("Candle already analysed.")
        save_state(state, original)
        return
    state["last_candle"] = str(last_ts)

    last_sig = state.get("last_signal")
    in_cooldown = last_sig and now - datetime.fromisoformat(last_sig) < timedelta(minutes=COOLDOWN_MIN)
    in_session = now.weekday() < 5 and SESSION_UTC[0] <= now.hour < SESSION_UTC[1]
    if in_cooldown or not in_session or d.iloc[-1].atr < MIN_ATR:
        print("Filtered (cooldown / session / low volatility).")
        save_state(state, original)
        return

    trend = htf_trend(fetch("1h", 60))
    decision = evaluate(d, trend)
    if not decision:
        print("No high-quality setup.")
        save_state(state, original)
        return

    side, conf, reasons = decision
    p = trade_plan(d, side)
    trend_txt = {1: "bullish", -1: "bearish", 0: "neutral"}[trend]
    msg = (f"{'🟢' if side == 'BUY' else '🔴'} XAUUSD {side} (M5)\n"
           f"Entry: {p['entry']}\nSL: {p['sl']}\nTP: {p['tp']}  (RR 1:{RR:g})\n"
           f"Confidence score: {conf * 100:.0f}/100\n\nWhy:\n"
           + "\n".join(f"• {r['reason']}" for r in reasons[:3])
           + f"\n• H1 trend: {trend_txt}\n\n{winrate_line(state)}\n"
           "Risk 1% or less per trade. Not financial advice.")
    send(msg)  # if this fails, state is not saved, so the next run retries
    state["open"] = dict(side=side, time=str(last_ts), **p)
    state["last_signal"] = now.isoformat()
    save_state(state, original)


if __name__ == "__main__":
    main()

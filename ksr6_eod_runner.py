#!/usr/bin/env python3
"""
KSR6 EOD Runner
===============
Fetches EOD data for the watchlist, resamples to weekly, scores every name
with KSR6 Engine v3, and writes docs/results.json + docs/index.html
(a self-contained dashboard, ready for GitHub Pages).

Modes:
  python ksr6_eod_runner.py                  # live: fetch from Yahoo Finance
  python ksr6_eod_runner.py --offline DIR    # test: read Investing.com CSVs from DIR
                                             #   (NIFTY500.csv required, others = ticker name)

Live mode requirements: pip install yfinance pandas numpy
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksr6_engine_v3 import (
    enhanced_score_stock, score_timer, parse_investing,
    OUTLAY_MIN, OUTLAY_MAX, RISK_REF,
)

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SYMBOL = "^CRSLDX"  # Nifty 500
DAILY_LOOKBACK = "3y"     # enough daily history for 52+ weekly bars + EMA warmup
OUT_DIR = "docs"          # GitHub Pages serves from /docs
SCORE_FLOOR = 45          # hide SKIPs below this in the table (queues unaffected)


# ---------------------------------------------------------------- data layer
def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> weekly bars (Mon-Sun, labeled by week start, partial week kept)."""
    d = daily.copy()
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d.set_index(pd.DatetimeIndex(d["Date"]))
    w = d.resample("W-SUN", label="left", closed="right").agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Close"])
    w = w.reset_index().rename(columns={"index": "Date"})
    w["Date"] = pd.to_datetime(w["Date"])
    return w.sort_values("Date").reset_index(drop=True)


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.Ticker(symbol).history(period=DAILY_LOOKBACK, interval="1d",
                                    auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    raw = raw.reset_index()
    df = pd.DataFrame({
        "Date": pd.to_datetime(raw["Date"]).dt.tz_localize(None),
        "Open": raw["Open"], "High": raw["High"],
        "Low": raw["Low"], "Close": raw["Close"],
        "Volume": raw["Volume"].fillna(0),
    })
    return resample_weekly(df)


def load_offline(csv_path: str) -> pd.DataFrame:
    """Read an Investing.com weekly CSV directly (already weekly)."""
    return parse_investing(csv_path)


# ---------------------------------------------------------------- scoring
def grade_all(tickers: pd.DataFrame, get_data, idx_df: pd.DataFrame) -> list:
    results = []
    for _, row in tickers.iterrows():
        name = row["name"]
        try:
            df = get_data(row)
            if df is None:
                continue
            r = enhanced_score_stock(name, df, idx_df, idx_df)
            if "Error" in r:
                results.append({"name": name, "sector": row["sector"],
                                "error": r["Error"]})
                continue
            pos = r.get("Position") or {}
            results.append({
                "name": name,
                "sector": row["sector"],
                "price": r["Price"],
                "score": r["Score_Pct"],
                "verdict": r["Verdict"],
                "bars": r["Bars_Color"],
                "zone": r["Entry_Zone"],
                "quality": r["Entry_Quality"],
                "pct10w": r["Pct_10w"],
                "rrs": r["VA_RRS"],
                "rs_signal": r["RS_Signal"],
                "rvol": r["RVol_Pct"],
                "awr": r["AWR_Pct"],
                "resistance": r.get("Resistance"),
                "vs_res": r.get("Price_vs_Res"),
                "tight_stop": r.get("Tight_Stop"),
                "stop_pct": r.get("Tight_Stop_Pct"),
                "vcp": r.get("VCP_Contractions"),
                "shares": pos.get("Shares"),
                "outlay": pos.get("Pos_Value"),
                "risk": pos.get("Risk_Amt"),
                "risk_flag": bool(pos.get("Risk_Flag")),
                "cap_constraint": bool(pos.get("Capital_Constraint")),
                "breakout": r.get("Breakout_Detail"),
            })
        except Exception as e:
            results.append({"name": name, "sector": row["sector"],
                            "error": f"{type(e).__name__}: {e}"})
    return results


# ---------------------------------------------------------------- dashboard
ZONE_ORDER = {"PULLBACK_BUY": 0, "BUY_ZONE": 1, "EXTENDED": 2,
              "CLIMACTIC": 3, "BROKEN": 4}

def render_dashboard(timer: dict, results: list, generated: str) -> str:
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    ok.sort(key=lambda r: (-r["score"], ZONE_ORDER.get(r["zone"], 9)))
    shown = [r for r in ok if r["score"] >= SCORE_FLOOR]
    hidden_count = len(ok) - len(shown)

    pullback_q = [r for r in ok if r["zone"] == "PULLBACK_BUY" and r["bars"] in ("BLUE", "NEUTRAL") and r["score"] >= 45]
    breakout_q = [r for r in ok if r["zone"] == "BUY_ZONE" and r["score"] >= 60]

    timer_up = timer["Timer"] == "UP"
    data = json.dumps({
        "timer": timer, "generated": generated,
        "results": shown, "errors": errs, "hidden": hidden_count, "total": len(ok),
        "pullback": [r["name"] for r in pullback_q],
        "breakout": [r["name"] for r in breakout_q],
        "outlay_band": [OUTLAY_MIN, OUTLAY_MAX], "risk_ref": RISK_REF,
    })

    gate_word = "ENTRIES SANCTIONED" if timer_up else "ENTRIES LOCKED"
    hidden_note = f"{hidden_count} below {SCORE_FLOOR}% hidden of {len(ok)} scanned"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KSR6 — EOD Scanner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink:#10141f; --ink2:#171c2b; --line:#262d40;
    --paper:#e8e4d8; --dim:#8b90a0;
    --amber:#e8a13c; --teal:#3fae8c; --red:#c75450; --blue:#5b8dd9;
    --lock:#3a4156;
  }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--ink); color:var(--paper);
         font-family:'IBM Plex Mono',monospace; font-size:13px; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 20px 60px; }}

  /* ---- Timer Gate (signature) ---- */
  .gate {{ display:flex; align-items:center; gap:18px; padding:22px 20px;
           border-bottom:2px solid {('var(--teal)' if timer_up else 'var(--amber)')};
           background:linear-gradient(180deg, var(--ink2), var(--ink)); }}
  .gate .lamp {{ width:14px; height:14px; border-radius:50%;
                 background:{('var(--teal)' if timer_up else 'var(--amber)')};
                 box-shadow:0 0 18px {('var(--teal)' if timer_up else 'var(--amber)')}; }}
  .gate h1 {{ font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:700;
              letter-spacing:.5px; }}
  .gate .state {{ margin-left:auto; text-align:right; }}
  .gate .state b {{ font-size:17px; color:{('var(--teal)' if timer_up else 'var(--amber)')};
                    letter-spacing:1.5px; }}
  .gate .state span {{ display:block; color:var(--dim); font-size:11px; margin-top:3px; }}

  .meta {{ color:var(--dim); font-size:11px; padding:10px 0 0; }}

  /* ---- queues ---- */
  h2 {{ font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:500;
        letter-spacing:2.5px; text-transform:uppercase; color:var(--dim);
        margin:34px 0 10px; }}
  .queues {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:760px) {{ .queues {{ grid-template-columns:1fr; }} }}
  .queue {{ border:1px solid var(--line); background:var(--ink2); padding:14px 16px; }}
  .queue .tag {{ font-size:10px; letter-spacing:2px; color:var(--dim); }}
  .queue .names {{ font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:500;
                   margin-top:6px; line-height:1.5; }}
  .queue.pb .names {{ color:var(--teal); }}
  .queue.bo .names {{ color:var(--blue); }}
  .queue .empty {{ color:var(--lock); }}

  /* ---- table ---- */
  table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
  th {{ text-align:left; font-weight:500; font-size:10px; letter-spacing:1.5px;
        color:var(--dim); padding:8px 8px; border-bottom:1px solid var(--line);
        cursor:pointer; user-select:none; white-space:nowrap; }}
  th:hover {{ color:var(--paper); }}
  td {{ padding:9px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
  tr:hover td {{ background:var(--ink2); }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  th.num {{ text-align:right; }}
  .name {{ font-family:'Space Grotesk',sans-serif; font-weight:500; font-size:14px; }}
  .sec {{ color:var(--dim); font-size:11px; }}

  .chip {{ display:inline-block; padding:2px 8px; font-size:10px; letter-spacing:1px;
           border:1px solid; border-radius:2px; }}
  .z-PULLBACK_BUY {{ color:var(--teal); border-color:var(--teal); }}
  .z-BUY_ZONE {{ color:var(--blue); border-color:var(--blue); }}
  .z-EXTENDED {{ color:var(--amber); border-color:var(--amber); }}
  .z-CLIMACTIC, .z-BROKEN {{ color:var(--red); border-color:var(--red); }}
  .b-BLUE {{ color:var(--blue); }} .b-BLACK {{ color:var(--dim); }} .b-NEUTRAL {{ color:var(--paper); }}
  .pos {{ color:var(--teal); }} .neg {{ color:var(--red); }}
  .warn {{ color:var(--amber); }}
  .locked td.action {{ color:var(--lock); }}
  td.action {{ font-size:11px; letter-spacing:.5px; }}
  .scorebar {{ display:inline-block; width:46px; height:5px; background:var(--line);
               vertical-align:middle; margin-left:8px; border-radius:2px; overflow:hidden; }}
  .scorebar i {{ display:block; height:100%; background:var(--teal); }}
  .err {{ color:var(--red); font-size:11px; margin-top:18px; }}
  footer {{ margin-top:40px; color:var(--lock); font-size:10px; letter-spacing:1px; }}
</style>
</head>
<body>
<div class="gate">
  <div class="lamp"></div>
  <h1>KSR6 / EOD</h1>
  <div class="state">
    <b>TIMER {timer['Timer']} {timer['Timer_Gap']:+.2f}%</b>
    <span>{gate_word} · 10w/20w EMA · Nifty 500</span>
  </div>
</div>
<div class="wrap">
  <div class="meta">generated {generated} IST · outlay ₹{OUTLAY_MIN:,}–{OUTLAY_MAX:,} · risk ref ₹{RISK_REF:,}</div>

  <h2>Entry queues {('' if timer_up else '· locked until Timer UP')}</h2>
  <div class="queues">
    <div class="queue pb">
      <div class="tag">PULLBACK QUEUE · at the 10w EMA</div>
      <div class="names" id="pbq"></div>
    </div>
    <div class="queue bo">
      <div class="tag">BREAKOUT QUEUE · buy-stops above pivot</div>
      <div class="names" id="boq"></div>
    </div>
  </div>

  <h2>Scans ≥{SCORE_FLOOR}% · click headers to sort <span style="color:var(--lock)">· {hidden_note}</span></h2>
  <table id="tbl">
    <thead><tr>
      <th data-k="name">STOCK</th>
      <th data-k="score" class="num">SCORE</th>
      <th data-k="verdict">VERDICT</th>
      <th data-k="zone">ZONE</th>
      <th data-k="pct10w" class="num">%10W</th>
      <th data-k="bars">BARS</th>
      <th data-k="rrs" class="num">RRS</th>
      <th data-k="rvol" class="num">RVOL</th>
      <th data-k="vs_res" class="num">VS RES</th>
      <th data-k="stop_pct" class="num">STOP%</th>
      <th data-k="risk" class="num">RISK ₹</th>
      <th>ENTRY PLAN</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <div class="err" id="errs"></div>
  <footer>KSR6 ENHANCED ENGINE V3 · NOT INVESTMENT ADVICE · DATA: YAHOO FINANCE EOD</footer>
</div>

<script>
const D = {data};
const timerUp = D.timer.Timer === 'UP';

document.getElementById('pbq').innerHTML =
  D.pullback.length ? D.pullback.join('<br>') : '<span class="empty">none at the EMA</span>';
document.getElementById('boq').innerHTML =
  D.breakout.length ? D.breakout.join('<br>') : '<span class="empty">none near pivot</span>';

function plan(r) {{
  if (r.cap_constraint) return '⚠ 1 share exceeds outlay band';
  let p = '';
  if (r.zone === 'PULLBACK_BUY') p = 'pullback entry · reversal bar at 10w EMA';
  else if (r.zone === 'BUY_ZONE' && r.resistance) p = 'buy-stop ₹' + r.resistance.toLocaleString('en-IN');
  else if (r.zone === 'EXTENDED') p = 'wait for retest';
  else p = '—';
  if (!timerUp && p !== '—') p = '🔒 ' + p;
  return p;
}}

function fmt(v, dec=1) {{ return (v===null||v===undefined) ? '—' : Number(v).toFixed(dec); }}

function row(r) {{
  const riskCell = r.risk == null ? '—'
    : (r.risk_flag ? '<span class="warn">' + Math.round(r.risk).toLocaleString('en-IN') + ' ⚠</span>'
                   : Math.round(r.risk).toLocaleString('en-IN'));
  return `<tr class="${{timerUp ? '' : 'locked'}}">
    <td><span class="name">${{r.name}}</span><br><span class="sec">${{r.sector}} · ₹${{Number(r.price).toLocaleString('en-IN')}}</span></td>
    <td class="num">${{fmt(r.score)}}%<span class="scorebar"><i style="width:${{Math.min(100,r.score)}}%"></i></span></td>
    <td>${{r.verdict}}</td>
    <td><span class="chip z-${{r.zone}}">${{r.zone.replace('_',' ')}}</span></td>
    <td class="num ${{r.pct10w>=0?'pos':'neg'}}">${{fmt(r.pct10w)}}%</td>
    <td class="b-${{r.bars}}">${{r.bars}}</td>
    <td class="num ${{r.rrs>=0?'pos':'neg'}}">${{fmt(r.rrs,2)}}</td>
    <td class="num">${{fmt(r.rvol,0)}}%</td>
    <td class="num">${{r.vs_res==null?'—':fmt(r.vs_res)+'%'}}</td>
    <td class="num">${{r.stop_pct==null?'—':fmt(r.stop_pct)+'%'}}</td>
    <td class="num">${{riskCell}}</td>
    <td class="action">${{plan(r)}}</td>
  </tr>`;
}}

let rows = D.results.slice();
function draw() {{
  document.querySelector('#tbl tbody').innerHTML = rows.map(row).join('');
}}
let sortK = 'score', sortAsc = false;
document.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {{
  const k = th.dataset.k;
  sortAsc = (k === sortK) ? !sortAsc : false;
  sortK = k;
  rows.sort((a,b) => {{
    const x=a[k], y=b[k];
    if (x==null) return 1; if (y==null) return -1;
    return (typeof x==='string' ? x.localeCompare(y) : x-y) * (sortAsc?1:-1);
  }});
  draw();
}});
draw();

if (D.errors.length)
  document.getElementById('errs').textContent =
    'fetch errors: ' + D.errors.map(e => e.name + ' (' + e.error + ')').join(' · ');
</script>
</body>
</html>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="DIR",
                    help="read Investing.com weekly CSVs from DIR instead of Yahoo")
    ap.add_argument("--tickers", default="tickers.csv")
    args = ap.parse_args()

    tickers = pd.read_csv(args.tickers)
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    if args.offline:
        idx_path = os.path.join(args.offline, "NIFTY500.csv")
        idx_df = parse_investing(idx_path)

        def get_data(row):
            p = os.path.join(args.offline, row["name"].replace(" ", "_") + ".csv")
            return load_offline(p) if os.path.exists(p) else None
    else:
        idx_df = fetch_yahoo(INDEX_SYMBOL)

        def get_data(row):
            return fetch_yahoo(row["yahoo"])

    timer = score_timer(idx_df)
    print(f"Timer: {timer['Timer']} ({timer['Timer_Gap']}%)")

    results = grade_all(tickers, get_data, idx_df)
    scanned = [r for r in results if 'error' not in r]
    print(f"Scored {len(scanned)}/{len(results)} names")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"timer": timer, "generated": now, "results": results}, f, indent=1)
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(render_dashboard(timer, results, now))
    print(f"Wrote {OUT_DIR}/results.json and {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()

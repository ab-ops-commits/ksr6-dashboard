"""
KSR6 Weekly Scoring Engine — v3 (Entry Zone + Fixed Outlay)
============================================
6-component scoring (100pts max + bonus) with pre-breakout intelligence.
Components: MA Stack, HH/HL Trend, Market Timer, AWR/ADR, RVol, VA-RRS
NEW in v3:
  - Entry Zone layer: PULLBACK_BUY / BUY_ZONE / EXTENDED / CLIMACTIC / BROKEN (vs 10w EMA)
  - Tight stop: max(last pivot low, 8w swing low) instead of base low
  - Entry Quality grade A-D (zone x tight stop width)
  - Fixed outlay sizing: ₹20,000-30,000 per position (risk amount now varies with stop)
  - Risk ceiling VETO: ACTIONABLE/LOADING downgrade to WAIT when rupee risk > ₹2,500
  - Capital constraint flag for high-priced shares
Capital: ₹5,00,000 | Outlay: ₹20-30K per trade
"""

import pandas as pd
import numpy as np

# === PARAMETERS (Weekly) ===
EMA_SHORT, EMA_MID, EMA_LONG, EMA_TREND = 10, 20, 40, 200
TIMER_FAST, TIMER_SLOW = 10, 20
ADR_PERIOD, RVOL_PERIOD = 10, 20
VA_RRS_PERIOD, VA_RRS_MA_LEN = 26, 10
VCP_PIVOT_LR = 5
BREAKOUT_VOL_MULT = 2.0
ZERO_CROSS_LOOKBACK = 3
SCORE_MAX = 110.0
OUTLAY_MIN, OUTLAY_MAX = 20000, 30000
OUTLAY_TARGET = 25000
RISK_REF = 2500  # reference ceiling for risk flag


# === UTILITY ===
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_atr(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

def find_pivots(highs, lows, lr=VCP_PIVOT_LR):
    n = len(highs)
    ph, pl = [], []
    for i in range(lr, n - lr):
        is_ph = all(highs[i] > highs[i - j] for j in range(1, lr + 1)) and \
                all(highs[i] > highs[i + j] for j in range(1, lr + 1))
        if is_ph:
            ph.append((i, highs[i]))
        is_pl = all(lows[i] < lows[i - j] for j in range(1, lr + 1)) and \
                all(lows[i] < lows[i + j] for j in range(1, lr + 1))
        if is_pl:
            pl.append((i, lows[i]))
    return ph, pl


# === COMPONENT 1: Moving Averages (20 pts max) ===
def score_ma(df):
    c = df['Close']
    ema10 = calc_ema(c, EMA_SHORT).iloc[-1]
    ema20 = calc_ema(c, EMA_MID).iloc[-1]
    ema40 = calc_ema(c, EMA_LONG).iloc[-1]
    ema200 = calc_ema(c, EMA_TREND).iloc[-1] if len(c) >= 200 else np.nan
    price = c.iloc[-1]

    pts = 0
    stack_ok = ema10 > ema20 > ema40
    if stack_ok:
        pts += 8
    elif ema10 > ema20:
        pts += 4

    # 40w EMA rising?
    ema40_series = calc_ema(c, EMA_LONG)
    if len(ema40_series) >= 5 and ema40_series.iloc[-1] > ema40_series.iloc[-5]:
        pts += 4

    # Distance from 10w EMA
    pct_10w = (price - ema10) / ema10 * 100
    if 0 <= pct_10w <= 5:
        pts += 8  # ideal
    elif 5 < pct_10w <= 10:
        pts += 5
    elif -5 <= pct_10w < 0:
        pts += 3
    elif pct_10w > 15:
        pts += 0  # extended

    return {
        'Score_MA': min(pts, 20),
        'EMA10': round(ema10, 2), 'EMA20': round(ema20, 2),
        'EMA40': round(ema40, 2), 'EMA200': round(ema200, 2) if not np.isnan(ema200) else None,
        'MA_Stack': 'YES' if stack_ok else 'NO',
        'Pct_10w': round(pct_10w, 2),
        'Price': round(price, 2)
    }


# === COMPONENT 2: HH/HL Trend Engine (20 pts max) ===
def score_trend(df):
    highs, lows = df['High'].values, df['Low'].values
    ph, pl = find_pivots(highs, lows)

    pts = 0
    bars_color = 'NEUTRAL'
    resistance = None
    support = None

    if len(ph) >= 2 and len(pl) >= 2:
        hh = ph[-1][1] > ph[-2][1]
        hl = pl[-1][1] > pl[-2][1]
        if hh and hl:
            bars_color = 'BLUE'
            pts = 20
        elif not hh and not hl:
            bars_color = 'BLACK'
            pts = 5
        else:
            bars_color = 'NEUTRAL'
            pts = 10

    if ph:
        resistance = max(p[1] for p in ph[-6:]) if len(ph) >= 6 else max(p[1] for p in ph)
    if pl:
        support = pl[-1][1]

    return {
        'Score_Trend': pts,
        'Bars_Color': bars_color,
        'Resistance': round(resistance, 2) if resistance else None,
        'Support': round(support, 2) if support else None,
        'Pivot_Highs': [(i, round(v, 2)) for i, v in ph[-5:]],
        'Pivot_Lows': [(i, round(v, 2)) for i, v in pl[-5:]]
    }


# === COMPONENT 3: Market Timer (15 pts max) ===
def score_timer(idx_df):
    if idx_df is None or len(idx_df) < TIMER_SLOW:
        return {'Score_Market': 0, 'Timer': 'UNKNOWN', 'Timer_Gap': 0}
    c = idx_df['Close']
    fast = calc_ema(c, TIMER_FAST).iloc[-1]
    slow = calc_ema(c, TIMER_SLOW).iloc[-1]
    gap = (fast - slow) / slow * 100
    timer = 'UP' if fast > slow else 'DOWN'
    pts = 15 if timer == 'UP' else 0
    return {'Score_Market': pts, 'Timer': timer, 'Timer_Gap': round(gap, 2)}


# === COMPONENT 4: AWR% / ADR% + RVol (25 pts: 10 ADR + 15 RVol) ===
def score_adr_rvol(df):
    # AWR%
    ranges = ((df['High'] - df['Low']) / df['Close'] * 100).tail(ADR_PERIOD)
    awr = ranges.mean()
    awr_pts = 0
    if 4 <= awr <= 10:
        awr_pts = 10
    elif 3 <= awr < 4 or 10 < awr <= 15:
        awr_pts = 6
    elif awr > 15:
        awr_pts = 2

    # RVol
    vol = df['Volume'].values
    curr_vol = vol[-1]
    avg_vol = np.mean(vol[-RVOL_PERIOD:]) if len(vol) >= RVOL_PERIOD else np.mean(vol)
    rvol = (curr_vol / avg_vol * 100) if avg_vol > 0 else 0

    rvol_pts = 0
    if rvol < 60:
        rvol_pts = 15  # dry-up = pre-breakout
    elif 60 <= rvol < 100:
        rvol_pts = 10
    elif 100 <= rvol < 200:
        rvol_pts = 8
    elif rvol >= 200:
        rvol_pts = 5  # climactic

    # 6-week RVol trend
    rvol_6w = []
    for i in range(-6, 0):
        if abs(i) <= len(vol):
            rv = vol[i] / avg_vol * 100 if avg_vol > 0 else 0
            rvol_6w.append(round(rv, 1))

    # Breakout volume flag
    bo_vol = curr_vol > avg_vol * BREAKOUT_VOL_MULT

    return {
        'Score_ADR': awr_pts, 'Score_RVol': rvol_pts,
        'AWR_Pct': round(awr, 2), 'RVol_Pct': round(rvol, 1),
        'RVol_6w': rvol_6w, 'Breakout_Vol': bo_vol,
        'Curr_Vol': curr_vol, 'Avg_Vol': round(avg_vol, 0)
    }


# === COMPONENT 5/6: VA-RRS (Volatility-Adjusted Real Relative Strength) (20 pts max) ===
def calc_va_rrs(df, idx_df):
    if idx_df is None or len(df) < VA_RRS_PERIOD + VA_RRS_MA_LEN or len(idx_df) < VA_RRS_PERIOD + VA_RRS_MA_LEN:
        return {'VA_RRS': np.nan, 'VA_RRS_MA': np.nan, 'RS_Signal': 'UNKNOWN',
                'RS_Zero_Cross': False, 'RS_Cross_Bar': None}

    # Align dates
    stock_dates = set(df['Date'].dt.date)
    idx_dates = set(idx_df['Date'].dt.date)
    common = sorted(stock_dates & idx_dates)

    if len(common) < VA_RRS_PERIOD + VA_RRS_MA_LEN:
        return {'VA_RRS': np.nan, 'VA_RRS_MA': np.nan, 'RS_Signal': 'UNKNOWN',
                'RS_Zero_Cross': False, 'RS_Cross_Bar': None}

    sdf = df[df['Date'].dt.date.isin(common)].sort_values('Date').reset_index(drop=True)
    idf = idx_df[idx_df['Date'].dt.date.isin(common)].sort_values('Date').reset_index(drop=True)

    # Returns over VA_RRS_PERIOD
    s_ret = sdf['Close'].pct_change(VA_RRS_PERIOD)
    i_ret = idf['Close'].pct_change(VA_RRS_PERIOD)

    # ATR normalization
    s_atr = calc_atr(sdf, 14)
    i_atr = calc_atr(idf, 14)
    s_atr_pct = s_atr / sdf['Close'].values
    i_atr_pct = i_atr / idf['Close'].values

    # VA-RRS = (stock_return / stock_atr%) - (index_return / index_atr%)
    va_rrs = pd.Series(np.where(
        (s_atr_pct > 0) & (i_atr_pct > 0),
        s_ret.values / s_atr_pct - i_ret.values / i_atr_pct,
        np.nan
    ))

    va_rrs_ma = va_rrs.rolling(VA_RRS_MA_LEN).mean()

    curr_rrs = va_rrs.iloc[-1] if not np.isnan(va_rrs.iloc[-1]) else 0
    curr_ma = va_rrs_ma.iloc[-1] if not np.isnan(va_rrs_ma.iloc[-1]) else 0

    # Signal
    if curr_rrs > 0 and curr_rrs > curr_ma:
        signal = 'STRONG_GAINING'
    elif curr_rrs > 0 and curr_rrs <= curr_ma:
        signal = 'STRONG_FADING'
    elif curr_rrs <= 0 and curr_rrs > curr_ma:
        signal = 'WEAK_IMPROVING'
    else:
        signal = 'WEAK_LOSING'

    # Zero-line cross detection
    zero_cross = False
    cross_bar = None
    for i in range(1, min(ZERO_CROSS_LOOKBACK + 1, len(va_rrs))):
        prev_val = va_rrs.iloc[-(i + 1)]
        curr_val = va_rrs.iloc[-i]
        if not np.isnan(prev_val) and not np.isnan(curr_val):
            if prev_val <= 0 < curr_val:
                zero_cross = True
                cross_bar = i
                break

    return {
        'VA_RRS': round(float(curr_rrs), 2),
        'VA_RRS_MA': round(float(curr_ma), 2),
        'RS_Signal': signal,
        'RS_Zero_Cross': zero_cross,
        'RS_Cross_Bar': cross_bar
    }


# === VCP DETECTION ===
def detect_vcp(df):
    highs, lows = df['High'].values, df['Low'].values
    ph, pl = find_pivots(highs, lows)

    if len(ph) < 2 or len(pl) < 2:
        return {'VCP_Stage': 0, 'VCP_Contractions': 0, 'VCP_Depths': [], 'VCP_Tightening': False}

    # Find contraction depths
    depths = []
    for i in range(max(0, len(ph) - 5), len(ph)):
        # Find nearest pivot low after this pivot high
        ph_idx, ph_val = ph[i]
        for j in range(len(pl)):
            if pl[j][0] > ph_idx:
                pl_val = pl[j][1]
                depth = (ph_val - pl_val) / ph_val * 100
                if depth > 0:
                    depths.append(depth)
                break

    c = sum(1 for i in range(1, len(depths)) if depths[i] < depths[i - 1])
    stage = [0, 1, 2, 3, 4, 5][min(c + 1, 5)] if c > 0 else 0

    # Tightness score (ratio of last depth to first)
    tightness = depths[-1] / depths[0] if len(depths) >= 2 and depths[0] > 0 else 1.0

    return {
        'VCP_Stage': stage,
        'VCP_Contractions': c,
        'VCP_Depths': [round(d, 2) for d in depths],
        'VCP_Tightening': c >= 2,
        'VCP_Tightness': round(tightness, 2) if len(depths) >= 2 else None
    }


# === BREAKOUT DETECTION ===
def detect_breakout(df, resistance, bo_vol):
    if resistance is None:
        return {'Is_Breakout': False, 'Breakout_Detail': 'NO_RESISTANCE', 'Price_vs_Res': None}

    c = df['Close'].iloc[-1]
    h = df['High'].iloc[-1]
    lo = df['Low'].iloc[-1]
    price_break = c > resistance
    intra_break = h > resistance and c > (c + lo) / 2

    if price_break and bo_vol:
        detail = 'FULL_BREAKOUT'
        pts = 20
    elif price_break:
        detail = 'PRICE_BREAK_LOW_VOL'
        pts = 8
    elif intra_break and bo_vol:
        detail = 'INTRADAY_BREAK_VOL'
        pts = 12
    else:
        detail = 'NO_BREAKOUT'
        pts = 0

    return {
        'Is_Breakout': pts >= 12,
        'Breakout_Detail': detail,
        'Price_vs_Res': round((c - resistance) / resistance * 100, 2)
    }


# === POSITION SIZING (v3: fixed outlay ₹20-30K) ===
def calc_position(price, stop, capital=None, risk_pct=None):
    """Fixed-outlay sizing: target ₹25K, accept ₹20-30K. Risk varies with stop width."""
    if stop is None or price <= stop:
        return None
    if price > OUTLAY_MAX:
        return {'Capital_Constraint': True, 'Note': f'1 share (₹{price:,.0f}) exceeds ₹{OUTLAY_MAX:,} outlay band'}
    shares = max(1, round(OUTLAY_TARGET / price))
    value = shares * price
    # nudge into band if rounding pushed us out
    if value > OUTLAY_MAX and shares > 1:
        shares -= 1
    elif value < OUTLAY_MIN:
        if (shares + 1) * price <= OUTLAY_MAX:
            shares += 1
    value = shares * price
    rps = price - stop
    risk = shares * rps
    return {
        'Stop': round(stop, 2),
        'Shares': shares,
        'Pos_Value': round(value, 2),
        'Risk_Amt': round(risk, 2),
        'Stop_Width_Pct': round(rps / price * 100, 2),
        'Risk_Flag': risk > RISK_REF,
        'Capital_Constraint': False
    }


# === ENTRY ZONE LAYER (v3) ===
def classify_entry_zone(pct_10w, price, pivot_lows, df):
    """Zone vs 10w EMA + tight stop + entry quality grade."""
    if -3 <= pct_10w <= 3:
        zone = 'PULLBACK_BUY'
    elif 3 < pct_10w <= 8:
        zone = 'BUY_ZONE'
    elif 8 < pct_10w <= 15:
        zone = 'EXTENDED'
    elif pct_10w > 15:
        zone = 'CLIMACTIC'
    else:
        zone = 'BROKEN'

    last_pl = pivot_lows[-1][1] if pivot_lows else None
    swing8 = float(df.tail(8)['Low'].min())
    candidates = [s for s in [last_pl, swing8] if s is not None and s < price]
    tight_stop = max(candidates) if candidates else None

    out = {'Entry_Zone': zone, 'Tight_Stop': None, 'Tight_Stop_Pct': None, 'Entry_Quality': 'D (no valid stop)'}
    if tight_stop:
        w = (price - tight_stop) / price * 100
        out['Tight_Stop'] = round(tight_stop, 2)
        out['Tight_Stop_Pct'] = round(w, 2)
        if zone == 'PULLBACK_BUY' and w <= 10:
            out['Entry_Quality'] = 'A (tight pullback)'
        elif zone in ('PULLBACK_BUY', 'BUY_ZONE') and w <= 15:
            out['Entry_Quality'] = 'B (workable)'
        elif zone == 'EXTENDED':
            out['Entry_Quality'] = 'C (wait for retest)'
        else:
            out['Entry_Quality'] = 'D (no entry)'
    return out


# === MASTER SCORING FUNCTION ===
def enhanced_score_stock(ticker, df, idx_df, nifty_df=None, capital=500000, risk_pct=0.5):
    if len(df) < 40:
        return {'Ticker': ticker, 'Error': f'Only {len(df)} bars (need 40+)'}

    ma = score_ma(df)
    tr = score_trend(df)
    tm = score_timer(idx_df)
    ar = score_adr_rvol(df)

    # VA-RRS
    rrs = calc_va_rrs(df, idx_df)
    sig = rrs['RS_Signal']
    rrs_pts = {'STRONG_GAINING': 20, 'STRONG_FADING': 12, 'WEAK_IMPROVING': 6, 'WEAK_LOSING': 0, 'UNKNOWN': 0}.get(sig, 0)
    if rrs.get('RS_Zero_Cross'):
        rrs_pts += 3

    # VCP & Breakout
    vcp = detect_vcp(df)
    bo = detect_breakout(df, tr['Resistance'], ar['Breakout_Vol'])

    # Total score
    raw = ma['Score_MA'] + tr['Score_Trend'] + tm['Score_Market'] + ar['Score_ADR'] + ar['Score_RVol'] + rrs_pts
    pct = round(raw / SCORE_MAX * 100, 1)

    # Entry zone layer (v3)
    zone = classify_entry_zone(ma['Pct_10w'], ma['Price'], tr['Pivot_Lows'], df)

    # Position sizing off TIGHT stop (v3), fall back to structural support
    stop_for_sizing = zone['Tight_Stop'] if zone['Tight_Stop'] else tr['Support']
    pos = calc_position(ma['Price'], stop_for_sizing)

    # Verdict
    stop_width = zone['Tight_Stop_Pct'] if zone['Tight_Stop_Pct'] else 99
    if pct >= 75 and stop_width <= 15 and tr['Bars_Color'] in ['BLUE', 'NEUTRAL']:
        verdict = 'ACTIONABLE'
    elif pct >= 65 and vcp['VCP_Contractions'] >= 1:
        verdict = 'LOADING'
    elif pct >= 60:
        verdict = 'WATCHLIST'
    elif pct >= 45:
        verdict = 'EARLY'
    else:
        verdict = 'SKIP'

    # Override: wide stop
    if stop_width > 20 and verdict == 'ACTIONABLE':
        verdict = 'WAIT (Stop Wide)'

    # Override (v3): risk veto — fixed outlay means wide stops blow past risk ceiling
    if pos and not pos.get('Capital_Constraint') and pos.get('Risk_Flag'):
        if verdict in ('ACTIONABLE', 'LOADING'):
            verdict = 'WAIT (Risk > Ceiling)'

    return {
        'Ticker': ticker,
        'Score_Pct': pct,
        'Score_Raw': raw,
        'Verdict': verdict,
        **ma, **tr, **tm, **ar,
        'RRS_Pts': rrs_pts,
        **rrs,
        **vcp, **bo,
        **zone,
        'Position': pos
    }


# === REPORT PRINTER ===
def print_enhanced_report(r):
    if 'Error' in r:
        print(f"  ⚠ {r['Ticker']}: {r['Error']}")
        return

    t = r['Ticker']
    v = r['Verdict']
    s = r['Score_Pct']
    emoji = {'ACTIONABLE': '🟢', 'LOADING': '🔄', 'WATCHLIST': '👁', 'EARLY': '🌱', 'SKIP': '⛔',
             'WAIT (Stop Wide)': '⏳', 'WAIT (Risk > Ceiling)': '⏳'}.get(v, '❓')

    print(f"\n  {emoji} {t} — {v} | Score: {s}% ({r['Score_Raw']}/{SCORE_MAX:.0f})")
    print(f"  ├─ Price: ₹{r['Price']:,.2f} | %10w: {r['Pct_10w']}%")
    print(f"  ├─ MA Stack: {r['MA_Stack']} | EMAs: {r['EMA10']}/{r['EMA20']}/{r['EMA40']}")
    print(f"  ├─ Bars: {r['Bars_Color']} | Resistance: {r.get('Resistance', 'N/A')} | Support: {r.get('Support', 'N/A')}")
    print(f"  ├─ Timer: {r['Timer']} ({r['Timer_Gap']}%) | AWR: {r['AWR_Pct']}%")
    print(f"  ├─ RVol: {r['RVol_Pct']}% | 6w: {r.get('RVol_6w', [])}")
    print(f"  ├─ VA-RRS: {r['VA_RRS']} (MA: {r['VA_RRS_MA']}) | Signal: {r['RS_Signal']}")
    if r.get('RS_Zero_Cross'):
        print(f"  │  └─ ⚡ Zero-line cross {r['RS_Cross_Bar']} bar(s) ago")
    print(f"  ├─ VCP: {r['VCP_Contractions']} contractions | Depths: {r['VCP_Depths']}")
    if r.get('VCP_Tightness') is not None:
        print(f"  │  └─ Tightness: {r['VCP_Tightness']}")
    print(f"  ├─ Breakout: {r['Breakout_Detail']} | vs Res: {r.get('Price_vs_Res', 'N/A')}%")

    print(f"  ├─ ZONE: {r['Entry_Zone']} | Quality: {r['Entry_Quality']} | Tight Stop: " +
          (f"₹{r['Tight_Stop']:,.2f} ({r['Tight_Stop_Pct']}%)" if r.get('Tight_Stop') else "N/A"))

    pos = r.get('Position')
    if pos and pos.get('Capital_Constraint'):
        print(f"  └─ SIZING: ⚠ {pos['Note']}")
    elif pos:
        flag = " ⚠ RISK > ₹2,500 REF" if pos.get('Risk_Flag') else ""
        print(f"  └─ SIZING: {pos['Shares']} shares @ ₹{r['Price']:,.2f} | Stop: ₹{pos['Stop']:,.2f} ({pos['Stop_Width_Pct']}%) | Outlay: ₹{pos['Pos_Value']:,.0f} | Risk: ₹{pos['Risk_Amt']:,.0f}{flag}")
    else:
        print(f"  └─ SIZING: N/A (invalid stop)")


# === INVESTING.COM CSV PARSER ===
def parse_investing(filepath):
    """Parse Investing.com weekly OHLCV CSV format."""
    df = pd.read_csv(filepath)
    dc = pd.DataFrame()
    dc['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    dc['Open'] = pd.to_numeric(df['Open'].astype(str).str.replace(',', ''), errors='coerce')
    dc['High'] = pd.to_numeric(df['High'].astype(str).str.replace(',', ''), errors='coerce')
    dc['Low'] = pd.to_numeric(df['Low'].astype(str).str.replace(',', ''), errors='coerce')
    dc['Close'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
    v = df['Vol.'].astype(str).str.strip().str.replace('"', '')
    dc['Volume'] = v.apply(lambda x:
        float(x.replace('B', '')) * 1e9 if 'B' in x else
        (float(x.replace('M', '')) * 1e6 if 'M' in x else
         (float(x.replace('K', '')) * 1e3 if 'K' in x else
          (float(x.replace(',', '')) if x not in ['-', ''] else 0))))
    dc = dc.sort_values('Date').reset_index(drop=True)
    return dc

# KSR6 EOD Dashboard

Automated end-of-day scanner for the KSR6 weekly pre-breakout system.
Every trading day at 18:45 IST it fetches EOD data, resamples to weekly,
scores every watchlist name with KSR6 Engine v3, and publishes a dashboard.

## How it works

```
GitHub Actions (cron, 18:45 IST Mon-Fri)
   └─ ksr6_eod_runner.py
        ├─ fetch daily OHLCV from Yahoo Finance (.NS tickers, ^CRSLDX index)
        ├─ resample daily → weekly bars (Mon-Sun)
        ├─ KSR6 Engine v3: score, zone, tight stop, fixed-outlay sizing, risk veto
        └─ write docs/index.html + docs/results.json
   └─ commit → GitHub Pages serves the dashboard
```

## One-time setup (~10 minutes)

1. **Create a GitHub repo** (private works fine) and push these files:
   ```
   ksr6_engine_v3.py
   ksr6_eod_runner.py
   tickers.csv
   .github/workflows/ksr6-eod.yml
   ```

2. **Enable workflow write access**: repo → Settings → Actions → General →
   Workflow permissions → "Read and write permissions" → Save.

3. **Enable GitHub Pages**: repo → Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, folder `/docs` → Save.
   (Note: Pages on a *private* repo needs a paid GitHub plan. On a free plan
   either make the repo public, or skip Pages and download
   `docs/index.html` from the repo after each run — it's fully self-contained.)

4. **Test it**: repo → Actions → "KSR6 EOD Scan" → Run workflow.
   After ~2 minutes the dashboard is at
   `https://<username>.github.io/<repo>/`

That's it. It runs itself every trading evening.

## Editing the watchlist

Edit `tickers.csv` (name, Yahoo symbol with `.NS` suffix, sector) and push.
Next run picks it up. Find Yahoo symbols by searching the stock on
finance.yahoo.com — NSE listings end in `.NS`.

## Local / offline use

```bash
pip install yfinance pandas numpy
python ksr6_eod_runner.py                      # live fetch
python ksr6_eod_runner.py --offline test_data  # from Investing.com CSVs
```

## Notes & caveats

- **Data source**: Yahoo Finance is free but occasionally has gaps or short
  delays on NSE EOD data. Volumes for some mid-caps can differ slightly from
  NSE official. If a ticker errors repeatedly, check its Yahoo symbol.
  The dashboard footer lists any per-ticker fetch errors.
- **Current week bar is partial** until Friday's close — scores shift slightly
  through the week, exactly like scanning mid-week manually.
- **Timer gate**: when DOWN, entry plans render locked (🔒). The engine's
  scoring and the risk veto are identical to the chat-session workflow.
- This is a scanning aid, not investment advice.

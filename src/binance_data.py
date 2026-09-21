"""
Binance data layer: USDT-margined perpetual futures.

What has FULL history (usable for backtesting):
  * perp daily klines  -- OHLCV + quote volume + trade count + taker-buy volume
  * spot daily klines  -- for the perp/spot basis
  * funding rate       -- 8h settlements, aggregated to daily

What Binance only serves for the LAST 30 DAYS (NOT backtestable):
  * openInterestHist, topLongShortAccountRatio, globalLongShortAccountRatio,
    takerlongshortRatio
  These are handled on a second track: `collect_short_history()` appends today's
  window to a local store so that history accumulates going forward.

Everything is cached as parquet under data/binance/.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

import config

FAPI = 'https://fapi.binance.com'
SAPI = 'https://api.binance.com'

KLINE_COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
              'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
NUMERIC_KLINE = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                 'trades', 'taker_buy_base', 'taker_buy_quote']


class BinanceError(RuntimeError):
    pass


def _request(url, params=None, retries=4, pause=0.15):
    """GET with backoff. Binance answers 429/418 when the weight budget is spent."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CryptoQuantPipeline/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                used = resp.headers.get('X-MBX-USED-WEIGHT-1M')
                if used and int(used) > 2000:      # soft limit is 2400/min
                    print(f"    weight {used}/2400, easing off")
                    time.sleep(10)
                time.sleep(pause)
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (418, 429):
                wait = 30 * attempt
                print(f"    rate limited ({exc.code}); sleeping {wait}s")
                time.sleep(wait)
                continue
            if exc.code == 400:
                raise BinanceError(f"400 from {url[:120]}: {exc.read()[:200]}") from exc
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)

    raise BinanceError(f"gave up on {url[:120]}")


def _ms(date):
    return int(pd.Timestamp(date).tz_localize('UTC').timestamp() * 1000)


def _day_index(series_ms):
    return pd.to_datetime(series_ms, unit='ms', utc=True).dt.tz_localize(None).dt.normalize()


# --- universe --------------------------------------------------------------

def top_perpetuals(n=50, quote='USDT', exclude_prefixes=('1000000',)):
    """
    The n most-traded USDT-margined PERPETUAL contracts, by 24h quote volume.

    Returns a DataFrame [symbol, quote_volume, onboard_date] sorted by volume.
    """
    info = _request(f'{FAPI}/fapi/v1/exchangeInfo')
    perps = {
        s['symbol']: s
        for s in info['symbols']
        if s.get('contractType') == 'PERPETUAL'
        and s.get('status') == 'TRADING'
        and s.get('quoteAsset') == quote
        and not s['symbol'].startswith(exclude_prefixes)
    }

    tickers = _request(f'{FAPI}/fapi/v1/ticker/24hr')
    rows = [
        {'symbol': t['symbol'],
         'quote_volume': float(t['quoteVolume']),
         'onboard_date': pd.to_datetime(perps[t['symbol']].get('onboardDate', 0),
                                        unit='ms').normalize()}
        for t in tickers if t['symbol'] in perps
    ]

    df = pd.DataFrame(rows).sort_values('quote_volume', ascending=False)
    return df.head(n).reset_index(drop=True)


# --- klines ----------------------------------------------------------------

def _fetch_klines(base, path, symbol, start_ms, end_ms, interval='1d', limit=1500):
    out = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _request(f'{base}{path}', {'symbol': symbol, 'interval': interval,
                                           'startTime': cursor, 'limit': limit})
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < limit:
            break
        cursor = last_open + 86_400_000

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out, columns=KLINE_COLS).drop(columns=['ignore', 'close_time'])
    df['date'] = _day_index(df['open_time'])
    df[NUMERIC_KLINE] = df[NUMERIC_KLINE].astype(float)
    return (df.drop(columns=['open_time'])
              .drop_duplicates('date', keep='last')
              .set_index('date')
              .sort_index())


def fetch_perp_klines(symbol, start, end):
    # Futures allows limit=1500 per request.
    return _fetch_klines(FAPI, '/fapi/v1/klines', symbol, _ms(start), _ms(end), limit=1500)


def fetch_spot_klines(symbol, start, end):
    """
    Spot pair for the basis. Not every perp has one -- returns empty if absent.

    NOTE the spot endpoint caps limit at 1000 (futures allows 1500); passing 1500
    silently truncates the pagination and leaves holes in the series.
    """
    try:
        return _fetch_klines(SAPI, '/api/v3/klines', symbol, _ms(start), _ms(end), limit=1000)
    except (BinanceError, urllib.error.HTTPError):
        return pd.DataFrame()


# --- funding ---------------------------------------------------------------

def fetch_funding(symbol, start, end):
    """
    8h funding settlements aggregated to a UTC day.

    funding_daily = sum of the day's settlements = what a 1x long actually pays
    (positive) or receives (negative) over that day.
    """
    out = []
    cursor = _ms(start)
    end_ms = _ms(end)

    while cursor < end_ms:
        batch = _request(f'{FAPI}/fapi/v1/fundingRate',
                         {'symbol': symbol, 'startTime': cursor, 'limit': 1000})
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1]['fundingTime'] + 1

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df['date'] = _day_index(df['fundingTime'])
    df['fundingRate'] = df['fundingRate'].astype(float)

    daily = df.groupby('date').agg(
        funding_daily=('fundingRate', 'sum'),
        funding_mean=('fundingRate', 'mean'),
        funding_n=('fundingRate', 'size'),
    )
    return daily.sort_index()


# --- assembly --------------------------------------------------------------

def build_symbol_frame(symbol, start, end, with_spot=True, drop_incomplete=True):
    """
    Perp klines + funding (+ spot close for the basis) on one daily index.

    drop_incomplete removes the current UTC day, whose bar is still forming --
    its OHLC is partial and its funding has not fully settled, so keeping it
    poisons every factor computed tonight.
    """
    perp = fetch_perp_klines(symbol, start, end)
    if perp.empty:
        return pd.DataFrame()

    if drop_incomplete:
        today_utc = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
        perp = perp[perp.index < today_utc]
        if perp.empty:
            return pd.DataFrame()

    perp = perp.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume', 'quote_volume': 'QuoteVolume', 'trades': 'Trades',
        'taker_buy_base': 'TakerBuyBase', 'taker_buy_quote': 'TakerBuyQuote',
    })

    funding = fetch_funding(symbol, start, end)
    if not funding.empty:
        perp = perp.join(funding, how='left')
    else:
        perp[['funding_daily', 'funding_mean', 'funding_n']] = 0.0, 0.0, 0

    if with_spot:
        spot = fetch_spot_klines(symbol, start, end)
        if not spot.empty:
            perp['SpotClose'] = spot['close']
            perp['SpotVolume'] = spot['volume']
        else:
            perp['SpotClose'] = pd.NA
            perp['SpotVolume'] = pd.NA

    # Funding only settles every 8h; a missing day means no settlement, i.e. zero.
    perp[['funding_daily', 'funding_mean']] = perp[['funding_daily', 'funding_mean']].fillna(0.0)
    perp['funding_n'] = perp['funding_n'].fillna(0).astype(int)
    perp['Symbol'] = symbol
    return perp


def download_universe(symbols=None, n=None, start=None, end=None,
                      cache_dir=None, with_spot=True, verbose=True,
                      drop_incomplete=True):
    """
    Downloads every symbol and caches one parquet per symbol.

    Written atomically and only after validation, so a rate-limit mid-run can
    never leave a truncated cache behind.
    """
    n = n or config.UNIVERSE_SIZE
    start = start or config.START_DATE
    end = end or pd.Timestamp.utcnow().tz_localize(None).normalize()
    cache_dir = str(cache_dir or config.BINANCE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    if symbols is None:
        universe = top_perpetuals(n)
        symbols = universe['symbol'].tolist()
        universe.to_csv(os.path.join(cache_dir, '_universe.csv'), index=False)
        if verbose:
            print(f"Universe: top {len(symbols)} USDT perps by 24h volume")

    frames, failures = {}, []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = build_symbol_frame(symbol, start, end, with_spot=with_spot,
                                    drop_incomplete=drop_incomplete)
            if df.empty or len(df) < config.MIN_HISTORY_DAYS:
                failures.append(f"{symbol}: only {len(df)} daily bars "
                                f"(need {config.MIN_HISTORY_DAYS})")
                continue
            path = os.path.join(cache_dir, f'{symbol}.parquet')
            tmp = f'{path}.tmp'
            df.to_parquet(tmp)
            os.replace(tmp, path)
            frames[symbol] = df
            if verbose:
                print(f"  [{i:>2}/{len(symbols)}] {symbol:<14} {len(df):>5} bars  "
                      f"{df.index.min().date()} -> {df.index.max().date()}")
        except Exception as exc:
            failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  [{i:>2}/{len(symbols)}] {symbol:<14} FAILED: {exc}")

    if verbose and failures:
        print(f"\n{len(failures)} symbol(s) skipped:")
        for f in failures:
            print(f"  - {f}")
    if not frames:
        raise BinanceError("No symbol downloaded successfully; cache untouched.")

    return frames


def load_universe(cache_dir=None, min_bars=None):
    """Loads every cached parquet. Returns {symbol: DataFrame}."""
    cache_dir = str(cache_dir or config.BINANCE_DIR)
    min_bars = min_bars or config.MIN_HISTORY_DAYS
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"No Binance cache at {cache_dir}. Run `python main.py download` first.")

    out = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith('.parquet'):
            continue
        df = pd.read_parquet(os.path.join(cache_dir, fname))
        if len(df) >= min_bars:
            out[fname[:-8]] = df.sort_index()

    if not out:
        raise FileNotFoundError(f"{cache_dir} holds no symbol with >= {min_bars} bars.")
    return out


# --- second track: 30-day-only endpoints -----------------------------------

SHORT_HISTORY_ENDPOINTS = {
    'open_interest':       ('/futures/data/openInterestHist', 'sumOpenInterest'),
    'top_account_ratio':   ('/futures/data/topLongShortAccountRatio', 'longShortRatio'),
    'top_position_ratio':  ('/futures/data/topLongShortPositionRatio', 'longShortRatio'),
    'global_account_ratio':('/futures/data/globalLongShortAccountRatio', 'longShortRatio'),
    'taker_ls_ratio':      ('/futures/data/takerlongshortRatio', 'buySellRatio'),
}


def collect_short_history(symbols=None, store_path=None, period='1d', verbose=True):
    """
    Second track. Binance serves these only for the last 30 days, so they cannot
    be backtested today -- but appending each night builds a usable history.

    Merges today's 30-day window into a long-format parquet store, de-duplicated
    on [date, symbol, metric]. Safe to run repeatedly.
    """
    store_path = str(store_path or config.SHORT_HISTORY_PATH)
    if symbols is None:
        symbols = sorted(load_universe().keys())

    rows = []
    for symbol in symbols:
        for metric, (path, field) in SHORT_HISTORY_ENDPOINTS.items():
            try:
                data = _request(f'{FAPI}{path}',
                                {'symbol': symbol, 'period': period, 'limit': 500})
            except Exception as exc:
                if verbose:
                    print(f"  {symbol} {metric}: {type(exc).__name__}")
                continue
            for rec in data or []:
                rows.append({
                    'date': pd.to_datetime(int(rec['timestamp']), unit='ms').normalize(),
                    'symbol': symbol,
                    'metric': metric,
                    'value': float(rec[field]),
                })

    if not rows:
        print("collect_short_history: nothing returned.")
        return pd.DataFrame()

    fresh = pd.DataFrame(rows)
    if os.path.exists(store_path):
        fresh = pd.concat([pd.read_parquet(store_path), fresh], ignore_index=True)

    fresh = (fresh.drop_duplicates(['date', 'symbol', 'metric'], keep='last')
                  .sort_values(['date', 'symbol', 'metric'])
                  .reset_index(drop=True))

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    tmp = f'{store_path}.tmp'
    fresh.to_parquet(tmp)
    os.replace(tmp, store_path)

    span = f"{fresh['date'].min().date()} -> {fresh['date'].max().date()}"
    if verbose:
        print(f"Short-history store: {len(fresh):,} rows, {fresh['symbol'].nunique()} symbols, "
              f"{span} -> {store_path}")
    return fresh

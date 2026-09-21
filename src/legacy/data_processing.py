"""
Price data: download, cache, load, and sanity-check.

Operational contract for the nightly job: a failed or degraded download must
NEVER overwrite a good cache. Yahoo rate-limits (HTTP 429) regularly, and
yfinance signals that by returning an empty or all-NaN frame rather than
raising -- so every download is validated before anything touches disk.
"""
import os
import time
import pandas as pd

import config

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


class DataDownloadError(RuntimeError):
    """Raised when one or more tickers could not be refreshed. Cache is untouched."""


def _validate(data, ticker, expected_index, min_coverage=0.8):
    """Returns an error string if the frame is unusable, else None."""
    if data is None or len(data) == 0:
        return f"{ticker}: empty response (rate-limited or delisted)"

    missing_cols = [c for c in OHLCV if c not in data.columns]
    if missing_cols:
        return f"{ticker}: response is missing columns {missing_cols}"

    close = data['Close']
    if close.isna().all():
        return f"{ticker}: every Close is NaN"

    coverage = close.notna().sum() / max(len(expected_index), 1)
    if coverage < min_coverage:
        return (f"{ticker}: only {coverage:.0%} of the requested calendar has a price "
                f"(need {min_coverage:.0%})")
    return None


def _atomic_to_csv(df, path):
    """Write via a temp file + rename so an interrupted run cannot truncate the cache."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp)
    os.replace(tmp, path)


def download_crypto_data(tickers, start_date, end_date, cache_dir=config.DATA_DIR,
                         retries=3, backoff=10, min_coverage=0.8):
    """
    Downloads daily OHLCV for each ticker, aligns to a unified calendar, and
    caches it.

    Every ticker is validated before being written. If any ticker fails after
    `retries` attempts, NOTHING is written for it and DataDownloadError is raised
    listing the failures -- the existing cache stays intact.
    """
    import yfinance as yf

    unified_index = pd.date_range(start=start_date, end=end_date, freq='D')
    cleaned_data = {}
    failures = []

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for ticker in tickers:
        coin = ticker.split('-')[0]
        error = None

        for attempt in range(1, retries + 1):
            print(f"Downloading {ticker} (attempt {attempt}/{retries})...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date,
                                   progress=False, auto_adjust=True)
            except Exception as exc:
                error = f"{ticker}: {type(exc).__name__}: {exc}"
                data = None
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                error = _validate(data, ticker, unified_index, min_coverage)

            if error is None:
                break
            print(f"  {error}")
            if attempt < retries:
                wait = backoff * attempt
                print(f"  retrying in {wait}s")
                time.sleep(wait)

        if error is not None:
            failures.append(error)
            continue

        # Align to the unified daily calendar. ffill covers genuine gaps; bfill
        # only ever fills leading rows before the asset's first print.
        data_filled = data[OHLCV].reindex(unified_index).ffill().bfill()
        cleaned_data[coin] = data_filled

        if cache_dir:
            cache_path = os.path.join(cache_dir, f"{coin}_daily.csv")
            _atomic_to_csv(data_filled, cache_path)
            print(f"  cached -> {cache_path}")

    if failures:
        raise DataDownloadError(
            "Refresh failed for "
            f"{len(failures)}/{len(tickers)} ticker(s); the cache was NOT modified "
            "for them:\n  - " + "\n  - ".join(failures) +
            "\n\nYahoo rate-limiting (HTTP 429) is the usual cause. Retry later, or "
            "run without --refresh to use the existing cache."
        )

    return cleaned_data


def load_cached_data(tickers, cache_dir=config.DATA_DIR):
    """
    Reads the cached daily CSVs. Returns (cleaned_data, missing_tickers) so the
    caller can decide whether to fall back to a download.
    """
    cleaned_data = {}
    missing = []

    for ticker in tickers:
        coin = ticker.split('-')[0]
        cache_path = os.path.join(cache_dir, f"{coin}_daily.csv")
        if not os.path.exists(cache_path):
            missing.append(ticker)
            continue
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = None
        if df['Close'].isna().all():
            missing.append(ticker)
            print(f"  {coin}: cached file has no usable prices; treating as missing")
            continue
        cleaned_data[coin] = df.sort_index()

    return cleaned_data, missing


def get_crypto_data(tickers=config.TICKERS, start_date=config.START_DATE,
                    end_date=config.END_DATE, cache_dir=config.DATA_DIR,
                    refresh=False, allow_stale=False):
    """
    Single entry point.

    refresh=False  serve the cache (download only what is missing)
    refresh=True   re-download everything; on failure raise DataDownloadError
                   unless allow_stale=True, in which case fall back to the cache
                   with a loud warning.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()

    if not refresh:
        cleaned_data, missing = load_cached_data(tickers, cache_dir)
        if not missing:
            spans = [(df.index.min().date(), df.index.max().date())
                     for df in cleaned_data.values()]
            print(f"Loaded {len(cleaned_data)} cached assets from {cache_dir} "
                  f"({spans[0][0]} -> {spans[0][1]})")
            return cleaned_data
        print(f"Cache incomplete (missing {', '.join(missing)}); downloading instead.")

    try:
        return download_crypto_data(tickers, start_date, end_date, cache_dir)
    except DataDownloadError as exc:
        if not allow_stale:
            raise
        print(f"\nDOWNLOAD FAILED, falling back to the cache:\n{exc}\n")
        cleaned_data, missing = load_cached_data(tickers, cache_dir)
        if missing:
            raise DataDownloadError(
                f"Download failed AND the cache is missing {', '.join(missing)}."
            ) from exc
        print("WARNING: running on CACHED prices. Check the freshness warnings below.")
        return cleaned_data


def check_freshness(cleaned_data, max_stale_days=2):
    """
    Nightly-job guard. The daily calendar is forward-filled, so an exchange
    outage or a not-yet-closed day shows up as the last bar being a byte-for-byte
    copy of the one before it. That is a stale price, not a flat market, and it
    silently poisons every factor. Returns a list of warning strings.
    """
    warnings = []
    today = pd.Timestamp.today().normalize()

    for coin, df in cleaned_data.items():
        last = df.index.max()
        age = (today - last).days
        if age > max_stale_days:
            warnings.append(f"{coin}: last bar {last.date()} is {age} days old")

        tail = df[OHLCV].tail(3)
        dup = int(tail.duplicated(keep='first').sum())
        if dup:
            warnings.append(f"{coin}: last {dup + 1} bars are identical "
                            f"(forward-filled / incomplete day through {last.date()})")

    return warnings

"""
Factor library: 305 price/volume/microstructure/derivatives factors.

Every factor is computed from data observable AT or BEFORE the bar it is
attached to. There is no bfill anywhere; the rolling warm-up period is dropped.

Families (single-asset: 241)
    A  return / momentum           35
    B  volatility                  30
    C  range & candle structure    27
    D  trend / moving average      38
    E  oscillators / reversion     22
    F  volume & liquidity          37
    G  taker flow (microstructure) 16
    H  derivatives (funding/basis) 34
       calendar (day-of-week)       2
Cross-sectional & market: 64        (computed across assets, see build_panel)

NOTE: walk-forward showed the 305-feature set DILUTES the selection head's
RankIC versus the previous 158-feature set at default LGB_PARAMS (the new
features improve MSE -- the market-level component -- at the expense of
cross-sectional ordering). Re-run `main.py backtest` before trusting any
strategy conclusion, and consider per-head feature selection.
"""
import numpy as np
import pandas as pd

import config

EPS = 1e-12


# --- helpers ---------------------------------------------------------------

def _z(series, window):
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / (sd + EPS)


def _slope(series, window):
    """OLS slope over a rolling window, normalised by the series level."""
    idx = np.arange(window)
    idx = idx - idx.mean()
    denom = (idx ** 2).sum()
    return (series.rolling(window)
                  .apply(lambda w: np.dot(w - w.mean(), idx) / denom, raw=True)
            / (series.abs().rolling(window).mean() + EPS))


def _rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + EPS))


def _true_range(df):
    prev_close = df['Close'].shift(1)
    return pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _adx(df, window):
    up = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _true_range(df).ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / (atr + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


# --- per-asset factor construction ----------------------------------------

def compute_factors(df):
    """
    Builds every single-asset factor for one symbol's daily frame.

    Expects the Binance schema: Open/High/Low/Close/Volume/QuoteVolume/Trades/
    TakerBuyBase/TakerBuyQuote/funding_daily/funding_mean/SpotClose/SpotVolume.
    """
    d = df.copy().sort_index()
    o, h, l, c, v = d['Open'], d['High'], d['Low'], d['Close'], d['Volume']
    F = {}                      # collected, then concatenated once
    class _Sink(dict):
        def __setitem__(self, k, v): dict.__setitem__(self, k, v)
        def __getitem__(self, k): return dict.__getitem__(self, k)
    f = _Sink()

    logret = np.log(c / c.shift(1))

    # ---- A. return / momentum ----
    for w in [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]:
        f[f'ret_{w}d'] = logret.rolling(w).sum()
    # momentum acceleration: this window's return vs the previous same window
    for w in [7, 14, 30]:
        f[f'mom_accel_{w}d'] = f[f'ret_{w}d'] - f[f'ret_{w}d'].shift(w)
    # risk-adjusted momentum (rolling Sharpe of daily returns)
    for w in [7, 14, 30, 60, 90]:
        f[f'sharpe_mom_{w}d'] = logret.rolling(w).mean() / (logret.rolling(w).std() + EPS)
    # sign consistency and extremes of the daily return distribution
    for w in [5, 10, 21, 60]:
        f[f'updays_frac_{w}d'] = (logret > 0).rolling(w).mean()
    for w in [14, 30, 60]:
        f[f'ret_skew_{w}d'] = logret.rolling(w).skew()
        f[f'ret_kurt_{w}d'] = logret.rolling(w).kurt()
    for w in [14, 30]:
        f[f'max_ret_{w}d'] = logret.rolling(w).max()
        f[f'min_ret_{w}d'] = logret.rolling(w).min()
    # signed run length: +n after n consecutive up days, -n after n down days
    sign = np.sign(logret)
    runs = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    f['ret_streak'] = (runs * sign).astype(float)

    # ---- B. volatility (18) ----
    for w in [5, 7, 10, 14, 21, 30, 60, 90]:
        f[f'vol_{w}d'] = logret.rolling(w).std()
    for a, b in [(7, 30), (14, 60), (5, 21)]:
        f[f'vol_ratio_{a}_{b}'] = f[f'vol_{a}d'] / (f[f'vol_{b}d'] + EPS)
    # Parkinson: uses the high-low range, ~5x more efficient than close-to-close
    park = np.log(h / (l + EPS)) ** 2 / (4 * np.log(2))
    for w in [5, 14, 30]:
        f[f'parkinson_{w}d'] = np.sqrt(park.rolling(w).mean())
    # Garman-Klass: adds the open-close move
    gk = 0.5 * np.log(h / (l + EPS)) ** 2 - (2 * np.log(2) - 1) * np.log(c / (o + EPS)) ** 2
    for w in [14, 30]:
        f[f'garman_klass_{w}d'] = np.sqrt(gk.rolling(w).mean().clip(lower=0))
    for w in [14, 30]:
        f[f'downside_vol_{w}d'] = logret.clip(upper=0).rolling(w).std()
        f[f'upside_vol_{w}d'] = logret.clip(lower=0).rolling(w).std()
        f[f'up_down_vol_ratio_{w}d'] = f[f'upside_vol_{w}d'] / (f[f'downside_vol_{w}d'] + EPS)
    # Rogers-Satchell: drift-independent OHLC estimator
    rs = np.log(h / (c + EPS)) * np.log(h / (o + EPS)) + \
        np.log(l / (c + EPS)) * np.log(l / (o + EPS))
    for w in [14, 30]:
        f[f'rogers_satchell_{w}d'] = np.sqrt(rs.rolling(w).mean().clip(lower=0))
    atr = _true_range(d)
    for w in [14, 30]:
        f[f'atr_ratio_{w}d'] = atr.rolling(w).mean() / (c + EPS)
    # vol-of-vol: how unstable the vol state itself is
    for w in [30, 60]:
        f[f'vol_of_vol_{w}d'] = f['vol_14d'].rolling(w).std() / (f['vol_14d'].rolling(w).mean() + EPS)
    # where today's vol sits inside its own trailing distribution
    for w in [30, 90]:
        f[f'vol_z_{w}d'] = _z(f['vol_14d'], w)

    # ---- C. range & candle structure (15) ----
    hl = (h - l) / (c + EPS)
    for w in [5, 14, 30]:
        f[f'hl_range_{w}d'] = hl.rolling(w).mean()
    for w in [5, 14, 30, 60]:
        hh, ll = h.rolling(w).max(), l.rolling(w).min()
        f[f'close_loc_{w}d'] = (c - ll) / (hh - ll + EPS)
    body = (c - o).abs() / (h - l + EPS)
    for w in [5, 14]:
        f[f'body_ratio_{w}d'] = body.rolling(w).mean()
    upper_wick = (h - np.maximum(o, c)) / (h - l + EPS)
    lower_wick = (np.minimum(o, c) - l) / (h - l + EPS)
    for w in [5, 14]:
        f[f'upper_wick_{w}d'] = upper_wick.rolling(w).mean()
        f[f'lower_wick_{w}d'] = lower_wick.rolling(w).mean()
    gap = (o - c.shift(1)) / (c.shift(1) + EPS)
    for w in [5, 14]:
        f[f'gap_{w}d'] = gap.rolling(w).mean()
    # drawdown from the running peak / run-up from the running trough
    for w in [30, 90]:
        f[f'drawdown_{w}d'] = c / (c.rolling(w).max() + EPS) - 1
        f[f'runup_{w}d'] = c / (c.rolling(w).min() + EPS) - 1
    # freshness of the extreme: 0 = the high/low was today, 1 = w bars ago
    for w in [30, 90]:
        f[f'days_since_high_{w}d'] = h.rolling(w).apply(
            lambda x: (len(x) - 1 - x.argmax()) / (len(x) - 1), raw=True)
        f[f'days_since_low_{w}d'] = l.rolling(w).apply(
            lambda x: (len(x) - 1 - x.argmin()) / (len(x) - 1), raw=True)
    for w in [14, 30]:
        f[f'range_z_{w}d'] = _z(hl, w)
    # mean intraday (open->close) move, the perp's session drift
    intraday = (c - o) / (o + EPS)
    for w in [5, 14]:
        f[f'intraday_ret_{w}d'] = intraday.rolling(w).mean()

    # ---- D. trend / moving average (22) ----
    for w in [5, 10, 20, 50, 100, 200]:
        f[f'sma_ratio_{w}d'] = c / (c.rolling(w).mean() + EPS) - 1
    for w in [5, 10, 20, 50]:
        f[f'ema_ratio_{w}d'] = c / (c.ewm(span=w, adjust=False).mean() + EPS) - 1
    for a, b in [(5, 20), (10, 50), (20, 100), (50, 200)]:
        f[f'sma_cross_{a}_{b}'] = (c.rolling(a).mean() / (c.rolling(b).mean() + EPS) - 1)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / (c + EPS)
    f['macd'] = macd
    f['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    for w in [14, 28]:
        f[f'adx_{w}d'] = _adx(d, w)
    aroon_up = h.rolling(25).apply(lambda x: x.argmax() / 24 * 100, raw=True)
    aroon_dn = l.rolling(25).apply(lambda x: x.argmin() / 24 * 100, raw=True)
    f['aroon_up_25'] = aroon_up
    f['aroon_down_25'] = aroon_dn
    f['aroon_osc_25'] = aroon_up - aroon_dn
    # normalised OLS slope of the price path itself
    for w in [10, 20, 50, 100]:
        f[f'price_slope_{w}d'] = _slope(c, w)
    # how linear the path is: R^2 of price against time
    t_idx = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    for w in [20, 50]:
        f[f'trend_r2_{w}d'] = c.rolling(w).corr(t_idx) ** 2
    # Kaufman efficiency ratio: net move / path length, 1 = pure trend
    for w in [10, 30]:
        f[f'eff_ratio_{w}d'] = (c - c.shift(w)).abs() / \
            (c.diff().abs().rolling(w).sum() + EPS)
    mom = c.diff()
    tsi_num = mom.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi_den = mom.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    f['tsi'] = 100 * tsi_num / (tsi_den + EPS)
    f['ppo'] = (ema12 - ema26) / (ema26 + EPS) * 100
    f['dpo_20'] = (c.shift(11) - c.rolling(20).mean()) / (c + EPS)
    # vortex: directional range movement scaled by true range
    tr14 = _true_range(d).rolling(14).sum()
    f['vortex_pos_14'] = (h - l.shift(1)).abs().rolling(14).sum() / (tr14 + EPS)
    f['vortex_neg_14'] = (l - h.shift(1)).abs().rolling(14).sum() / (tr14 + EPS)
    f['vortex_diff_14'] = f['vortex_pos_14'] - f['vortex_neg_14']
    for w in [14, 28]:
        gains = logret.clip(lower=0).rolling(w).sum()
        losses = (-logret.clip(upper=0)).rolling(w).sum()
        f[f'cmo_{w}d'] = 100 * (gains - losses) / (gains + losses + EPS)

    # ---- E. oscillators / mean reversion (16) ----
    for w in [7, 14, 21, 30]:
        f[f'rsi_{w}d'] = _rsi(c, w)
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = 100 * (c - ll14) / (hh14 - ll14 + EPS)
    f['stoch_k_14'] = stoch_k
    f['stoch_d_14'] = stoch_k.rolling(3).mean()
    f['williams_r_14'] = -100 * (hh14 - c) / (hh14 - ll14 + EPS)
    tp = (h + l + c) / 3
    f['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + EPS)
    for w in [10, 20, 50, 100]:
        f[f'zscore_{w}d'] = _z(c, w)
    for w in [20, 50]:
        mu, sd = c.rolling(w).mean(), c.rolling(w).std()
        f[f'bb_pct_{w}d'] = (c - (mu - 2 * sd)) / (4 * sd + EPS)
        f[f'bb_width_{w}d'] = 4 * sd / (mu + EPS)
    # stochastic of the RSI itself: where RSI sits in its own recent range
    rsi14 = f['rsi_14d']
    rsi_min, rsi_max = rsi14.rolling(14).min(), rsi14.rolling(14).max()
    f['stoch_rsi_14'] = (rsi14 - rsi_min) / (rsi_max - rsi_min + EPS)
    # ultimate oscillator: buying pressure over three blended horizons
    prev_c = c.shift(1)
    bp = c - np.minimum(l, prev_c)
    tr = np.maximum(h, prev_c) - np.minimum(l, prev_c)
    avgs = [bp.rolling(w).sum() / (tr.rolling(w).sum() + EPS) for w in (7, 14, 28)]
    f['ultimate_osc'] = 100 * (4 * avgs[0] + 2 * avgs[1] + avgs[2]) / 7
    f['williams_r_28'] = -100 * (h.rolling(28).max() - c) / \
        (h.rolling(28).max() - l.rolling(28).min() + EPS)
    f['cci_50'] = (tp - tp.rolling(50).mean()) / (0.015 * tp.rolling(50).std() + EPS)
    # position inside the Keltner channel (EMA20 +/- 2 ATR)
    ema20 = c.ewm(span=20, adjust=False).mean()
    atr20 = _true_range(d).rolling(20).mean()
    f['keltner_pct_20'] = (c - (ema20 - 2 * atr20)) / (4 * atr20 + EPS)
    f['bb_pct_100d'] = (lambda mu, sd: (c - (mu - 2 * sd)) / (4 * sd + EPS))(
        c.rolling(100).mean(), c.rolling(100).std())

    # ---- F. volume & liquidity (19) ----
    qv = d['QuoteVolume']
    for w in [5, 14, 30]:
        f[f'volume_z_{w}d'] = _z(v, w)
    for a, b in [(7, 30), (5, 60)]:
        f[f'volume_ratio_{a}_{b}'] = v.rolling(a).mean() / (v.rolling(b).mean() + EPS)
    for w in [7, 30]:
        f[f'log_dollar_vol_{w}d'] = np.log1p(qv.rolling(w).mean())
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    for w in [14, 30]:
        f[f'obv_slope_{w}d'] = _slope(obv, w)
    raw_mf = tp * v
    pos_mf = raw_mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp.diff() < 0, 0).rolling(14).sum()
    f['mfi_14'] = 100 - 100 / (1 + pos_mf / (neg_mf + EPS))
    for w in [7, 30]:
        vwap = (tp * v).rolling(w).sum() / (v.rolling(w).sum() + EPS)
        f[f'vwap_ratio_{w}d'] = c / (vwap + EPS) - 1
    for w in [14, 30]:
        f[f'amihud_{w}d'] = (logret.abs() / (qv + EPS)).rolling(w).mean() * 1e9
    f['turnover_z_14'] = _z(qv, 14)
    for w in [7, 30]:
        f[f'trades_z_{w}d'] = _z(d['Trades'], w)
    avg_trade = qv / (d['Trades'] + EPS)
    for w in [7, 30]:
        f[f'avg_trade_size_z_{w}d'] = _z(avg_trade, w)
    # Chaikin money flow: close-location-weighted volume balance
    mfv = ((c - l) - (h - c)) / (h - l + EPS) * v
    for w in [14, 30]:
        f[f'cmf_{w}d'] = mfv.rolling(w).sum() / (v.rolling(w).sum() + EPS)
    # accumulation/distribution line slope (same units as obv_slope)
    ad_line = mfv.cumsum()
    for w in [14, 30]:
        f[f'ad_slope_{w}d'] = _slope(ad_line, w)
    force = c.diff() * v
    for w in [14, 30]:
        f[f'force_z_{w}d'] = _z(force, w)
    # ease of movement: range travelled per unit volume, z-scored for scale
    eom = ((h + l) / 2 - (h.shift(1) + l.shift(1)) / 2) * (h - l) / (v + EPS)
    f['eom_z_14'] = _z(eom, 14)
    # price-volume coupling: does volume expansion accompany the move?
    dlogv = np.log1p(v).diff()
    for w in [14, 30, 60]:
        f[f'pv_corr_{w}d'] = logret.rolling(w).corr(dlogv)
    for w in [14, 30]:
        f[f'vol_slope_{w}d'] = _slope(v, w)
    # share of volume printed on up days
    upvol = v.where(logret > 0, 0.0)
    for w in [14, 30]:
        f[f'upvol_frac_{w}d'] = upvol.rolling(w).sum() / (v.rolling(w).sum() + EPS)
    # Roll (1984) implied spread from negative return autocovariance
    for w in [14, 30]:
        autocov = logret.rolling(w).cov(logret.shift(1))
        f[f'roll_spread_{w}d'] = 2 * np.sqrt((-autocov).clip(lower=0))
    f['amihud_z_30'] = _z(f['amihud_30d'], 30)
    f['turnover_z_30'] = _z(qv, 30)

    # ---- G. taker flow / microstructure (8) ----
    # Binance klines carry taker-BUY volume; sell volume is the remainder.
    taker_imb = (2 * d['TakerBuyBase'] - v) / (v + EPS)
    for w in [1, 3, 7, 14, 30]:
        f[f'taker_imb_{w}d'] = taker_imb.rolling(w).mean()
    f['taker_imb_z_14'] = _z(taker_imb, 14)
    taker_q_imb = (2 * d['TakerBuyQuote'] - qv) / (qv + EPS)
    for w in [7, 30]:
        f[f'taker_quote_imb_{w}d'] = taker_q_imb.rolling(w).mean()
    f['taker_imb_z_30'] = _z(taker_imb, 30)
    f['taker_quote_imb_z_14'] = _z(taker_q_imb, 14)
    # is the buying pressure building or fading?
    for w in [7, 14]:
        f[f'taker_imb_chg_{w}d'] = taker_imb.rolling(w).mean() - \
            taker_imb.rolling(w).mean().shift(w)
    f['taker_imb_slope_14'] = _slope(taker_imb.rolling(3).mean(), 14)
    f['taker_imb_cum_30d'] = taker_imb.rolling(30).sum()
    # does aggressive buying actually move price here?
    for w in [14, 30]:
        f[f'taker_ret_corr_{w}d'] = taker_imb.rolling(w).corr(logret)

    # ---- H. derivatives: funding & basis (20) ----
    fund = d['funding_daily']
    f['funding_1d'] = fund
    for w in [3, 7, 14, 30, 60]:
        f[f'funding_mean_{w}d'] = fund.rolling(w).mean()
    for w in [14, 30]:
        f[f'funding_z_{w}d'] = _z(fund, w)
    for w in [7, 30]:
        f[f'funding_cum_{w}d'] = fund.rolling(w).sum()
    for w in [14, 30]:
        f[f'funding_pos_frac_{w}d'] = (fund > 0).rolling(w).mean()
    f['funding_vol_30d'] = fund.rolling(30).std()
    f['funding_chg_7d'] = fund.rolling(7).mean() - fund.rolling(7).mean().shift(7)
    # where today's funding sits in its own trailing distribution
    for w in [90, 180]:
        f[f'funding_pctile_{w}d'] = fund.rolling(w).rank(pct=True)
    f['funding_min_30d'] = fund.rolling(30).min()
    f['funding_max_30d'] = fund.rolling(30).max()
    f['funding_skew_60d'] = fund.rolling(60).skew()
    f['funding_slope_14d'] = _slope(fund, 14)
    f['funding_dev_30d'] = fund - fund.rolling(30).mean()
    f['funding_pos_frac_60d'] = (fund > 0).rolling(60).mean()

    # to_numeric: symbols without a spot pair carry pd.NA, which turns the
    # column into object dtype and breaks both rolling ops and LightGBM.
    spot_close = pd.to_numeric(d['SpotClose'], errors='coerce') if 'SpotClose' in d \
        else pd.Series(np.nan, index=d.index)
    basis = (c - spot_close) / (spot_close + EPS)
    f['basis_1d'] = basis
    for w in [3, 7, 14, 30]:
        f[f'basis_mean_{w}d'] = basis.rolling(w).mean()
    f['basis_z_14d'] = _z(basis, 14)
    f['basis_vol_30d'] = basis.rolling(30).std()
    f['basis_chg_7d'] = basis.rolling(7).mean() - basis.rolling(7).mean().shift(7)
    f['basis_pctile_90d'] = basis.rolling(90).rank(pct=True)
    spot_vol = pd.to_numeric(d['SpotVolume'], errors='coerce') if 'SpotVolume' in d \
        else pd.Series(np.nan, index=d.index)
    perp_spot = v / (spot_vol + EPS)
    for w in [7, 30]:
        f[f'perp_spot_vol_{w}d'] = np.log1p(perp_spot.rolling(w).mean())
    f['perp_spot_vol_z_14'] = _z(np.log1p(perp_spot), 14)

    # ---- I. calendar ----
    dow = pd.Series(d.index.dayofweek, index=d.index).astype(float)
    f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    f['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    out = pd.concat({k: pd.Series(v, index=d.index) if not isinstance(v, pd.Series) else v
                     for k, v in f.items()}, axis=1)

    # Carry the raw series the backtest and labels need.
    carry = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume',
                             'funding_daily', 'Symbol'] if col in d]
    return pd.concat([out, d[carry]], axis=1)


FACTOR_PREFIX_SKIP = {'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume',
                      'funding_daily', 'Symbol'}

# The 2026-09-21 factor expansion, grouped by family. Family-level ablation on
# the selection head (wf_step=40, vs the previous 158-feature baseline RankIC
# +0.031) showed C/E/G/H/cross-sectional/calendar additions HELP ranking while
# A/B/D/F additions hurt it; models.HEADS excludes the hurtful ones for the
# selection head only. All heads keep the full panel available.
V2_MOMENTUM = ([f'mom_accel_{w}d' for w in (7, 14, 30)] +
               [f'sharpe_mom_{w}d' for w in (7, 14, 30, 60, 90)] +
               [f'updays_frac_{w}d' for w in (5, 10, 21, 60)] +
               [f'ret_skew_{w}d' for w in (14, 30, 60)] +
               [f'ret_kurt_{w}d' for w in (14, 30, 60)] +
               [f'max_ret_{w}d' for w in (14, 30)] +
               [f'min_ret_{w}d' for w in (14, 30)] + ['ret_streak'])
V2_VOLATILITY = ([f'upside_vol_{w}d' for w in (14, 30)] +
                 [f'up_down_vol_ratio_{w}d' for w in (14, 30)] +
                 [f'rogers_satchell_{w}d' for w in (14, 30)] +
                 [f'atr_ratio_{w}d' for w in (14, 30)] +
                 [f'vol_of_vol_{w}d' for w in (30, 60)] +
                 [f'vol_z_{w}d' for w in (30, 90)])
V2_TREND = ([f'price_slope_{w}d' for w in (10, 20, 50, 100)] +
            [f'trend_r2_{w}d' for w in (20, 50)] +
            [f'eff_ratio_{w}d' for w in (10, 30)] +
            ['tsi', 'ppo', 'dpo_20', 'vortex_pos_14', 'vortex_neg_14',
             'vortex_diff_14', 'cmo_14d', 'cmo_28d'])
V2_VOLUME = ([f'cmf_{w}d' for w in (14, 30)] +
             [f'ad_slope_{w}d' for w in (14, 30)] +
             [f'force_z_{w}d' for w in (14, 30)] + ['eom_z_14'] +
             [f'pv_corr_{w}d' for w in (14, 30, 60)] +
             [f'vol_slope_{w}d' for w in (14, 30)] +
             [f'upvol_frac_{w}d' for w in (14, 30)] +
             [f'roll_spread_{w}d' for w in (14, 30)] +
             ['amihud_z_30', 'turnover_z_30'])

# What the selection head does NOT see (69 features).
SELECTION_EXCLUDE = V2_MOMENTUM + V2_VOLATILITY + V2_TREND + V2_VOLUME


def factor_columns(df):
    return [c for c in df.columns if c not in FACTOR_PREFIX_SKIP]


# --- cross-sectional factors & panel assembly ------------------------------

CS_RANK_BASE = [
    'ret_7d', 'ret_30d', 'vol_14d', 'funding_mean_7d', 'taker_imb_7d',
    'log_dollar_vol_30d', 'rsi_14d', 'basis_mean_7d', 'sma_ratio_50d',
    'close_loc_14d',
    # second wave: momentum spectrum, risk-adjusted momentum, flow and liquidity
    'ret_1d', 'ret_14d', 'ret_60d', 'ret_90d', 'sharpe_mom_30d', 'mom_accel_14d',
    'vol_30d', 'parkinson_14d', 'taker_imb_14d', 'funding_z_14d',
    'funding_cum_30d', 'basis_z_14d', 'obv_slope_14d', 'amihud_14d', 'mfi_14',
    'macd_hist', 'bb_pct_20d', 'zscore_20d', 'drawdown_30d', 'eff_ratio_30d',
]
CS_Z_BASE = ['ret_7d', 'vol_14d', 'funding_mean_7d',
             'ret_30d', 'taker_imb_7d', 'basis_mean_7d', 'log_dollar_vol_30d',
             'sharpe_mom_30d']


def add_cross_sectional(panel, benchmark='BTCUSDT'):
    """
    Factors that only exist relative to the rest of the universe. These are what
    a cross-sectional ranker actually needs -- a raw momentum value says nothing
    about whether this coin is the strongest one today.

    All are computed within a single date, so no information crosses time.
    """
    p = panel.copy()
    g = p.groupby(level='Date')

    for col in CS_RANK_BASE:
        if col in p:
            p[f'cs_rank_{col}'] = g[col].rank(pct=True)
    for col in CS_Z_BASE:
        if col in p:
            mu, sd = g[col].transform('mean'), g[col].transform('std')
            p[f'cs_z_{col}'] = (p[col] - mu) / (sd + EPS)

    # --- market aggregates (same value for every asset on a date) ---
    p['mkt_ret_1d'] = g['ret_1d'].transform('mean')
    p['mkt_ret_7d'] = g['ret_7d'].transform('mean')
    p['mkt_vol_14d'] = g['vol_14d'].transform('mean')
    p['mkt_funding_7d'] = g['funding_mean_7d'].transform('mean')
    p['mkt_breadth_7d'] = g['ret_7d'].transform(lambda s: (s > 0).mean())
    p['mkt_dispersion_1d'] = g['ret_1d'].transform('std')
    p['mkt_ret_30d'] = g['ret_30d'].transform('mean')
    p['mkt_vol_30d'] = g['vol_30d'].transform('mean')
    p['mkt_breadth_30d'] = g['ret_30d'].transform(lambda s: (s > 0).mean())
    p['mkt_breadth_1d'] = g['ret_1d'].transform(lambda s: (s > 0).mean())
    p['mkt_dispersion_7d'] = g['ret_7d'].transform('std')
    p['mkt_taker_imb_7d'] = g['taker_imb_7d'].transform('mean')
    p['mkt_basis_7d'] = g['basis_mean_7d'].transform('mean')
    p['mkt_funding_z_14d'] = g['funding_z_14d'].transform('mean')

    # --- excess over the market (cross-sectionally demeaned momentum) ---
    for w in [1, 7, 30]:
        p[f'exc_ret_{w}d'] = p[f'ret_{w}d'] - p[f'mkt_ret_{w}d']

    # --- relative to the benchmark ---
    bench = p.xs(benchmark, level='Symbol') if benchmark in p.index.get_level_values('Symbol') else None
    if bench is not None:
        for w in [7, 14, 30, 90]:
            b = bench[f'ret_{w}d'].reindex(p.index.get_level_values('Date')).values
            p[f'rel_ret_{w}d_vs_bench'] = p[f'ret_{w}d'].values - b

        bench_r1 = bench['ret_1d']
        b_aligned = pd.Series(bench_r1.reindex(p.index.get_level_values('Date')).values,
                              index=p.index)
        for w in [30, 90]:
            def _beta(sub):
                r = sub['_r']; bb = sub['_b']
                cov = r.rolling(w).cov(bb)
                var = bb.rolling(w).var()
                return cov / (var + EPS)

            def _corr(sub):
                return sub['_r'].rolling(w).corr(sub['_b'])

            tmp = pd.DataFrame({'_r': p['ret_1d'], '_b': b_aligned})
            by_sym = tmp.groupby(level='Symbol', group_keys=False)
            p[f'beta_bench_{w}d'] = by_sym.apply(_beta)
            p[f'corr_bench_{w}d'] = by_sym.apply(_corr)

        resid_var = p['vol_30d'] ** 2 - (p['beta_bench_30d'] ** 2) * \
            pd.Series(bench['ret_1d'].rolling(30).std().reindex(
                p.index.get_level_values('Date')).values, index=p.index) ** 2
        p['idio_vol_30d'] = np.sqrt(resid_var.clip(lower=0))

    return p


def add_labels(panel, selection_horizon=7):
    """
    Labels for all three model heads. Every one is strictly FORWARD-looking and
    is dropped (not filled) where the future is unknown.

      target_ret_7d   selection : forward 7d simple return
      target_ret_1d   timing    : forward 1d simple return (price only)
      target_up_1d    timing    : 1 if tomorrow closes up, else 0
      target_high_1d  range     : tomorrow's HIGH  / today's close - 1
      target_low_1d   range     : tomorrow's LOW   / today's close - 1
      funding_next_1d PnL       : funding a long actually pays over tomorrow
      target_net_1d   PnL       : target_ret_1d - funding_next_1d
    """
    p = panel.copy()
    by = p.groupby(level='Symbol', group_keys=False)

    p[f'target_ret_{selection_horizon}d'] = by['Close'].apply(
        lambda s: s.shift(-selection_horizon) / s - 1)
    p['target_ret_1d'] = by['Close'].apply(lambda s: s.shift(-1) / s - 1)
    p['target_up_1d'] = (p['target_ret_1d'] > 0).astype('float')
    p['target_high_1d'] = by.apply(lambda d: d['High'].shift(-1) / d['Close'] - 1)
    p['target_low_1d'] = by.apply(lambda d: d['Low'].shift(-1) / d['Close'] - 1)

    # A long perp pays tomorrow's funding while it holds the position.
    p['funding_next_1d'] = by['funding_daily'].apply(lambda s: s.shift(-1))
    p['target_net_1d'] = p['target_ret_1d'] - p['funding_next_1d']

    # target_up_1d must be NaN, not 0, where tomorrow is unknown.
    p.loc[p['target_ret_1d'].isna(), 'target_up_1d'] = np.nan
    return p


LABEL_COLS = ['target_ret_7d', 'target_ret_1d', 'target_up_1d', 'target_high_1d',
              'target_low_1d', 'funding_next_1d', 'target_net_1d']
CARRY_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume', 'funding_daily']


def build_panel(frames, warmup_days=None, selection_horizon=7, benchmark=None,
                min_assets_per_date=5, verbose=True):
    """
    Full panel: per-asset factors -> cross-sectional factors -> labels.

    warmup_days rows are dropped from the head of each symbol. Factors with
    windows longer than that stay NaN, which is fine and intentional: LightGBM
    handles missing values natively, and a NaN here honestly means "not yet
    observable" -- there is no filling anywhere in this pipeline.
    """
    warmup_days = warmup_days if warmup_days is not None else config.WARMUP_DAYS
    benchmark = benchmark or config.BENCHMARK_SYMBOL

    per_symbol = []
    for symbol, df in frames.items():
        f = compute_factors(df)
        f = f.iloc[warmup_days:]
        if f.empty:
            continue
        f['Symbol'] = symbol
        f.index.name = 'Date'
        per_symbol.append(f.reset_index().set_index(['Date', 'Symbol']))

    if not per_symbol:
        raise ValueError("No symbol produced factors.")

    panel = pd.concat(per_symbol).sort_index()

    # Dates with too few live assets cannot support a cross-sectional ranking.
    counts = panel.groupby(level='Date').size()
    keep = counts[counts >= min_assets_per_date].index
    panel = panel[panel.index.get_level_values('Date').isin(keep)]

    panel = add_cross_sectional(panel, benchmark=benchmark)
    panel = add_labels(panel, selection_horizon=selection_horizon)

    if verbose:
        dates = panel.index.get_level_values('Date')
        print(f"Panel: {len(panel):,} rows | {dates.nunique():,} dates | "
              f"{panel.index.get_level_values('Symbol').nunique()} symbols | "
              f"{dates.min().date()} -> {dates.max().date()}")
        print(f"       {len(feature_columns(panel))} features, "
              f"{len(LABEL_COLS)} labels, "
              f"median {counts.median():.0f} assets/date")

    return panel


def feature_columns(panel):
    """Everything that is a model input: not a label, not a raw carried series."""
    excluded = set(LABEL_COLS) | set(CARRY_COLS) | {'Symbol'}
    return [c for c in panel.columns if c not in excluded]

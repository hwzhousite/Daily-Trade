import pandas as pd
import numpy as np

import config

FACTOR_COLUMNS = [
    'momentum_7d', 'momentum_14d', 'volatility_7d', 'volatility_14d',
    'rsi_14d', 'volume_ratio',
]


def calculate_daily_factors(df, mom_period_1=7, mom_period_2=14,
                            vol_period_1=7, vol_period_2=14, rsi_period=14,
                            vol_ratio_short=7, vol_ratio_long=30,
                            drop_warmup=True):
    """
    Calculates technical factors on a daily DataFrame.

    With drop_warmup=True (default) the leading rows where any rolling window is
    still incomplete are dropped rather than back-filled, so no factor value is
    ever derived from data that was not yet observable at that timestamp.
    """
    df = df.copy().sort_index()

    # Daily Log Returns
    df['daily_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 1. Momentum factors
    df['momentum_7d'] = df['daily_return'].rolling(window=mom_period_1).sum()
    df['momentum_14d'] = df['daily_return'].rolling(window=mom_period_2).sum()

    # 2. Volatility factors (std of *daily* log returns)
    df['volatility_7d'] = df['daily_return'].rolling(window=vol_period_1).std()
    df['volatility_14d'] = df['daily_return'].rolling(window=vol_period_2).std()

    # 3. Mean Reversion factor (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14d'] = 100 - (100 / (1 + rs))

    # 4. Volume Dynamics factor
    df['volume_ratio'] = df['Volume'].rolling(window=vol_ratio_short).mean() / (df['Volume'].rolling(window=vol_ratio_long).mean() + 1e-9)

    # Forward-fill only. A bfill() here would populate the rolling warm-up rows
    # with values computed from FUTURE bars -- a look-ahead leak. Instead the
    # warm-up period is dropped, which is what those rows honestly are: unknown.
    df = df.ffill()

    if drop_warmup:
        first_valid = [df[c].first_valid_index() for c in FACTOR_COLUMNS]
        if any(idx is None for idx in first_valid):
            raise ValueError(
                f"Series of length {len(df)} is shorter than the factor warm-up "
                f"period (longest window = {vol_ratio_long}); no usable rows."
            )
        df = df.loc[max(first_valid):]

    return df


def build_panel(cleaned_data, horizon_days, resample_rule=None, **factor_params):
    """
    Builds the supervised panel for one layer.

    horizon_days   label = simple return from this bar's close to the close
                   `horizon_days` bars ahead.
    resample_rule  None (default) -> sample DAILY. Labels then overlap when
                   horizon_days > 1, which is fine for fitting but requires an
                   embargo in every validation split (see model_pipeline).
                   Pass e.g. 'W-SUN' for non-overlapping calendar bars.

    Returns a panel indexed by [Date, Asset] with the factor columns, 'Close',
    'bar_return' (trailing) and 'target_return' (forward, the label).
    """
    params = {**config.FACTOR_PARAMS, **factor_params}
    frames = []

    agg_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum',
        **{c: 'last' for c in FACTOR_COLUMNS},
    }

    for coin, df in cleaned_data.items():
        df_factors = calculate_daily_factors(df, **params)

        bars = df_factors.resample(resample_rule).agg(agg_dict) if resample_rule else df_factors

        bars = bars.copy()
        bars['bar_return'] = bars['Close'].pct_change()
        # Forward return over `horizon_days` bars -- the supervision signal.
        bars['target_return'] = bars['Close'].shift(-horizon_days) / bars['Close'] - 1.0
        bars['Asset'] = coin

        bars = bars.dropna(subset=['target_return', 'bar_return'] + FACTOR_COLUMNS)
        frames.append(bars)

    panel = pd.concat(frames)
    panel.index.name = 'Date'
    panel = panel.reset_index().set_index(['Date', 'Asset']).sort_index()
    return panel


def build_layer_panels(cleaned_data, layers=None, **factor_params):
    """Builds one panel per configured layer: {'selection': df, 'timing': df}."""
    layers = layers or config.LAYERS
    return {
        name: build_panel(cleaned_data, horizon_days=spec['horizon_days'], **factor_params)
        for name, spec in layers.items()
    }


def latest_factor_snapshot(cleaned_data, **factor_params):
    """
    Inference-time feature row for each asset: the most recent daily bar's
    factors. Identical construction to build_panel's daily sampling, so the
    features match the training distribution exactly.
    """
    params = {**config.FACTOR_PARAMS, **factor_params}
    rows = []

    for coin, df in cleaned_data.items():
        df_factors = calculate_daily_factors(df, **params)
        latest_row = df_factors.iloc[[-1]].copy()
        latest_row['Asset'] = coin
        rows.append(latest_row)

    snapshot = pd.concat(rows)
    snapshot.index.name = 'Date'
    return snapshot.reset_index()


def engineer_factors(cleaned_data, resample_rule='W-SUN', horizon_days=1, **factor_params):
    """Legacy wrapper: non-overlapping weekly bars, label = next bar's return."""
    return build_panel(cleaned_data, horizon_days=horizon_days,
                       resample_rule=resample_rule, **factor_params)

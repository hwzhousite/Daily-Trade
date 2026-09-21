"""
Nightly inference for the two-layer strategy.

Both models score the SAME latest daily factor snapshot:
  selection -> expected 7d return  (which coins are worth owning)
  timing    -> expected 1d return  (whether to be in them tomorrow)

The trade horizon is one day: the book is rebalanced every night, so price
boundaries and risk levels are computed for 1 day, not 7.
"""
import os
import joblib
import numpy as np
import pandas as pd

import config
from features import latest_factor_snapshot


def load_bundle(role, model_path=None):
    """Loads a layer's persisted bundle {model, features, horizon_days, ...}."""
    model_path = str(model_path or config.model_path(role))
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained '{role}' model at {model_path}. Run `python main.py train` first."
        )
    obj = joblib.load(model_path)
    if not (isinstance(obj, dict) and 'model' in obj):
        raise ValueError(
            f"{model_path} holds a bare estimator from an older pipeline version "
            f"(no feature metadata). Retrain to regenerate it."
        )
    return obj


def generate_signals(cleaned_data, top_n=config.TOP_N,
                     threshold=config.TIMING_THRESHOLD, model_paths=None):
    """
    Scores the latest snapshot with both layers and applies the position rule.

    Returns (signals_df, plan_assets, meta). signals_df is ranked by the
    selection score and carries a Decision column per asset.
    """
    model_paths = model_paths or {}
    sel = load_bundle('selection', model_paths.get('selection'))
    tim = load_bundle('timing', model_paths.get('timing'))

    if sel['factor_params'] != tim['factor_params']:
        raise ValueError("Selection and timing models were trained with different "
                         "factor windows; retrain both together.")

    snapshot = latest_factor_snapshot(cleaned_data, **sel['factor_params'])

    for bundle in (sel, tim):
        missing = [f for f in bundle['features'] if f not in snapshot.columns]
        if missing:
            raise KeyError(f"Snapshot is missing {bundle['role']} features: {missing}")

    # Column order must match training exactly -- taken from each bundle.
    snapshot['sel_score'] = sel['model'].predict(snapshot[sel['features']])
    snapshot['timing_score'] = tim['model'].predict(snapshot[tim['features']])

    df = snapshot[['Date', 'Asset', 'Close', 'sel_score', 'timing_score',
                   'volatility_7d']].copy()
    df = df.sort_values('sel_score', ascending=False).reset_index(drop=True)

    df['Rank'] = np.arange(1, len(df) + 1)
    df['Candidate'] = df['Rank'] <= top_n
    df['Timing_OK'] = df['timing_score'] > threshold
    df['Decision'] = np.where(
        df['Candidate'] & df['Timing_OK'], 'HOLD',
        np.where(df['Candidate'], 'BLOCKED (timing)', 'NOT SELECTED'))

    plan_assets = df.loc[df['Decision'] == 'HOLD', 'Asset'].tolist()

    as_of = pd.Timestamp(df['Date'].max())
    meta = {
        'as_of': as_of,
        'selection_horizon': sel['horizon_days'],
        'timing_horizon': tim['horizon_days'],
        'top_n': top_n,
        'threshold': threshold,
        'n_candidates': int(df['Candidate'].sum()),
        'n_held': len(plan_assets),
        'selection_trained_at': sel.get('trained_at'),
        'timing_trained_at': tim.get('trained_at'),
        'selection_wf_metrics': sel.get('wf_metrics'),
        'timing_wf_metrics': tim.get('wf_metrics'),
    }
    return df, plan_assets, meta


def price_boundaries(signals_df, horizon_days=1,
                     confidence_level=config.CONFIDENCE_LEVEL,
                     score_col='timing_score'):
    """
    Lognormal price boundaries over `horizon_days`.

    Units:
      * the score is a SIMPLE return -> log1p() before exponentiating
      * volatility_7d is the std of DAILY log returns -> scaled by sqrt(horizon)
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    sqrt_h = np.sqrt(horizon_days)

    out = signals_df[['Asset', 'Close']].copy()
    mu = np.log1p(signals_df[score_col].clip(lower=-0.999))
    sigma = signals_df['volatility_7d'] * sqrt_h

    out['Expected_Return'] = signals_df[score_col].values
    out['Horizon_Days'] = horizon_days
    out['Sigma_Horizon'] = sigma.values
    out['Lower_Boundary'] = (signals_df['Close'] * np.exp(mu - z * sigma)).values
    out['Upper_Boundary'] = (signals_df['Close'] * np.exp(mu + z * sigma)).values
    return out.reset_index(drop=True)

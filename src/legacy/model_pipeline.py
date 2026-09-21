"""
Training and validation for one layer (selection or timing).

The panel is sampled daily, so when a layer's horizon is > 1 day its labels
overlap across consecutive rows. Every split below therefore leaves an `embargo`
gap between the last training bar and the test bar: a training row dated t
carries a label spanning (t, t+h], so t must satisfy t + h < T for the test bar
T. Without that gap the walk-forward score is inflated by label leakage.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import config


def _cross_sectional_ic(df, pred_col='predicted_return', target_col='target_return'):
    """Mean per-date Spearman rank correlation -- the metric selection lives on."""
    ics = []
    for _, grp in df.groupby('Date'):
        if grp[pred_col].nunique() < 2 or grp[target_col].nunique() < 2:
            continue
        ic = spearmanr(grp[pred_col], grp[target_col]).statistic
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return np.nan, np.nan
    ics = np.asarray(ics)
    # IC t-stat: mean / (std / sqrt(n))
    t_stat = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan
    return float(ics.mean()), float(t_stat)


def _score(df, target_col='target_return'):
    y, p = df[target_col].values, df['predicted_return'].values
    ic, ic_t = _cross_sectional_ic(df, target_col=target_col)
    return {
        'mse': float(mean_squared_error(y, p)),
        'r2': float(r2_score(y, p)),
        'rank_ic': ic,
        'ic_t_stat': ic_t,
        'hit_rate': float(np.mean(np.sign(p) == np.sign(y))),
        'n': int(len(df)),
    }


def train_layer(panel_df, layer, features=None, target=config.TARGET,
                wf_step=1, rf_params=None, model_save_path=None, verbose=True):
    """
    Walk-forward validates and then fits a production model for one layer.

    layer    a dict from config.LAYERS (role / horizon_days / train_window / embargo)
    wf_step  refit cadence in bars. 1 = refit every day (what a nightly job does);
             larger values trade fidelity for speed when iterating in a notebook.

    Returns {'wf_predictions', 'wf_metrics', 'holdout_metrics', 'importances',
             'bundle', 'model_path'}.
    """
    features = list(features or config.FEATURES)
    rf_params = dict(config.RF_PARAMS if rf_params is None else rf_params)
    role = layer['role']
    horizon = layer['horizon_days']
    train_window = layer['train_window']
    embargo = layer['embargo']
    model_save_path = str(model_save_path or config.model_path(role))

    df = panel_df.reset_index().sort_values(['Date', 'Asset'])
    dates = np.array(sorted(df['Date'].unique()))
    n_dates = len(dates)

    start = train_window + embargo
    if start >= n_dates:
        raise ValueError(
            f"[{role}] need > {start} bars (train_window {train_window} + embargo "
            f"{embargo}) but the panel has {n_dates}."
        )

    by_date = {d: g for d, g in df.groupby('Date')}

    # --- Walk-forward with embargo ----------------------------------------
    folds = []
    for i in range(start, n_dates, wf_step):
        train_dates = dates[i - embargo - train_window : i - embargo]
        test_dates = dates[i : i + wf_step]

        train_fold = df[df['Date'].isin(train_dates)]
        test_fold = pd.concat([by_date[d] for d in test_dates if d in by_date]).copy()
        if train_fold.empty or test_fold.empty:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(train_fold[features], train_fold[target])
        test_fold['predicted_return'] = model.predict(test_fold[features])
        folds.append(test_fold)

    if not folds:
        raise ValueError(f"[{role}] walk-forward produced no folds.")

    wf_predictions = pd.concat(folds).sort_values(['Date', 'Asset']).reset_index(drop=True)
    wf_metrics = _score(wf_predictions, target)

    # --- Held-out tail (last 20% of bars), embargoed -----------------------
    split = int(n_dates * 0.8)
    holdout = wf_predictions[wf_predictions['Date'] >= dates[split]]
    holdout_metrics = _score(holdout, target) if len(holdout) else {}

    # --- Production fit on everything --------------------------------------
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    production_model = RandomForestRegressor(**rf_params)
    production_model.fit(df[features], df[target])  # DataFrame -> feature names kept

    bundle = {
        'model': production_model,
        'role': role,
        'features': features,
        'target': target,
        'horizon_days': horizon,
        'train_window': train_window,
        'embargo': embargo,
        'wf_step': wf_step,
        'factor_params': dict(config.FACTOR_PARAMS),
        'rf_params': rf_params,
        'trained_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'train_rows': int(len(df)),
        'train_end': pd.Timestamp(dates[-1]).isoformat(),
        'wf_metrics': wf_metrics,
    }
    joblib.dump(bundle, model_save_path)

    importances = pd.DataFrame({
        'Feature': features,
        'Importance': production_model.feature_importances_,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"[{role}] horizon={horizon}d  window={train_window}  embargo={embargo}  "
              f"folds={len(folds)}  refit every {wf_step}d")
        print(f"[{role}] walk-forward : MSE {wf_metrics['mse']:.6f} | R2 {wf_metrics['r2']:+.4f} | "
              f"RankIC {wf_metrics['rank_ic']:+.4f} (t={wf_metrics['ic_t_stat']:+.2f}) | "
              f"hit {wf_metrics['hit_rate']:.3f}  n={wf_metrics['n']}")
        if holdout_metrics:
            print(f"[{role}] last 20%     : MSE {holdout_metrics['mse']:.6f} | R2 {holdout_metrics['r2']:+.4f} | "
                  f"RankIC {holdout_metrics['rank_ic']:+.4f} (t={holdout_metrics['ic_t_stat']:+.2f}) | "
                  f"hit {holdout_metrics['hit_rate']:.3f}  n={holdout_metrics['n']}")
        print(f"[{role}] saved -> {model_save_path}")

    return {
        'wf_predictions': wf_predictions,
        'wf_metrics': wf_metrics,
        'holdout_metrics': holdout_metrics,
        'importances': importances,
        'bundle': bundle,
        'model_path': model_save_path,
    }


def train_all_layers(panels, layers=None, wf_step=1, verbose=True, **kwargs):
    """Trains every configured layer; returns {'selection': result, 'timing': result}."""
    layers = layers or config.LAYERS
    out = {}
    for name, spec in layers.items():
        out[name] = train_layer(panels[name], spec, wf_step=wf_step, verbose=verbose, **kwargs)
        if verbose:
            print()
    return out

"""
LightGBM model heads.

Gradient-boosted trees are the strongest general-purpose model for tabular
financial panels: they handle 150+ correlated features, need no scaling, and --
crucially here -- treat NaN as an informative "not yet observable" rather than
forcing an imputation that would fabricate history.

Four heads, all trained on the same panel, all walk-forward validated with an
embargo sized to their label horizon:

    selection   regression  -> expected 7d return, used for cross-sectional rank
    timing      binary      -> P(tomorrow closes up)          <- a probability
    range_high  quantile    -> q90 of tomorrow's high / close
    range_low   quantile    -> q10 of tomorrow's low  / close
"""
import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_squared_error, roc_auc_score)

import config

HEADS = {
    'selection':  dict(task='regression', target='target_ret_7d',  horizon=7, embargo=7),
    'timing':     dict(task='binary',     target='target_up_1d',   horizon=1, embargo=1),
    'range_high': dict(task='quantile',   target='target_high_1d', horizon=1, embargo=1, alpha=0.90),
    'range_low':  dict(task='quantile',   target='target_low_1d',  horizon=1, embargo=1, alpha=0.10),
}


def _make_model(head, params=None):
    p = dict(config.LGB_PARAMS if params is None else params)
    task = head['task']
    if task == 'binary':
        return LGBMClassifier(objective='binary', **p)
    if task == 'quantile':
        return LGBMRegressor(objective='quantile', alpha=head['alpha'], **p)
    return LGBMRegressor(objective='regression', **p)


def _predict(model, X, task):
    if task == 'binary':
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _pinball(y, pred, alpha):
    delta = y - pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1) * delta)))


def _cross_sectional_ic(df, pred_col, target_col):
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
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan
    return float(ics.mean()), float(t)


def score_head(df, head, pred_col='prediction'):
    """Task-appropriate metrics. Each head is judged on what it is actually for."""
    target = head['target']
    y, p = df[target].values, df[pred_col].values
    task = head['task']
    out = {'n': int(len(df))}

    if task == 'binary':
        out['auc'] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        out['accuracy'] = float(accuracy_score(y, (p > 0.5).astype(int)))
        # Brier / log-loss test whether the PROBABILITY is honest, not just ranked.
        out['brier'] = float(brier_score_loss(y, p))
        out['log_loss'] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        out['base_rate'] = float(np.mean(y))
        out['mean_pred'] = float(np.mean(p))
    elif task == 'quantile':
        alpha = head['alpha']
        out['pinball'] = _pinball(y, p, alpha)
        # Coverage is the metric that matters: for alpha=0.9, 90% of realised
        # values should fall BELOW the prediction. Far from that = miscalibrated.
        out['coverage'] = float(np.mean(y <= p))
        out['target_coverage'] = alpha
        out['mean_pred'] = float(np.mean(p))
    else:
        out['mse'] = float(mean_squared_error(y, p))
        ic, t = _cross_sectional_ic(df, pred_col, target)
        out['rank_ic'] = ic
        out['ic_t_stat'] = t
        out['hit_rate'] = float(np.mean(np.sign(p) == np.sign(y)))

    return out


def format_metrics(name, m):
    if 'auc' in m:
        return (f"[{name}] AUC {m['auc']:.4f} | acc {m['accuracy']:.4f} | "
                f"Brier {m['brier']:.4f} | logloss {m['log_loss']:.4f} | "
                f"base {m['base_rate']:.3f} vs pred {m['mean_pred']:.3f} | n={m['n']:,}")
    if 'pinball' in m:
        return (f"[{name}] pinball {m['pinball']:.5f} | coverage {m['coverage']:.3f} "
                f"(target {m['target_coverage']:.2f}) | mean pred {m['mean_pred']:+.4f} | n={m['n']:,}")
    return (f"[{name}] MSE {m['mse']:.6f} | RankIC {m['rank_ic']:+.4f} "
            f"(t={m['ic_t_stat']:+.2f}) | hit {m['hit_rate']:.3f} | n={m['n']:,}")


def fit_conformal(wf_preds, head, calib_days=None):
    """
    Conformal calibration for a quantile head.

    Plain quantile regression systematically UNDER-covers on financial data:
    it is fitted on a trailing window, and when volatility expands the realised
    high/low blows through the predicted band. The fix is an additive offset
    chosen so that empirical coverage on a recent calibration window equals the
    nominal level -- the standard conformalized-quantile-regression correction.

        delta = quantile(y - prediction, alpha)
        calibrated = prediction + delta

    Returns (delta, diagnostics).
    """
    calib_days = calib_days or config.CONFORMAL_CALIB_DAYS
    alpha = head['alpha']
    target = head['target']

    df = wf_preds.dropna(subset=[target, 'prediction'])
    cutoff = df['Date'].max() - pd.Timedelta(days=calib_days)
    calib = df[df['Date'] >= cutoff]
    if len(calib) < 200:
        calib = df

    residual = calib[target].values - calib['prediction'].values
    delta = float(np.quantile(residual, alpha))

    raw_cov = float(np.mean(df[target].values <= df['prediction'].values))
    adj_cov = float(np.mean(df[target].values <= df['prediction'].values + delta))
    return delta, {'delta': delta, 'raw_coverage': raw_cov,
                   'calibrated_coverage': adj_cov, 'target_coverage': alpha,
                   'calib_rows': int(len(calib))}


def walk_forward(panel, head_name, features=None, wf_step=None, train_window=None,
                 params=None, verbose=True):
    """
    Embargoed rolling walk-forward for one head.

    A training row dated t carries a label spanning (t, t+horizon], so the last
    training bar must sit `embargo` bars before the test bar. Shrinking the
    embargo is the single easiest way to fabricate a good score here.
    """
    head = HEADS[head_name]
    wf_step = wf_step or config.WF_STEP
    train_window = train_window or config.TRAIN_WINDOW
    embargo = head['embargo']
    target = head['target']

    df = panel.reset_index()
    features = list(features) if features is not None else \
        [c for c in df.columns if c not in set(config.NON_FEATURE_COLS)]

    df = df.dropna(subset=[target]).sort_values(['Date', 'Symbol'])
    dates = np.array(sorted(df['Date'].unique()))
    start = train_window + embargo
    if start >= len(dates):
        raise ValueError(f"[{head_name}] need > {start} dates, panel has {len(dates)}")

    folds = []
    for i in range(start, len(dates), wf_step):
        train_dates = dates[i - embargo - train_window: i - embargo]
        test_dates = dates[i: i + wf_step]

        tr = df[df['Date'].isin(train_dates)]
        te = df[df['Date'].isin(test_dates)].copy()
        if tr.empty or te.empty or tr[target].nunique() < 2:
            continue

        model = _make_model(head, params)
        model.fit(tr[features], tr[target])
        te['prediction'] = _predict(model, te[features], head['task'])
        folds.append(te)

    if not folds:
        raise ValueError(f"[{head_name}] no folds produced")

    preds = pd.concat(folds).sort_values(['Date', 'Symbol']).reset_index(drop=True)
    metrics = score_head(preds, head)
    if verbose:
        print(f"  {format_metrics(head_name, metrics)}  ({len(folds)} folds, refit/{wf_step}d)")
    return preds, metrics


def fit_production(panel, head_name, features=None, params=None, save=True,
                   conformal=None, wf_metrics=None):
    """Fits on all available history and persists a bundle with its metadata."""
    head = HEADS[head_name]
    target = head['target']

    df = panel.reset_index()
    features = list(features) if features is not None else \
        [c for c in df.columns if c not in set(config.NON_FEATURE_COLS)]
    df = df.dropna(subset=[target])

    model = _make_model(head, params)
    model.fit(df[features], df[target])

    importances = pd.DataFrame({
        'Feature': features,
        'Gain': model.booster_.feature_importance('gain'),
        'Split': model.booster_.feature_importance('split'),
    }).sort_values('Gain', ascending=False).reset_index(drop=True)

    bundle = {
        'model': model, 'head': head_name, 'task': head['task'],
        'target': target, 'horizon_days': head['horizon'], 'embargo': head['embargo'],
        'alpha': head.get('alpha'), 'features': features,
        'lgb_params': dict(config.LGB_PARAMS if params is None else params),
        'trained_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'train_rows': int(len(df)),
        'train_end': pd.Timestamp(df['Date'].max()).isoformat(),
        'n_symbols': int(df['Symbol'].nunique()),
        'conformal': conformal,
        'conformal_delta': (conformal or {}).get('delta', 0.0),
        'wf_metrics': wf_metrics,
    }

    # A refit that skipped validation must not silently drop a calibration that
    # an earlier validated run established.
    if conformal is None and head['task'] == 'quantile':
        existing = config.model_path(head_name)
        if os.path.exists(existing):
            try:
                prev = joblib.load(existing)
                bundle['conformal'] = prev.get('conformal')
                bundle['conformal_delta'] = prev.get('conformal_delta', 0.0)
            except Exception:
                pass

    if save:
        path = str(config.model_path(head_name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp'
        joblib.dump(bundle, tmp)
        os.replace(tmp, path)
        bundle['path'] = path

    return bundle, importances


def train_all(panel, features=None, wf_step=None, params=None, verbose=True,
              validate=True):
    """
    Fits a production model for every head, and (when validate=True) also runs
    the embargoed walk-forward that the quantile heads need for their conformal
    calibration.

    validate=False is the fast nightly path: production refits only.
    """
    results = {}
    if verbose:
        print("Walk-forward validation:" if validate else "Production refit only:")

    for name in HEADS:
        head = HEADS[name]
        preds = metrics = None
        conformal = None

        if validate:
            preds, metrics = walk_forward(panel, name, features=features,
                                          wf_step=wf_step, params=params, verbose=verbose)
            if head['task'] == 'quantile':
                delta, diag = fit_conformal(preds, head)
                conformal = diag
                preds['prediction_calibrated'] = preds['prediction'] + delta
                if verbose:
                    print(f"      conformal delta {delta:+.5f} -> coverage "
                          f"{diag['raw_coverage']:.3f} => {diag['calibrated_coverage']:.3f} "
                          f"(target {diag['target_coverage']:.2f})")

        bundle, importances = fit_production(panel, name, features=features,
                                             params=params, conformal=conformal,
                                             wf_metrics=metrics)
        results[name] = {'wf_predictions': preds, 'wf_metrics': metrics,
                         'conformal': conformal, 'bundle': bundle,
                         'importances': importances}
    return results


def load_head(head_name):
    path = str(config.model_path(head_name))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained '{head_name}' model at {path}. Run `python main.py train` first.")
    return joblib.load(path)

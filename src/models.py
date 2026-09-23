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

Before each fit, features are pre-selected on the training window only
(config.FEATURE_SELECTOR): 'lasso' walks an ElasticNet L1 path (per-date
standardized for the cross-sectional selection head), 'ic' thresholds the
mean daily |rank IC|, 'off' disables selection. Heads therefore train on
DIFFERENT feature subsets. Per-head hyperparameters come from
config.head_params(), which layers `main.py tune` results on top of
LGB_PARAMS.
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
import factors as _factors

EPS = 1e-12

# Per-head feature policy, settled by wf_step=40 comparisons (2026-09-21):
#   selection   selector='off' + exclude: the curated 236-feature pool beat the
#               158 baseline (RankIC +0.037 vs +0.031); both the univariate IC
#               filter and the L1 path DESTROY its ranking signal, because
#               LightGBM needs the very redundancy those selectors remove.
#   timing/range selector='ic': beat 'lasso' on Brier/pinball across all three.
# cs=True marks the head judged on cross-sectional ordering (the lasso
# selector then standardizes features and target within each date).
# config.FEATURE_SELECTOR, when set, overrides every head (for experiments).
HEADS = {
    'selection':  dict(task='regression', target='target_ret_7d',  horizon=7, embargo=7,
                       cs=True, selector='off', exclude=_factors.SELECTION_EXCLUDE,
                       ensemble=True),
    'timing':     dict(task='binary',     target='target_up_1d',   horizon=1, embargo=1,
                       cs=False, selector='ic'),
    'range_high': dict(task='quantile',   target='target_high_1d', horizon=1, embargo=1,
                       alpha=0.90, cs=False, selector='ic'),
    'range_low':  dict(task='quantile',   target='target_low_1d',  horizon=1, embargo=1,
                       alpha=0.10, cs=False, selector='ic'),
}


class EnsembleRegressor:
    """
    Bag of independently seeded LGBM regressors.

    Boosting has no random-forest OOB property, so per-prediction uncertainty
    comes from re-fitting the same spec on K different seeds (which reshuffle
    the row/feature subsampling): the bag's disagreement (std across members)
    is an out-of-sample-style confidence estimate for each prediction.
    """

    def __init__(self, models):
        self.models = models

    def _stack(self, X):
        return np.stack([m.predict(X) for m in self.models])

    def predict(self, X):
        return self._stack(X).mean(axis=0)

    def predict_stats(self, X):
        preds = self._stack(X)
        return preds.mean(axis=0), preds.std(axis=0)


def _fit_head_model(head, params, X, y):
    """One fitted estimator: an EnsembleRegressor for an ensemble head."""
    n = config.N_ENSEMBLE if head.get('ensemble') else 1
    if head['task'] != 'regression' or n <= 1:
        model = _make_model(head, params)
        model.fit(X, y)
        return model
    members = []
    for k in range(n):
        p = dict(params or config.LGB_PARAMS)
        p['random_state'] = int(p.get('random_state', 42)) + k
        m = _make_model(head, p)
        m.fit(X, y)
        members.append(m)
    return EnsembleRegressor(members)


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


def daily_feature_ic(df, features, target, min_assets=5):
    """
    Per-date cross-sectional Spearman IC of every feature against the target.

    Returns a DataFrame indexed by Date with one column per feature. Row t is
    computed from data whose latest information is the label at t (observable
    through t + horizon), so averaging rows over a training window that ends
    `embargo` bars before the test window leaks nothing the fit itself doesn't.
    """
    rows = {}
    # Date-broadcast features (mkt_*, btc_*, eth_*) are CONSTANT within a
    # date: their cross-sectional correlation is undefined (NaN), which is
    # semantically right -- but np.corrcoef under corrwith warns about the
    # zero-variance divide. Silence exactly that, locally.
    with np.errstate(invalid='ignore', divide='ignore'):
        for date, grp in df.groupby('Date'):
            if len(grp) < min_assets:
                continue
            tgt_rank = grp[target].rank()
            if tgt_rank.nunique() < 2:
                continue
            rows[date] = grp[features].rank().corrwith(tgt_rank)
    ic = pd.DataFrame(rows).T
    ic.index.name = 'Date'
    return ic


def select_features_ic(daily_ic, dates, threshold=None, min_features=None):
    """
    Features whose mean daily |IC| over `dates` clears the threshold.

    Falls back to the top `min_features` by |IC| when the threshold keeps too
    few -- an empty feature set is worse than a mildly diluted one, and the 1d
    heads (timing especially) rarely have many features above 0.03.
    """
    threshold = config.IC_THRESHOLD if threshold is None else threshold
    min_features = min_features or config.IC_MIN_FEATURES

    ic = daily_ic.loc[daily_ic.index.isin(dates)].mean().dropna()
    keep = ic[ic.abs() >= threshold]
    if len(keep) < min_features:
        keep = ic.reindex(ic.abs().sort_values(ascending=False).index).head(min_features)
    return sorted(keep.index), ic


def select_features_lasso(df_train, features, target, cross_sectional=False,
                          max_features=None, min_features=None, l1_ratio=None):
    """
    ElasticNet L1-path feature selection on the training window.

    Within a block of correlated features the L1 penalty keeps one
    representative and zeroes the rest, which is exactly what the univariate
    IC filter cannot do. The path is walked from the strongest penalty down;
    we keep the densest support that still has <= max_features (and walk
    further only if min_features hasn't been reached).

    cross_sectional=True z-scores features AND target within each date first:
    the market-level component of both disappears, so the path selects what
    explains the cross-sectional ordering -- the thing a ranking head is for.
    """
    from sklearn.linear_model import enet_path

    max_features = max_features or config.LASSO_MAX_FEATURES
    min_features = min_features or config.LASSO_MIN_FEATURES
    l1_ratio = l1_ratio or config.LASSO_L1_RATIO

    X, y = df_train[features], df_train[target]
    if cross_sectional:
        g = df_train.groupby('Date')
        Xz = (X - g[features].transform('mean')) / (g[features].transform('std') + EPS)
        yz = (y - g[target].transform('mean')) / (g[target].transform('std') + EPS)
    else:
        Xz = (X - X.mean()) / (X.std() + EPS)
        yz = (y - y.mean()) / (y.std() + EPS)
    # 0 = the (per-date or global) mean: a neutral value, honest for "unknown".
    Xz = Xz.fillna(0.0).values
    yz = yz.fillna(0.0).values

    alphas, coefs, _ = enet_path(Xz, yz, l1_ratio=l1_ratio, n_alphas=50, eps=3e-3)
    support = np.abs(coefs) > 1e-10          # (n_features, n_alphas)
    counts = support.sum(axis=0)             # path runs large alpha -> small

    within = np.where(counts <= max_features)[0]
    idx = within[-1] if len(within) else 0
    if counts[idx] < min_features:
        reach = np.where(counts >= min_features)[0]
        idx = reach[0] if len(reach) else len(alphas) - 1

    return sorted(f for f, keep in zip(features, support[:, idx]) if keep)


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


def signal_health(wf_preds, pred_col='prediction', window=None,
                  min_periods=None, embargo=None):
    """
    Rolling t-stat of the selection signal's daily cross-sectional IC.

    The IC at day d uses the 7d label, observable only at d+7 -- so the
    series is shifted by `embargo` bars and the value AT day d is fully
    point-in-time. Returns a frame [ic, roll_t, multiplier]; multiplier is
    HEALTH_SCALE while roll_t < HEALTH_T_THRESHOLD, 1.0 otherwise (and 1.0
    while the window is still warming up -- the monitor abstains).
    """
    window = window or config.HEALTH_WINDOW
    min_periods = min_periods or config.HEALTH_MIN_PERIODS
    embargo = embargo if embargo is not None else config.HEALTH_EMBARGO

    ics = {}
    for d, g in wf_preds.groupby('Date'):
        if g[pred_col].nunique() > 1 and g['target_ret_7d'].nunique() > 1:
            ics[d] = g[pred_col].rank().corr(g['target_ret_7d'].rank())
    ic = pd.Series(ics).sort_index()
    n = ic.rolling(window, min_periods=min_periods).count()
    mu = ic.rolling(window, min_periods=min_periods).mean()
    sd = ic.rolling(window, min_periods=min_periods).std()
    t = (mu / (sd / np.sqrt(n))).shift(embargo)
    mult = t.apply(lambda v: 1.0 if (pd.isna(v) or v >= config.HEALTH_T_THRESHOLD)
                   else config.HEALTH_SCALE)
    return pd.DataFrame({'ic': ic, 'roll_t': t, 'multiplier': mult})


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
                 params=None, verbose=True, selector=None, daily_ic=None):
    """
    Embargoed rolling walk-forward for one head.

    A training row dated t carries a label spanning (t, t+horizon], so the last
    training bar must sit `embargo` bars before the test bar. Shrinking the
    embargo is the single easiest way to fabricate a good score here.

    With a feature selector on (config.FEATURE_SELECTOR, default 'lasso'),
    every fold re-selects features on ITS OWN training window -- the selector
    sees exactly what the fit sees, never the test period. For selector='ic',
    pass a precomputed `daily_ic` (from daily_feature_ic) to avoid recomputing
    it across repeated calls; 'lasso' runs per fold by construction.
    """
    head = HEADS[head_name]
    wf_step = wf_step or config.WF_STEP
    train_window = train_window or config.TRAIN_WINDOW
    if selector is None:
        selector = config.FEATURE_SELECTOR or head.get('selector', 'off')
    params = params if params is not None else config.head_params(head_name)
    embargo = head['embargo']
    target = head['target']

    df = panel.reset_index()
    if features is not None:
        features = list(features)
    else:
        features = [c for c in df.columns if c not in set(config.NON_FEATURE_COLS)]
        excluded = set(head.get('exclude') or [])
        features = [c for c in features if c not in excluded]

    df = df.dropna(subset=[target]).sort_values(['Date', 'Symbol'])
    dates = np.array(sorted(df['Date'].unique()))
    start = train_window + embargo
    if start >= len(dates):
        raise ValueError(f"[{head_name}] need > {start} dates, panel has {len(dates)}")

    if selector == 'ic' and daily_ic is None:
        daily_ic = daily_feature_ic(df, features, target)

    folds, n_feats_used = [], []
    for i in range(start, len(dates), wf_step):
        train_dates = dates[i - embargo - train_window: i - embargo]
        test_dates = dates[i: i + wf_step]

        tr = df[df['Date'].isin(train_dates)]
        te = df[df['Date'].isin(test_dates)].copy()
        if tr.empty or te.empty or tr[target].nunique() < 2:
            continue

        fold_feats = features
        if selector == 'ic':
            fold_feats, _ = select_features_ic(daily_ic, train_dates)
        elif selector == 'lasso':
            fold_feats = select_features_lasso(tr, features, target,
                                               cross_sectional=head.get('cs', False))
        n_feats_used.append(len(fold_feats))

        model = _fit_head_model(head, params, tr[fold_feats], tr[target])
        if isinstance(model, EnsembleRegressor):
            mean, std = model.predict_stats(te[fold_feats])
            te['prediction'] = mean
            te['prediction_std'] = std
            te['prediction_tstat'] = mean / (std + EPS)
            te['prediction_lcb'] = mean - std
        else:
            te['prediction'] = _predict(model, te[fold_feats], head['task'])
        folds.append(te)

    if not folds:
        raise ValueError(f"[{head_name}] no folds produced")

    preds = pd.concat(folds).sort_values(['Date', 'Symbol']).reset_index(drop=True)
    metrics = score_head(preds, head)
    if selector != 'off':
        metrics['n_features_mean'] = float(np.mean(n_feats_used))
    if verbose:
        extra = f", {np.mean(n_feats_used):.0f}/{len(features)} feats" if selector != 'off' else ""
        print(f"  {format_metrics(head_name, metrics)}  ({len(folds)} folds, refit/{wf_step}d{extra})")
    return preds, metrics


def fit_production(panel, head_name, features=None, params=None, save=True,
                   conformal=None, wf_metrics=None, selector=None, health=None):
    """
    Fits on all available history and persists a bundle with its metadata.

    Feature selection here uses all history -- the production model only ever
    predicts bars after its training window, so nothing leaks.
    """
    head = HEADS[head_name]
    target = head['target']
    if selector is None:
        selector = config.FEATURE_SELECTOR or head.get('selector', 'off')
    params = params if params is not None else config.head_params(head_name)

    df = panel.reset_index()
    if features is not None:
        features = list(features)
    else:
        features = [c for c in df.columns if c not in set(config.NON_FEATURE_COLS)]
        excluded = set(head.get('exclude') or [])
        features = [c for c in features if c not in excluded]
    df = df.dropna(subset=[target])

    n_features_raw = len(features)
    if selector == 'ic':
        daily_ic = daily_feature_ic(df, features, target)
        features, _ = select_features_ic(daily_ic, daily_ic.index)
    elif selector == 'lasso':
        features = select_features_lasso(df, features, target,
                                         cross_sectional=head.get('cs', False))

    model = _fit_head_model(head, params, df[features], df[target])

    booster_owner = model.models[0] if isinstance(model, EnsembleRegressor) else model
    importances = pd.DataFrame({
        'Feature': features,
        'Gain': booster_owner.booster_.feature_importance('gain'),
        'Split': booster_owner.booster_.feature_importance('split'),
    }).sort_values('Gain', ascending=False).reset_index(drop=True)

    bundle = {
        'model': model, 'head': head_name, 'task': head['task'],
        'target': target, 'horizon_days': head['horizon'], 'embargo': head['embargo'],
        'alpha': head.get('alpha'), 'features': features,
        'selector': selector, 'n_features_raw': n_features_raw,
        'lgb_params': dict(params),
        'trained_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'train_rows': int(len(df)),
        'train_end': pd.Timestamp(df['Date'].max()).isoformat(),
        'n_symbols': int(df['Symbol'].nunique()),
        'conformal': conformal,
        'conformal_delta': (conformal or {}).get('delta', 0.0),
        'health': health,
        'wf_metrics': wf_metrics,
    }

    # A refit that skipped validation must not silently drop a calibration or
    # health reading that an earlier validated run established.
    if (conformal is None and head['task'] == 'quantile') or \
            (health is None and head_name == 'selection'):
        existing = config.model_path(head_name)
        if os.path.exists(existing):
            try:
                prev = joblib.load(existing)
                if conformal is None and head['task'] == 'quantile':
                    bundle['conformal'] = prev.get('conformal')
                    bundle['conformal_delta'] = prev.get('conformal_delta', 0.0)
                if health is None and head_name == 'selection':
                    bundle['health'] = prev.get('health')
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

        health = None
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
            if name == 'selection' and config.USE_SIGNAL_HEALTH:
                h = signal_health(preds)
                last = h.dropna(subset=['roll_t']).iloc[-1] if h['roll_t'].notna().any() else None
                health = {'as_of': str(h.index[-1].date()),
                          'roll_t': float(last['roll_t']) if last is not None else None,
                          'multiplier': float(last['multiplier']) if last is not None else 1.0}
                if verbose:
                    print(f"      signal health: rolling t {health['roll_t']:+.2f} "
                          f"-> exposure x{health['multiplier']:.2f}")

        bundle, importances = fit_production(panel, name, features=features,
                                             params=params, conformal=conformal,
                                             wf_metrics=metrics, health=health)
        results[name] = {'wf_predictions': preds, 'wf_metrics': metrics,
                         'conformal': conformal, 'bundle': bundle,
                         'importances': importances}
    return results


# --- market head: date-level P(market up tomorrow) --------------------------

MARKET_FEATURE_PREFIXES = ('btc_', 'eth_', 'mkt_', 'dow_')


def build_market_frame(panel):
    """
    One row per date: the date-broadcast features (BTC/ETH leader state,
    mkt_* aggregates, calendar) plus BTC/ETH's own 1d returns, labelled with
    whether the equal-weight market closes up TOMORROW.

    This is a time-series problem (~1000 rows), not a cross-sectional one --
    hence its own tiny, heavily regularized model (config.MARKET_PARAMS).
    """
    p = panel.reset_index()
    date_cols = [c for c in p.columns if c.startswith(MARKET_FEATURE_PREFIXES)]
    frame = p.groupby('Date')[date_cols].first()

    for tag, sym in _factors.LEADERS.items():
        if sym in p['Symbol'].values:
            s = p[p['Symbol'] == sym].set_index('Date')
            frame[f'{tag}_ret_1d'] = s['ret_1d']
            frame[f'{tag}_close_loc_14d'] = s['close_loc_14d']

    mkt_next = p.groupby('Date')['target_ret_1d'].mean()
    frame['mkt_ret_next_1d'] = mkt_next
    frame['mkt_up_next_1d'] = (mkt_next > 0).astype(float)
    frame.loc[mkt_next.isna(), 'mkt_up_next_1d'] = np.nan
    return frame


def market_feature_cols(frame):
    return [c for c in frame.columns if c not in ('mkt_ret_next_1d', 'mkt_up_next_1d')]


def walk_forward_market(panel, wf_step=None, train_window=None, verbose=True):
    """Embargoed walk-forward for the market head (embargo=1, like timing)."""
    wf_step = wf_step or config.WF_STEP
    train_window = train_window or config.TRAIN_WINDOW
    frame = build_market_frame(panel).dropna(subset=['mkt_up_next_1d'])
    feats = market_feature_cols(frame)
    dates = frame.index.to_numpy()

    folds = []
    for i in range(train_window + 1, len(dates), wf_step):
        tr = frame.iloc[i - 1 - train_window: i - 1]
        te = frame.iloc[i: i + wf_step].copy()
        if tr.empty or te.empty or tr['mkt_up_next_1d'].nunique() < 2:
            continue
        m = LGBMClassifier(objective='binary', **config.MARKET_PARAMS)
        m.fit(tr[feats], tr['mkt_up_next_1d'])
        te['prediction'] = m.predict_proba(te[feats])[:, 1]
        folds.append(te)

    preds = pd.concat(folds)
    y, pr = preds['mkt_up_next_1d'].values, preds['prediction'].values
    out = {
        'n': int(len(preds)),
        'auc': float(roc_auc_score(y, pr)) if len(np.unique(y)) > 1 else np.nan,
        'brier': float(brier_score_loss(y, pr)),
        'base_rate': float(np.mean(y)), 'mean_pred': float(np.mean(pr)),
    }
    if verbose:
        print(f"  [market] AUC {out['auc']:.4f} | Brier {out['brier']:.4f} | "
              f"base {out['base_rate']:.3f} vs pred {out['mean_pred']:.3f} | "
              f"n={out['n']:,}  ({len(folds)} folds)")
    return preds, out


def fit_market_calibration(wf_preds):
    """
    Platt scaling on the walk-forward predictions: p' = sigmoid(a*logit(p)+b).

    The raw head is ANTI-calibrated at the tails (its confident calls were no
    better than the base rate), so this 2-parameter map compresses the output
    toward honesty -- the same philosophy as the range heads' conformal step.
    """
    from sklearn.linear_model import LogisticRegression
    p = np.clip(wf_preds['prediction'].values, 1e-4, 1 - 1e-4)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    y = wf_preds['mkt_up_next_1d'].values
    lr = LogisticRegression(C=1e6)
    lr.fit(X, y)
    return {'a': float(lr.coef_[0, 0]), 'b': float(lr.intercept_[0])}


def apply_market_calibration(prob, calib):
    if not calib:
        return prob
    p = np.clip(prob, 1e-4, 1 - 1e-4)
    z = calib['a'] * np.log(p / (1 - p)) + calib['b']
    return 1.0 / (1.0 + np.exp(-z))


def fit_market(panel, save=True, wf_metrics=None, wf_preds=None):
    """Production market head: fit on all history, persist, return the bundle."""
    frame = build_market_frame(panel)
    fit = frame.dropna(subset=['mkt_up_next_1d'])
    feats = market_feature_cols(frame)
    m = LGBMClassifier(objective='binary', **config.MARKET_PARAMS)
    m.fit(fit[feats], fit['mkt_up_next_1d'])

    calib = fit_market_calibration(wf_preds) if wf_preds is not None else None
    bundle = {'model': m, 'head': 'market', 'task': 'binary',
              'target': 'mkt_up_next_1d', 'features': feats,
              'lgb_params': dict(config.MARKET_PARAMS), 'calib': calib,
              'trained_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'train_rows': int(len(fit)), 'wf_metrics': wf_metrics,
              'train_end': pd.Timestamp(fit.index.max()).isoformat()}
    # A nightly refit without validation must not drop an existing calibration.
    if calib is None:
        existing = str(config.model_path('market'))
        if os.path.exists(existing):
            try:
                bundle['calib'] = joblib.load(existing).get('calib')
            except Exception:
                pass
    if save:
        path = str(config.model_path('market'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp'
        joblib.dump(bundle, tmp)
        os.replace(tmp, path)
        bundle['path'] = path
    return bundle


def market_forecast(panel, bundle=None):
    """P(equal-weight market up tomorrow) from the latest bar."""
    if bundle is None:
        path = str(config.model_path('market'))
        if not os.path.exists(path):
            return None
        bundle = joblib.load(path)
    frame = build_market_frame(panel)
    last = frame.iloc[[-1]]
    raw = float(bundle['model'].predict_proba(last[bundle['features']])[0, 1])
    prob = float(apply_market_calibration(raw, bundle.get('calib')))
    return {'as_of': frame.index[-1], 'prob_up': prob, 'prob_up_raw': raw,
            'calibrated': bundle.get('calib') is not None,
            'wf_metrics': bundle.get('wf_metrics')}


# --- regime head: date-level P(market up over the next 7 days) ---------------

REGIME_LEADERS = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT'}
REGIME_LEADER_COLS = ['ret_1d', 'ret_7d', 'ret_30d', 'vol_14d', 'rsi_14d',
                      'funding_mean_7d', 'sma_ratio_50d', 'close_loc_14d',
                      'breakout_up_20d', 'macd_hist', 'basis_mean_7d']


def build_regime_frame(panel):
    """
    One row per date: BTC/ETH/SOL single-asset states (pulled straight from
    their panel rows) plus the mkt_*/calendar aggregates, labelled with
    whether the equal-weight market closes up over the NEXT 7 DAYS.
    """
    p = panel.reset_index()
    date_cols = [c for c in p.columns if c.startswith(('mkt_', 'dow_'))]
    frame = p.groupby('Date')[date_cols].first()

    for tag, sym in REGIME_LEADERS.items():
        s = p[p['Symbol'] == sym].set_index('Date')
        for col in REGIME_LEADER_COLS:
            if col in s:
                frame[f'{tag}_{col}'] = s[col]

    fwd = p.groupby('Date')['target_ret_7d'].mean()
    frame['mkt_ret_next_7d'] = fwd
    frame['mkt_up_next_7d'] = (fwd > 0).astype(float)
    frame.loc[fwd.isna(), 'mkt_up_next_7d'] = np.nan
    return frame


def regime_feature_cols(frame):
    return [c for c in frame.columns if c not in ('mkt_ret_next_7d', 'mkt_up_next_7d')]


def _monday_rows(frame):
    return frame[frame.index.dayofweek == 0]


def walk_forward_regime(panel, train_weeks=None, verbose=True, **_):
    """
    MONDAYS ONLY: the head is fitted and evaluated on Monday rows, whose 7d
    labels run Monday-to-Monday and therefore do NOT overlap -- every test
    point is independent. Weekly refit (the model is tiny), 1-Monday embargo.
    """
    train_weeks = train_weeks or config.REGIME_TRAIN_WEEKS
    frame = build_regime_frame(panel).dropna(subset=['mkt_up_next_7d'])
    mon = _monday_rows(frame)
    feats = regime_feature_cols(frame)

    folds = []
    for i in range(train_weeks + 1, len(mon)):
        tr = mon.iloc[i - 1 - train_weeks: i - 1]
        te = mon.iloc[[i]].copy()
        if tr['mkt_up_next_7d'].nunique() < 2:
            continue
        m = LGBMClassifier(objective='binary', **config.REGIME_PARAMS)
        m.fit(tr[feats], tr['mkt_up_next_7d'])
        te['prediction'] = m.predict_proba(te[feats])[:, 1]
        folds.append(te)

    preds = pd.concat(folds)
    y, pr = preds['mkt_up_next_7d'].values, preds['prediction'].values
    out = {'n': int(len(preds)),          # independent Mondays
           'auc': float(roc_auc_score(y, pr)) if len(np.unique(y)) > 1 else np.nan,
           'brier': float(brier_score_loss(y, pr)),
           'base_rate': float(np.mean(y)), 'mean_pred': float(np.mean(pr))}
    if verbose:
        print(f"  [regime-weekly] AUC {out['auc']:.4f} | Brier {out['brier']:.4f} | "
              f"base {out['base_rate']:.3f} vs pred {out['mean_pred']:.3f} | "
              f"n={out['n']} independent Mondays")
    return preds, out


def fit_regime(panel, save=True, wf_metrics=None, wf_preds=None):
    """Production regime head with Platt calibration from the walk-forward."""
    frame = build_regime_frame(panel)
    fit = _monday_rows(frame.dropna(subset=['mkt_up_next_7d']))
    feats = regime_feature_cols(frame)
    m = LGBMClassifier(objective='binary', **config.REGIME_PARAMS)
    m.fit(fit[feats], fit['mkt_up_next_7d'])

    calib = None
    if wf_preds is not None:
        calib = fit_market_calibration(
            wf_preds.rename(columns={'mkt_up_next_7d': 'mkt_up_next_1d'}))
    base_rate = float((wf_metrics or {}).get('base_rate',
                                             fit['mkt_up_next_7d'].mean()))
    bundle = {'model': m, 'head': 'regime', 'task': 'binary',
              'target': 'mkt_up_next_7d', 'features': feats, 'calib': calib,
              'base_rate': base_rate,
              'lgb_params': dict(config.REGIME_PARAMS),
              'trained_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'train_rows': int(len(fit)), 'wf_metrics': wf_metrics,
              'train_end': pd.Timestamp(fit.index.max()).isoformat()}
    if calib is None:
        existing = str(config.model_path('regime'))
        if os.path.exists(existing):
            try:
                bundle['calib'] = joblib.load(existing).get('calib')
            except Exception:
                pass
    if save:
        path = str(config.model_path('regime'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp'
        joblib.dump(bundle, tmp)
        os.replace(tmp, path)
        bundle['path'] = path
    return bundle


def regime_forecast(panel, bundle=None):
    """
    The week's stance: predicted from the MOST RECENT MONDAY's features and
    held fixed until the next Monday, as the weekly prior the cascade uses.
    """
    if bundle is None:
        path = str(config.model_path('regime'))
        if not os.path.exists(path):
            return None
        bundle = joblib.load(path)
    frame = build_regime_frame(panel)
    mon = _monday_rows(frame)
    if mon.empty:
        return None
    row = mon.iloc[[-1]]
    raw = float(bundle['model'].predict_proba(row[bundle['features']])[0, 1])
    prob = float(apply_market_calibration(raw, bundle.get('calib')))
    # The stance compares against the BASE RATE, not 0.5: in a history where
    # most weeks were down, a calibrated output below 50% is normal, not a
    # bear call. Within the neutral margin the honest answer is "no view".
    base = float(bundle.get('base_rate', 0.5))
    edge = prob - base
    if abs(edge) < config.REGIME_NEUTRAL_MARGIN:
        stance = 'NEUTRAL'
    else:
        stance = 'LONG' if edge > 0 else 'SHORT'
    return {'as_of': frame.index[-1], 'based_on_monday': mon.index[-1],
            'prob_up': prob, 'prob_up_raw': raw, 'base_rate': base,
            'edge': edge, 'stance': stance,
            'calibrated': bundle.get('calib') is not None,
            'wf_metrics': bundle.get('wf_metrics')}


# --- hyperparameter tuning --------------------------------------------------

TUNE_SPACE = {
    'num_leaves': [15, 31, 63],
    'max_depth': [4, 6, 8],
    'min_child_samples': [30, 50, 100, 150],
    'learning_rate': [0.02, 0.03, 0.05],
    'colsample_bytree': [0.2, 0.3, 0.4, 0.6],
    'subsample': [0.7, 0.8, 0.9],
    'reg_alpha': [0.0, 0.1, 0.3, 1.0],
    'reg_lambda': [0.5, 1.0, 3.0, 10.0],
}

# What "better" means per task. Selection is judged on ranking, not MSE --
# tuning it on MSE would reward predicting the market level. Timing is judged
# on Brier, not AUC: its deliverable is an HONEST probability (README), and an
# AUC-picked config traded calibration away for ordering it barely has.
TUNE_METRIC = {
    'regression': ('rank_ic', +1),
    'binary':     ('brier', -1),
    'quantile':   ('pinball', -1),
}


def tune_head(panel, head_name, n_trials=20, wf_step=None, seed=42, verbose=True):
    """
    Random search over TUNE_SPACE, each trial scored with the embargoed
    walk-forward (IC filter on, so tuning sees the same pipeline production
    uses). Trial 0 is always the current config as the baseline to beat.

    wf_step defaults to 40 for speed; confirm the winner at wf_step=10 before
    trusting it. Returns (best_params, trials DataFrame sorted best-first).
    """
    head = HEADS[head_name]
    wf_step = wf_step or 40
    metric, sign = TUNE_METRIC[head['task']]
    rng = np.random.default_rng(seed)

    df = panel.reset_index().dropna(subset=[head['target']])
    features = [c for c in df.columns if c not in set(config.NON_FEATURE_COLS)]
    sel = config.FEATURE_SELECTOR or head.get('selector', 'off')
    daily_ic = daily_feature_ic(df, features, head['target']) if sel == 'ic' else None

    # learning_rate and n_estimators move together: halve one, double the other.
    def _n_estimators(lr):
        return int(round(300 * 0.03 / lr / 50) * 50)

    trials, seen = [], set()
    base = config.head_params(head_name)
    candidates = [dict(base)]
    while len(candidates) < n_trials:
        p = dict(base)
        for k, vals in TUNE_SPACE.items():
            p[k] = vals[rng.integers(len(vals))]
        p['num_leaves'] = min(p['num_leaves'], 2 ** p['max_depth'] - 1)
        p['n_estimators'] = _n_estimators(p['learning_rate'])
        key = tuple(sorted((k, v) for k, v in p.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    for i, p in enumerate(candidates):
        _, m = walk_forward(panel, head_name, wf_step=wf_step, params=p,
                            verbose=False, daily_ic=daily_ic)
        row = {k: p[k] for k in TUNE_SPACE}
        row['n_estimators'] = p['n_estimators']
        row['score'] = m.get(metric, np.nan)
        trials.append(row)
        if verbose:
            tag = 'baseline' if i == 0 else f'trial {i:>2}'
            print(f"  [{head_name}] {tag}: {metric} {row['score']:+.4f}")

    out = pd.DataFrame(trials).sort_values('score', ascending=(sign < 0)).reset_index(drop=True)
    best = out.iloc[0]
    best_params = dict(config.LGB_PARAMS)
    best_params.update({k: best[k] for k in TUNE_SPACE})
    best_params['n_estimators'] = int(best['n_estimators'])
    for k in ('num_leaves', 'max_depth', 'min_child_samples'):
        best_params[k] = int(best_params[k])
    if verbose:
        print(f"  [{head_name}] best {metric}: {best['score']:+.4f} "
              f"(baseline {trials[0]['score']:+.4f})")
    return best_params, out


def load_head(head_name):
    path = str(config.model_path(head_name))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained '{head_name}' model at {path}. Run `python main.py train` first.")
    return joblib.load(path)

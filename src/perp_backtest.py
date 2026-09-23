"""
Daily-rebalanced backtest for USDT-margined perpetuals.

Differences from a spot backtest that matter:

  * FUNDING IS REAL PnL. A long perp pays funding when the rate is positive.
    At 0.01%/8h (the Binance default) that is ~11%/year bled from a long book --
    far too large to leave out. It is charged on the position actually held.
  * The universe is UNBALANCED: symbols enter as they are listed. Each date is
    ranked against whatever was tradeable on that date.

Position rule: rank by the 7d selection score, take the top N, then keep only
those whose timing probability clears the threshold. Everything else is cash.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def build_signal_frame(results):
    """Merges every head's walk-forward predictions onto one daily grid."""
    sel_preds = results['selection']['wf_predictions']
    # Rank by the ensemble's confidence-adjusted score when available:
    # t-stat ranking beat the raw mean +0.031 -> +0.038 RankIC at wf40.
    conf_col = {'tstat': 'prediction_tstat', 'lcb': 'prediction_lcb'}.get(config.CONF_RANKING)
    score_col = conf_col if conf_col and conf_col in sel_preds.columns else 'prediction'
    keep = ['Date', 'Symbol', score_col] + \
        (['target_ret_7d'] if 'target_ret_7d' in sel_preds.columns else [])
    sel = sel_preds[keep].rename(columns={score_col: 'sel_score'})
    tim = results['timing']['wf_predictions'][
        ['Date', 'Symbol', 'prediction', 'target_ret_1d', 'funding_next_1d',
         'target_net_1d', 'Close']].rename(columns={'prediction': 'prob_up'})

    df = sel.merge(tim, on=['Date', 'Symbol'], how='inner')

    for head in ('range_high', 'range_low'):
        if results.get(head) and results[head].get('wf_predictions') is not None:
            col = 'prediction_calibrated' if 'prediction_calibrated' in \
                results[head]['wf_predictions'].columns else 'prediction'
            part = results[head]['wf_predictions'][['Date', 'Symbol', col]].rename(
                columns={col: head})
            df = df.merge(part, on=['Date', 'Symbol'], how='left')

    return df.sort_values(['Date', 'Symbol']).reset_index(drop=True)


# --- position rules --------------------------------------------------------

def _equal(held):
    return {s: 1.0 / len(held) for s in held} if held else {}


def rule_selection_hysteresis(grp, prev_weights, top_n, exit_rank_mult=2,
                              max_entries=None, **_):
    """
    DEFAULT rule. Enter the top N; hold until the name drops out of the top
    N*exit_rank_mult.

    Rebalancing to an exact top-N every single day costs ~0.6 turnover/day. The
    ranking is not that precise -- a name sitting at rank 9 today and 7
    tomorrow carries no real signal, but round-tripping it costs real fees.
    The band cuts turnover ~60% with no systematic loss of return.

    At most `max_entries` NEW names enter per day (exits are never throttled);
    each held name is weighted 1/top_n, so an under-filled book holds the
    remainder in cash instead of concentrating.
    """
    max_entries = config.MAX_ENTRIES_PER_DAY if max_entries is None else max_entries
    ranked = grp.sort_values('sel_score', ascending=False)['Symbol'].tolist()
    entries = ranked[:top_n]
    keep_zone = set(ranked[:top_n * exit_rank_mult])

    held = [s for s in (prev_weights or {}) if s in keep_zone]
    adds = 0
    for s in entries:
        if s not in held and len(held) < top_n and adds < max_entries:
            held.append(s)
            adds += 1
    return {s: 1.0 / top_n for s in held[:top_n]}


def rule_selection_timing(grp, prev_weights, top_n, prob_threshold, **_):
    cand = grp.nlargest(top_n, 'sel_score')
    return _equal(cand[cand['prob_up'] > prob_threshold]['Symbol'].tolist())


def rule_selection_only(grp, prev_weights, top_n, **_):
    return _equal(grp.nlargest(top_n, 'sel_score')['Symbol'].tolist())


def rule_timing_only(grp, prev_weights, top_n, prob_threshold, **_):
    cand = grp.nlargest(top_n, 'prob_up')
    return _equal(cand[cand['prob_up'] > prob_threshold]['Symbol'].tolist())


def rule_equal_weight(grp, prev_weights, **_):
    return _equal(grp['Symbol'].tolist())


def rule_long_short(grp, prev_weights, top_n, exit_rank_mult=None, gross=1.0, **_):
    """
    Long the top N, short the bottom N of the selection ranking, equal weight
    per side, dollar-neutral. `gross` is total exposure (1.0 = 50% long + 50%
    short). With exit_rank_mult set, hysteresis applies symmetrically: a long
    survives while it stays in the top N*mult, a short while it stays in the
    bottom N*mult.

    The engine handles signed weights natively: a short's price PnL is
    -w*ret, and its funding flow flips sign too -- a short RECEIVES positive
    funding, which matters here because the ranking's favourite longs tend to
    have negative funding (longs get paid) while bottom names skew positive.
    """
    ranked = grp.sort_values('sel_score', ascending=False)['Symbol'].tolist()
    if len(ranked) < 2 * top_n:
        top_n = len(ranked) // 2
    if top_n == 0:
        return {}
    per_side = gross / 2.0

    if exit_rank_mult:
        keep_long = set(ranked[:top_n * exit_rank_mult])
        keep_short = set(ranked[-top_n * exit_rank_mult:])
        longs = [s for s, w in (prev_weights or {}).items() if w > 0 and s in keep_long]
        for s in ranked[:top_n]:
            if s not in longs and len(longs) < top_n:
                longs.append(s)
        longs = longs[:top_n]
        long_set = set(longs)
        shorts = [s for s, w in (prev_weights or {}).items()
                  if w < 0 and s in keep_short and s not in long_set]
        for s in reversed(ranked[-top_n:]):
            if s not in shorts and s not in long_set and len(shorts) < top_n:
                shorts.append(s)
        shorts = shorts[:top_n]
    else:
        longs = ranked[:top_n]
        shorts = ranked[-top_n:]

    w = {s: per_side / len(longs) for s in longs}
    w.update({s: -per_side / len(shorts) for s in shorts})
    return w


def rule_buy_and_hold(symbol):
    def _rule(grp, prev_weights, **_):
        return {symbol: 1.0} if symbol in set(grp['Symbol']) else {}
    return _rule


# --- engine ----------------------------------------------------------------

def simulate(signal_df, rule, cost_bps=None, charge_funding=True, exposure=None,
             **rule_kwargs):
    """`exposure` is an optional {date: multiplier} map (signal-health sizing)
    applied to the rule's weights before accounting."""
    cost_bps = config.COST_BPS if cost_bps is None else cost_bps
    prev_w = {}
    rows = []

    for date, grp in signal_df.groupby('Date', sort=True):
        w = rule(grp, prev_weights=prev_w, **rule_kwargs)
        if exposure is not None:
            m = exposure.get(date, 1.0)
            w = {s: wt * m for s, wt in w.items()}
        price_ret = dict(zip(grp['Symbol'], grp['target_ret_1d']))
        funding = dict(zip(grp['Symbol'], grp['funding_next_1d'].fillna(0.0)))

        gross = sum(wt * price_ret.get(s, 0.0) for s, wt in w.items())
        fund_cost = sum(wt * funding.get(s, 0.0) for s, wt in w.items()) if charge_funding else 0.0
        turnover = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0))
                       for s in set(w) | set(prev_w))
        fee = turnover * cost_bps / 1e4
        net = gross - fund_cost - fee

        rows.append({'Date': date, 'gross_return': gross, 'funding_cost': fund_cost,
                     'fee': fee, 'net_return': net, 'turnover': turnover,
                     'n_held': len(w), 'held': ','.join(sorted(w))})
        prev_w = w

    out = pd.DataFrame(rows)
    out['equity'] = (1 + out['net_return']).cumprod()
    out['equity_gross'] = (1 + out['gross_return']).cumprod()
    return out


def metrics(daily, ann=None):
    ann = ann or config.ANNUALIZATION
    r = daily['net_return']
    total = float(daily['equity'].iloc[-1] - 1)
    years = len(r) / ann
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    sd = r.std(ddof=1)
    dd = daily['equity'] / daily['equity'].cummax() - 1
    return {
        'Total Return': total,
        'CAGR': cagr,
        'Ann. Vol': float(sd * np.sqrt(ann)),
        'Sharpe': float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan,
        'Max Drawdown': float(dd.min()),
        'Win Rate': float((r > 0).mean()),
        'Avg Turnover': float(daily['turnover'].mean()),
        'Days Invested': float((daily['n_held'] > 0).mean()),
        'Funding Drag': float(daily['funding_cost'].sum()),
        'Fee Drag': float(daily['fee'].sum()),
        'Days': len(r),
    }


def run_all(signal_df, top_n=None, prob_threshold=None, cost_bps=None,
            benchmark=None, charge_funding=True, **kwargs):
    top_n = top_n or config.TOP_N
    prob_threshold = config.PROB_THRESHOLD if prob_threshold is None else prob_threshold
    benchmark = benchmark or config.BENCHMARK_SYMBOL

    exit_mult = kwargs.get('exit_rank_mult', config.EXIT_RANK_MULT)
    specs = {
        'Selection + Hysteresis': (rule_selection_hysteresis,
                                   dict(top_n=top_n, exit_rank_mult=exit_mult)),
        'Selection (daily rebal)': (rule_selection_only, dict(top_n=top_n)),
        'L/S 50-50 + Hysteresis': (rule_long_short,
                                   dict(top_n=top_n, exit_rank_mult=exit_mult, gross=1.0)),
        'L/S 50-50 (daily)':      (rule_long_short, dict(top_n=top_n, gross=1.0)),
        'L/S 100-100 + Hyst.':    (rule_long_short,
                                   dict(top_n=top_n, exit_rank_mult=exit_mult, gross=2.0)),
        'Selection + Timing gate': (rule_selection_timing,
                                    dict(top_n=top_n, prob_threshold=prob_threshold)),
        'Timing only':            (rule_timing_only, dict(top_n=top_n, prob_threshold=prob_threshold)),
        'Equal-Weight (all)':     (rule_equal_weight, {}),
        f'{benchmark} Buy&Hold':  (rule_buy_and_hold(benchmark), {}),
    }
    curves = {label: simulate(signal_df, rule, cost_bps=cost_bps,
                              charge_funding=charge_funding, **kw)
              for label, (rule, kw) in specs.items()}

    # Signal-health sizing on the default rule (needs the realized 7d label).
    if config.USE_SIGNAL_HEALTH and 'target_ret_7d' in signal_df.columns:
        import models as _M
        h = _M.signal_health(signal_df, pred_col='sel_score')
        curves['Sel + Hyst + Health'] = simulate(
            signal_df, rule_selection_hysteresis, cost_bps=cost_bps,
            charge_funding=charge_funding, exposure=h['multiplier'].to_dict(),
            top_n=top_n, exit_rank_mult=exit_mult)

    summary = pd.DataFrame({k: metrics(v) for k, v in curves.items()}).T
    return summary, curves


def format_summary(summary):
    out = summary.copy()
    for c in ['Total Return', 'CAGR', 'Ann. Vol', 'Max Drawdown', 'Win Rate',
              'Days Invested', 'Funding Drag', 'Fee Drag']:
        out[c] = out[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else 'n/a')
    for c in ['Sharpe', 'Avg Turnover']:
        out[c] = out[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else 'n/a')
    out['Days'] = out['Days'].astype(int)
    return out


def plot_curves(curves, save_path=None, show=False, title=None):
    styles = {
        'Selection + Hysteresis':  dict(color='darkgreen', lw=2.5),
        'Sel + Hyst + Health':     dict(color='black', lw=2.0),
        'Selection (daily rebal)': dict(color='seagreen', lw=1.5, ls='--'),
        'L/S 50-50 + Hysteresis':  dict(color='darkorange', lw=2.0),
        'L/S 50-50 (daily)':       dict(color='goldenrod', lw=1.3, ls='--'),
        'L/S 100-100 + Hyst.':     dict(color='sienna', lw=1.3, ls=':'),
        'Selection + Timing gate': dict(color='crimson', lw=1.5),
        'Timing only':             dict(color='purple', lw=1.2, ls='-.'),
        'Equal-Weight (all)':      dict(color='royalblue', lw=1.8, ls='--'),
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    for label, df in curves.items():
        st = styles.get(label, dict(color='orange', lw=1.8, ls=':'))
        ax1.plot(df['Date'], (df['equity'] - 1) * 100, label=label, **st)
        ax2.plot(df['Date'], (df['equity'] / df['equity'].cummax() - 1) * 100, **st)

    ax1.set_title(title or 'Binance perps: 7d selection x 1d timing '
                           f'({config.COST_BPS:.0f}bps + funding, daily rebalance)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Net Return (%)')
    ax1.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax1.grid(True, ls=':', alpha=0.6); ax1.legend(fontsize=10, loc='upper left')
    ax2.set_ylabel('Drawdown (%)'); ax2.set_xlabel('Date')
    ax2.grid(True, ls=':', alpha=0.6)
    fig.tight_layout()

    if save_path:
        import os
        save_path = str(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=110)
        print(f"Backtest plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

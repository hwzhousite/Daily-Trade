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
    sel = results['selection']['wf_predictions'][
        ['Date', 'Symbol', 'prediction']].rename(columns={'prediction': 'sel_score'})
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


def rule_selection_hysteresis(grp, prev_weights, top_n, exit_rank_mult=2, **_):
    """
    DEFAULT rule. Enter the top N; hold until the name drops out of the top
    N*exit_rank_mult.

    Rebalancing to an exact top-N every single day costs ~0.6 turnover/day. The
    ranking is not that precise -- a name sitting at rank 9 today and 7
    tomorrow carries no real signal, but round-tripping it costs real fees.
    The band cuts turnover ~60% with no systematic loss of return.
    """
    ranked = grp.sort_values('sel_score', ascending=False)['Symbol'].tolist()
    entries = ranked[:top_n]
    keep_zone = set(ranked[:top_n * exit_rank_mult])

    held = [s for s in (prev_weights or {}) if s in keep_zone]
    for s in entries:
        if s not in held and len(held) < top_n:
            held.append(s)
    return _equal(held[:top_n])


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


def rule_buy_and_hold(symbol):
    def _rule(grp, prev_weights, **_):
        return {symbol: 1.0} if symbol in set(grp['Symbol']) else {}
    return _rule


# --- engine ----------------------------------------------------------------

def simulate(signal_df, rule, cost_bps=None, charge_funding=True, **rule_kwargs):
    cost_bps = config.COST_BPS if cost_bps is None else cost_bps
    prev_w = {}
    rows = []

    for date, grp in signal_df.groupby('Date', sort=True):
        w = rule(grp, prev_weights=prev_w, **rule_kwargs)
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
        'Selection + Timing gate': (rule_selection_timing,
                                    dict(top_n=top_n, prob_threshold=prob_threshold)),
        'Timing only':            (rule_timing_only, dict(top_n=top_n, prob_threshold=prob_threshold)),
        'Equal-Weight (all)':     (rule_equal_weight, {}),
        f'{benchmark} Buy&Hold':  (rule_buy_and_hold(benchmark), {}),
    }
    curves = {label: simulate(signal_df, rule, cost_bps=cost_bps,
                              charge_funding=charge_funding, **kw)
              for label, (rule, kw) in specs.items()}
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
        'Selection (daily rebal)': dict(color='seagreen', lw=1.5, ls='--'),
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

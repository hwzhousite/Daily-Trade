"""
Daily-rebalanced backtest of the two-layer strategy.

Selection (7d-supervised) ranks the universe cross-sectionally; timing
(1d-supervised) decides whether each candidate is actually held tonight. PnL is
always accounted on the realised NEXT-DAY return, because the book is rebalanced
daily -- the 7d label only supervises the ranking, it is never the PnL.

Turnover is charged at COST_BPS one-way. With daily rebalancing this is not a
rounding error: ignoring it makes any daily strategy look better than it is.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def build_signal_frame(selection_wf, timing_wf):
    """Merges both layers' walk-forward predictions onto one daily grid."""
    sel = selection_wf[['Date', 'Asset', 'predicted_return']].rename(
        columns={'predicted_return': 'sel_score'})
    tim = timing_wf[['Date', 'Asset', 'predicted_return', 'target_return']].rename(
        columns={'predicted_return': 'timing_score', 'target_return': 'ret_fwd_1d'})
    df = sel.merge(tim, on=['Date', 'Asset'], how='inner')
    return df.sort_values(['Date', 'Asset']).reset_index(drop=True)


# --- position rules --------------------------------------------------------

def rule_selection_only(grp, top_n=config.TOP_N, **_):
    """Top-N by 7d selection score, always invested (no timing gate)."""
    held = grp.nlargest(top_n, 'sel_score')['Asset'].tolist()
    return {a: 1.0 / len(held) for a in held} if held else {}


def rule_selection_timing(grp, top_n=config.TOP_N, threshold=config.TIMING_THRESHOLD, **_):
    """Top-N candidates, then keep only those the timing layer likes. Rest -> cash."""
    cand = grp.nlargest(top_n, 'sel_score')
    held = cand[cand['timing_score'] > threshold]['Asset'].tolist()
    return {a: 1.0 / len(held) for a in held} if held else {}


def rule_timing_only(grp, top_n=config.TOP_N, threshold=config.TIMING_THRESHOLD, **_):
    """Ablation: rank by the timing score alone, ignoring selection."""
    cand = grp.nlargest(top_n, 'timing_score')
    held = cand[cand['timing_score'] > threshold]['Asset'].tolist()
    return {a: 1.0 / len(held) for a in held} if held else {}


def rule_equal_weight(grp, **_):
    """Benchmark: hold the whole universe, equal weight."""
    assets = grp['Asset'].tolist()
    return {a: 1.0 / len(assets) for a in assets} if assets else {}


def rule_buy_and_hold(asset):
    def _rule(grp, **_):
        return {asset: 1.0} if asset in set(grp['Asset']) else {}
    return _rule


# --- engine ----------------------------------------------------------------

def simulate(signal_df, rule, cost_bps=config.COST_BPS, **rule_kwargs):
    """
    Walks the daily grid applying `rule`, charging turnover, and returning a
    per-day frame with gross/net returns.
    """
    prev_w = {}
    rows = []

    for date, grp in signal_df.groupby('Date', sort=True):
        w = rule(grp, **rule_kwargs)
        rets = dict(zip(grp['Asset'], grp['ret_fwd_1d']))

        gross = sum(wt * rets.get(a, 0.0) for a, wt in w.items())
        turnover = sum(abs(w.get(a, 0.0) - prev_w.get(a, 0.0))
                       for a in set(w) | set(prev_w))
        net = gross - turnover * cost_bps / 1e4

        rows.append({
            'Date': date,
            'gross_return': gross,
            'net_return': net,
            'turnover': turnover,
            'n_held': len(w),
            'held': ','.join(sorted(w)),
        })
        prev_w = w

    out = pd.DataFrame(rows)
    out['equity'] = (1 + out['net_return']).cumprod()
    out['equity_gross'] = (1 + out['gross_return']).cumprod()
    return out


def metrics(daily, ann=config.ANNUALIZATION):
    """Standard performance stats on the NET return series."""
    r = daily['net_return']
    n = len(r)
    total = float(daily['equity'].iloc[-1] - 1)
    years = n / ann
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(ann)) if r.std(ddof=1) > 0 else np.nan
    dd = daily['equity'] / daily['equity'].cummax() - 1
    return {
        'Total Return': total,
        'CAGR': cagr,
        'Ann. Vol': vol,
        'Sharpe': sharpe,
        'Max Drawdown': float(dd.min()),
        'Win Rate': float((r > 0).mean()),
        'Avg Turnover': float(daily['turnover'].mean()),
        'Days Invested': float((daily['n_held'] > 0).mean()),
        'Cost Drag': float(daily['equity_gross'].iloc[-1] - daily['equity'].iloc[-1]),
        'Days': n,
    }


def run_all(signal_df, top_n=config.TOP_N, threshold=config.TIMING_THRESHOLD,
            cost_bps=config.COST_BPS, benchmark_asset=config.BENCHMARK_ASSET):
    """
    Runs the strategy and its ablations/benchmarks.

    Returns (summary_df, curves) where curves maps a label to its daily frame.
    """
    specs = {
        'Selection + Timing': (rule_selection_timing, dict(top_n=top_n, threshold=threshold)),
        'Selection only':     (rule_selection_only, dict(top_n=top_n)),
        'Timing only':        (rule_timing_only, dict(top_n=top_n, threshold=threshold)),
        'Equal-Weight (all)': (rule_equal_weight, {}),
        f'{benchmark_asset} Buy&Hold': (rule_buy_and_hold(benchmark_asset), {}),
    }

    curves = {label: simulate(signal_df, rule, cost_bps=cost_bps, **kw)
              for label, (rule, kw) in specs.items()}

    summary = pd.DataFrame({label: metrics(df) for label, df in curves.items()}).T
    return summary, curves


def format_summary(summary):
    """Human-readable copy of the summary table."""
    out = summary.copy()
    for col in ['Total Return', 'CAGR', 'Ann. Vol', 'Max Drawdown', 'Win Rate', 'Days Invested']:
        out[col] = out[col].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
    for col in ['Sharpe', 'Avg Turnover', 'Cost Drag']:
        out[col] = out[col].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "n/a")
    out['Days'] = out['Days'].astype(int)
    return out


def plot_curves(curves, save_path=None, show=False, title=None):
    # No backend switching here: a headless run already defaults to Agg, and
    # forcing it would break `%matplotlib inline` in the notebook.
    styles = {
        'Selection + Timing': dict(color='darkgreen', lw=2.5),
        'Selection only':     dict(color='crimson', lw=1.8),
        'Timing only':        dict(color='purple', lw=1.4, ls='-.'),
        'Equal-Weight (all)': dict(color='royalblue', lw=1.8, ls='--'),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    for label, df in curves.items():
        style = styles.get(label, dict(color='orange', lw=1.8, ls=':'))
        ax1.plot(df['Date'], (df['equity'] - 1) * 100, label=label, **style)
        dd = (df['equity'] / df['equity'].cummax() - 1) * 100
        ax2.plot(df['Date'], dd, **style)

    ax1.set_title(title or 'Two-Layer Strategy: 7d Selection x 1d Timing '
                           f'(daily rebalance, {config.COST_BPS:.0f}bps one-way)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Net Return (%)')
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.axhline(0, color='black', lw=0.8, alpha=0.5)

    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
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

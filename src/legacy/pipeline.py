"""
End-to-end orchestration shared by main.py and the notebook.

Nightly flow:
    data -> factors -> train both layers -> backtest -> tonight's signals -> plan
"""
import pandas as pd

import config
import backtest as bt
from data_processing import get_crypto_data, check_freshness
from features import build_layer_panels
from model_pipeline import train_all_layers
from daily_inference import generate_signals, price_boundaries
from trading_plan import build_trading_plan, format_plan, plan_summary


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load_data(refresh=False, warn=True, allow_stale=False):
    """Loads cached (or freshly downloaded) OHLCV and checks for stale bars."""
    data = get_crypto_data(refresh=refresh, allow_stale=allow_stale)
    if warn:
        for w in check_freshness(data):
            print(f"  DATA WARNING: {w}")
    return data


def train(data=None, refresh=False, wf_step=1, verbose=True, allow_stale=False):
    """Builds both panels and walk-forward trains both layers."""
    data = load_data(refresh=refresh, allow_stale=allow_stale) if data is None else data

    banner("Building panels")
    panels = build_layer_panels(data)
    for name, panel in panels.items():
        dates = panel.index.get_level_values('Date')
        spec = config.LAYERS[name]
        print(f"  {name:10} label=fwd {spec['horizon_days']}d  rows={len(panel):5}  "
              f"bars={dates.nunique():5}  {dates.min().date()} -> {dates.max().date()}")

    banner(f"Walk-forward training (refit every {wf_step}d)")
    results = train_all_layers(panels, wf_step=wf_step, verbose=verbose)

    if verbose:
        for name, res in results.items():
            print(f"--- {name} feature importances ---")
            print(res['importances'].to_string(index=False))
            print()

    return data, panels, results


def run_backtest(results, top_n=config.TOP_N, threshold=config.TIMING_THRESHOLD,
                 cost_bps=config.COST_BPS, show_plot=False, save_plot=True):
    """Merges both layers' out-of-sample predictions and runs the daily backtest."""
    banner(f"Backtest (top {top_n}, timing > {threshold:+.3f}, {cost_bps:.0f}bps one-way)")

    signals = bt.build_signal_frame(results['selection']['wf_predictions'],
                                    results['timing']['wf_predictions'])
    summary, curves = bt.run_all(signals, top_n=top_n, threshold=threshold,
                                 cost_bps=cost_bps)
    print(bt.format_summary(summary).to_string())

    fig = bt.plot_curves(curves,
                         save_path=config.plot_path('backtest_two_layer') if save_plot else None,
                         show=show_plot)
    return signals, summary, curves, fig


def tonight(data, top_n=config.TOP_N, threshold=config.TIMING_THRESHOLD,
            capital=config.CAPITAL, weighting='equal', save_plan=True):
    """Scores the latest snapshot and builds the executable plan."""
    banner("Tonight's signals")
    signals_df, plan_assets, meta = generate_signals(data, top_n=top_n, threshold=threshold)

    print(f"As of {pd.Timestamp(meta['as_of']).date()}  |  "
          f"selection {meta['selection_horizon']}d / timing {meta['timing_horizon']}d")
    print(signals_df[['Rank', 'Asset', 'sel_score', 'timing_score', 'Close', 'Decision']]
          .to_string(index=False))

    banner("Trading plan")
    plan, orders = build_trading_plan(signals_df, plan_assets, meta, capital=capital,
                                      weighting=weighting, save=save_plan)
    for k, v in plan_summary(plan, meta).items():
        print(f"  {k:<42} {v}")

    print("\n--- Positions ---")
    print(format_plan(plan).to_string(index=False))

    print("\n--- Orders to execute ---")
    if orders.empty:
        print("  (no change from the previous plan)")
    else:
        print(orders.to_string(index=False))

    return signals_df, plan, orders, meta


def run_nightly(refresh=False, wf_step=1, top_n=config.TOP_N,
                threshold=config.TIMING_THRESHOLD, capital=config.CAPITAL,
                cost_bps=config.COST_BPS, weighting='equal',
                show_plot=False, save_plan=True, allow_stale=False):
    """The whole nightly job. Returns every intermediate for further inspection."""
    print(f"Project root: {config.PROJECT_ROOT}")
    data, panels, results = train(refresh=refresh, wf_step=wf_step, allow_stale=allow_stale)
    signals, summary, curves, fig = run_backtest(
        results, top_n=top_n, threshold=threshold, cost_bps=cost_bps, show_plot=show_plot)
    signals_df, plan, orders, meta = tonight(
        data, top_n=top_n, threshold=threshold, capital=capital,
        weighting=weighting, save_plan=save_plan)

    return {
        'data': data, 'panels': panels, 'results': results,
        'backtest_signals': signals, 'summary': summary, 'curves': curves, 'figure': fig,
        'signals': signals_df, 'plan': plan, 'orders': orders, 'meta': meta,
    }

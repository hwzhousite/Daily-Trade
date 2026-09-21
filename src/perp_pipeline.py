"""
Orchestration for the Binance perp pipeline. Shared by main.py and the notebook.

Two cadences, deliberately separated:

  nightly    refit the production heads on all history, score tonight's bar,
             write the plan.                                   (~1 min)
  backtest   full embargoed walk-forward across every head, the ablation table
             and the conformal calibration for the range heads.  (~6 min)

Re-validating on one extra day adds nothing, so the slow path is weekly. But the
range heads need a validated walk-forward to calibrate against, so `nightly`
reuses the calibration the last backtest stored in the bundle.
"""
import pandas as pd

import config
import binance_data as bd
import factors as F
import models as M
import perp_backtest as pb
import perp_inference as pi
import perp_plan as pp


def banner(t):
    print(f"\n{'=' * 82}\n{t}\n{'=' * 82}")


def load_panel(refresh=False, collect_short=False, verbose=True):
    """Cached parquet -> factors -> labelled panel."""
    if refresh:
        banner("Refreshing Binance data")
        bd.download_universe(verbose=verbose)
        if collect_short:
            print("\nSecond track: 30-day-only endpoints")
            bd.collect_short_history()

    frames = bd.load_universe()
    if verbose:
        spans = [(df.index.min(), df.index.max()) for df in frames.values()]
        print(f"Loaded {len(frames)} symbols | latest bar "
              f"{max(s[1] for s in spans).date()}")

    banner("Building factor panel")
    return F.build_panel(frames, verbose=verbose)


def train(panel, wf_step=None, validate=True, verbose=True):
    banner("Training LightGBM heads" + ("" if validate else " (production refit only)"))
    return M.train_all(panel, wf_step=wf_step, validate=validate, verbose=verbose)


def backtest(results, top_n=None, prob_threshold=None, cost_bps=None,
             exit_rank_mult=None, show_plot=False, save_plot=True):
    top_n = top_n or config.TOP_N
    exit_rank_mult = exit_rank_mult or config.EXIT_RANK_MULT
    banner(f"Backtest (top {top_n}, exit below rank {top_n*exit_rank_mult}, "
           f"{cost_bps or config.COST_BPS:.0f}bps + funding)")

    signals = pb.build_signal_frame(results)
    summary, curves = pb.run_all(signals, top_n=top_n, prob_threshold=prob_threshold,
                                 cost_bps=cost_bps, exit_rank_mult=exit_rank_mult)
    print(pb.format_summary(summary).to_string())
    fig = pb.plot_curves(curves,
                         save_path=config.plot_path('backtest_perp') if save_plot else None,
                         show=show_plot)
    return signals, summary, curves, fig


def tonight(panel, top_n=None, capital=None, use_timing_gate=None,
            prob_threshold=None, exit_rank_mult=None, save_plan=True, top_show=15):
    """Score tonight's bar and build the plan. This is the nightly deliverable."""
    banner("Tonight's forecast")

    _prev_df, _prev_date, prev_holdings = pp.load_previous_plan(
        panel.index.get_level_values('Date').max())

    signals, held, meta = pi.generate_signals(
        panel, top_n=top_n, prob_threshold=prob_threshold,
        prev_holdings=prev_holdings, exit_rank_mult=exit_rank_mult,
        use_timing_gate=use_timing_gate)

    print(f"as of {pd.Timestamp(meta['as_of']).date()} (UTC bar close) | "
          f"{meta['n_universe']} symbols | {meta['n_features']} features")
    print(f"\n--- Next-day forecast (top {top_show} by 7d selection score) ---")
    print(pi.format_signals(signals, top_show).to_string(index=False))

    banner("Trading plan")
    plan, orders = pp.build_plan(signals, held, meta, capital=capital, save=save_plan)
    for k, v in pp.plan_summary(plan, meta).items():
        print(f"  {k:<26} {v}")

    print("\n--- Orders to execute ---")
    print(orders.to_string(index=False) if not orders.empty
          else "  (no change from the previous plan)")

    return signals, plan, orders, meta


def run_nightly(refresh=True, capital=None, top_n=None, save_plan=True,
                collect_short=True, **kwargs):
    """Fast path: refresh -> refit -> forecast -> plan. No walk-forward."""
    print(f"Project root: {config.PROJECT_ROOT}")
    panel = load_panel(refresh=refresh, collect_short=collect_short)
    results = train(panel, validate=False)
    signals, plan, orders, meta = tonight(panel, top_n=top_n, capital=capital,
                                          save_plan=save_plan, **kwargs)
    return {'panel': panel, 'results': results, 'signals': signals,
            'plan': plan, 'orders': orders, 'meta': meta}


def run_full(refresh=False, wf_step=None, top_n=None, capital=None, cost_bps=None,
             show_plot=False, save_plan=True, collect_short=False, **kwargs):
    """Slow path: everything, including the walk-forward and the ablation table."""
    print(f"Project root: {config.PROJECT_ROOT}")
    panel = load_panel(refresh=refresh, collect_short=collect_short)
    results = train(panel, wf_step=wf_step, validate=True)

    for name, res in results.items():
        print(f"\n--- {name}: top 12 features by gain ---")
        print(res['importances'].head(12).to_string(index=False))

    signals_bt, summary, curves, fig = backtest(
        results, top_n=top_n, cost_bps=cost_bps, show_plot=show_plot)
    signals, plan, orders, meta = tonight(panel, top_n=top_n, capital=capital,
                                          save_plan=save_plan, **kwargs)
    return {'panel': panel, 'results': results, 'backtest_signals': signals_bt,
            'summary': summary, 'curves': curves, 'figure': fig,
            'signals': signals, 'plan': plan, 'orders': orders, 'meta': meta}

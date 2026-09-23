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
    results = M.train_all(panel, wf_step=wf_step, validate=validate, verbose=verbose)
    mkt_preds = mkt_metrics = None
    rg_preds = rg_metrics = None
    if validate:
        mkt_preds, mkt_metrics = M.walk_forward_market(panel, wf_step=wf_step,
                                                       verbose=verbose)
        rg_preds, rg_metrics = M.walk_forward_regime(panel, verbose=verbose)
    results['market'] = {'bundle': M.fit_market(panel, wf_metrics=mkt_metrics,
                                                wf_preds=mkt_preds),
                         'wf_metrics': mkt_metrics}
    results['regime'] = {'bundle': M.fit_regime(panel, wf_metrics=rg_metrics,
                                                wf_preds=rg_preds),
                         'wf_metrics': rg_metrics}
    return results


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

    mkt = M.market_forecast(panel)
    if mkt:
        wfm = mkt.get('wf_metrics') or {}
        cal = (f" (walk-forward Brier {wfm['brier']:.3f}, base {wfm['base_rate']:.1%})"
               if wfm else " (not walk-forward validated yet; run `backtest`)")
        print(f"Market (44-coin equal weight): P(up tomorrow) = "
              f"{mkt['prob_up']:.1%}{cal}")

    _prev_df, _prev_date, prev_holdings = pp.load_previous_plan(
        panel.index.get_level_values('Date').max())

    signals, held, meta = pi.generate_signals(
        panel, top_n=top_n, prob_threshold=prob_threshold,
        prev_holdings=prev_holdings, exit_rank_mult=exit_rank_mult,
        use_timing_gate=use_timing_gate)

    nf = meta['n_features']
    nf_txt = '/'.join(str(v) for v in nf.values()) if isinstance(nf, dict) else str(nf)
    print(f"as of {pd.Timestamp(meta['as_of']).date()} (UTC bar close) | "
          f"{meta['n_universe']} symbols | {nf_txt} features per head")
    print(f"\n--- Next-day forecast (top {top_show} by 7d selection score) ---")
    print(pi.format_signals(signals, top_show).to_string(index=False))

    banner("Trading plan")
    plan, orders = pp.build_plan(signals, held, meta, capital=capital, save=save_plan)
    for k, v in pp.plan_summary(plan, meta).items():
        print(f"  {k:<26} {v}")

    print("\n--- Orders to execute ---")
    print(orders.to_string(index=False) if not orders.empty
          else "  (no change from the previous plan)")

    # --- regime cascade: 7d stance -> 3 recommendations -> next-day bands ---
    rg = M.regime_forecast(panel)
    if rg and 'p_up_7d' in signals.columns and signals['p_up_7d'].notna().any():
        banner("Regime cascade (weekly stance from BTC/ETH/SOL, set each Monday)")
        wfm = rg.get('wf_metrics') or {}
        cal = (f"walk-forward Brier {wfm['brier']:.3f}, base {wfm['base_rate']:.1%}, "
               f"{wfm.get('n', '?')} independent Mondays"
               if wfm else "not walk-forward validated yet; run `backtest`")
        print(f"Week's stance: {rg['stance']}  |  P(up over 7d) = {rg['prob_up']:.1%}"
              f"  |  set on Monday {rg['based_on_monday']:%Y-%m-%d}  ({cal})")
        n = config.REGIME_N_RECOMMEND
        if rg['stance'] == 'LONG':
            picks = signals.nlargest(n, 'p_up_7d')
            print(f"Top {n} by P(7d up):")
        else:
            picks = signals.nsmallest(n, 'p_up_7d')
            print(f"Top {n} by P(7d DOWN)  [informational -- the pipeline "
                  f"trades LONG-ONLY; short backtests were net-negative]:")
        for _, r in picks.iterrows():
            p7 = r['p_up_7d'] if rg['stance'] == 'LONG' else 1 - r['p_up_7d']
            print(f"  {r['Symbol']:<14} P(7d {'up' if rg['stance']=='LONG' else 'down'}) "
                  f"{p7:.0%} | P(up 1d) {r['prob_up_1d']:.0%} | next-day band "
                  f"{r['low_price']:,.6g} ~ {r['high_price']:,.6g}")

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
        if res.get('importances') is None:      # the market head carries none
            continue
        print(f"\n--- {name}: top 12 features by gain ---")
        print(res['importances'].head(12).to_string(index=False))

    signals_bt, summary, curves, fig = backtest(
        results, top_n=top_n, cost_bps=cost_bps, show_plot=show_plot)
    signals, plan, orders, meta = tonight(panel, top_n=top_n, capital=capital,
                                          save_plan=save_plan, **kwargs)
    return {'panel': panel, 'results': results, 'backtest_signals': signals_bt,
            'summary': summary, 'curves': curves, 'figure': fig,
            'signals': signals, 'plan': plan, 'orders': orders, 'meta': meta}

#!/usr/bin/env python3
"""
CryptoQuantPipeline -- Binance USDT perpetuals.

  selection   LightGBM regression on forward 7d return   -> cross-sectional rank
  timing      LightGBM binary                            -> P(tomorrow closes up)
  range_high  LightGBM quantile (q90, conformal)         -> tomorrow's high
  range_low   LightGBM quantile (q10, conformal)         -> tomorrow's low

    python main.py nightly        # refresh -> refit -> forecast -> plan   (~1 min)
    python main.py backtest       # full walk-forward + ablation table     (~6 min)
    python main.py full           # backtest + tonight's plan
    python main.py download       # (re)download the universe only
    python main.py signals        # forecast from the saved models, no refit

Run `nightly` after 00:15 UTC so the previous UTC day's bar is closed.
All paths come from src/config.py and are anchored to this file's directory.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd

import config
import perp_pipeline as P


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['nightly', 'backtest', 'full', 'download', 'signals'])
    p.add_argument('--no-refresh', action='store_true',
                   help='use the cached parquet instead of hitting Binance')
    p.add_argument('--wf-step', type=int, default=config.WF_STEP,
                   help=f'walk-forward refit cadence in days (default {config.WF_STEP})')
    p.add_argument('--top-n', type=int, default=config.TOP_N,
                   help=f'positions held (default {config.TOP_N})')
    p.add_argument('--exit-rank-mult', type=int, default=config.EXIT_RANK_MULT,
                   help=f'hysteresis band: exit below rank top_n*this (default {config.EXIT_RANK_MULT})')
    p.add_argument('--timing-gate', action='store_true',
                   help='gate positions on P(up) > --prob-threshold (net-negative; see README)')
    p.add_argument('--prob-threshold', type=float, default=config.PROB_THRESHOLD)
    p.add_argument('--capital', type=float, default=config.CAPITAL)
    p.add_argument('--cost-bps', type=float, default=config.COST_BPS)
    p.add_argument('--universe-size', type=int, default=config.UNIVERSE_SIZE)
    p.add_argument('--show-plot', action='store_true')
    p.add_argument('--no-save-plan', action='store_true')
    p.add_argument('--no-collect-short', action='store_true',
                   help='skip the 30-day OI / long-short collector')
    args = p.parse_args()

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 60)

    common = dict(top_n=args.top_n, capital=args.capital,
                  exit_rank_mult=args.exit_rank_mult,
                  use_timing_gate=args.timing_gate,
                  prob_threshold=args.prob_threshold,
                  save_plan=not args.no_save_plan)

    if args.command == 'download':
        import binance_data as bd
        print(f"Project root: {config.PROJECT_ROOT}")
        bd.download_universe(n=args.universe_size)
        if not args.no_collect_short:
            bd.collect_short_history()
    elif args.command == 'nightly':
        P.run_nightly(refresh=not args.no_refresh,
                      collect_short=not args.no_collect_short, **common)
    elif args.command == 'backtest':
        print(f"Project root: {config.PROJECT_ROOT}")
        panel = P.load_panel(refresh=not args.no_refresh)
        results = P.train(panel, wf_step=args.wf_step, validate=True)
        P.backtest(results, top_n=args.top_n, cost_bps=args.cost_bps,
                   exit_rank_mult=args.exit_rank_mult,
                   prob_threshold=args.prob_threshold, show_plot=args.show_plot)
    elif args.command == 'full':
        P.run_full(refresh=not args.no_refresh, wf_step=args.wf_step,
                   cost_bps=args.cost_bps, show_plot=args.show_plot,
                   collect_short=not args.no_collect_short, **common)
    else:
        print(f"Project root: {config.PROJECT_ROOT}")
        panel = P.load_panel(refresh=not args.no_refresh)
        P.tonight(panel, **common)

    print("\nDone.")


if __name__ == '__main__':
    main()

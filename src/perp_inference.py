"""
Nightly inference: score the latest bar with all four heads.

Produces exactly the three things the strategy needs:
  1. a 7d cross-sectional ranking            (which coins to own)
  2. P(tomorrow closes up) per coin          (a calibrated probability)
  3. tomorrow's expected high/low price band (conformal-calibrated quantiles)
"""
import numpy as np
import pandas as pd

import config
import factors as F
import models as M


def latest_rows(panel):
    """The most recent bar per symbol -- the features tonight's decision uses."""
    last_date = panel.index.get_level_values('Date').max()
    snap = panel.xs(last_date, level='Date', drop_level=False)
    return snap.reset_index(), pd.Timestamp(last_date)


def generate_signals(panel, top_n=None, prob_threshold=None, prev_holdings=None,
                     exit_rank_mult=None, use_timing_gate=None):
    """
    Scores the latest bar with every head and applies the position rule.

    The default rule is hysteresis: enter the top N, hold until a name drops out
    of the top N*exit_rank_mult. `prev_holdings` is what is currently held, so
    the rule is path-dependent by design -- pass last night's book.

    The timing gate is OFF by default: P(up) is reported for every coin, but
    using it to flip positions daily costs more in fees than its edge is worth
    (see README). Set use_timing_gate=True to switch it on.

    Returns (signals_df, held_symbols, meta).
    """
    top_n = top_n or config.TOP_N
    prob_threshold = config.PROB_THRESHOLD if prob_threshold is None else prob_threshold
    exit_rank_mult = exit_rank_mult or config.EXIT_RANK_MULT
    use_timing_gate = config.USE_TIMING_GATE if use_timing_gate is None else use_timing_gate
    prev_holdings = set(prev_holdings or [])

    snap, as_of = latest_rows(panel)
    bundles = {name: M.load_head(name) for name in M.HEADS}

    # The IC pre-filter selects per head, so each head has its OWN feature
    # subset; the snapshot only needs to cover each of them.
    for n, b in bundles.items():
        missing = [f for f in b['features'] if f not in snap.columns]
        if missing:
            raise KeyError(f"Snapshot missing {len(missing)} features for head "
                           f"'{n}', e.g. {missing[:5]}. Re-run training after "
                           f"a factor-layer change.")

    def _X(name):
        return snap[bundles[name]['features']]

    out = snap[['Symbol', 'Close']].copy()
    out['sel_score_7d'] = bundles['selection']['model'].predict(_X('selection'))
    out['prob_up_1d'] = bundles['timing']['model'].predict_proba(_X('timing'))[:, 1]

    # Quantile heads + their conformal offsets.
    for head, label in [('range_high', 'high'), ('range_low', 'low')]:
        b = bundles[head]
        raw = b['model'].predict(_X(head))
        delta = b.get('conformal_delta', 0.0) or 0.0
        out[f'{label}_ret'] = raw + delta
        out[f'{label}_price'] = out['Close'] * (1 + out[f'{label}_ret'])

    out['expected_move'] = out['high_ret'] - out['low_ret']

    out = out.sort_values('sel_score_7d', ascending=False).reset_index(drop=True)
    out['Rank'] = np.arange(1, len(out) + 1)
    out['In_Entry_Zone'] = out['Rank'] <= top_n
    out['In_Keep_Zone'] = out['Rank'] <= top_n * exit_rank_mult
    out['Was_Held'] = out['Symbol'].isin(prev_holdings)

    # --- hysteresis: keep what is still inside the band, then top up ---
    held = [s for s in out.loc[out['Was_Held'] & out['In_Keep_Zone'], 'Symbol']]
    for sym in out.loc[out['In_Entry_Zone'], 'Symbol']:
        if sym not in held and len(held) < top_n:
            held.append(sym)
    held = held[:top_n]

    out['Timing_OK'] = out['prob_up_1d'] > prob_threshold
    if use_timing_gate:
        held = [s for s in held if out.set_index('Symbol').loc[s, 'Timing_OK']]

    held_set = set(held)

    def _decide(row):
        if row['Symbol'] in held_set:
            return 'HOLD' if row['Was_Held'] else 'ENTER'
        if row['Was_Held']:
            return 'EXIT'
        if use_timing_gate and row['In_Entry_Zone'] and not row['Timing_OK']:
            return 'BLOCKED (prob)'
        return 'WATCH' if row['In_Keep_Zone'] else '-'

    out['Decision'] = out.apply(_decide, axis=1)

    meta = {
        'as_of': as_of,
        'n_universe': int(len(out)),
        'top_n': top_n,
        'exit_rank_mult': exit_rank_mult,
        'use_timing_gate': use_timing_gate,
        'prob_threshold': prob_threshold,
        'n_candidates': int(out['In_Entry_Zone'].sum()),
        'n_held': len(held),
        'trained_at': {n: b.get('trained_at') for n, b in bundles.items()},
        'conformal': {n: b.get('conformal') for n, b in bundles.items()
                      if b.get('task') == 'quantile'},
        'wf_metrics': {n: b.get('wf_metrics') for n, b in bundles.items()},
        'n_features': {n: len(b['features']) for n, b in bundles.items()},
    }
    return out, held, meta


def format_signals(signals, n=None):
    """The nightly forecast table: probability of up, and the price band."""
    df = signals if n is None else signals.head(n)
    out = pd.DataFrame({
        'Rank': df['Rank'],
        'Symbol': df['Symbol'],
        'Close': df['Close'].map(lambda v: f"{v:,.6g}"),
        'P(up)': df['prob_up_1d'].map(lambda v: f"{v:.1%}"),
        'Exp7d': df['sel_score_7d'].map(lambda v: f"{v:+.2%}"),
        'Low': df['low_price'].map(lambda v: f"{v:,.6g}"),
        'High': df['high_price'].map(lambda v: f"{v:,.6g}"),
        'Band': df['expected_move'].map(lambda v: f"{v:.1%}"),
        'Decision': df['Decision'],
    })
    return out

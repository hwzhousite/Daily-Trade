"""
Tonight's executable plan for the perp book.

Current holdings come from the last saved plan, which the hysteresis rule needs
(the position rule is path-dependent -- what you hold changes what you should
hold). That file is therefore assumed to reflect your REAL book; if you did not
execute a night, delete or edit it.
"""
import os
import numpy as np
import pandas as pd

import config


def load_previous_plan(as_of, plans_dir=None):
    """Most recent saved plan strictly before `as_of`. Returns (df, date, holdings)."""
    plans_dir = str(plans_dir or config.PLANS_DIR)
    if not os.path.isdir(plans_dir):
        return None, None, []

    stamp = f"{pd.Timestamp(as_of):%Y-%m-%d}"
    prior = sorted(f for f in os.listdir(plans_dir)
                   if f.startswith('plan_') and f.endswith('.csv') and f[5:15] < stamp)

    # Walk backwards to the most recent plan this pipeline can actually read.
    # Files written by the superseded yfinance/RF pipeline key on 'Asset', not
    # 'Symbol'; skipping them beats crashing tonight's run on a stale artifact.
    for fname in reversed(prior):
        df = pd.read_csv(os.path.join(plans_dir, fname))
        if {'Symbol', 'Target_Weight'}.issubset(df.columns):
            holdings = df.loc[df['Target_Weight'] > 0, 'Symbol'].tolist()
            return df, fname[5:15], holdings
        print(f"  skipping {fname}: written by an older pipeline (no Symbol column)")

    return None, None, []


def build_plan(signals, held, meta, capital=None, plans_dir=None, save=True):
    """Target book + the orders that move you from last night's book to it."""
    capital = capital if capital is not None else config.CAPITAL
    plans_dir = str(plans_dir or config.PLANS_DIR)

    plan = signals[['Symbol', 'Rank', 'Close', 'sel_score_7d', 'prob_up_1d',
                    'low_price', 'high_price', 'expected_move', 'Decision']].copy()
    plan = plan.rename(columns={'Close': 'Ref_Price', 'sel_score_7d': 'Exp_Return_7d',
                                'prob_up_1d': 'Prob_Up_1d',
                                'low_price': 'Pred_Low_1d', 'high_price': 'Pred_High_1d',
                                'expected_move': 'Pred_Band_1d'})

    held_set = set(held)
    # 1/top_n per name, NOT 1/len(held): with the daily entry limit an
    # under-filled book keeps the empty slots in cash instead of concentrating.
    # The signal-health multiplier (rolling IC t-stat monitor) scales every
    # position when the selection signal is unhealthy.
    health_mult = float((meta.get('health') or {}).get('multiplier', 1.0)) \
        if config.USE_SIGNAL_HEALTH else 1.0
    per_name = health_mult / meta['top_n']
    plan['Target_Weight'] = plan['Symbol'].map(
        lambda s: per_name if s in held_set else 0.0)
    plan['Target_Notional'] = plan['Target_Weight'] * capital
    plan['Target_Units'] = np.where(plan['Ref_Price'] > 0,
                                    plan['Target_Notional'] / plan['Ref_Price'], 0.0)

    prev_df, prev_date, _ = load_previous_plan(meta['as_of'], plans_dir)
    prev_w = (dict(zip(prev_df['Symbol'], prev_df['Target_Weight']))
              if prev_df is not None and 'Target_Weight' in prev_df else {})
    plan['Prev_Weight'] = plan['Symbol'].map(lambda s: prev_w.get(s, 0.0))
    plan['Delta_Weight'] = plan['Target_Weight'] - plan['Prev_Weight']
    plan['Delta_Notional'] = plan['Delta_Weight'] * capital

    def _action(r):
        if abs(r['Delta_Weight']) < 1e-9:
            return 'HOLD' if r['Target_Weight'] > 0 else 'FLAT'
        if r['Prev_Weight'] == 0:
            return 'BUY (open)'
        if r['Target_Weight'] == 0:
            return 'SELL (close)'
        return 'BUY (add)' if r['Delta_Weight'] > 0 else 'SELL (trim)'

    plan['Action'] = plan.apply(_action, axis=1)
    plan['As_Of'] = pd.Timestamp(meta['as_of']).date()
    plan['Prev_Plan_Date'] = prev_date or 'none'
    plan['Capital'] = capital
    plan = plan.sort_values(['Target_Weight', 'Rank'], ascending=[False, True]).reset_index(drop=True)

    orders = plan[plan['Action'].str.startswith(('BUY', 'SELL'))][
        ['Symbol', 'Action', 'Delta_Weight', 'Delta_Notional', 'Ref_Price',
         'Prob_Up_1d', 'Pred_Low_1d', 'Pred_High_1d']].reset_index(drop=True)

    if save:
        os.makedirs(plans_dir, exist_ok=True)
        path = str(config.plan_path(pd.Timestamp(meta['as_of'])))
        tmp = f'{path}.tmp'
        plan.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"Plan saved to {path}")

    return plan, orders


def format_plan(plan, n=None):
    df = plan if n is None else plan.head(n)
    return pd.DataFrame({
        'Symbol': df['Symbol'], 'Rank': df['Rank'], 'Decision': df['Decision'],
        'Action': df['Action'],
        'Exp7d': df['Exp_Return_7d'].map(lambda v: f"{v:+.2%}"),
        'P(up)': df['Prob_Up_1d'].map(lambda v: f"{v:.1%}"),
        'Weight': df['Target_Weight'].map(lambda v: f"{v:.1%}"),
        'Notional': df['Target_Notional'].map(lambda v: f"{v:,.0f}"),
        'Ref': df['Ref_Price'].map(lambda v: f"{v:,.6g}"),
        'PredLow': df['Pred_Low_1d'].map(lambda v: f"{v:,.6g}"),
        'PredHigh': df['Pred_High_1d'].map(lambda v: f"{v:,.6g}"),
    })


def plan_summary(plan, meta):
    held = plan[plan['Target_Weight'] > 0]
    gross = float(plan['Target_Notional'].sum())
    turnover = float(plan['Delta_Weight'].abs().sum())
    cap = float(plan['Capital'].iloc[0])
    conf = meta.get('conformal', {})
    return {
        'As of (UTC bar close)': f"{pd.Timestamp(meta['as_of']).date()}",
        'Universe': f"{meta['n_universe']} symbols",
        'Rule': f"top {meta['top_n']}, exit below rank {meta['top_n']*meta['exit_rank_mult']}"
                + ("  + P(up) gate" if meta['use_timing_gate'] else "  (timing gate OFF)"),
        'Holding': f"{len(held)} ({', '.join(held['Symbol']) or 'cash'})",
        'Gross exposure': f"{gross:,.0f} / {cap:,.0f} ({gross/cap:.0%})",
        'Turnover vs prev plan': f"{turnover:.2f} (~{turnover*config.COST_BPS/1e4*cap:,.2f} in fees)",
        'Prev plan': plan['Prev_Plan_Date'].iloc[0],
        'Signal health': (lambda h: f"rolling t {h['roll_t']:+.2f} -> exposure x{h['multiplier']:.2f}"
                          if h and h.get('roll_t') is not None
                          else 'not measured (run backtest)')(meta.get('health')),
        'Range calibration': ', '.join(
            f"{k.replace('range_','')} {v['calibrated_coverage']:.1%}/{v['target_coverage']:.0%}"
            for k, v in (conf or {}).items() if v) or 'not calibrated (run backtest)',
    }

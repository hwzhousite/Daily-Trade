"""
Turns tonight's signals into an executable plan, and diffs it against last
night's plan so the output is a list of ORDERS rather than a list of positions.
"""
import os
import numpy as np
import pandas as pd

import config
from daily_inference import price_boundaries


def _previous_plan(as_of, plans_dir=config.PLANS_DIR):
    """Most recent saved plan strictly before `as_of`, or None."""
    if not os.path.isdir(plans_dir):
        return None, None
    files = sorted(f for f in os.listdir(plans_dir)
                   if f.startswith('plan_') and f.endswith('.csv'))
    prior = [f for f in files if f[5:15] < f"{as_of:%Y-%m-%d}"]
    if not prior:
        return None, None
    path = os.path.join(plans_dir, prior[-1])
    return pd.read_csv(path), prior[-1][5:15]


def build_trading_plan(signals_df, plan_assets, meta, capital=config.CAPITAL,
                       confidence_level=config.CONFIDENCE_LEVEL,
                       weighting='equal', plans_dir=config.PLANS_DIR, save=True):
    """
    Builds the nightly plan.

    weighting  'equal'       -> 1/N across held assets (what the backtest validates)
               'inverse_vol' -> risk-parity-ish; NOT what the backtest measures,
                                so its performance is unverified.

    Returns (plan_df, orders_df).
    """
    as_of = meta['as_of']
    horizon = meta['timing_horizon']

    bounds = price_boundaries(signals_df, horizon_days=horizon,
                              confidence_level=confidence_level).set_index('Asset')

    plan = signals_df[['Asset', 'Rank', 'Close', 'sel_score', 'timing_score',
                       'Decision', 'volatility_7d']].copy()
    plan = plan.rename(columns={'Close': 'Ref_Price',
                                'sel_score': 'Exp_Return_7d',
                                'timing_score': 'Exp_Return_1d'})

    # --- target weights ----------------------------------------------------
    held = set(plan_assets)
    if not held:
        plan['Target_Weight'] = 0.0
    elif weighting == 'inverse_vol':
        print("NOTE: inverse-vol weighting is NOT the scheme the backtest validates "
              "(that one is equal-weight). Its live behaviour is unverified.")
        inv = plan['Asset'].map(lambda a: 1.0 / plan.loc[plan['Asset'] == a, 'volatility_7d'].iloc[0]
                                if a in held else 0.0)
        plan['Target_Weight'] = inv / inv.sum()
    else:
        plan['Target_Weight'] = plan['Asset'].map(lambda a: 1.0 / len(held) if a in held else 0.0)

    plan['Target_Notional'] = plan['Target_Weight'] * capital
    plan['Target_Units'] = np.where(plan['Ref_Price'] > 0,
                                    plan['Target_Notional'] / plan['Ref_Price'], 0.0)

    # --- risk levels from the 1d lognormal interval ------------------------
    plan['Stop_Price'] = plan['Asset'].map(bounds['Lower_Boundary'])
    plan['Target_Price'] = plan['Asset'].map(bounds['Upper_Boundary'])
    plan['Risk_Per_Unit'] = plan['Ref_Price'] - plan['Stop_Price']
    plan['Notional_At_Risk'] = (plan['Target_Units'] * plan['Risk_Per_Unit']).clip(lower=0)

    plan['As_Of'] = pd.Timestamp(as_of).date()
    plan['Horizon_Days'] = horizon
    plan['Capital'] = capital

    # --- diff against the previous plan -> orders --------------------------
    prev, prev_date = _previous_plan(as_of, plans_dir)
    prev_w = (dict(zip(prev['Asset'], prev['Target_Weight']))
              if prev is not None and 'Target_Weight' in prev else {})

    plan['Prev_Weight'] = plan['Asset'].map(lambda a: prev_w.get(a, 0.0))
    plan['Delta_Weight'] = plan['Target_Weight'] - plan['Prev_Weight']
    plan['Delta_Notional'] = plan['Delta_Weight'] * capital

    def _action(row):
        if abs(row['Delta_Weight']) < 1e-9:
            return 'HOLD' if row['Target_Weight'] > 0 else 'FLAT'
        if row['Prev_Weight'] == 0:
            return 'BUY (open)'
        if row['Target_Weight'] == 0:
            return 'SELL (close)'
        return 'BUY (add)' if row['Delta_Weight'] > 0 else 'SELL (trim)'

    plan['Action'] = plan.apply(_action, axis=1)
    plan['Prev_Plan_Date'] = prev_date or 'none'

    plan = plan.sort_values(['Target_Weight', 'Rank'], ascending=[False, True]).reset_index(drop=True)

    orders = plan[plan['Action'].str.startswith(('BUY', 'SELL'))][
        ['Asset', 'Action', 'Delta_Weight', 'Delta_Notional', 'Ref_Price',
         'Stop_Price', 'Target_Price']
    ].reset_index(drop=True)

    if save:
        os.makedirs(plans_dir, exist_ok=True)
        path = config.plan_path(pd.Timestamp(as_of))
        plan.to_csv(path, index=False)
        print(f"Plan saved to {path}")

    return plan, orders


def format_plan(plan):
    """Compact, human-readable view of the plan."""
    cols = ['Asset', 'Rank', 'Decision', 'Action', 'Exp_Return_7d', 'Exp_Return_1d',
            'Target_Weight', 'Target_Notional', 'Ref_Price', 'Stop_Price', 'Target_Price']
    out = plan[cols].copy()
    for c in ['Exp_Return_7d', 'Exp_Return_1d', 'Target_Weight']:
        out[c] = out[c].map(lambda v: f"{v:+.2%}")
    for c in ['Target_Notional']:
        out[c] = out[c].map(lambda v: f"{v:,.0f}")
    for c in ['Ref_Price', 'Stop_Price', 'Target_Price']:
        out[c] = out[c].map(lambda v: f"{v:,.6g}")
    return out


def plan_summary(plan, meta):
    """One-paragraph description of tonight's book."""
    held = plan[plan['Target_Weight'] > 0]
    gross = float(plan['Target_Notional'].sum())
    at_risk = float(plan['Notional_At_Risk'].sum())
    turnover = float(plan['Delta_Weight'].abs().sum())
    return {
        'As of': f"{pd.Timestamp(meta['as_of']).date()}",
        'Candidates (7d selection)': f"{meta['n_candidates']} of {len(plan)}",
        'Held after timing gate': f"{meta['n_held']} ({', '.join(held['Asset']) or 'cash'})",
        'Gross exposure': f"{gross:,.0f} / {plan['Capital'].iloc[0]:,.0f} "
                          f"({gross / plan['Capital'].iloc[0]:.0%})",
        'Cash': f"{plan['Capital'].iloc[0] - gross:,.0f}",
        f"Notional at risk (1d {config.CONFIDENCE_LEVEL:.0%} stop)": f"{at_risk:,.0f}",
        'Turnover vs prev plan': f"{turnover:.2f} "
                                 f"(~{turnover * config.COST_BPS / 1e4 * plan['Capital'].iloc[0]:,.2f} in fees)",
        'Prev plan': plan['Prev_Plan_Date'].iloc[0],
    }

"""
Central configuration. Every path derives from the project root, so the pipeline
runs from any working directory.

Architecture -- four LightGBM heads on one daily panel of Binance USDT perps:

  selection   regression  forward 7d return   -> cross-sectional ranking (选币)
  timing      binary      P(tomorrow up)      -> reported probability   (择时)
  range_high  quantile    q90 of next high    -> upper price boundary
  range_low   quantile    q10 of next low     -> lower price boundary

All four are sampled DAILY, so the 7d label overlaps across consecutive rows;
every validation split carries an `embargo` equal to the label horizon.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
BINANCE_DIR = DATA_DIR / 'binance'
SHORT_HISTORY_PATH = DATA_DIR / 'short_history.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output'
PLANS_DIR = OUTPUT_DIR / 'plans'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'


def model_path(head):
    """Bundle path for a head: selection | timing | range_high | range_low."""
    return MODELS_DIR / f'lgb_{head}.joblib'


def plot_path(name):
    return MODELS_DIR / f'{name}.png'


def plan_path(as_of):
    return PLANS_DIR / f'plan_{as_of:%Y-%m-%d}.csv'


# --- Universe --------------------------------------------------------------
EXCHANGE = 'binance_perp'
UNIVERSE_SIZE = 50          # top N USDT perps by 24h quote volume
MIN_HISTORY_DAYS = 400      # a symbol needs at least this much history to join
START_DATE = '2023-09-21'
END_DATE = None             # None -> today
BENCHMARK_SYMBOL = 'BTCUSDT'
WARMUP_DAYS = 90            # head rows dropped per symbol (rolling warm-up)

# --- Factors ---------------------------------------------------------------
# 332 features: 248 single-asset (price / volume / microstructure / funding /
# basis / calendar / box breakout) + 84 cross-sectional & market (incl.
# BTC/ETH leader state and leadership/breakout ranks). See src/factors.py.
FACTOR_PARAMS = {}          # factors.py owns its windows; kept for API symmetry

# --- Model -----------------------------------------------------------------
# LightGBM: strongest general-purpose learner for a tabular panel this size, and
# it treats NaN as "not yet observable" instead of forcing an imputation.
LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,      # main overfit brake with 305 features
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6,      # decorrelates trees across correlated factors
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=-1,
)
TRAIN_WINDOW = 252          # rolling walk-forward window, in daily bars
WF_STEP = 10                # refit cadence for VALIDATION (production refits nightly)
CONFORMAL_CALIB_DAYS = 250  # trailing window used to calibrate the range heads

# --- Feature pre-selection --------------------------------------------------
# Before a head is fitted, features are selected ON THE TRAINING WINDOW ONLY;
# each walk-forward fold re-selects on its own window, so the filter is as
# leak-free as the fit. Two selectors are available:
#
#   'lasso'  ElasticNet L1 path. Inside a block of correlated features the L1
#            penalty keeps a representative and zeroes the rest -- the failure
#            mode of 'ic' (keeping the whole volatility block) cannot happen.
#            For the selection head, features AND target are z-scored within
#            each date first, so the path picks what explains the CROSS-SECTION,
#            not the market level.
#   'ic'     univariate: keep features with mean daily |rank IC| >= threshold.
#            Best for the 1d heads (timing / range) -- but it killed the
#            selection head's RankIC (see README).
#   'off'    all features go in (minus any per-head exclude list).
#
# None = each head uses its own default from models.HEADS (selection: 'off'
# with the curated pool; timing/range: 'ic'). Setting a value here overrides
# every head at once -- an experiment knob, not the normal configuration.
FEATURE_SELECTOR = None
IC_THRESHOLD = 0.03         # 'ic': keep features with mean |daily rank IC| >= this
IC_MIN_FEATURES = 30        # 'ic': never go below this many (fall back to top-|IC|)
LASSO_MAX_FEATURES = 100    # 'lasso': densest point on the path that is kept
LASSO_MIN_FEATURES = 30     # 'lasso': walk further down the path to reach this
LASSO_L1_RATIO = 0.9        # 1.0 = pure lasso; <1 adds L2, stabler within blocks

# Per-head hyperparameter overrides on top of LGB_PARAMS (from `main.py tune`,
# 20 random-search trials/head at wf_step=40, 2026-09-21, each head tuned
# under its FINAL feature policy: selection on the curated pool by RankIC
# (+0.037 -> +0.042), timing under 'ic' by Brier (0.263 -> 0.256), ranges
# under 'ic' by pinball).
HEAD_PARAMS = {
    'selection':  dict(n_estimators=450, learning_rate=0.02, num_leaves=63,
                       max_depth=8, min_child_samples=30, subsample=0.7,
                       reg_alpha=1.0),
    'timing':     dict(num_leaves=15, max_depth=4, min_child_samples=150,
                       colsample_bytree=0.2, reg_alpha=1.0, reg_lambda=0.5),
    'range_high': dict(num_leaves=15, max_depth=4, min_child_samples=100,
                       subsample=0.7, colsample_bytree=0.3, reg_alpha=0.3,
                       reg_lambda=3.0),
    'range_low':  dict(num_leaves=15, max_depth=8, min_child_samples=100,
                       subsample=0.9, colsample_bytree=0.3, reg_alpha=0.0,
                       reg_lambda=3.0),
}


def head_params(head):
    """LGB_PARAMS with any tuned per-head overrides applied."""
    p = dict(LGB_PARAMS)
    p.update(HEAD_PARAMS.get(head, {}))
    return p

# Columns that are never model inputs.
NON_FEATURE_COLS = [
    'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume',
    'funding_daily', 'target_ret_7d', 'target_ret_1d', 'target_up_1d',
    'target_high_1d', 'target_low_1d', 'funding_next_1d', 'target_net_1d',
]

# --- Market head ------------------------------------------------------------
# A fifth, DATE-LEVEL head: P(the equal-weight market closes up tomorrow),
# driven by the BTC/ETH leader-state and mkt_* aggregate features. One row per
# date (~1000 samples), so the tree is kept tiny and heavily regularized.
MARKET_PARAMS = dict(
    n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=3.0,
    random_state=42, n_jobs=-1, verbosity=-1,
)

# --- Selection ensemble & confidence ---------------------------------------
# LightGBM is boosting, so it has no random-forest OOB property; the
# equivalent is a BAG of independently seeded fits, whose disagreement is an
# out-of-sample uncertainty estimate per prediction.
N_ENSEMBLE = 5             # bagged fits for the selection head (1 = single model)
# 'tstat': rank by mean/std across the bag (confidence-weighted score)
# 'lcb':   rank by mean - std (lower confidence bound)
# 'off':   rank by the plain ensemble mean
#
# Default 'off', decided by the wf10 ablation: tstat ranks BETTER on average
# (RankIC +0.0269 vs +0.0184) and its instantaneous top-8 even earns more
# (+0.234%/d vs +0.169%/d) -- but its ordering decays fast, so the hysteresis
# book goes stale and turnover doubles (0.42 vs 0.26): strategy +20% vs +132%.
# The mean ranking is sticky, which is what a hysteresis strategy needs.
CONF_RANKING = 'off'

# --- Strategy --------------------------------------------------------------
TOP_N = 8                  # positions held from the selection ranking
EXIT_RANK_MULT = 2         # hysteresis: sell only when a name leaves top N*mult
MAX_ENTRIES_PER_DAY = 3    # new BUYs per day; unfilled slots stay in cash
                           # (exits are never throttled -- risk control first)
USE_TIMING_GATE = False    # gating on P(up) is net-negative -- see README
PROB_THRESHOLD = 0.50      # only applies when USE_TIMING_GATE is True
COST_BPS = 10.0            # one-way taker fee, basis points of notional
ANNUALIZATION = 365        # perps trade every calendar day

# --- Execution -------------------------------------------------------------
CAPITAL = 10_000.0         # notional used to size the nightly plan

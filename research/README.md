# Research 工作区

在生产管线(`src/`)之上做研究的 notebook 集合。所有 notebook 直接 import
`src/` 的模块——数据、因子、模型、回测都复用生产代码,保证研究与生产口径一致。

| Notebook | 内容 | 耗时 |
|---|---|---|
| `01_data_exploration.ipynb` | 数据模块读取缓存;每币概览、质量检查、资金费/基差/相关性/宽度 | ~1 分钟 |
| `02_factor_research.ipynb` | 305 因子逐个 IC(+按族聚合)、Lasso L1 路径筛选、随机森林 variable importance、三方法交集 | ~5 分钟 |
| `03_backtest_research.ipynb` | 完整回测框架:walk-forward → 九组策略对比 → 回撤/滚动 Sharpe/换手 → top_n 敏感度 | wf40 约 6 分钟;正式结论用 wf10(~20 分钟) |

## 使用

```bash
cd research && jupyter lab
```

依赖同主管线(见根 README),另需 `jupyter`。数据要先有缓存:`python main.py download`。

## 两条纪律

1. **模型层更好 ≠ 策略层更好**。本管线已两次实证(单变量 IC 筛选、t-stat 置信度排名
   都是模型层赢、策略层输)。任何筛选/排名想法先过 `models.walk_forward`
   (每折训练窗内重选,防泄漏),再看策略层,且正式结论只认 `wf_step=10`。
2. **策略层总收益噪声极大**(同等信号质量给过 +56%~+176%),只信相对结论与模型层指标。

`output/` 存放 notebook 产物(信号 parquet 等),已 gitignore。

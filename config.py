# ============================================================
# 配置文件
# ============================================================

import pandas as pd
from datetime import datetime

# ========== Tushare 配置 ==========
TUSHARE_TOKEN = "11f93d2e16eda6837167b7eb1eb552f3c6641ed531bf89923ba9c85c"

# ========== 时间配置 ==========
TRAIN_START = "20100101"
TRAIN_END = "20201231"
VAL_START = "20210101"
VAL_END = "20231231"
TEST_START = "20240101"
TEST_END = "20241231"

# ========== 因子配置 ==========
FACTOR_CONFIG = {
    # 动量因子
    "momentum_5d": {"type": "momentum", "window": 5},
    "momentum_10d": {"type": "momentum", "window": 10},
    "momentum_20d": {"type": "momentum", "window": 20},
    
    # 反转因子
    "reversal_5d": {"type": "reversal", "window": 5},
    "reversal_10d": {"type": "reversal", "window": 10},
    
    # 估值因子
    "pb_ratio": {"type": "valuation"},
    "pe_ratio": {"type": "valuation"},
    
    # 质量因子
    "roe": {"type": "quality"},
    "roa": {"type": "quality"},
    
    # 流动性因子
    "turnover_ratio": {"type": "liquidity", "window": 20},
    "volume_ma_ratio": {"type": "liquidity", "window": 20},
    
    # 波动率因子
    "volatility_20d": {"type": "volatility", "window": 20},
    "volatility_60d": {"type": "volatility", "window": 60},
    
    # 技术面因子
    "rsi_14": {"type": "technical", "window": 14},
    "macd_signal": {"type": "technical"},
    
    # 规模因子
    "market_cap_log": {"type": "size"},
    
    # 增长因子
    "revenue_growth_yoy": {"type": "growth"},
    "earnings_growth_yoy": {"type": "growth"},
}

# ========== 模型配置 ==========
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 7,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "random_state": 42,
}

LGBM_TRAIN_PARAMS = {
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
}

# ========== 回测配置 ==========
BACKTEST_CONFIG = {
    "rebalance_freq": "M",  # M: 月度, W: 周度, D: 日度
    "holding_period": 20,  # 预测周期
    "top_k": 30,  # 选取排名前30的股票
    "min_price": 3.0,  # 最低股价(元)
    "transaction_cost": 0.001,  # 交易成本
    "initial_capital": 1000000,  # 初始资金(元)
}

# ========== 数据配置 ==========
DATA_CONFIG = {
    "exclude_st": True,  # 剔除ST股
    "exclude_suspended": True,  # 剔除停牌股
    "min_history_days": 252,  # 最少历史数据天数
    "cache_dir": "./data/cache",
    "output_dir": "./output",
}

# ========== 特征工程配置 ==========
FEATURE_ENGINEERING_CONFIG = {
    "standardize": True,
    "remove_outliers": True,
    "outlier_std": 3,  # 3倍标准差
    "handle_missing": "forward_fill",  # forward_fill, backward_fill, interpolate
    "missing_threshold": 0.3,  # 缺失值比例阈值
}

# A 股多因子 LightGBM 量化交易系统

## 项目概述

这是一个完整的 A 股量化交易系统，使用多因子模型 + LightGBM 预测 20 日收益，进行月度调仓回测。

### 核心特性

- ✅ **数据源**: Tushare A 股数据
- ✅ **时间范围**: 2010-2023 年（训练 2010-2020 / 验证 2021-2023 / 测试 2024）
- ✅ **多因子模型**: 20+ 个量化因子
- ✅ **预测目标**: 20 日收益率
- ✅ **机器学习**: LightGBM 回归模型
- ✅ **无未来函数**: 严格时序切分
- ✅ **回测框架**: 月度调仓，等权重分配
- ✅ **数据过滤**: 自动剔除 ST 股、停牌股

## 系统架构

```
├── config.py                  # 全局配置
├── data_loader.py             # Tushare 数据加载
├── factor_engineering.py      # 因子计算引擎
├── label_generation.py        # 标签生成（20日收益）
├── model_train.py             # LightGBM 模型训练
├── backtester.py              # 回测框架
├── main.py                    # 主程序（完整流程）
└── requirements.txt           # 依赖包
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

注意：TA-Lib 可能需要额外配置，参考 [TA-Lib 安装指南](https://github.com/mrjbq7/ta-lib)

### 2. 配置 Tushare Token

在 `config.py` 中设置您的 Tushare Token：

```python
TUSHARE_TOKEN = "11f93d2e16eda6837167b7eb1eb552f3c6641ed531bf89923ba9c85c"
```

在 [Tushare](https://www.tushare.pro) 网站注册并获取免费 Token

### 3. 运行回测

```bash
python main.py
```

系统将自动执行以下步骤：
1. 数据加载与预处理
2. 因子计算
3. 标签生成
4. 模型训练
5. 模型评估
6. 回测
7. 结果输出

## 因子体系

### 动量因子
- `momentum_5d`: 5 日动量
- `momentum_10d`: 10 日动量
- `momentum_20d`: 20 日动量

### 反转因子
- `reversal_5d`: 5 日反转
- `reversal_10d`: 10 日反转

### 技术面因子
- `rsi_14`: 14 周期 RSI
- `macd_signal`: MACD 信号
- `volatility_20d`: 20 日波动率
- `volatility_60d`: 60 日波动率

### 流动性因子
- `volume_ma_20`: 20 日成交量均线
- `volume_ratio`: 成交量比率

### 规模因子
- `market_cap_log`: 对数市值

### 估值因子
- `pb_ratio`: 市净率
- `pe_ratio`: 市盈率

### 质量因子
- `roe`: 净资产收益率
- `roa`: 总资产收益率

### 增长因子
- `revenue_growth_yoy`: 营收同比增速
- `earnings_growth_yoy`: 净利润同比增速

## 输出文件

回测结果保存在 `./output/` 目录：

- `summary.csv`: 回测摘要
- `predictions.csv`: 模型预测结果
- `trade_records.csv`: 交易明细
- `feature_importance.csv`: 特征重要性排序
- `lgb_model.bin`: 训练好的 LightGBM 模型
- `backtest.log`: 执行日志

## 回测指标说明

| 指标 | 含义 |
|------|------|
| Total Return | 总收益率 |
| Annual Return | 年化收益率 |
| Annual Volatility | 年化波动率 |
| Sharpe Ratio | 夏普比率（风险调整收益） |
| Max Drawdown | 最大回撤 |
| IC | 信息系数（预测与实际收益的相关性） |

## 模型配置

### LightGBM 参数

```python
LGBM_PARAMS = {
    "objective": "regression",        # 回归问题
    "learning_rate": 0.05,            # 学习率
    "num_leaves": 31,                 # 叶子数
    "max_depth": 7,                   # 最大深度
    "feature_fraction": 0.8,          # 特征采样
    "bagging_fraction": 0.8,          # 样本采样
    "min_child_samples": 20,          # 最小样本数
    "reg_alpha": 0.1,                 # L1 正则
    "reg_lambda": 0.1,                # L2 正则
}
```

### 回测配置

```python
BACKTEST_CONFIG = {
    "rebalance_freq": "M",            # 月度调仓
    "holding_period": 20,             # 20 日持仓
    "top_k": 30,                      # 选取前 30 名
    "transaction_cost": 0.001,        # 0.1% 交易成本
    "initial_capital": 1000000,       # 初始 100 万
}
```

## 关键设计

### 1. 时序切分（无未来函数）

- **训练集**: 2010-2020（11 年数据）
- **验证集**: 2021-2023（3 年数据）
- **测试集**: 2024（1 年数据）

严格按时间顺序切分，不存在数据泄露问题。

### 2. 因子延迟

所有因子计算使用历史数据，预测值在当日计算，用于明天或之后的交易。

### 3. 标签定义

```
label = (price[t+20] - price[t]) / price[t]
```

从 t 日收盘买入，20 日后卖出，计算收益率。

### 4. 数据过滤

- ✓ 自动剔除 ST 股
- ✓ 自动剔除停牌股
- ✓ 处理缺失值（向前填充）
- ✓ 移除极端收益
- ✓ 特征标准化

## 性能优化

### 缓存机制

第一次运行会下载数据并缓存到本地，后续运行可直接使用缓存，加快速度。

```python
data = data_loader.batch_get_daily_data(
    ts_codes, start_date, end_date, cache=True
)
```

### 批量操作

使用 Tushare 的批量接口减少请求次数，加快数据下载。

## 常见问题

### Q: Tushare Token 不可用

**A**: 确保已在 [Tushare](https://www.tushare.pro) 注册，获取免费 Token，并在 `config.py` 中正确配置。

### Q: TA-Lib 导入失败

**A**: 参考 [TA-Lib 安装指南](https://github.com/mrjbq7/ta-lib)，不同操作系统安装方式不同。

### Q: 内存不足

**A**: 可以分期间运行，或减少 `BACKTEST_CONFIG["top_k"]` 的值。

### Q: 回测结果不好

**A**: 
- 尝试调整 `LGBM_PARAMS` 中的超参数
- 增加因子数量
- 改变 `holding_period` 的值
- 调整 `top_k` 和 `transaction_cost`

## 扩展功能

### 1. 添加自定义因子

在 `factor_engineering.py` 中添加新的因子计算方法，然后在 `config.py` 的 `FACTOR_CONFIG` 中配置。

### 2. 优化超参数

使用网格搜索或贝叶斯优化来找到最优参数：

```python
from sklearn.model_selection import GridSearchCV
# ...
```

### 3. 多头空头策略

选择排名靠后的股票作为空头头寸，增加策略复杂性。

### 4. 实盘交易

集成券商 API（如同花顺、东方财富等），实现自动下单。

## 风险提示

⚠️ **免责声明**：
- 本项目仅供学习和研究使用
- 过去表现不代表未来收益
- 实际交易需谨慎，建议先进行充分的风险评估
- 量化交易存在系统性风险和模型风险
- 建议与专业投资顾问合作

## 参考资源

- [Tushare 文档](https://tushare.pro/)
- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)
- [量化交易基础](https://www.joinquant.com/)
- [A 股因子研究](https://www.ricequant.com/)

## License

MIT License

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

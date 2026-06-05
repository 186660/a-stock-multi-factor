# ============================================================
# 回测引擎
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from config import BACKTEST_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """回测引擎"""
    
    def __init__(self, predictions_df, daily_data_dict):
        """
        初始化回测引擎
        
        Parameters:
        -----------
        predictions_df: pd.DataFrame
            包含预测结果的数据框
            必需列: trade_date, ts_code, pred_return
        daily_data_dict: dict
            {ts_code: daily_data_df}
        """
        self.predictions = predictions_df.copy()
        self.daily_data = daily_data_dict
        self.portfolio_value = []
        self.positions = []
    
    def backtest(self):
        """
        执行回测
        
        Returns:
        --------
        dict
            回测结果
        """
        # 按日期分组
        self.predictions['trade_date'] = pd.to_datetime(self.predictions['trade_date'])
        dates = sorted(self.predictions['trade_date'].unique())
        
        cash = BACKTEST_CONFIG['initial_capital']
        portfolio_values = []
        trade_records = []
        
        logger.info(f"回测期间: {dates[0]} 到 {dates[-1]}")
        logger.info(f"初始资金: {cash:,.0f}")
        
        for date in dates:
            # 获取该日期的预测结果
            day_preds = self.predictions[self.predictions['trade_date'] == date].copy()
            
            # 选取排名前 top_k 的股票
            top_k = BACKTEST_CONFIG['top_k']
            day_preds = day_preds.nlargest(top_k, 'pred_return')
            
            selected_stocks = day_preds[['ts_code', 'pred_return']].to_dict('records')
            
            if len(selected_stocks) == 0:
                portfolio_values.append(cash)
                continue
            
            # 等权重分配
            allocation_per_stock = cash / len(selected_stocks)
            
            # 计算未来 holding_period 日的实际收益
            holding_period = BACKTEST_CONFIG['holding_period']
            future_date = self._get_future_date(date, holding_period)
            
            period_pnl = 0
            
            for stock_info in selected_stocks:
                ts_code = stock_info['ts_code']
                
                # 获取该股票的数据
                if ts_code not in self.daily_data:
                    continue
                
                stock_data = self.daily_data[ts_code].copy()
                stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                
                # 获取买入价
                buy_price_data = stock_data[stock_data['trade_date'] == date]
                if len(buy_price_data) == 0:
                    continue
                
                buy_price = buy_price_data.iloc[0]['close']
                
                # 获取卖出价
                sell_price_data = stock_data[stock_data['trade_date'] == future_date]
                if len(sell_price_data) == 0:
                    continue
                
                sell_price = sell_price_data.iloc[0]['close']
                
                # 计算交易收益
                actual_return = (sell_price - buy_price) / buy_price
                
                # 扣除交易成本
                cost = 2 * BACKTEST_CONFIG['transaction_cost']  # 买入 + 卖出
                net_return = actual_return - cost
                
                pnl = allocation_per_stock * net_return
                period_pnl += pnl
                
                trade_record = {
                    'date': date,
                    'ts_code': ts_code,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': actual_return,
                    'net_return': net_return,
                    'pnl': pnl,
                }
                trade_records.append(trade_record)
            
            cash += period_pnl
            portfolio_values.append(cash)
        
        # 计算回测指标
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - BACKTEST_CONFIG['initial_capital']) / BACKTEST_CONFIG['initial_capital']
        annual_return = (portfolio_values[-1] / BACKTEST_CONFIG['initial_capital']) ** (252 / len(dates)) - 1
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'final_value': portfolio_values[-1],
            'trade_records': pd.DataFrame(trade_records),
        }
        
        logger.info(f"\n\n回测结果:")
        logger.info(f"总收益率: {total_return*100:.2f}%")
        logger.info(f"年化收益率: {annual_return*100:.2f}%")
        logger.info(f"年化波动率: {annual_volatility*100:.2f}%")
        logger.info(f"Sharpe 比率: {sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {max_dd*100:.2f}%")
        logger.info(f"最终资产: {portfolio_values[-1]:,.0f}")
        
        return results
    
    @staticmethod
    def _get_future_date(date, periods):
        """
        获取未来日期（交易日）
        
        简化实现：假设每年 252 个交易日
        """
        date_obj = pd.to_datetime(date)
        future_date = date_obj + pd.Timedelta(days=periods)
        return future_date
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_values):
        """
        计算最大回撤
        """
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        return np.min(drawdown)

# ============================================================
# 标签生成模块 - 预测 20 日收益
# ============================================================

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """标签生成器 - 计算未来 N 日收益"""
    
    @staticmethod
    def calculate_future_returns(daily_data, holding_period=20):
        """
        计算未来 holding_period 日的收益率
        注意: 无未来函数，只使用历史数据
        """
        df = daily_data.copy()
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算未来 holding_period 日的收益
        df['future_close'] = df['close'].shift(-holding_period)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # 移除无法计算标签的行（最后 holding_period 行）
        df = df[:-holding_period].copy()
        
        return df
    
    @staticmethod
    def generate_labels(daily_data_dict, holding_period=20, remove_extreme=True):
        """
        为所有股票生成标签
        
        Parameters:
        -----------
        daily_data_dict: dict
            {ts_code: daily_data_df}
        holding_period: int
            持有期天数（预测周期）
        remove_extreme: bool
            是否移除极端收益
        
        Returns:
        --------
        pd.DataFrame
            包含标签的数据框
        """
        all_data = []
        
        for ts_code, df in daily_data_dict.items():
            df_with_label = LabelGenerator.calculate_future_returns(
                df, holding_period
            )
            if len(df_with_label) > 0:
                all_data.append(df_with_label)
        
        if len(all_data) == 0:
            logger.warning("没有生成任何标签")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 移除极端收益
        if remove_extreme:
            q1 = result['future_return'].quantile(0.01)
            q99 = result['future_return'].quantile(0.99)
            result = result[(result['future_return'] >= q1) & (result['future_return'] <= q99)]
        
        result = result.rename(columns={'future_return': 'label'})
        result = result.drop('future_close', axis=1)
        
        logger.info(f"标签生成完成: {len(result)} 条记录")
        logger.info(f"标签均值: {result['label'].mean():.4f}, 标准差: {result['label'].std():.4f}")
        
        return result
    
    @staticmethod
    def add_returns_bins(df, n_bins=5):
        """
        将连续收益分组（用于分类评估）
        
        Returns:
        --------
        pd.DataFrame
            添加 'label_bin' 列
        """
        df = df.copy()
        df['label_bin'] = pd.qcut(df['label'], q=n_bins, labels=False, duplicates='drop')
        return df

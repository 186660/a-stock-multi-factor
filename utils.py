# ============================================================
# 工具函数
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DateUtils:
    """日期工具类"""
    
    @staticmethod
    def get_trading_dates(start_date, end_date, date_list=None):
        """
        获取交易日期列表
        
        Parameters:
        -----------
        start_date: str
            开始日期 (YYYYMMDD)
        end_date: str
            结束日期 (YYYYMMDD)
        date_list: list, optional
            已知的交易日期列表
        
        Returns:
        --------
        list
            交易日期列表
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 简单实现：排除周末
        all_dates = pd.date_range(start=start, end=end, freq='D')
        trading_dates = [d for d in all_dates if d.weekday() < 5]
        
        return [d.strftime('%Y%m%d') for d in trading_dates]
    
    @staticmethod
    def get_last_trading_date(date_str, date_list):
        """
        获取前一个交易日
        """
        date_list = sorted(date_list)
        date = pd.to_datetime(date_str)
        
        previous_dates = [d for d in date_list if pd.to_datetime(d) < date]
        
        if len(previous_dates) == 0:
            return None
        
        return previous_dates[-1]
    
    @staticmethod
    def get_next_trading_date(date_str, date_list):
        """
        获取后一个交易日
        """
        date_list = sorted(date_list)
        date = pd.to_datetime(date_str)
        
        next_dates = [d for d in date_list if pd.to_datetime(d) > date]
        
        if len(next_dates) == 0:
            return None
        
        return next_dates[0]


class PerformanceUtils:
    """性能评估工具类"""
    
    @staticmethod
    def calculate_ic(predictions, actuals):
        """
        计算信息系数 (Information Coefficient)
        """
        return np.corrcoef(predictions, actuals)[0, 1]
    
    @staticmethod
    def calculate_rank_ic(predictions, actuals):
        """
        计算排序信息系数
        """
        pred_rank = pd.Series(predictions).rank()
        actual_rank = pd.Series(actuals).rank()
        return np.corrcoef(pred_rank, actual_rank)[0, 1]
    
    @staticmethod
    def calculate_hit_ratio(predictions, actuals, threshold=0):
        """
        计算命中率（预测方向正确的比例）
        """
        pred_sign = np.sign(predictions - threshold)
        actual_sign = np.sign(actuals - threshold)
        
        hits = np.sum(pred_sign == actual_sign)
        total = len(predictions)
        
        return hits / total
    
    @staticmethod
    def calculate_group_returns(predictions, actuals, groups=5):
        """
        按预测值分组，计算各组的平均实际收益
        
        Returns:
        --------
        list
            各组的平均收益
        """
        df = pd.DataFrame({
            'pred': predictions,
            'actual': actuals
        })
        
        df['group'] = pd.qcut(df['pred'], q=groups, labels=False)
        group_returns = df.groupby('group')['actual'].mean().tolist()
        
        return group_returns


class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def ensure_dir(directory):
        """
        确保目录存在
        """
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    @staticmethod
    def save_df(df, filepath, index=False):
        """
        保存数据框
        """
        df.to_csv(filepath, index=index)
        logger.info(f"文件已保存: {filepath}")
    
    @staticmethod
    def load_df(filepath):
        """
        加载数据框
        """
        df = pd.read_csv(filepath)
        logger.info(f"文件已加载: {filepath}")
        return df

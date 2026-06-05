# ============================================================
# 数据加载模块 - Tushare 数据源
# ============================================================

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TushareDataLoader:
    """Tushare 数据加载器"""
    
    def __init__(self, token=TUSHARE_TOKEN):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = DATA_CONFIG["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_stock_list(self, date_str):
        """获取指定日期的 A 股列表"""
        try:
            # 获取基础信息
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',  # L: 上市, D: 停牌, P: 暂停上市
                fields='ts_code,symbol,name,market,list_date'
            )
            # 过滤上市股票
            df['list_date'] = pd.to_datetime(df['list_date'])
            date_obj = pd.to_datetime(date_str)
            df = df[df['list_date'] <= date_obj]
            return df['ts_code'].tolist()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_daily_data(self, ts_code, start_date, end_date):
        """获取日线数据"""
        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )
            if df is None or len(df) == 0:
                return None
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['close'] = df['close'].astype(float)
            df['vol'] = df['vol'].astype(float)
            df['amount'] = df['amount'].astype(float)
            return df
        except Exception as e:
            logger.debug(f"获取 {ts_code} 日线数据失败: {e}")
            return None
    
    def get_stock_valuation(self, ts_code, start_date, end_date):
        """获取估值数据（PE, PB等）"""
        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,pb,pe,ps,dv_ratio,dv_ttm,total_mv'
            )
            if df is None or len(df) == 0:
                return None
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            return df
        except Exception as e:
            logger.debug(f"获取 {ts_code} 估值数据失败: {e}")
            return None
    
    def get_financial_data(self, ts_code):
        """获取财务数据"""
        try:
            # 获取年度财务数据
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                fields='ts_code,ann_date,roe,roa,gross_profit_margin,net_profit_margin,'
                       'total_assets,revenue_yoy,profit_yoy'
            )
            if df is None or len(df) == 0:
                return None
            
            df['ann_date'] = pd.to_datetime(df['ann_date'])
            df = df.sort_values('ann_date').reset_index(drop=True)
            return df
        except Exception as e:
            logger.debug(f"获取 {ts_code} 财务数据失败: {e}")
            return None
    
    def get_st_info(self, date_str):
        """获取 ST 股票列表"""
        try:
            df = self.pro.namechange(
                start_date=date_str,
                end_date=date_str,
                fields='ts_code,name_before,name_after,change_date,change_reason'
            )
            if df is None or len(df) == 0:
                return []
            # 过滤包含 ST 的股票
            st_codes = df[df['name_after'].str.contains('ST', na=False)]['ts_code'].unique()
            return st_codes.tolist()
        except Exception as e:
            logger.debug(f"获取 ST 股票列表失败: {e}")
            return []
    
    def batch_get_daily_data(self, ts_codes, start_date, end_date, cache=True):
        """批量获取日线数据"""
        cache_file = os.path.join(
            self.cache_dir,
            f"daily_data_{start_date}_{end_date}.pkl"
        )
        
        if cache and os.path.exists(cache_file):
            logger.info(f"从缓存加载数据: {cache_file}")
            return pd.read_pickle(cache_file)
        
        all_data = {}
        total = len(ts_codes)
        
        for idx, ts_code in enumerate(ts_codes):
            if (idx + 1) % 100 == 0:
                logger.info(f"进度: {idx + 1}/{total}")
            
            df = self.get_daily_data(ts_code, start_date, end_date)
            if df is not None and len(df) > 0:
                all_data[ts_code] = df
        
        if cache:
            pd.to_pickle(all_data, cache_file)
            logger.info(f"数据已保存到缓存: {cache_file}")
        
        return all_data
    
    def filter_stocks(self, ts_codes, date_str):
        """过滤股票（剔除ST、停牌等）"""
        filtered_codes = []
        
        # 剔除ST股
        if DATA_CONFIG["exclude_st"]:
            st_codes = self.get_st_info(date_str)
            ts_codes = [code for code in ts_codes if code not in st_codes]
        
        # 剔除停牌股
        if DATA_CONFIG["exclude_suspended"]:
            for ts_code in ts_codes:
                df = self.get_daily_data(ts_code, date_str, date_str)
                if df is not None and len(df) > 0:
                    filtered_codes.append(ts_code)
        else:
            filtered_codes = ts_codes
        
        return filtered_codes

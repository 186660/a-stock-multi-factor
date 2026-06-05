# ============================================================
# 因子工程模块
# ============================================================

import pandas as pd
import numpy as np
from config import *
import talib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorEngine:
    """因子计算引擎"""
    
    @staticmethod
    def calculate_momentum(prices, window=5):
        """动量因子: (当前价 - N日前价) / N日前价"""
        return prices.pct_change(window)
    
    @staticmethod
    def calculate_reversal(prices, window=5):
        """反转因子: 取反的动量"""
        return -FactorEngine.calculate_momentum(prices, window)
    
    @staticmethod
    def calculate_volatility(prices, returns, window=20):
        """波动率因子"""
        return returns.rolling(window).std()
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """相对强弱指数 (RSI)"""
        try:
            return pd.Series(
                talib.RSI(prices.values, timeperiod=window),
                index=prices.index
            ) / 100.0  # 归一化到 [0,1]
        except:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            return 100 - 100 / (1 + rs)
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD 指标"""
        try:
            macd = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(macd[0] - macd[1], index=prices.index)
        except:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            return macd_line - signal_line
    
    @staticmethod
    def calculate_turnover(volume, vol_ma, window=20):
        """成交量变化率"""
        vol_ratio = volume / vol_ma
        return vol_ratio.rolling(window).mean()
    
    @staticmethod
    def calculate_features(daily_data, valuation_data=None, financial_data=None):
        """计算所有因子"""
        df = daily_data.copy()
        features = pd.DataFrame(index=df.index)
        features['trade_date'] = df['trade_date']
        features['ts_code'] = df['ts_code']
        
        close = df['close'].values
        volume = df['vol'].values
        returns = df['close'].pct_change().values
        
        # 动量因子
        for window in [5, 10, 20]:
            features[f'momentum_{window}d'] = FactorEngine.calculate_momentum(
                df['close'], window
            )
        
        # 反转因子
        for window in [5, 10]:
            features[f'reversal_{window}d'] = FactorEngine.calculate_reversal(
                df['close'], window
            )
        
        # 波动率因子
        for window in [20, 60]:
            features[f'volatility_{window}d'] = FactorEngine.calculate_volatility(
                df['close'], df['close'].pct_change(), window
            )
        
        # 技术面因子
        features['rsi_14'] = FactorEngine.calculate_rsi(df['close'], 14)
        features['macd_signal'] = FactorEngine.calculate_macd(df['close'])
        
        # 成交量因子
        features['volume_ma_20'] = df['vol'].rolling(20).mean()
        features['volume_ratio'] = df['vol'] / features['volume_ma_20']
        
        # 对数市值（用收盘价 * 成交量估计）
        features['market_cap_log'] = np.log1p(df['close'] * df['vol'])
        
        # 估值因子（如果有数据）
        if valuation_data is not None:
            val_df = valuation_data.set_index('trade_date')
            features['pb_ratio'] = val_df['pb'] if 'pb' in val_df.columns else np.nan
            features['pe_ratio'] = val_df['pe'] if 'pe' in val_df.columns else np.nan
        
        # 财务因子（如果有数据）
        if financial_data is not None:
            # 注意：财务数据频率低，需要 forward fill
            fin_df = financial_data.copy()
            fin_df.set_index('ann_date', inplace=True)
            
            for col in ['roe', 'roa', 'revenue_yoy', 'profit_yoy']:
                if col in fin_df.columns:
                    features[col] = np.nan
                    for date in features['trade_date']:
                        recent_fin = fin_df[fin_df.index <= date]
                        if len(recent_fin) > 0:
                            features.loc[features['trade_date'] == date, col] = recent_fin[col].iloc[-1]
        
        return features
    
    @staticmethod
    def handle_missing_values(df, method='forward_fill'):
        """处理缺失值"""
        df = df.copy()
        
        # 按 ts_code 分组处理
        for ts_code in df['ts_code'].unique():
            mask = df['ts_code'] == ts_code
            
            if method == 'forward_fill':
                df.loc[mask] = df.loc[mask].fillna(method='ffill')
            elif method == 'backward_fill':
                df.loc[mask] = df.loc[mask].fillna(method='bfill')
            elif method == 'interpolate':
                df.loc[mask] = df.loc[mask].interpolate(method='linear')
        
        # 删除仍有缺失值的行
        df = df.dropna()
        
        return df
    
    @staticmethod
    def remove_outliers(df, std_threshold=3):
        """移除异常值"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            mask = (df[col] - mean).abs() <= std_threshold * std
            df = df[mask]
        
        return df
    
    @staticmethod
    def standardize_features(df, exclude_cols=['trade_date', 'ts_code', 'label']):
        """特征标准化"""
        from sklearn.preprocessing import StandardScaler
        
        df = df.copy()
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        return df, scaler

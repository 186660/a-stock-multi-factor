# ============================================================
# 模型训练模块 - LightGBM
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """LightGBM 模型训练器"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
    
    def prepare_data(self, df, exclude_cols=['trade_date', 'ts_code', 'label']):
        """
        准备训练数据
        
        Parameters:
        -----------
        df: pd.DataFrame
            包含特征和标签的数据框
        exclude_cols: list
            排除的列
        
        Returns:
        --------
        X, y, feature_names
        """
        # 提取特征
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if not col.startswith('future_')]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        # 处理缺失值
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练 LightGBM 模型
        
        Parameters:
        -----------
        X_train: pd.DataFrame
            训练特征
        y_train: pd.Series
            训练标签
        X_val: pd.DataFrame, optional
            验证特征
        y_val: pd.Series, optional
            验证标签
        """
        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_names=self.feature_names)
        
        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        logger.info("开始训练 LightGBM 模型...")
        self.model = lgb.train(
            params=LGBM_PARAMS,
            train_set=train_data,
            valid_sets=[valid_data] if valid_data is not None else None,
            num_boost_round=LGBM_TRAIN_PARAMS['num_boost_round'],
            early_stopping_rounds=LGBM_TRAIN_PARAMS['early_stopping_rounds'],
            callbacks=[
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=LGBM_TRAIN_PARAMS['early_stopping_rounds'])
            ]
        )
        
        logger.info(f"模型训练完成，共 {self.model.num_trees()} 棵树")
    
    def predict(self, X_test):
        """
        模型预测
        
        Parameters:
        -----------
        X_test: pd.DataFrame
            测试特征
        
        Returns:
        --------
        np.array
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)
    
    def evaluate(self, y_true, y_pred, set_name="Test"):
        """
        模型评估
        
        Parameters:
        -----------
        y_true: np.array
            真实标签
        y_pred: np.array
            预测值
        set_name: str
            数据集名称
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算 IC（信息系数）
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        logger.info("-" * 50)
        logger.info(f"{set_name} Set Performance:")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE:  {mae:.6f}")
        logger.info(f"R2:   {r2:.6f}")
        logger.info(f"IC:   {ic:.6f}")
        logger.info("-" * 50)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
        }
    
    def get_feature_importance(self, top_k=20):
        """
        获取特征重要性
        
        Parameters:
        -----------
        top_k: int
            返回排名前 k 的特征
        
        Returns:
        --------
        pd.DataFrame
            特征重要性排序
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        importance = self.model.feature_importance()
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n特征重要性 (Top {top_k}):")
        logger.info(feature_importance_df.head(top_k).to_string(index=False))
        
        return feature_importance_df.head(top_k)
    
    def save_model(self, filepath):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.save_model(filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        """
        self.model = lgb.Booster(model_file=filepath)
        logger.info(f"模型已加载: {filepath}")

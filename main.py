# ============================================================
# 主程序 - 完整流程
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from config import *
from data_loader import TushareDataLoader
from factor_engineering import FactorEngine
from label_generation import LabelGenerator
from model_train import LightGBMTrainer
from backtester import Backtester

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./output/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建输出目录
os.makedirs(DATA_CONFIG['output_dir'], exist_ok=True)
os.makedirs(DATA_CONFIG['cache_dir'], exist_ok=True)


def main():
    logger.info("="*60)
    logger.info("A 股多因子 LightGBM 量化交易系统")
    logger.info("="*60)
    
    # ========== 第一步：数据加载 ==========
    logger.info("\n[第一步] 数据加载...")
    data_loader = TushareDataLoader()
    
    # 获取训练期股票列表
    train_stocks = data_loader.get_stock_list(TRAIN_START)
    train_stocks = data_loader.filter_stocks(train_stocks, TRAIN_START)
    logger.info(f"训练期股票数: {len(train_stocks)}")
    
    # 获取训练数据
    train_data = data_loader.batch_get_daily_data(
        train_stocks, TRAIN_START, TRAIN_END
    )
    logger.info(f"训练数据: {len(train_data)} 只股票")
    
    # 获取验证数据
    val_stocks = data_loader.get_stock_list(VAL_START)
    val_stocks = data_loader.filter_stocks(val_stocks, VAL_START)
    val_data = data_loader.batch_get_daily_data(
        val_stocks, VAL_START, VAL_END
    )
    logger.info(f"验证数据: {len(val_data)} 只股票")
    
    # 获取测试数据
    test_stocks = data_loader.get_stock_list(TEST_START)
    test_stocks = data_loader.filter_stocks(test_stocks, TEST_START)
    test_data = data_loader.batch_get_daily_data(
        test_stocks, TEST_START, TEST_END
    )
    logger.info(f"测试数据: {len(test_data)} 只股票")
    
    # ========== 第二步：因子计算 ==========
    logger.info("\n[第二步] 因子计算...")
    
    def calculate_factors(daily_data_dict):
        """为所有股票计算因子"""
        all_features = []
        for ts_code, daily_df in daily_data_dict.items():
            features_df = FactorEngine.calculate_features(daily_df)
            all_features.append(features_df)
        
        if len(all_features) == 0:
            logger.error("未生成任何因子")
            return pd.DataFrame()
        
        return pd.concat(all_features, ignore_index=True)
    
    train_features = calculate_factors(train_data)
    logger.info(f"训练特征: {train_features.shape}")
    
    val_features = calculate_factors(val_data)
    logger.info(f"验证特征: {val_features.shape}")
    
    test_features = calculate_factors(test_data)
    logger.info(f"测试特征: {test_features.shape}")
    
    # ========== 第三步：标签生成 ==========
    logger.info("\n[第三步] 标签生成 (20日收益)...")
    
    train_labels = LabelGenerator.generate_labels(
        train_data, 
        holding_period=BACKTEST_CONFIG['holding_period']
    )
    
    val_labels = LabelGenerator.generate_labels(
        val_data,
        holding_period=BACKTEST_CONFIG['holding_period']
    )
    
    test_labels = LabelGenerator.generate_labels(
        test_data,
        holding_period=BACKTEST_CONFIG['holding_period']
    )
    
    # ========== 第四步：数据合并和预处理 ==========
    logger.info("\n[第四步] 数据合并和预处理...")
    
    def merge_features_labels(features, labels):
        """合并特征和标签"""
        merged = features.merge(
            labels[['trade_date', 'ts_code', 'label']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        return merged
    
    train_df = merge_features_labels(train_features, train_labels)
    val_df = merge_features_labels(val_features, val_labels)
    test_df = merge_features_labels(test_features, test_labels)
    
    logger.info(f"合并后 - 训练集: {train_df.shape}, 验证集: {val_df.shape}, 测试集: {test_df.shape}")
    
    # 处理缺失值和异常值
    train_df = FactorEngine.handle_missing_values(
        train_df,
        method=FEATURE_ENGINEERING_CONFIG['handle_missing']
    )
    val_df = FactorEngine.handle_missing_values(
        val_df,
        method=FEATURE_ENGINEERING_CONFIG['handle_missing']
    )
    test_df = FactorEngine.handle_missing_values(
        test_df,
        method=FEATURE_ENGINEERING_CONFIG['handle_missing']
    )
    
    if FEATURE_ENGINEERING_CONFIG['remove_outliers']:
        train_df = FactorEngine.remove_outliers(
            train_df,
            std_threshold=FEATURE_ENGINEERING_CONFIG['outlier_std']
        )
    
    logger.info(f"预处理后 - 训练集: {train_df.shape}, 验证集: {val_df.shape}, 测试集: {test_df.shape}")
    
    # ========== 第五步：模型训练 ==========
    logger.info("\n[第五步] 模型训练...")
    
    trainer = LightGBMTrainer()
    
    # 准备数据
    X_train, y_train, feature_names = trainer.prepare_data(train_df)
    X_val, y_val, _ = trainer.prepare_data(val_df)
    X_test, y_test, _ = trainer.prepare_data(test_df)
    
    logger.info(f"特征维度: {len(feature_names)}")
    logger.info(f"特征列表: {feature_names[:10]}...")  # 显示前10个
    
    # 训练模型
    trainer.train(X_train, y_train, X_val, y_val)
    
    # ========== 第六步：模型评估 ==========
    logger.info("\n[第六步] 模型评估...")
    
    pred_val = trainer.predict(X_val)
    val_metrics = trainer.evaluate(y_val, pred_val, "Validation")
    
    pred_test = trainer.predict(X_test)
    test_metrics = trainer.evaluate(y_test, pred_test, "Test")
    
    # 特征重要性
    feature_importance = trainer.get_feature_importance(top_k=20)
    
    # ========== 第七步：模型预测 ==========
    logger.info("\n[第七步] 生成预测结果...")
    
    test_df['pred_return'] = pred_test
    predictions = test_df[['trade_date', 'ts_code', 'pred_return', 'label']].copy()
    predictions.to_csv(
        os.path.join(DATA_CONFIG['output_dir'], 'predictions.csv'),
        index=False
    )
    logger.info(f"预测结果已保存: {len(predictions)} 条")
    
    # ========== 第八步：执行回测 ==========
    logger.info("\n[第八步] 执行回测...")
    
    backtester = Backtester(predictions, test_data)
    backtest_results = backtester.backtest()
    
    # 保存回测结果
    if len(backtest_results['trade_records']) > 0:
        backtest_results['trade_records'].to_csv(
            os.path.join(DATA_CONFIG['output_dir'], 'trade_records.csv'),
            index=False
        )
    
    # ========== 第九步：结果汇总 ==========
    logger.info("\n[第九步] 结果汇总...")
    
    summary = {
        'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Train Period': f"{TRAIN_START} - {TRAIN_END}",
        'Val Period': f"{VAL_START} - {VAL_END}",
        'Test Period': f"{TEST_START} - {TEST_END}",
        'Train Samples': len(train_df),
        'Val Samples': len(val_df),
        'Test Samples': len(test_df),
        'Feature Dimension': len(feature_names),
        'Model Trees': trainer.model.num_trees(),
        'Val RMSE': val_metrics['rmse'],
        'Test RMSE': test_metrics['rmse'],
        'Test IC': test_metrics['ic'],
        'Backtest Total Return': backtest_results['total_return'],
        'Backtest Annual Return': backtest_results['annual_return'],
        'Backtest Annual Volatility': backtest_results['annual_volatility'],
        'Backtest Sharpe Ratio': backtest_results['sharpe_ratio'],
        'Backtest Max Drawdown': backtest_results['max_drawdown'],
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(DATA_CONFIG['output_dir'], 'summary.csv'),
        index=False
    )
    
    # 保存特征重要性
    feature_importance.to_csv(
        os.path.join(DATA_CONFIG['output_dir'], 'feature_importance.csv'),
        index=False
    )
    
    # 保存模型
    trainer.save_model(
        os.path.join(DATA_CONFIG['output_dir'], 'lgb_model.bin')
    )
    
    logger.info("\n" + "="*60)
    logger.info("回测完成！")
    logger.info(f"输出目录: {DATA_CONFIG['output_dir']}")
    logger.info("="*60)
    
    return summary_df, backtest_results


if __name__ == '__main__':
    try:
        summary, results = main()
        print("\n" + summary.to_string(index=False))
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

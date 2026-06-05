# ============================================================
# 回测结果报告生成脚本
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_backtest_report():
    """生成完整的回测结果报告"""
    
    output_dir = "./output"
    
    logger.info("="*80)
    logger.info("A股多因子LightGBM量化交易系统 - 回测结果报告")
    logger.info("="*80)
    
    # ========== 1. 读取摘要数据 ==========
    logger.info("\n[1] 加载回测摘要数据...")
    
    summary_files = [f for f in os.listdir(output_dir) if f.startswith('summary_') and f.endswith('.csv')]
    
    if not summary_files:
        logger.error("未找到摘要文件，请先运行回测脚本")
        return
    
    for summary_file in sorted(summary_files):
        summary_path = os.path.join(output_dir, summary_file)
        summary_df = pd.read_csv(summary_path)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"报告文件: {summary_file}")
        logger.info(f"{'='*80}\n")
        
        # 打印基本信息
        print("\n📅 【时间段信息】")
        print(f"  报告生成时间: {summary_df['Report Date'].values[0]}")
        print(f"  训练周期: {summary_df['Training Period'].values[0]}")
        print(f"  验证周期: {summary_df['Validation Period'].values[0]}")
        print(f"  测试周期: {summary_df['Test Period'].values[0]}")
        print(f"  回测周期: {summary_df['Backtest Period'].values[0]}")
        
        # 打印数据量信息
        print("\n📊 【数据量统计】")
        print(f"  训练样本数: {int(summary_df['Train Samples'].values[0]):,}")
        print(f"  验证样本数: {int(summary_df['Val Samples'].values[0]):,}")
        print(f"  测试样本数: {int(summary_df['Test Samples'].values[0]):,}")
        print(f"  回测样本数: {int(summary_df['Backtest Samples'].values[0]):,}")
        
        # 打印模型信息
        print("\n🤖 【模型信息】")
        print(f"  特征维度: {int(summary_df['Feature Dimension'].values[0])}")
        print(f"  决策树数: {int(summary_df['Model Trees'].values[0])}")
        
        # 打印模型性能指标
        print("\n📈 【模型性能指标】")
        print(f"  验证集 RMSE: {summary_df['Validation RMSE'].values[0]:.6f}")
        print(f"  测试集 RMSE: {summary_df['Test RMSE'].values[0]:.6f}")
        print(f"  测试集 IC (信息系数): {summary_df['Test IC'].values[0]:.6f}")
        print(f"  测试集 R²: {summary_df['Test R2'].values[0]:.6f}")
        
        # 打印回测结果
        print("\n💰 【回测结果】")
        total_return = summary_df['Backtest Total Return'].values[0]
        annual_return = summary_df['Backtest Annual Return'].values[0]
        annual_vol = summary_df['Backtest Annual Volatility'].values[0]
        sharpe_ratio = summary_df['Backtest Sharpe Ratio'].values[0]
        max_dd = summary_df['Backtest Max Drawdown'].values[0]
        final_value = summary_df['Final Portfolio Value'].values[0]
        
        print(f"  总收益率: {total_return*100:>8.2f}%")
        print(f"  年化收益率: {annual_return*100:>8.2f}%")
        print(f"  年化波动率: {annual_vol*100:>8.2f}%")
        print(f"  Sharpe比率: {sharpe_ratio:>8.4f}")
        print(f"  最大回���: {max_dd*100:>8.2f}%")
        print(f"  最终资产值: ¥{final_value:>15,.0f}")
        
        # 计算风险调整指标
        if annual_vol > 0:
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
            print(f"  Calmar比率: {calmar_ratio:>8.4f}")
        
        # ========== 2. 特征重要性 ==========
        logger.info("\n[2] 加载特征重要性数据...")
        
        feature_file = summary_file.replace('summary_', 'feature_importance_')
        feature_path = os.path.join(output_dir, feature_file)
        
        if os.path.exists(feature_path):
            feature_importance_df = pd.read_csv(feature_path)
            print("\n🎯 【特征重要性排序 (Top 10)】")
            print(feature_importance_df.head(10).to_string(index=False))
        
        # ========== 3. 交易记录统计 ==========
        logger.info("\n[3] 分析交易记录...")
        
        trade_file = summary_file.replace('summary_', 'trade_records_')
        trade_path = os.path.join(output_dir, trade_file)
        
        if os.path.exists(trade_path):
            trades_df = pd.read_csv(trade_path)
            
            print("\n📋 【交易统计】")
            print(f"  总交易数: {len(trades_df)}")
            print(f"  盈利交易: {len(trades_df[trades_df['pnl'] > 0])}")
            print(f"  亏损交易: {len(trades_df[trades_df['pnl'] < 0])}")
            print(f"  平盘交易: {len(trades_df[trades_df['pnl'] == 0])}")
            
            # 计算胜率
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
            print(f"  胜率: {win_rate*100:.2f}%")
            
            # 计算平均收益
            avg_return = trades_df['net_return'].mean()
            print(f"  平均收益率: {avg_return*100:.4f}%")
            
            # 打印收益分布
            print("\n📊 【收益分布统计】")
            returns = trades_df['net_return']
            print(f"  最大单笔收益: {returns.max()*100:>8.4f}%")
            print(f"  最小单笔收益: {returns.min()*100:>8.4f}%")
            print(f"  平均收益: {returns.mean()*100:>8.4f}%")
            print(f"  收益中位数: {returns.median()*100:>8.4f}%")
            print(f"  收益标准差: {returns.std()*100:>8.4f}%")
            
            # 打印前10大盈利交易
            print("\n🏆 【前10大盈利交易】")
            top_wins = trades_df.nlargest(10, 'pnl')[['date', 'ts_code', 'buy_price', 'sell_price', 'return', 'pnl']]
            print(top_wins.to_string(index=False))
            
            # 打印前10大亏损交易
            print("\n⚠️  【前10大亏损交易】")
            top_loss = trades_df.nsmallest(10, 'pnl')[['date', 'ts_code', 'buy_price', 'sell_price', 'return', 'pnl']]
            print(top_loss.to_string(index=False))
        
        # ========== 4. 预测准确性 ==========
        logger.info("\n[4] 分析预测准确性...")
        
        pred_file = summary_file.replace('summary_', 'predictions_')
        pred_path = os.path.join(output_dir, pred_file)
        
        if os.path.exists(pred_path):
            predictions_df = pd.read_csv(pred_path)
            
            # 计算IC (相关系数)
            ic = predictions_df['pred_return'].corr(predictions_df['label'])
            
            # 计算分组收益
            predictions_df['pred_group'] = pd.qcut(predictions_df['pred_return'], q=5, labels=False, duplicates='drop')
            group_returns = predictions_df.groupby('pred_group')['label'].mean()
            
            print("\n🎯 【预测准确性】")
            print(f"  整体IC (信息系数): {ic:.6f}")
            print(f"  有效预测样本: {len(predictions_df):,}")
            
            print("\n📊 【按预测排名的分组收益】")
            print("  预测排名 | 实际平均收益")
            for group, ret in group_returns.items():
                print(f"    {int(group)+1}分位  |  {ret*100:>8.4f}%")
    
    logger.info("\n" + "="*80)
    logger.info("回测报告生成完成！")
    logger.info("="*80)

if __name__ == '__main__':
    try:
        generate_backtest_report()
    except Exception as e:
        logger.error(f"报告生成失败: {e}", exc_info=True)
        sys.exit(1)

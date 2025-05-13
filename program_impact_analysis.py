#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import platform

# 设置中文显示
# Check if on macOS
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB']
else:  # Windows or other systems
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun']

# Universal settings
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs properly
sns.set(style="whitegrid", font_scale=1.1)

# 创建输出目录
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """加载所有数据文件"""
    data_dir = "processed_data"
    
    # 加载月度汇总数据
    monthly_all = pd.read_csv(os.path.join(data_dir, "all_programs_monthly.csv"))
    monthly_all['month'] = pd.to_datetime(monthly_all['month'])
    
    # 加载每个项目的月度数据
    program_files = {
        0: pd.read_csv(os.path.join(data_dir, "program_0_monthly.csv")),
        1: pd.read_csv(os.path.join(data_dir, "program_1_monthly.csv")),
        4: pd.read_csv(os.path.join(data_dir, "program_4_monthly.csv")),
        22: pd.read_csv(os.path.join(data_dir, "program_22_monthly.csv")),
        5: pd.read_csv(os.path.join(data_dir, "program_5_monthly.csv")),
        15: pd.read_csv(os.path.join(data_dir, "program_15_monthly.csv"))
    }
    
    # 转换日期
    for program_id, df in program_files.items():
        df['month'] = pd.to_datetime(df['month'])
        df['intervention_date'] = pd.to_datetime(df['intervention_date'])
    
    # 加载日度数据(用于更详细分析)
    daily_all = pd.read_csv(os.path.join(data_dir, "all_programs_daily.csv"))
    
    return monthly_all, program_files, daily_all

def plot_sentiment_before_after(monthly_all, program_files):
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB']
    else:  # Windows or other systems
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun']

    # Universal settings
    plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs properly    
    """绘制项目实施前后的情感得分对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    program_names = {
        0: "同车不同温",
        1: "智能动态地图显示系统",
        4: "成功推出乘车码二维码扫码",
        22: "降低票价",
        5: "完成82个站点卫生间改造",
        15: "移动母婴室",
    }
    
    # 修改这里: 直接遍历 program_files 字典的 items()
    for i, (program_id, df) in enumerate(program_files.items()):
        if i >= len(axes):  # 安全检查，确保不会超出子图数量
            break
            
        # 分离实施前后的数据
        before = df[df['post_intervention'] == 0]
        after = df[df['post_intervention'] == 1]
        
        # 绘制时间序列
        ax = axes[i]
        ax.plot(before['month'], before['mean_sentiment'], 'o-', color='blue', label='实施前')
        ax.plot(after['month'], after['mean_sentiment'], 'o-', color='red', label='实施后')
        
        # 添加干预日期的垂直线
        intervention_date = df['intervention_date'].iloc[0]
        ax.axvline(x=intervention_date, color='green', linestyle='--', label='干预日期')
        
        # 设置标题和标签
        ax.set_title(f'项目 {program_id}: {program_names[program_id]}')
        ax.set_xlabel('日期')
        ax.set_ylabel('平均情感得分')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_before_after.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_distribution(monthly_all):
    """绘制不同项目情感得分分布对比图"""
    plt.figure(figsize=(14, 8))
    
    # 为每个项目创建实施前后的分组
    programs = monthly_all['program_id'].unique()
    data_to_plot = []
    labels = []
    
    for program_id in programs:
        program_data = monthly_all[monthly_all['program_id'] == program_id]
        program_name = program_data['program_name'].iloc[0]
        
        # 实施前
        before_data = program_data[program_data['post_intervention'] == 0]['mean_sentiment']
        data_to_plot.append(before_data)
        labels.append(f"{program_name}\n(实施前)")
        
        # 实施后
        after_data = program_data[program_data['post_intervention'] == 1]['mean_sentiment']
        data_to_plot.append(after_data)
        labels.append(f"{program_name}\n(实施后)")
    
    # 绘制箱线图
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
    
    # 设置颜色
    colors = []
    for i in range(len(programs)):
        colors.extend(['lightblue', 'lightcoral'])
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('各项目实施前后情感得分分布')
    plt.ylabel('平均情感得分')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_impact_statistics(program_files):
    """计算项目实施前后的统计差异"""
    impact_stats = []
    
    for program_id, df in program_files.items():
        before = df[df['post_intervention'] == 0]['mean_sentiment']
        after = df[df['post_intervention'] == 1]['mean_sentiment']
        
        # 计算基本统计量
        stats = {
            'program_id': program_id,
            'program_name': df['program_name'].iloc[0],
            'before_mean': before.mean(),
            'after_mean': after.mean(),
            'mean_diff': after.mean() - before.mean(),
            'before_std': before.std(),
            'after_std': after.std(),
            'before_min': before.min(),
            'after_min': after.min(),
            'before_max': before.max(),
            'after_max': after.max(),
            'sample_size_before': len(before),
            'sample_size_after': len(after)
        }
        
        impact_stats.append(stats)
    
    impact_df = pd.DataFrame(impact_stats)
    return impact_df

def plot_impact_comparison(impact_stats):
    """绘制项目影响对比图"""
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    programs = impact_stats['program_name']
    mean_diff = impact_stats['mean_diff']
    
    # 创建颜色映射
    colors = ['red' if x < 0 else 'green' for x in mean_diff]
    
    # 绘制条形图
    bars = plt.bar(programs, mean_diff, color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + 0.01 if height >= 0 else height - 0.03,
                 f'{height:.3f}',
                 ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('各项目实施前后平均情感得分变化')
    plt.ylabel('情感得分变化（实施后 - 实施前）')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "impact_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_decomposition(program_files):
    """绘制时间序列分解图，分析趋势、季节性和残差"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    for program_id, df in program_files.items():
        # 确保数据按时间排序
        df = df.sort_values('month')
        
        # 设置时间索引
        ts_data = df.set_index('month')['mean_sentiment']
        
        # 进行时间序列分解
        try:
            # 假设数据是月度数据，周期为12
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            # 创建图形
            fig = plt.figure(figsize=(14, 10))
            
            # 原始数据
            ax1 = plt.subplot(411)
            ax1.plot(ts_data.index, ts_data.values)
            ax1.set_ylabel('原始数据')
            ax1.set_title(f'项目 {program_id}: {df["program_name"].iloc[0]} - 时间序列分解')
            
            # 趋势
            ax2 = plt.subplot(412)
            ax2.plot(decomposition.trend.index, decomposition.trend.values)
            ax2.set_ylabel('趋势')
            
            # 季节性
            ax3 = plt.subplot(413)
            ax3.plot(decomposition.seasonal.index, decomposition.seasonal.values)
            ax3.set_ylabel('季节性')
            
            # 残差
            ax4 = plt.subplot(414)
            ax4.scatter(decomposition.resid.index, decomposition.resid.values)
            ax4.set_ylabel('残差')
            
            # 添加干预日期的垂直线
            intervention_date = df['intervention_date'].iloc[0]
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=intervention_date, color='red', linestyle='--', label='干预日期')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"time_series_decomposition_program_{program_id}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"无法为项目 {program_id} 进行时间序列分解: {e}")

def plot_sample_size_analysis(program_files):
    """分析样本量与情感得分的关系"""
    plt.figure(figsize=(14, 8))
    
    for i, (program_id, df) in enumerate(program_files.items()):
        if i >= 4:  # 限制子图数量，以确保不超出
            break
            
        plt.subplot(2, 2, i+1)
        
        # 分离实施前后的数据
        before = df[df['post_intervention'] == 0]
        after = df[df['post_intervention'] == 1]
        
        # 绘制散点图
        plt.scatter(before['sample_size'], before['mean_sentiment'],
                    label='实施前', alpha=0.7, color='blue')
        plt.scatter(after['sample_size'], after['mean_sentiment'],
                    label='实施后', alpha=0.7, color='red')
        
        # 添加趋势线
        if len(before) > 1:
            z1 = np.polyfit(before['sample_size'], before['mean_sentiment'], 1)
            p1 = np.poly1d(z1)
            plt.plot(before['sample_size'], p1(before['sample_size']),
                     linestyle='--', color='blue')
            
        if len(after) > 1:
            z2 = np.polyfit(after['sample_size'], after['mean_sentiment'], 1)
            p2 = np.poly1d(z2)
            plt.plot(after['sample_size'], p2(after['sample_size']),
                     linestyle='--', color='red')
            
        plt.title(f'项目 {program_id}: {df["program_name"].iloc[0]}')
        plt.xlabel('样本量')
        plt.ylabel('平均情感得分')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_size_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap_monthly_patterns(program_files):
    """绘制月度模式热图，分析不同月份的情感得分模式"""
    for program_id, df in program_files.items():
        # 提取年份和月份
        df['year'] = df['month'].dt.year
        df['month_num'] = df['month'].dt.month
        
        # 创建数据透视表
        pivot_data = df.pivot_table(index='year', columns='month_num',
                                    values='mean_sentiment', aggfunc='mean')
        
        # 绘制热图
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': '平均情感得分'})
        
        # 设置标题和标签
        plt.title(f'项目 {program_id}: {df["program_name"].iloc[0]} - 月度情感得分模式')
        plt.xlabel('月份')
        plt.ylabel('年份')
        
        # 设置月份标签
        month_labels = ['一月', '二月', '三月', '四月', '五月', '六月',
                        '七月', '八月', '九月', '十月', '十一月', '十二月']
        plt.xticks(np.arange(12) + 0.5, month_labels, rotation=45)
        
        # 添加干预时间标记
        intervention_date = df['intervention_date'].iloc[0]
        intervention_year = intervention_date.year
        intervention_month = intervention_date.month
        
        # 在热图上标记干预时间点
        if intervention_year in pivot_data.index and intervention_month in pivot_data.columns:
            plt.plot(intervention_month - 0.5, pivot_data.index.get_loc(intervention_year) + 0.5,
                     'o', markersize=12, color='green', mfc='none', mew=2, label='干预时间点')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_monthly_pattern_program_{program_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_intervention_effect_over_time(program_files):
    """分析干预效果随时间的变化"""
    plt.figure(figsize=(14, 8))
    
    for i, (program_id, df) in enumerate(program_files.items()):
        if i >= 4:  # 限制子图数量，以确保不超出
            break
            
        # 只使用干预后的数据
        after_data = df[df['post_intervention'] == 1].copy()
        
        # 计算干预前的平均情感得分作为基准
        before_mean = df[df['post_intervention'] == 0]['mean_sentiment'].mean()
        
        # 计算与基准的差异
        after_data['diff_from_baseline'] = after_data['mean_sentiment'] - before_mean
        
        # 绘制子图
        plt.subplot(2, 2, i+1)
        plt.plot(after_data['time_since_intervention'], after_data['diff_from_baseline'], 'o-')
        
        # 添加趋势线
        if len(after_data) > 1:
            z = np.polyfit(after_data['time_since_intervention'], after_data['diff_from_baseline'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(after_data['time_since_intervention'].min(),
                                after_data['time_since_intervention'].max(), 100)
            plt.plot(x_line, p(x_line), '--', color='red')
            
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'项目 {program_id}: {df["program_name"].iloc[0]}')
        plt.xlabel('干预后月数')
        plt.ylabel('与基准的差异')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intervention_effect_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(program_files, impact_stats):
    """创建汇总仪表盘，展示所有项目的关键指标"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 1])
    
    # 顶部标题
    ax_title = plt.subplot(gs[0, :])
    ax_title.text(0.5, 0.5, '项目影响分析汇总仪表盘',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20, fontweight='bold')
    ax_title.axis('off')
    
    # 项目影响对比图
    ax_impact = plt.subplot(gs[1, :])
    programs = impact_stats['program_name']
    mean_diff = impact_stats['mean_diff']
    colors = ['red' if x < 0 else 'green' for x in mean_diff]
    bars = ax_impact.bar(programs, mean_diff, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax_impact.text(bar.get_x() + bar.get_width()/2.,
                     height + 0.01 if height >= 0 else height - 0.03,
                     f'{height:.3f}',
                     ha='center', va='bottom' if height >= 0 else 'top')
    
    ax_impact.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_impact.set_title('各项目实施前后平均情感得分变化')
    ax_impact.set_ylabel('情感得分变化（实施后 - 实施前）')
    ax_impact.set_xticks(range(len(programs)))
    ax_impact.set_xticklabels(programs, rotation=45, ha='right')
    ax_impact.grid(True, axis='y', alpha=0.3)
    
    # 关键统计指标表格
    ax_stats = plt.subplot(gs[2, :])
    ax_stats.axis('tight')
    ax_stats.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['项目名称', '实施前均值', '实施后均值', '变化量', '实施前样本数', '实施后样本数']
    
    for _, row in impact_stats.iterrows():
        table_data.append([
            row['program_name'],
            f"{row['before_mean']:.3f}",
            f"{row['after_mean']:.3f}",
            f"{row['mean_diff']:.3f}",
            str(row['sample_size_before']),
            str(row['sample_size_after'])
        ])
    
    # 创建表格
    table = ax_stats.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 为表格添加颜色
    for i in range(len(table_data)):
        # 根据变化量设置颜色
        change_val = float(table_data[i][3])
        cell_color = 'lightgreen' if change_val > 0 else 'lightcoral'
        table[(i+1, 3)].set_facecolor(cell_color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    print("加载数据...")
    monthly_all, program_files, daily_all = load_data()
    
    # 计算项目影响统计数据
    print("计算项目影响统计数据...")
    impact_stats = calculate_impact_statistics(program_files)
    
    # 生成各种图表
    print("生成项目实施前后对比图...")
    plot_sentiment_before_after(monthly_all, program_files)
    
    print("生成情感得分分布图...")
    plot_sentiment_distribution(monthly_all)
    
    print("生成项目影响对比图...")
    plot_impact_comparison(impact_stats)
    
    print("生成时间序列分解图...")
    plot_time_series_decomposition(program_files)
    
    print("分析样本量与情感得分的关系...")
    plot_sample_size_analysis(program_files)
    
    print("生成月度模式热图...")
    plot_heatmap_monthly_patterns(program_files)
    
    print("分析干预效果随时间的变化...")
    plot_intervention_effect_over_time(program_files)
    
    print("创建汇总仪表盘...")
    create_summary_dashboard(program_files, impact_stats)
    
    print(f"分析完成！所有图表已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()

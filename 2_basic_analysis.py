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
output_dir = "basic_analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 新增：定义统计数据汇总文件名
summary_file_path = os.path.join(output_dir, "analysis_data_summary.txt")

def append_to_summary_file(filepath, title, content_df=None, content_text=None):
    """向汇总文件追加内容"""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n\n******************************************\n")
        f.write(f"{title}\n")
        f.write(f"******************************************\n")
        if content_df is not None:
            f.write(content_df.to_string(index=True))
            f.write("\n")
        if content_text is not None:
            f.write(content_text)
            f.write("\n")

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
        0: "Different Temperature Zones in Same Car",
        1: "Smart Dynamic Map Display System",
        4: "Successful Launch of QR Code Scanning",
        22: "Fare Reduction",
        5: "Renovation of 82 Station Restrooms",
        15: "Mobile Nursing Room",
    }
    
    for i, (program_id, df) in enumerate(program_files.items()):
        if i >= len(axes):
            break
            
        before = df[df['post_intervention'] == 0]
        after = df[df['post_intervention'] == 1]
        
        ax = axes[i]
        ax.plot(before['month'], before['mean_sentiment'], 'o-', color='blue', label='Before Implementation')
        ax.plot(after['month'], after['mean_sentiment'], 'o-', color='red', label='After Implementation')
        
        intervention_date = df['intervention_date'].iloc[0]
        ax.axvline(x=intervention_date, color='green', linestyle='--', label='Intervention Date')
        
        ax.set_title(f'Program {program_id}: {program_names.get(program_id, f"Unknown Program {program_id}")}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_before_after.svg"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_distribution(monthly_all):
    """绘制不同项目情感得分分布对比图"""
    plt.figure(figsize=(14, 8))
    
    programs = monthly_all['program_id'].unique()
    data_to_plot = []
    labels = []
    
    program_names_map = {
        0: "同车不同温", 1: "智能动态地图显示系统", 4: "成功推出乘车码二维码扫码",
        22: "降低票价", 5: "完成82个站点卫生间改造", 15: "移动母婴室"
    }

    for program_id in programs:
        program_data = monthly_all[monthly_all['program_id'] == program_id]
        # Use the map, fall back to a default if program_id not found or name is missing
        program_name = program_names_map.get(program_id, f"项目 {program_id}")
        if not program_data.empty and 'program_name' in program_data.columns:
             # Prefer actual name from data if available and consistent
            actual_name = program_data['program_name'].iloc[0]
            if pd.notna(actual_name) : program_name = actual_name

        before_data = program_data[program_data['post_intervention'] == 0]['mean_sentiment']
        if not before_data.empty:
            data_to_plot.append(before_data)
            labels.append(f"{program_name}\n(Before Implementation)")
        
        after_data = program_data[program_data['post_intervention'] == 1]['mean_sentiment']
        if not after_data.empty:
            data_to_plot.append(after_data)
            labels.append(f"{program_name}\n(After Implementation)")
    
    if not data_to_plot:
        plt.text(0.5, 0.5, "No data available for boxplot", ha='center', va='center')
        plt.title('Sentiment Score Distribution Before and After Implementation')
        plt.savefig(os.path.join(output_dir, "sentiment_distribution.svg"), dpi=300, bbox_inches='tight')
        plt.close()
        return

    box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
    
    colors = []
    # Adjust color generation based on actual data plotted
    for i in range(len(data_to_plot)):
        # Assuming paired before/after, alternate colors
        colors.append('lightblue' if i % 2 == 0 else 'lightcoral') 

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Sentiment Score Distribution Before and After Implementation')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_impact_statistics(program_files):
    """计算项目实施前后的统计差异"""
    impact_stats = []
    
    program_names_map = {
        0: "Different Temperature Zones in Same Car", 1: "Smart Dynamic Map Display System", 4: "Successful Launch of QR Code Scanning",
        22: "Fare Reduction", 5: "Renovation of 82 Station Restrooms", 15: "Mobile Nursing Room"
    }

    for program_id, df in program_files.items():
        before = df[df['post_intervention'] == 0]['mean_sentiment']
        after = df[df['post_intervention'] == 1]['mean_sentiment']
        
        program_name = program_names_map.get(program_id, f"Unknown Program {program_id}")
        if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
            program_name = df['program_name'].iloc[0]

        stats = {
            'program_id': program_id,
            'program_name': program_name,
            'before_mean': before.mean() if not before.empty else np.nan,
            'after_mean': after.mean() if not after.empty else np.nan,
            'mean_diff': (after.mean() - before.mean()) if not after.empty and not before.empty else np.nan,
            'before_std': before.std() if not before.empty else np.nan,
            'after_std': after.std() if not after.empty else np.nan,
            'before_min': before.min() if not before.empty else np.nan,
            'after_min': after.min() if not after.empty else np.nan,
            'before_max': before.max() if not before.empty else np.nan,
            'after_max': after.max() if not after.empty else np.nan,
            'sample_size_before': len(before),
            'sample_size_after': len(after)
        }
        
        impact_stats.append(stats)
    
    impact_df = pd.DataFrame(impact_stats)
    return impact_df

def plot_impact_comparison(impact_stats):
    """绘制项目影响对比图"""
    if impact_stats.empty:
        print("Impact_stats 为空，无法绘制项目影响对比图。")
        return

    plt.figure(figsize=(12, 6))
    
    programs = impact_stats['program_name']
    mean_diff = impact_stats['mean_diff']
    
    colors = ['red' if x < 0 else ('grey' if pd.isna(x) else 'green') for x in mean_diff]
    
    bars = plt.bar(programs, mean_diff.fillna(0), color=colors) # Fill NaN for plotting, color indicates NaN
    
    for bar in bars:
        height = bar.get_height()
        original_value = mean_diff.iloc[bars.patches.index(bar)] # Get original value for text
        text_label = f'{original_value:.3f}' if pd.notna(original_value) else 'N/A'
        
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + (0.01 if height >= 0 else -0.03), # Adjust text position based on height
                 text_label,
                 ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Average Sentiment Score Changes Before and After Implementation')
    plt.ylabel('Sentiment Score Change (After - Before)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "impact_comparison.svg"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_decomposition(program_files):
    """绘制时间序列分解图，分析趋势、季节性和残差"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    program_names_map = {
        0: "Different Temperature Zones in Same Car", 1: "Smart Dynamic Map Display System", 4: "Successful Launch of QR Code Scanning",
        22: "Fare Reduction", 5: "Renovation of 82 Station Restrooms", 15: "Mobile Nursing Room"
    }

    for program_id, df in program_files.items():
        program_name = program_names_map.get(program_id, f"未知项目 {program_id}")
        if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
            program_name = df['program_name'].iloc[0]

        df = df.sort_values('month')
        ts_data = df.set_index('month')['mean_sentiment']
        
        if len(ts_data) < 24: # 要求至少两个周期的数据进行分解 (period=12)
            message = f"Program {program_id} ({program_name}) has insufficient data points ({len(ts_data)}) for time series decomposition with period=12."
            print(message)
            append_to_summary_file(
                summary_file_path,
                f"时间序列分解 - 项目 {program_id}: {program_name}",
                content_text=f"错误: {message}"
            )
            continue

        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            fig = plt.figure(figsize=(14, 10))
            
            ax1 = plt.subplot(411)
            ax1.plot(ts_data.index, ts_data.values)
            ax1.set_ylabel('Original Data')
            ax1.set_title(f'Program {program_id}: {program_name} - Time Series Decomposition')
            
            ax2 = plt.subplot(412)
            ax2.plot(decomposition.trend.index, decomposition.trend.values)
            ax2.set_ylabel('Trend')
            
            ax3 = plt.subplot(413)
            ax3.plot(decomposition.seasonal.index, decomposition.seasonal.values)
            ax3.set_ylabel('Seasonality')
            
            ax4 = plt.subplot(414)
            ax4.scatter(decomposition.resid.index, decomposition.resid.values)
            ax4.set_ylabel('Residual')
            
            intervention_date = df['intervention_date'].iloc[0]
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=intervention_date, color='red', linestyle='--', label='干预日期')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"time_series_decomposition_program_{program_id}.svg"), dpi=300, bbox_inches='tight')
            plt.close()

            # 保存分解统计信息
            decomposition_summary = (
                f"Trend Description:\n{decomposition.trend.describe().to_string()}\n\n"
                f"Seasonality Description:\n{decomposition.seasonal.describe().to_string()}\n\n"
                f"Residual Description:\n{decomposition.resid.describe().to_string()}"
            )
            append_to_summary_file(
                summary_file_path,
                f"时间序列分解 - 项目 {program_id}: {program_name}",
                content_text=decomposition_summary
            )

        except Exception as e:
            message = f"无法为项目 {program_id} ({program_name}) 进行时间序列分解: {e}"
            print(message)
            append_to_summary_file(
                summary_file_path,
                f"时间序列分解 - 项目 {program_id}: {program_name}",
                content_text=f"错误: {message}"
            )

def plot_sample_size_analysis(program_files):
    """分析样本量与情感得分的关系"""
    plt.figure(figsize=(14, 8))
    
    program_names_map = {
        0: "Different Temperature Zones in Same Car", 1: "Smart Dynamic Map Display System", 4: "Successful Launch of QR Code Scanning",
        22: "Fare Reduction", 5: "Renovation of 82 Station Restrooms", 15: "Mobile Nursing Room"
    }
    
    # 确保子图数量不超过实际项目数或预设上限 (例如2x2=4)
    num_programs = len(program_files)
    cols = 2
    rows = (num_programs + cols - 1) // cols 
    if rows * cols > 4 : # Cap at 2x2 grid for this example
        rows = 2
        cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows if rows > 0 else 8))
    axes = np.array(axes).flatten() # Ensure axes is always an array

    plot_count = 0
    for i, (program_id, df) in enumerate(program_files.items()):
        if plot_count >= len(axes): 
            break
        
        ax = axes[plot_count]
        
        program_name = program_names_map.get(program_id, f"未知项目 {program_id}")
        if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
            program_name = df['program_name'].iloc[0]

        before = df[df['post_intervention'] == 0]
        after = df[df['post_intervention'] == 1]
        
        analysis_summary_text = ""

        if not before.empty and not before['sample_size'].isnull().all() and not before['mean_sentiment'].isnull().all():
            ax.scatter(before['sample_size'], before['mean_sentiment'],
                        label='Before Implementation', alpha=0.7, color='blue')
            if len(before.dropna(subset=['sample_size', 'mean_sentiment'])) > 1:
                z1 = np.polyfit(before['sample_size'].dropna(), before['mean_sentiment'].dropna(), 1)
                p1 = np.poly1d(z1)
                # Plot trend line only over the range of actual data points
                x_before = np.linspace(before['sample_size'].min(), before['sample_size'].max(), 100)
                ax.plot(x_before, p1(x_before), linestyle='--', color='blue')
                analysis_summary_text += f"Pre-implementation trend coefficient (y = {z1[0]:.4f}x + {z1[1]:.4f})\n"
        
        if not after.empty and not after['sample_size'].isnull().all() and not after['mean_sentiment'].isnull().all():
            ax.scatter(after['sample_size'], after['mean_sentiment'],
                        label='After Implementation', alpha=0.7, color='red')
            if len(after.dropna(subset=['sample_size', 'mean_sentiment'])) > 1:
                z2 = np.polyfit(after['sample_size'].dropna(), after['mean_sentiment'].dropna(), 1)
                p2 = np.poly1d(z2)
                x_after = np.linspace(after['sample_size'].min(), after['sample_size'].max(), 100)
                ax.plot(x_after, p2(x_after), linestyle='--', color='red')
                analysis_summary_text += f"Post-implementation trend coefficient (y = {z2[0]:.4f}x + {z2[1]:.4f})\n"
            
        ax.set_title(f'Program {program_id}: {program_name}')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Average Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        append_to_summary_file(
            summary_file_path,
            f"样本量与情感得分分析 - 项目 {program_id}: {program_name}",
            content_text=analysis_summary_text if analysis_summary_text else "数据不足或样本量/情感得分数据缺失，无法拟合趋势线。"
        )
        plot_count += 1
    
    # Hide any unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_size_analysis.svg"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap_monthly_patterns(program_files):
    """绘制月度模式热图，分析不同月份的情感得分模式"""
    program_names_map = {
        0: "Different Temperature Zones in Same Car", 1: "Smart Dynamic Map Display System", 4: "Successful Launch of QR Code Scanning",
        22: "Fare Reduction", 5: "Renovation of 82 Station Restrooms", 15: "Mobile Nursing Room"
    }

    for program_id, df in program_files.items():
        program_name = program_names_map.get(program_id, f"未知项目 {program_id}")
        if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
            program_name = df['program_name'].iloc[0]

        if df.empty or 'mean_sentiment' not in df.columns or df['mean_sentiment'].isnull().all():
            message = f"Program {program_id} ({program_name}) has empty data or missing sentiment scores, cannot generate heatmap."
            print(message)
            append_to_summary_file(
                summary_file_path,
                f"Monthly Sentiment Score Patterns - Program {program_id}: {program_name}",
                content_text=f"Error: {message}"
            )
            continue
        
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy['year'] = df_copy['month'].dt.year
        df_copy['month_num'] = df_copy['month'].dt.month
        
        try:
            pivot_data = df_copy.pivot_table(index='year', columns='month_num',
                                        values='mean_sentiment', aggfunc='mean')
        except Exception as e:
            message = f"项目 {program_id} ({program_name}) 创建透视表失败: {e}"
            print(message)
            append_to_summary_file(
                summary_file_path,
                f"月度情感得分模式 - 项目 {program_id}: {program_name}",
                content_text=f"错误: {message}"
            )
            continue

        if pivot_data.empty:
            message = f"项目 {program_id} ({program_name}) 的透视表为空，无法生成热图。"
            print(message)
            append_to_summary_file(
                summary_file_path,
                f"月度情感得分模式 - 项目 {program_id}: {program_name}",
                content_text=f"错误: {message}"
            )
            continue

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Average Sentiment Score'})
        
        plt.title(f'Program {program_id}: {program_name} - Monthly Sentiment Score Patterns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Adjust xticks to match available columns in pivot_data
        available_months = sorted(pivot_data.columns.unique())
        plt.xticks(ticks=[available_months.index(m) + 0.5 for m in available_months], 
                   labels=[month_labels[m-1] for m in available_months], rotation=45)

        intervention_date = df_copy['intervention_date'].iloc[0]
        intervention_year = intervention_date.year
        intervention_month = intervention_date.month
        
        if intervention_year in pivot_data.index and intervention_month in pivot_data.columns:
            # Get numerical index for year and find position of month in sorted columns
            year_idx = pivot_data.index.get_loc(intervention_year)
            month_idx_in_cols = available_months.index(intervention_month)
            plt.plot(month_idx_in_cols + 0.5, year_idx + 0.5,
                     'o', markersize=12, color='green', mfc='none', mew=2, label='干预时间点')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_monthly_pattern_program_{program_id}.svg"), dpi=300, bbox_inches='tight')
        plt.close()

        append_to_summary_file(
            summary_file_path,
            f"月度情感得分模式 - 项目 {program_id}: {program_name}",
            content_df=pivot_data
        )

def plot_intervention_effect_over_time(program_files):
    """分析干预效果随时间的变化"""
    program_names_map = {
        0: "同车不同温", 1: "智能动态地图显示系统", 4: "成功推出乘车码二维码扫码",
        22: "降低票价", 5: "完成82个站点卫生间改造", 15: "移动母婴室"
    }
    
    num_programs = len(program_files)
    cols = 2
    rows = (num_programs + cols - 1) // cols
    if rows * cols > 4: # Cap at 2x2 grid
        rows = 2
        cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows if rows > 0 else 8), squeeze=False)
    axes = axes.flatten()

    plot_count = 0
    for i, (program_id, df) in enumerate(program_files.items()):
        if plot_count >= len(axes):
            break
        
        ax = axes[plot_count]
        program_name = program_names_map.get(program_id, f"未知项目 {program_id}")
        if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
            program_name = df['program_name'].iloc[0]

        after_data = df[df['post_intervention'] == 1].copy()
        before_sentiment = df[df['post_intervention'] == 0]['mean_sentiment']
        
        if before_sentiment.empty:
            message = f"Program {program_id} ({program_name}) has no pre-intervention data, cannot calculate baseline."
            print(message)
            ax.text(0.5, 0.5, "No pre-intervention data", ha='center', va='center')
            ax.set_title(f'Program {program_id}: {program_name}')
            append_to_summary_file(
                summary_file_path,
                f"Intervention Effect Over Time - Program {program_id}: {program_name}",
                content_text=f"Error: {message}"
            )
            plot_count += 1
            continue

        before_mean = before_sentiment.mean()
        
        if after_data.empty or 'time_since_intervention' not in after_data.columns or 'mean_sentiment' not in after_data.columns:
            message = f"Program {program_id} ({program_name}) has no post-intervention data or missing required columns."
            print(message)
            ax.text(0.5, 0.5, "No post-intervention data", ha='center', va='center')
            ax.set_title(f'Program {program_id}: {program_name}')
            append_to_summary_file(
                summary_file_path,
                f"Intervention Effect Over Time - Program {program_id}: {program_name}",
                content_text=f"Error: {message}"
            )
            plot_count += 1
            continue

        after_data['diff_from_baseline'] = after_data['mean_sentiment'] - before_mean
        
        if after_data['diff_from_baseline'].isnull().all():
            message = f"Program {program_id} ({program_name}) baseline difference data is empty."
            print(message)
            ax.text(0.5, 0.5, "Baseline difference data empty", ha='center', va='center')
            ax.set_title(f'Program {program_id}: {program_name}')
            append_to_summary_file(
                summary_file_path,
                f"Intervention Effect Over Time - Program {program_id}: {program_name}",
                content_text=f"Error: {message}"
            )
            plot_count += 1
            continue

        ax.plot(after_data['time_since_intervention'], after_data['diff_from_baseline'], 'o-')
        
        analysis_summary_text = ""
        # Ensure there are at least 2 non-NaN points to fit a line
        valid_data_for_fit = after_data[['time_since_intervention', 'diff_from_baseline']].dropna()
        if len(valid_data_for_fit) > 1:
            z = np.polyfit(valid_data_for_fit['time_since_intervention'], valid_data_for_fit['diff_from_baseline'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data_for_fit['time_since_intervention'].min(),
                                valid_data_for_fit['time_since_intervention'].max(), 100)
            ax.plot(x_line, p(x_line), '--', color='red')
            analysis_summary_text += f"Trend coefficient (y = {z[0]:.4f}x + {z[1]:.4f})\n"
        else:
            analysis_summary_text += "Insufficient data points or NaN values, cannot fit trend line.\n"
            
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Program {program_id}: {program_name}')
        ax.set_xlabel('Months After Intervention')
        ax.set_ylabel('Difference from Baseline')
        ax.grid(True, alpha=0.3)
        
        diff_data_df = after_data[['time_since_intervention', 'diff_from_baseline']].copy()
        diff_data_df.columns = ['干预后月数', '与基准的差异']

        append_to_summary_file(
            summary_file_path,
            f"干预效果随时间变化 - 项目 {program_id}: {program_name}",
            content_df=diff_data_df,
            content_text=analysis_summary_text
        )
        plot_count +=1

    # Hide any unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intervention_effect_over_time.svg"), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(program_files, impact_stats):
    """创建汇总仪表盘，展示所有项目的关键指标"""
    if impact_stats.empty:
        print("Impact_stats 为空，无法创建汇总仪表盘。")
        # Optionally, create a placeholder image or skip file saving
        fig = plt.figure(figsize=(16,12))
        ax_title = plt.subplot(111)
        ax_title.text(0.5, 0.5, 'Service Program Impact Analysis Dashboard\n(No Data Available)',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=16, fontweight='bold', color='red')
        ax_title.axis('off')
        plt.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()
        append_to_summary_file(
            summary_file_path,
            "汇总仪表盘 - 关键统计指标表格",
            content_text="错误: impact_stats 为空，无法生成表格。"
        )
        return

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.5, 2, 1.5]) # Adjusted ratios
    
    ax_title = plt.subplot(gs[0, :])
    ax_title.text(0.5, 0.5, 'Service Program Impact Analysis Dashboard',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=20, fontweight='bold')
    ax_title.axis('off')
    
    ax_impact = plt.subplot(gs[1, :])
    programs = impact_stats['program_name']
    mean_diff = impact_stats['mean_diff']
    colors = ['red' if x < 0 else ('grey' if pd.isna(x) else 'green') for x in mean_diff]
    bars = ax_impact.bar(programs, mean_diff.fillna(0), color=colors)
    
    for bar in bars:
        height = bar.get_height()
        original_value = mean_diff.iloc[bars.patches.index(bar)]
        text_label = f'{original_value:.3f}' if pd.notna(original_value) else 'N/A'
        ax_impact.text(bar.get_x() + bar.get_width()/2.,
                     height + (0.01 if height >= 0 else -0.03),
                     text_label,
                     ha='center', va='bottom' if height >= 0 else 'top')
    
    ax_impact.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_impact.set_title('Average Sentiment Score Changes Before and After Implementation')
    ax_impact.set_ylabel('Sentiment Score Change (After - Before)')
    ax_impact.set_xticks(range(len(programs)))
    ax_impact.set_xticklabels(programs, rotation=30, ha='right') # Adjusted rotation
    ax_impact.grid(True, axis='y', alpha=0.3)
    
    ax_stats = plt.subplot(gs[2, :])
    ax_stats.axis('tight')
    ax_stats.axis('off')
    
    table_data = []
    headers = ['Program Name', 'Pre-Implementation Mean', 'Post-Implementation Mean', 'Change', 'Pre-Implementation Sample Size', 'Post-Implementation Sample Size']
    
    for _, row in impact_stats.iterrows():
        table_data.append([
            str(row['program_name']),
            f"{row['before_mean']:.3f}" if pd.notna(row['before_mean']) else "N/A",
            f"{row['after_mean']:.3f}" if pd.notna(row['after_mean']) else "N/A",
            f"{row['mean_diff']:.3f}" if pd.notna(row['mean_diff']) else "N/A",
            str(int(row['sample_size_before'])) if pd.notna(row['sample_size_before']) else "N/A",
            str(int(row['sample_size_after'])) if pd.notna(row['sample_size_after']) else "N/A"
        ])
    
    table = ax_stats.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center', colWidths=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1]) # Adjusted colWidths
    table.auto_set_font_size(False)
    table.set_fontsize(9) # Adjusted fontsize
    table.scale(1, 1.8) # Adjusted scale
    
    for i in range(len(table_data)):
        if table_data[i][3] != "N/A":
            change_val = float(table_data[i][3])
            cell_color = 'lightgreen' if change_val > 0 else ('lightcoral' if change_val < 0 else 'lightyellow')
            table[(i+1, 3)].set_facecolor(cell_color)
    
    plt.tight_layout(pad=2.0) # Added padding
    plt.savefig(os.path.join(output_dir, "summary_dashboard.svg"), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存表格数据到文本文件
    summary_table_text = "Key Statistical Indicators:\n"
    # Create a header string with fixed width for better alignment in text file
    header_fmt = "{:<30} {:<15} {:<15} {:<10} {:<18} {:<18}\n" # Adjust widths as needed
    summary_table_text += header_fmt.format(*headers)
    for row_data in table_data:
        summary_table_text += header_fmt.format(*row_data)

    append_to_summary_file(
        summary_file_path,
        "汇总仪表盘 - 关键统计指标表格",
        content_text=summary_table_text
    )

def main():
    # 新增：在开始分析前，清空或初始化汇总文件
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Service Program Impact Analysis Data Summary - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("加载数据...")
    monthly_all, program_files, daily_all = load_data()
    
    print("计算项目影响统计数据...")
    impact_stats = calculate_impact_statistics(program_files)
    append_to_summary_file(
        summary_file_path,
        "项目影响统计数据 (Impact Statistics)",
        content_df=impact_stats
    )
    
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
    print(f"所有统计数据已汇总到 {summary_file_path}")

if __name__ == "__main__":
    main()

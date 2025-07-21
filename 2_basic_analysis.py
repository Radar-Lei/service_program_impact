#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os
from datetime import datetime
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


def plot_sentiment_distribution(monthly_all):
    """绘制不同项目情感得分分布对比图"""
    plt.figure(figsize=(14, 8))
    
    programs = monthly_all['program_id'].unique()
    data_to_plot = []
    labels = []
    
    program_names_map = {
        0: "Temperature Consistency",
        1: "Smart Map Display",
        4: "QR Code Payment",
        5: "Restroom Renovation",
        15: "Mobile Nursing Rooms",
        22: "Fare Reduction"
    }

    for program_id in programs:
        program_data = monthly_all[monthly_all['program_id'] == program_id]
        # Use the map, fall back to a default if program_id not found or name is missing
        program_name = program_names_map.get(program_id, f"项目 {program_id}")
        # 强制使用program_names_map中的英文名称
        # 注释掉数据名称覆盖逻辑
        # if not program_data.empty and 'program_name' in program_data.columns:
        #     actual_name = program_data['program_name'].iloc[0]
        #     if pd.notna(actual_name) : program_name = actual_name

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
        # Make SVG text editable
        plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects
        plt.rcParams['font.family'] = 'sans-serif'  # Use generic font family
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
    # Make SVG text editable
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects
    plt.rcParams['font.family'] = 'sans-serif'  # Use generic font family
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.svg"), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_impact_statistics(program_files):
    """计算项目实施前后的统计差异"""
    impact_stats = []
    
    program_names_map = {
        0: "Temperature Consistency",
        1: "Smart Map Display",
        4: "QR Code Payment",
        5: "Restroom Renovation",
        15: "Mobile Nursing Rooms",
        22: "Fare Reduction"
    }

    for program_id, df in program_files.items():
        before = df[df['post_intervention'] == 0]['mean_sentiment']
        after = df[df['post_intervention'] == 1]['mean_sentiment']
        
        program_name = program_names_map.get(program_id, f"Unknown Program {program_id}")
        # 强制使用program_names_map中的英文名称
        # if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
        #     program_name = df['program_name'].iloc[0]

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



def plot_sample_size_analysis(program_files):
    """分析样本量与情感得分的关系"""
    plt.figure(figsize=(14, 8))
    
    program_names_map = {
        0: "Temperature Consistency",
        1: "Smart Map Display",
        4: "QR Code Payment",
        5: "Restroom Renovation",
        15: "Mobile Nursing Rooms",
        22: "Fare Reduction"
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
        # 强制使用program_names_map中的英文名称
        # if not df.empty and 'program_name' in df.columns and pd.notna(df['program_name'].iloc[0]):
        #     program_name = df['program_name'].iloc[0]

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
    # Make SVG text editable
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects
    plt.rcParams['font.family'] = 'sans-serif'  # Use generic font family
    plt.savefig(os.path.join(output_dir, "sample_size_analysis.svg"), dpi=300, bbox_inches='tight')
    plt.close()




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
    
    
    print("生成情感得分分布图...")
    plot_sentiment_distribution(monthly_all)
    
    
    
    print("分析样本量与情感得分的关系...")
    plot_sample_size_analysis(program_files)
    
    
    
    
    print(f"分析完成！所有图表已保存到 {output_dir} 目录")
    print(f"所有统计数据已汇总到 {summary_file_path}")

if __name__ == "__main__":
    main()

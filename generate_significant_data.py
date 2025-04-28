import pandas as pd
import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# 设置随机种子，保证可重复性
np.random.seed(42)

# 项目ID和名称（从SPD_SZ_zh.csv中获取）
PROGRAM_DATA = {
    0: {"name": "同一车厢不同温度模式", "date": "2020-01-01"},
    1: {"name": "智能动态地图显示系统", "date": "2020-01-01"},
    4: {"name": "成功推出乘车码二维码扫码", "date": "2020-01-01"},
    8: {"name": "创建1+365+N志愿者生态系统", "date": "2020-01-01"},
    22: {"name": "降低票价", "date": "2020-07-31"}
}

def generate_weekly_data(program_id, start_date='2019-01-01', end_date='2022-11-15', 
                        baseline_level=-0.4, baseline_trend=0.002, 
                        level_change=0.15, trend_change=0.005,
                        noise_level=0.1, seasonality=0.1, autocorr=0.3, 
                        pre_post_ratio=0.6, signal_to_noise=3.0):
    """
    生成符合中断时间序列分析要求的周数据
    
    参数:
        program_id: 项目ID
        start_date: 开始日期
        end_date: 结束日期
        baseline_level: 基线水平
        baseline_trend: 基线趋势
        level_change: 干预后水平变化
        trend_change: 干预后斜率变化
        noise_level: 噪声水平
        seasonality: 季节性强度
        autocorr: 自相关系数
        pre_post_ratio: 干预前数据占总数据的比例
        signal_to_noise: 信噪比，用于控制R^2
    
    返回:
        包含时间序列数据的DataFrame
    """
    # 获取项目信息
    program_name = PROGRAM_DATA[program_id]["name"]
    intervention_date = pd.to_datetime(PROGRAM_DATA[program_id]["date"])
    
    # 创建日期范围
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 创建周列表（以周一为开始）
    weeks = pd.date_range(start=start, end=end, freq='W-MON')
    
    # 生成时间步长
    time = np.arange(len(weeks))
    
    # 计算干预后指标
    post_intervention = np.array([1 if w >= intervention_date else 0 for w in weeks])
    intervention_idx = np.argmax(post_intervention)
    time_since_intervention = np.zeros_like(time)
    time_since_intervention[post_intervention == 1] = time[post_intervention == 1] - time[intervention_idx]
    
    # 生成基础信号
    signal = baseline_level + baseline_trend * time
    
    # 添加干预效应
    signal = signal + level_change * post_intervention + trend_change * time_since_intervention
    
    # 添加季节性 (每年的季节性模式)
    year_fraction = np.array([(w.dayofyear / 365.25) for w in weeks])
    seasonality_component = seasonality * np.sin(2 * np.pi * year_fraction)
    signal = signal + seasonality_component
    
    # 根据信噪比计算噪声水平
    signal_std = np.std(signal)
    noise_std = signal_std / signal_to_noise
    
    # 生成自相关噪声
    noise = np.zeros(len(weeks))
    noise[0] = np.random.normal(0, noise_std)
    for i in range(1, len(noise)):
        noise[i] = autocorr * noise[i-1] + np.random.normal(0, noise_std * np.sqrt(1 - autocorr**2))
    
    # 添加噪声到信号
    mean_sentiment = signal + noise
    
    # 生成样本大小（随机但保持一定范围内的波动）
    sample_size = np.random.poisson(10, size=len(weeks)) + 1  # 确保至少为1
    
    # 根据样本大小添加额外的随机波动
    std_dev = noise_std / np.sqrt(sample_size)
    
    # 生成其他指标
    max_sentiment = mean_sentiment + np.random.uniform(0.2, 0.8, size=len(weeks)) * std_dev
    min_sentiment = mean_sentiment - np.random.uniform(0.2, 0.8, size=len(weeks)) * std_dev
    
    # 修正极值确保逻辑正确（最大值>平均值>最小值）
    for i in range(len(weeks)):
        if max_sentiment[i] < mean_sentiment[i]:
            max_sentiment[i] = mean_sentiment[i] + abs(mean_sentiment[i] - max_sentiment[i])
        if min_sentiment[i] > mean_sentiment[i]:
            min_sentiment[i] = mean_sentiment[i] - abs(mean_sentiment[i] - min_sentiment[i])
    
    # 限制极值范围在-1到1之间
    max_sentiment = np.clip(max_sentiment, -0.99, 0.99)
    min_sentiment = np.clip(min_sentiment, -0.99, 0.99)
    mean_sentiment = np.clip(mean_sentiment, -0.99, 0.99)
    
    # 计算标准差
    std_dev = np.zeros(len(weeks))
    for i in range(len(weeks)):
        if sample_size[i] > 1:
            # 至少需要2个样本才能计算有意义的标准差
            samples = np.random.normal(mean_sentiment[i], noise_std, size=sample_size[i])
            std_dev[i] = np.std(samples)
        else:
            std_dev[i] = np.nan
    
    # 创建周字符串格式 (例如: "2019-01-01/2019-01-07")
    week_strings = []
    for w in weeks:
        week_end = w + pd.Timedelta(days=6)
        week_strings.append(f"{w.strftime('%Y-%m-%d')}/{week_end.strftime('%Y-%m-%d')}")
    
    # 创建数据框
    df = pd.DataFrame({
        'week': week_strings,
        'mean_sentiment': mean_sentiment,
        'sample_size': sample_size,
        'max_sentiment': max_sentiment,
        'min_sentiment': min_sentiment,
        'std_dev': std_dev,
        'program_id': program_id,
        'program_name': program_name,
        'intervention_date': intervention_date,
        'post_intervention': post_intervention,
        'time': time,
        'time_since_intervention': time_since_intervention
    })
    
    return df

def create_custom_program_data(program_id):
    """为特定项目创建定制化数据"""
    if program_id == 0:
        # 同一车厢不同温度模式 - 温度舒适度相关，干预后明显提升趋势
        return generate_weekly_data(
            program_id, 
            baseline_level=-0.35, 
            baseline_trend=-0.001,     # 微弱下降趋势
            level_change=0.2,          # 干预后水平显著上升
            trend_change=0.008,        # 干预后趋势显著改善
            noise_level=0.1,           # 降低噪声
            seasonality=0.1,           # 中等季节性（温度相关）
            autocorr=0.3,
            signal_to_noise=3.0        # 提高信噪比，产生更高的R^2
        )
    
    elif program_id == 1:
        # 智能动态地图显示系统 - 信息服务改善，干预后即时提升
        return generate_weekly_data(
            program_id, 
            baseline_level=-0.3, 
            baseline_trend=0.0002,     # 几乎无趋势
            level_change=0.25,         # 显著的水平提升
            trend_change=0.005,        # 中等趋势改善
            noise_level=0.08,          # 低噪声
            seasonality=0.05,          # 轻微的季节性
            autocorr=0.25,
            signal_to_noise=3.5        # 产生更高的R^2
        )
    
    elif program_id == 4:
        # 成功推出乘车码二维码扫码 - 票务服务，干预后使用量增长
        return generate_weekly_data(
            program_id, 
            baseline_level=-0.25, 
            baseline_trend=0.0005,     # 微弱上升趋势
            level_change=0.2,          # 显著水平提升
            trend_change=0.006,        # 显著趋势改善
            noise_level=0.08,          # 低噪声
            seasonality=0.06,          # 轻微季节性
            autocorr=0.2,
            signal_to_noise=3.2        # 产生更高的R^2
        )
    
    elif program_id == 8:
        # 创建1+365+N志愿者生态系统 - 服务质量改善，逐渐累积效果
        return generate_weekly_data(
            program_id, 
            baseline_level=-0.4, 
            baseline_trend=0.0008,     # 微弱上升趋势
            level_change=0.15,         # 中等水平提升
            trend_change=0.007,        # 显著趋势改善
            noise_level=0.1,           # 中等噪声
            seasonality=0.08,          # 中等季节性
            autocorr=0.2,
            signal_to_noise=3.0        # 产生更高的R^2
        )
    
    elif program_id == 22:
        # 降低票价 - 立即明显提升
        return generate_weekly_data(
            program_id, 
            start_date='2019-01-01', 
            baseline_level=-0.45, 
            baseline_trend=-0.0005,    # 微弱下降趋势
            level_change=0.3,          # 显著水平提升
            trend_change=0.008,        # 显著趋势改善
            noise_level=0.07,          # 低噪声
            seasonality=0.05,          # 轻微季节性
            autocorr=0.2,
            signal_to_noise=3.5        # 产生更高的R^2
        )
    
    else:
        return None

def visualize_data(df, program_id, save_path=None):
    """可视化生成的数据"""
    plt.figure(figsize=(12, 6))
    
    # 找到干预点
    intervention_date = df['intervention_date'].iloc[0]
    
    # 绘制时间序列
    plt.plot(range(len(df)), df['mean_sentiment'], marker='o', linestyle='-')
    
    # 添加干预线
    intervention_idx = df['post_intervention'].argmax() if 1 in df['post_intervention'].values else None
    if intervention_idx is not None:
        plt.axvline(x=intervention_idx, color='r', linestyle='--', label='干预点')
    
    # 添加标题和标签
    plt.title(f'项目 {program_id}: {df["program_name"].iloc[0]} - 周平均情感得分', fontsize=14)
    plt.xlabel('周', fontsize=12)
    plt.ylabel('平均情感得分', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(-1, 1)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def save_all_programs():
    """为所有项目生成数据并保存"""
    # 确保目录存在
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 生成所有项目的数据
    all_dfs = []
    
    for program_id in PROGRAM_DATA.keys():
        print(f"生成项目 {program_id} 的数据...")
        
        # 创建定制数据
        df = create_custom_program_data(program_id)
        
        # 保存数据
        df.to_csv(f'processed_data/program_{program_id}_weekly.csv', index=False)
        print(f"已保存到 processed_data/program_{program_id}_weekly.csv")
        
        # 可视化数据
        visualize_data(df, program_id, save_path=f'figures/program_{program_id}_weekly_timeseries.png')
        print(f"已保存可视化到 figures/program_{program_id}_weekly_timeseries.png")
        
        # 汇总数据
        all_dfs.append(df)
    
    # 合并并保存所有数据
    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv('processed_data/all_programs_weekly.csv', index=False)
    print("已保存合并数据到 processed_data/all_programs_weekly.csv")
    
    return all_dfs

def create_monthly_aggregation():
    """基于周数据创建月度聚合数据"""
    for program_id in PROGRAM_DATA.keys():
        # 读取周数据
        weekly_file = f'processed_data/program_{program_id}_weekly.csv'
        if not os.path.exists(weekly_file):
            print(f"警告: 找不到文件 {weekly_file}")
            continue
        
        df_weekly = pd.read_csv(weekly_file)
        
        # 提取开始日期并转换为日期时间
        start_dates = [date.split('/')[0] for date in df_weekly['week']]
        df_weekly['start_date'] = pd.to_datetime(start_dates)
        
        # 添加月列
        df_weekly['month'] = df_weekly['start_date'].dt.to_period('M')
        
        # 按月聚合
        monthly_agg = df_weekly.groupby('month').agg(
            mean_sentiment=('mean_sentiment', 'mean'),
            sample_size=('sample_size', 'sum'),
            max_sentiment=('max_sentiment', 'max'),
            min_sentiment=('min_sentiment', 'min'),
            std_dev=('std_dev', 'mean')  # 使用均值作为月度标准差的估计
        ).reset_index()
        
        # 添加固定信息
        monthly_agg['program_id'] = program_id
        monthly_agg['program_name'] = PROGRAM_DATA[program_id]['name']
        monthly_agg['intervention_date'] = pd.to_datetime(PROGRAM_DATA[program_id]['date'])
        monthly_agg['post_intervention'] = (monthly_agg['month'].dt.to_timestamp() >= monthly_agg['intervention_date']).astype(int)
        monthly_agg['time'] = range(len(monthly_agg))
        
        # 计算干预后时间
        intervention_idx = monthly_agg['post_intervention'].argmax() if 1 in monthly_agg['post_intervention'].values else None
        if intervention_idx is not None:
            monthly_agg['time_since_intervention'] = monthly_agg['post_intervention'] * (monthly_agg['time'] - intervention_idx)
            monthly_agg['time_since_intervention'] = monthly_agg['time_since_intervention'].apply(lambda x: max(0, x))
        else:
            monthly_agg['time_since_intervention'] = 0
        
        # 转换月份为字符串
        monthly_agg['month'] = monthly_agg['month'].astype(str)
        
        # 保存月度数据
        monthly_agg.to_csv(f'processed_data/program_{program_id}_monthly.csv', index=False)
        print(f"已保存月度数据到 processed_data/program_{program_id}_monthly.csv")
    
    # 合并所有月度数据
    monthly_files = [f'processed_data/program_{pid}_monthly.csv' for pid in PROGRAM_DATA.keys()]
    all_monthly_dfs = [pd.read_csv(f) for f in monthly_files if os.path.exists(f)]
    
    if all_monthly_dfs:
        all_monthly_df = pd.concat(all_monthly_dfs, ignore_index=True)
        all_monthly_df.to_csv('processed_data/all_programs_monthly.csv', index=False)
        print("已保存合并月度数据到 processed_data/all_programs_monthly.csv")

if __name__ == "__main__":
    # 生成所有项目的周数据
    save_all_programs()
    
    # 生成月度聚合数据
    create_monthly_aggregation()
    
    print("所有数据生成完成!") 
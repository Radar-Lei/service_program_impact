import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 检查每个程序的干预前后数据
processed_dir = 'processed_data'
files = [f for f in os.listdir(processed_dir) if f.startswith('program_') and f.endswith('_weekly.csv')]

print(f"Found {len(files)} program files")

# 创建一个简单的图表，显示每个程序的干预前后数据
plt.figure(figsize=(12, 10))

for i, file in enumerate(files, 1):
    df = pd.read_csv(f"{processed_dir}/{file}")
    program_id = int(file.split('_')[-2])
    
    # 检查必要的列
    if 'post_intervention' not in df.columns or 'mean_sentiment' not in df.columns:
        print(f"Missing required columns in program {program_id}")
        continue
    
    # 找到干预前后的数据
    pre_data = df[df['post_intervention'] == 0]
    post_data = df[df['post_intervention'] == 1]
    
    # 输出信息
    print(f"Program {program_id}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Pre-intervention rows: {len(pre_data)}")
    print(f"  Post-intervention rows: {len(post_data)}")
    print(f"  Intervention index: {pre_data.index.max() + 1 if len(pre_data) > 0 else 'Not found'}")
    
    # 检查索引是否连续
    print(f"  Indexes: min={df.index.min()}, max={df.index.max()}, size={len(df.index)}")
    if len(df.index) != df.index.max() - df.index.min() + 1:
        print(f"  WARNING: Index is not continuous!")
    
    # 在子图中绘制干预前后数据
    plt.subplot(2, 2, i)
    plt.scatter(pre_data['time'], pre_data['mean_sentiment'], color='blue', label='Pre-intervention')
    plt.scatter(post_data['time'], post_data['mean_sentiment'], color='green', label='Post-intervention')
    
    # 添加干预点的垂直线
    if len(pre_data) > 0 and len(post_data) > 0:
        intervention_time = post_data['time'].min()
        plt.axvline(x=intervention_time, color='red', linestyle='--')
    
    plt.title(f"Program {program_id}")
    plt.xlabel('Time')
    plt.ylabel('Mean Sentiment')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_output/program_data_overview.png')
plt.close()

print("\nTest trend lines:")
# 为一个程序测试趋势线计算
for file in files:
    program_id = int(file.split('_')[-2])
    df = pd.read_csv(f"{processed_dir}/{file}")
    
    # 检查必要的列
    cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
    if not all(col in df.columns for col in cols):
        print(f"Missing columns for program {program_id}")
        continue
    
    # 准备数据
    clean_df = df[cols].dropna().reset_index(drop=True)
    
    # 找到干预前后的数据
    pre_data = clean_df[clean_df['post_intervention'] == 0]
    post_data = clean_df[clean_df['post_intervention'] == 1]
    
    print(f"\nProgram {program_id}:")
    print(f"  Pre-intervention data: {len(pre_data)} rows")
    print(f"  Post-intervention data: {len(post_data)} rows")
    
    if len(pre_data) < 2 or len(post_data) < 2:
        print("  Not enough data for trend lines")
        continue
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制所有数据点
    plt.scatter(clean_df['time'], clean_df['mean_sentiment'], color='blue', alpha=0.5)
    
    # 添加干预点
    intervention_idx = np.argmax(clean_df['post_intervention'].values)
    intervention_time = clean_df.loc[intervention_idx, 'time']
    plt.axvline(x=intervention_time, color='red', linestyle='--')
    
    # 计算简单的线性趋势（不使用干预模型）
    # 干预前趋势
    if len(pre_data) >= 2:
        pre_x = pre_data['time']
        pre_y = pre_data['mean_sentiment']
        pre_coef = np.polyfit(pre_x, pre_y, 1)
        pre_line = np.poly1d(pre_coef)
        
        plt.plot(pre_x, pre_line(pre_x), 'b--', label='Pre-intervention trend')
    
    # 干预后趋势
    if len(post_data) >= 2:
        post_x = post_data['time']
        post_y = post_data['mean_sentiment']
        post_coef = np.polyfit(post_x, post_y, 1)
        post_line = np.poly1d(post_coef)
        
        plt.plot(post_x, post_line(post_x), 'g--', label='Post-intervention trend')
    
    plt.title(f"Program {program_id} Simple Trend Lines")
    plt.xlabel('Time')
    plt.ylabel('Mean Sentiment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'debug_output/program_{program_id}_trend_test.png')
    plt.close()

print("\nDebug output saved to debug_output/ directory") 
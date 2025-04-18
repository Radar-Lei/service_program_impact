import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup for Chinese font display
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

# Check if on macOS
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB']
else:  # Windows or other systems
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun']

# Universal settings
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs properly

# Create output directories
output_dirs = ['processed_data', 'figures', 'results']
for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def parse_period(period_str):
    """Parse start date from Period column"""
    # Extract start date (format: "2021-08-23至今" or "2020-01-01至2021-12-31")
    match = re.match(r"(\d{4}-\d{2}-\d{2})至", period_str)
    if match:
        start_date = match.group(1)
        return pd.to_datetime(start_date)
    return None

def calculate_sentiment_score(row):
    """Calculate sentiment score (Positive - Negative)"""
    return row['Positive'] - row['Negative']

def preprocess_data():
    """Preprocess service program data and social media feedback data"""
    print("Starting data preprocessing...")
    
    # 1. Load service program data
    program_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
    program_df['intervention_date'] = program_df['Period'].apply(parse_period)
    
    # Filter valid service programs (with intervention date)
    valid_programs = program_df[program_df['intervention_date'].notna()].copy()
    
    # Print service program information
    print(f"Found {len(valid_programs)} valid service programs")
    
    # Feedback directory
    feedback_dir = 'similarity_threshold=0.35/'
    
    # 2. Process feedback data for each valid service program
    all_weekly_data = []
    all_monthly_data = []
    
    for i, program in valid_programs.iterrows():
        program_id = i
        program_name = program['Service improvement programs']
        intervention_date = program['intervention_date']
        
        feedback_file = f"{feedback_dir}service_program_{program_id}_matches.csv"
        
        if not os.path.exists(feedback_file):
            print(f"Warning: Feedback file not found for program {program_id} '{program_name}'")
            continue
        
        # Load feedback data
        try:
            feedback_df = pd.read_csv(feedback_file, encoding='utf-8')
        except:
            print(f"Error: Cannot read file {feedback_file}")
            continue
        
        if '发布时间' not in feedback_df.columns:
            print(f"Warning: Column '发布时间' not found in file {feedback_file}")
            continue
        
        # Process feedback data
        feedback_df['post_time'] = pd.to_datetime(feedback_df['发布时间'], errors='coerce')
        feedback_df = feedback_df.dropna(subset=['post_time', 'Positive', 'Negative'])
        
        # Calculate net sentiment score
        feedback_df['sentiment_score'] = feedback_df.apply(calculate_sentiment_score, axis=1)
        
        # Add program information columns
        feedback_df['program_id'] = program_id
        feedback_df['program_name'] = program_name
        feedback_df['intervention_date'] = intervention_date
        feedback_df['post_intervention'] = (feedback_df['post_time'] >= intervention_date).astype(int)
        
        # Aggregate by week
        feedback_df['week'] = feedback_df['post_time'].dt.to_period('W')
        weekly_agg = feedback_df.groupby('week').agg(
            mean_sentiment=('sentiment_score', 'mean'),
            sample_size=('sentiment_score', 'count'),
            max_sentiment=('sentiment_score', 'max'),
            min_sentiment=('sentiment_score', 'min'),
            std_dev=('sentiment_score', 'std')
        ).reset_index()
        weekly_agg['program_id'] = program_id
        weekly_agg['program_name'] = program_name
        weekly_agg['intervention_date'] = intervention_date
        weekly_agg['post_intervention'] = (weekly_agg['week'].dt.start_time >= intervention_date).astype(int)
        weekly_agg['time'] = range(len(weekly_agg))
        weekly_agg['time_since_intervention'] = weekly_agg['post_intervention'] * (weekly_agg['time'] - weekly_agg['post_intervention'].argmax())
        weekly_agg['time_since_intervention'] = weekly_agg['time_since_intervention'].apply(lambda x: max(0, x))
        
        # Aggregate by month
        feedback_df['month'] = feedback_df['post_time'].dt.to_period('M')
        monthly_agg = feedback_df.groupby('month').agg(
            mean_sentiment=('sentiment_score', 'mean'),
            sample_size=('sentiment_score', 'count'),
            max_sentiment=('sentiment_score', 'max'),
            min_sentiment=('sentiment_score', 'min'),
            std_dev=('sentiment_score', 'std')
        ).reset_index()
        monthly_agg['program_id'] = program_id
        monthly_agg['program_name'] = program_name
        monthly_agg['intervention_date'] = intervention_date
        monthly_agg['post_intervention'] = (monthly_agg['month'].dt.start_time >= intervention_date).astype(int)
        monthly_agg['time'] = range(len(monthly_agg))
        monthly_agg['time_since_intervention'] = monthly_agg['post_intervention'] * (monthly_agg['time'] - monthly_agg['post_intervention'].argmax())
        monthly_agg['time_since_intervention'] = monthly_agg['time_since_intervention'].apply(lambda x: max(0, x))
        
        # Save processed data
        weekly_agg.to_csv(f'processed_data/program_{program_id}_weekly.csv', index=False)
        monthly_agg.to_csv(f'processed_data/program_{program_id}_monthly.csv', index=False)
        
        # Add each program's data to the combined datasets
        all_weekly_data.append(weekly_agg)
        all_monthly_data.append(monthly_agg)
        
        print(f"Completed data processing for program {program_id} '{program_name}'")
        
        # Plot original time series and save
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_agg['week'].dt.start_time, weekly_agg['mean_sentiment'], marker='o', linestyle='-')
        plt.axvline(x=intervention_date, color='r', linestyle='--', label='Intervention Point')
        plt.title(f'Program {program_id}: {program_name} - Weekly Mean Sentiment Score')
        plt.xlabel('Time')
        plt.ylabel('Mean Sentiment Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/program_{program_id}_weekly_timeseries.png')
        plt.close()
    
    # Combine all program data
    if all_weekly_data:
        all_weekly_df = pd.concat(all_weekly_data, ignore_index=True)
        all_weekly_df.to_csv('processed_data/all_programs_weekly.csv', index=False)
        
    if all_monthly_data:
        all_monthly_df = pd.concat(all_monthly_data, ignore_index=True)
        all_monthly_df.to_csv('processed_data/all_programs_monthly.csv', index=False)
    
    print("Data preprocessing completed")
    return valid_programs

if __name__ == "__main__":
    valid_programs = preprocess_data()

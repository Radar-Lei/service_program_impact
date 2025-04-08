import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories if they don't exist
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load service program data
service_programs = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
print(f"Loaded {len(service_programs)} service programs")

# Create a mapping to match program indexes with file numbers
program_mapping = {idx: idx for idx in range(len(service_programs))}

# Container for all sentiment data
all_sentiment_data = []

# Process each service program's feedback
for program_idx in range(len(service_programs)):
    file_path = f'service_program_matches_SZ/service_program_{program_idx}_matches.csv'
    
    try:
        feedback_data = pd.read_csv(file_path)
        
        # Add program index for identification
        feedback_data['program_idx'] = program_idx
        feedback_data['program_name'] = service_programs.iloc[program_idx]['Service improvement programs']
        feedback_data['program_category'] = service_programs.iloc[program_idx]['Related service quality']
        
        # Calculate sentiment scores
        # Make sure Positive and Neutral columns exist
        if 'Positive' in feedback_data.columns and 'Neutral' in feedback_data.columns:
            # Ensure values are numeric
            feedback_data['Positive'] = pd.to_numeric(feedback_data['Positive'], errors='coerce')
            feedback_data['Neutral'] = pd.to_numeric(feedback_data['Neutral'], errors='coerce')
            
            # Calculate Negative as 1 - (Positive + Neutral)
            # Already exists in data but we'll recalculate for consistency
            if 'Negative' not in feedback_data.columns:
                feedback_data['Negative'] = 1 - (feedback_data['Positive'] + feedback_data['Neutral'])
            
            # Add to our combined data
            all_sentiment_data.append(feedback_data)
            
            print(f"Processed program {program_idx}: {len(feedback_data)} feedback entries")
        else:
            print(f"Warning: Missing sentiment columns in program {program_idx}")
    except Exception as e:
        print(f"Error processing program {program_idx}: {e}")

# Combine all feedback data
if all_sentiment_data:
    combined_data = pd.concat(all_sentiment_data, ignore_index=True)
    print(f"Combined data contains {len(combined_data)} feedback entries")
else:
    print("No sentiment data was successfully processed")
    exit(1)

# Basic statistics by service program
program_stats = combined_data.groupby(['program_idx', 'program_name', 'program_category'])[['Positive', 'Neutral', 'Negative']].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).reset_index()

# Save statistics to CSV
program_stats.to_csv('results/program_stats.csv', index=False)
print("Saved basic statistics to results/program_stats.csv")

# Create a simpler summary for easier reading
summary_stats = combined_data.groupby(['program_idx', 'program_name', 'program_category'])[['Positive', 'Neutral', 'Negative']].agg([
    'count', 'mean', 'std'
]).reset_index()

# Rename multiindex columns to flatten
summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns.values]

# Save summary statistics
summary_stats.to_csv('results/program_summary_stats.csv', index=False)
print("Saved summary statistics to results/program_summary_stats.csv")

# Calculate comparative metrics
program_metrics = combined_data.groupby(['program_idx', 'program_name', 'program_category']).agg({
    'Positive': ['mean', 'count'],
    'Neutral': 'mean',
    'Negative': 'mean'
}).reset_index()

# Flatten column names
program_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in program_metrics.columns.values]

# Calculate additional metrics
program_metrics['sentiment_ratio'] = program_metrics['Positive_mean'] / program_metrics['Negative_mean']
program_metrics['feedback_volume'] = program_metrics['Positive_count']

# Normalize sentiment ratio for better visualization
max_ratio = program_metrics['sentiment_ratio'].max()
program_metrics['normalized_sentiment'] = program_metrics['sentiment_ratio'] / max_ratio

# Save program metrics
program_metrics.to_csv('results/program_metrics.csv', index=False)
print("Saved program metrics to results/program_metrics.csv")

# Group by service category
category_stats = combined_data.groupby('program_category')[['Positive', 'Neutral', 'Negative']].agg([
    'count', 'mean', 'std'
]).reset_index()

# Rename multiindex columns to flatten
category_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in category_stats.columns.values]

# Save category statistics
category_stats.to_csv('results/category_stats.csv', index=False)
print("Saved category statistics to results/category_stats.csv")

# Print some key statistics to console and save to a summary file
with open('results/statistical_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("# Statistical Analysis of Service Program Impact\n\n")
    
    # Overall sentiment
    f.write("## Overall Sentiment Metrics\n")
    overall_positive = combined_data['Positive'].mean()
    overall_neutral = combined_data['Neutral'].mean()
    overall_negative = combined_data['Negative'].mean()
    
    f.write(f"Average Positive Sentiment: {overall_positive:.4f}\n")
    f.write(f"Average Neutral Sentiment: {overall_neutral:.4f}\n")
    f.write(f"Average Negative Sentiment: {overall_negative:.4f}\n\n")
    
    # Top and bottom programs by positive sentiment
    f.write("## Top 5 Programs by Positive Sentiment\n")
    top_programs = program_metrics.sort_values('Positive_mean', ascending=False).head(5)
    for idx, row in top_programs.iterrows():
        f.write(f"{row['program_name']}: {row['Positive_mean']:.4f}\n")
    
    f.write("\n## Bottom 5 Programs by Positive Sentiment\n")
    bottom_programs = program_metrics.sort_values('Positive_mean').head(5)
    for idx, row in bottom_programs.iterrows():
        f.write(f"{row['program_name']}: {row['Positive_mean']:.4f}\n")
    
    # Service categories comparison
    f.write("\n## Service Categories by Average Positive Sentiment\n")
    categories = category_stats.sort_values('Positive_mean', ascending=False)
    for idx, row in categories.iterrows():
        if pd.notna(row['program_category']) and row['program_category'] != '':
            f.write(f"{row['program_category']}: {row['Positive_mean']:.4f} (n={row['Positive_count']})\n")
    
    print("Saved statistical analysis summary to results/statistical_analysis.txt")

print("Basic statistical analysis completed successfully.")

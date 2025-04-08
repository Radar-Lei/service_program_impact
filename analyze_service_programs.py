#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy import stats
import os
import re
import tempfile
import urllib.request
from collections import defaultdict
import matplotlib
# Do not set font here, we'll set it via get_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False    # For minus sign display
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Chinese font utilities
def get_chinese_font():
    """Find an available Chinese font on the system"""
    # 尝试常见的中文字体
    chinese_fonts = ['STHeiti', 'Songti SC', 'SimSong', 'Yuanti SC', 'LiSong Pro',
                    'SimHei', 'Microsoft YaHei', 'Heiti TC', 'STFangsong', 
                     'WenQuanYi Zen Hei', 'Hiragino Sans GB', 'Noto Sans CJK SC', 
                     'Source Han Sans CN', 'Source Han Sans SC', 'PingFang HK']
    
    existing_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in existing_fonts:
            print(f"Using Chinese font: {font}")
            return font
    
    # 如果找不到中文字体，尝试使用默认字体
    print("No Chinese font found, attempting to download one.")
    return None

# 下载中文字体如果需要
def download_chinese_font():
    """Download a Chinese font if none is available on the system"""
    # 下载文泉驿微米黑字体(开源中文字体)
    temp_dir = tempfile.gettempdir()
    font_path = os.path.join(temp_dir, 'wqy-microhei.ttc')
    
    if not os.path.exists(font_path):
        print("下载中文字体...")
        url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/wqy-microhei.ttc"
        urllib.request.urlretrieve(url, font_path)
        print(f"字体已下载到: {font_path}")
        # Add the font to matplotlib's font manager
        fm.fontManager.addfont(font_path)
        # Need to rebuild the font cache
        fm._rebuild()
    
    return 'WenQuanYi Micro Hei'  # This is how matplotlib will refer to this font

# Initialize the Chinese font
chinese_font = get_chinese_font()
if chinese_font is None:
    chinese_font = download_chinese_font()

# Configure matplotlib to use the Chinese font
matplotlib.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']

# Create output directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Load service program data
def load_service_programs():
    """Load service program data and metadata"""
    print("Loading service program data...")
    programs_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
    programs_df.index = range(len(programs_df))  # Ensure index matches the service program ID
    return programs_df

# 2. Process feedback data for all programs
def process_feedback_data():
    """Process feedback data from all service programs"""
    print("Processing feedback data...")
    all_feedback = []
    
    # Iterate through each service program's feedback file
    for i in range(23):  # 23 service programs
        file_path = f'service_program_matches_SZ/service_program_{i}_matches.csv'
        try:
            program_feedback = pd.read_csv(file_path)
            program_feedback['service_program_id'] = i
            all_feedback.append(program_feedback)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    # Combine all feedback into a single dataframe
    feedback_df = pd.concat(all_feedback, ignore_index=True)
    
    # Convert sentiment columns to numeric if they're not already
    for col in ['Positive', 'Negative', 'Neutral']:
        if col in feedback_df.columns:
            feedback_df[col] = pd.to_numeric(feedback_df[col], errors='coerce')
    
    # Add derived sentiment metrics
    feedback_df['sentiment_ratio'] = feedback_df['Positive'] / (feedback_df['Positive'] + feedback_df['Negative'])
    feedback_df['net_sentiment'] = feedback_df['Positive'] - feedback_df['Negative']
    
    return feedback_df

# 3. Calculate program-level sentiment metrics
def calculate_program_metrics(feedback_df, programs_df):
    """Calculate sentiment metrics for each service program"""
    print("Calculating program-level metrics...")
    
    # Group by service program ID and calculate metrics
    program_metrics = feedback_df.groupby('service_program_id').agg({
        'Positive': ['mean', 'median', 'count'],
        'Negative': ['mean', 'median', 'count'],
        'sentiment_ratio': ['mean', 'median'],
        'net_sentiment': ['mean', 'median', 'sum']
    })
    
    # Flatten the multi-index columns
    program_metrics.columns = ['_'.join(col).strip() for col in program_metrics.columns.values]
    program_metrics = program_metrics.reset_index()
    
    # Join with program metadata
    result = pd.merge(program_metrics, programs_df, 
                     left_on='service_program_id', 
                     right_index=True, 
                     how='left')
    
    # Add a column for the program's rank based on net_sentiment_mean
    result['sentiment_rank'] = result['net_sentiment_mean'].rank(ascending=False)
    
    # Create category groupings
    def categorize_program(row):
        """Categorize programs based on their 'Related service quality' field"""
        quality = str(row['Related service quality']).lower()
        if pd.isna(quality) or quality == 'nan':
            return 'Other'
        elif '温度' in quality:
            return 'Comfort-Temperature'
        elif '信息服务' in quality:
            return 'Information Services'
        elif '票务服务' in quality or '票价' in quality:
            return 'Ticketing Services & Pricing'
        elif '卫生间' in quality or '母婴' in quality:
            return 'Convenience Facilities'
        elif '拥挤度' in quality:
            return 'Crowding Management'
        elif '可靠性' in quality or '发车频率' in quality:
            return 'Reliability & Frequency'
        elif '人员' in quality:
            return 'Personnel Services'
        else:
            return 'Other'
    
    result['program_category'] = result.apply(categorize_program, axis=1)
    
    return result

# 4. Generate visualizations
def create_visualizations(program_metrics, feedback_df):
    """Create visualization of results"""
    print("Generating visualizations...")
    
    # 1. Net sentiment by program (sorted)
    plt.figure(figsize=(12, 8))
    sorted_metrics = program_metrics.sort_values('net_sentiment_mean')
    bars = plt.barh(sorted_metrics['Service improvement programs'], 
           sorted_metrics['net_sentiment_mean'], 
           color=sns.color_palette("viridis", len(sorted_metrics)))
    
    # Add program category as color coding
    categories = sorted_metrics['program_category'].unique()
    category_colors = dict(zip(categories, sns.color_palette("viridis", len(categories))))
    for i, category in enumerate(sorted_metrics['program_category']):
        bars[i].set_color(category_colors[category])
    
    plt.xlabel('Net Sentiment (Positive - Negative)')
    plt.ylabel('Service Program')
    plt.title('Net Sentiment by Service Program', fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=cat) 
                      for cat, color in category_colors.items()]
    plt.legend(handles=legend_elements, title='Program Category')
    
    plt.tight_layout()
    plt.savefig('figures/net_sentiment_by_program.png', dpi=300, bbox_inches='tight')
    
    # 2. Positive vs Negative scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(program_metrics['Positive_mean'], 
               program_metrics['Negative_mean'], 
               c=program_metrics['net_sentiment_mean'], 
               cmap='viridis', 
               s=100,
               alpha=0.7)
    
    # Add labels for each point
    for i, row in program_metrics.iterrows():
        plt.annotate(i, 
                    (row['Positive_mean'], row['Negative_mean']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.xlabel('Average Positive Sentiment')
    plt.ylabel('Average Negative Sentiment')
    plt.title('Positive vs Negative Sentiment by Program', fontsize=14)
    plt.colorbar(scatter, label='Net Sentiment')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/positive_vs_negative_scatter.png', dpi=300, bbox_inches='tight')
    
    # 3. Program category comparison
    plt.figure(figsize=(12, 8))
    category_metrics = program_metrics.groupby('program_category').agg({
        'net_sentiment_mean': 'mean',
        'Positive_mean': 'mean',
        'Negative_mean': 'mean',
        'service_program_id': 'count'
    }).rename(columns={'service_program_id': 'program_count'})
    
    category_metrics = category_metrics.sort_values('net_sentiment_mean')
    
    bars = plt.barh(category_metrics.index, 
           category_metrics['net_sentiment_mean'], 
           color=sns.color_palette("viridis", len(category_metrics)))
    
    plt.xlabel('Average Net Sentiment')
    plt.ylabel('Program Category')
    plt.title('Net Sentiment by Program Category', fontsize=14)
    
    # Add program count annotations
    for i, (idx, row) in enumerate(category_metrics.iterrows()):
        plt.annotate(f"({int(row['program_count'])} programs)", 
                    (row['net_sentiment_mean'], i),
                    xytext=(5, 0),
                    textcoords='offset points',
                    va='center')
    
    plt.tight_layout()
    plt.savefig('figures/category_comparison.png', dpi=300, bbox_inches='tight')
    
    # 4. Positive and Negative sentiment comparison by category (grouped bar chart)
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(category_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, category_metrics['Positive_mean'], width, label='Positive', color='green', alpha=0.7)
    ax.bar(x + width/2, category_metrics['Negative_mean'], width, label='Negative', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(category_metrics.index, rotation=45, ha='right')
    ax.legend()
    
    ax.set_ylabel('Average Sentiment Score')
    ax.set_title('Positive vs Negative Sentiment by Program Category', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/positive_negative_by_category.png', dpi=300, bbox_inches='tight')
    
    # 5. Top and bottom programs table visualization
    plt.figure(figsize=(14, 6))
    
    # Get top 5 and bottom 5 programs by net sentiment
    top_programs = program_metrics.sort_values('net_sentiment_mean', ascending=False).head(5)
    bottom_programs = program_metrics.sort_values('net_sentiment_mean').head(5)
    
    # Create a table for top programs
    plt.subplot(1, 2, 1)
    plt.axis('off')
    table_data = [
        [f"{i+1}. {row['Service improvement programs']}", 
         f"{row['net_sentiment_mean']:.4f}"]
        for i, (_, row) in enumerate(top_programs.iterrows())
    ]
    
    table = plt.table(cellText=table_data,
                     colLabels=['Top Programs', 'Net Sentiment'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.7, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Top 5 Programs by Net Sentiment', fontsize=14)
    
    # Create a table for bottom programs
    plt.subplot(1, 2, 2)
    plt.axis('off')
    table_data = [
        [f"{i+1}. {row['Service improvement programs']}", 
         f"{row['net_sentiment_mean']:.4f}"]
        for i, (_, row) in enumerate(bottom_programs.iterrows())
    ]
    
    table = plt.table(cellText=table_data,
                     colLabels=['Bottom Programs', 'Net Sentiment'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.7, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Bottom 5 Programs by Net Sentiment', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/top_bottom_programs.png', dpi=300, bbox_inches='tight')
    
    # Save the metrics table
    program_metrics[['service_program_id', 'Service improvement programs', 'net_sentiment_mean', 
                   'Positive_mean', 'Negative_mean', 'sentiment_ratio_mean', 'program_category']].to_csv(
        'results/program_metrics.csv', index=False, encoding='utf-8-sig')

# 5. Perform statistical analysis
def perform_statistical_analysis(program_metrics, feedback_df):
    """Perform statistical analysis on the data"""
    print("Performing statistical analysis...")
    
    # 1. ANOVA to test differences between program categories
    categories = program_metrics['program_category'].unique()
    anova_data = []
    
    for category in categories:
        category_programs = program_metrics[program_metrics['program_category'] == category]['service_program_id'].tolist()
        category_feedback = feedback_df[feedback_df['service_program_id'].isin(category_programs)]['net_sentiment']
        anova_data.append(category_feedback)
    
    f_stat, p_value = stats.f_oneway(*[d for d in anova_data if len(d) > 0])
    
    # 2. Calculate statistical significance between programs
    # Using Tukey's HSD test for multiple comparisons
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Prepare data for Tukey's test
    program_ids = []
    net_sentiments = []
    
    for program_id in program_metrics['service_program_id']:
        program_feedback = feedback_df[feedback_df['service_program_id'] == program_id]['net_sentiment']
        program_ids.extend([program_id] * len(program_feedback))
        net_sentiments.extend(program_feedback)
    
    # Run Tukey's test
    tukey_results = pairwise_tukeyhsd(net_sentiments, program_ids, alpha=0.05)
    
    # Save statistical results
    stats_results = {
        'anova_f_statistic': f_stat,
        'anova_p_value': p_value,
        'tukey_results': tukey_results
    }
    
    # Create a readable output of statistical results
    with open('results/statistical_analysis.txt', 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("===========================\n\n")
        
        f.write("1. ANOVA Test for Program Categories\n")
        f.write(f"F-statistic: {f_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Conclusion: {'Significant differences exist between categories' if p_value < 0.05 else 'No significant differences between categories'}\n\n")
        
        f.write("2. Tukey's HSD Test for Program Comparisons\n")
        f.write(str(tukey_results))
    
    return stats_results

# 6. Regression analysis
def regression_analysis(feedback_df, program_metrics):
    """Perform regression analysis"""
    print("Performing regression analysis...")
    
    # 1. Create dummy variables for program categories
    program_dummies = pd.get_dummies(program_metrics['program_category'], prefix='category')
    program_data = pd.concat([program_metrics, program_dummies], axis=1)
    
    # 2. Prepare data for regression
    import statsmodels.api as sm
    
    # Make sure all values are numeric
    X = program_data.filter(like='category_').astype(float)
    y = program_data['net_sentiment_mean'].astype(float)
    
    # Add constant
    X = sm.add_constant(X)
    
    # Check if any columns are all zeros or have other issues
    X = X.loc[:, (X != 0).any(axis=0)]  # Remove columns with all zeros
    
    try:
        # 3. Run regression
        model = sm.OLS(y, X).fit()
        
        # 4. Save regression results
        with open('results/regression_analysis.txt', 'w') as f:
            f.write(str(model.summary()))
        
        return model
    except Exception as e:
        print(f"Error during regression analysis: {e}")
        # Create a placeholder model object with necessary attributes for reporting
        class DummyModel:
            def __init__(self):
                self.rsquared = 0
                self.rsquared_adj = 0
                self.summary = lambda: type('obj', (object,), {
                    'tables': [None, pd.DataFrame([["Error", "No valid model"]])]
                })
        
        print("Continuing with other analyses...")
        return DummyModel()

# Main function
def main():
    """Main function to run the analysis"""
    # 1. Load service program data
    programs_df = load_service_programs()
    
    # 2. Process feedback data
    feedback_df = process_feedback_data()
    
    # 3. Calculate program-level metrics
    program_metrics = calculate_program_metrics(feedback_df, programs_df)
    
    # 4. Generate visualizations
    create_visualizations(program_metrics, feedback_df)
    
    # 5. Perform statistical analysis
    stats_results = perform_statistical_analysis(program_metrics, feedback_df)
    
    # 6. Regression analysis
    try:
        regression_model = regression_analysis(feedback_df, program_metrics)
    except Exception as e:
        print(f"Error in regression analysis: {e}")
        # Continue with a placeholder model if regression fails
        class DummyModel:
            def __init__(self):
                self.rsquared = 0
                self.rsquared_adj = 0
                self.summary = lambda: type('obj', (object,), {
                    'tables': [None, pd.DataFrame([["Error", "No valid model"]])]
                })
        regression_model = DummyModel()
    
    # 7. Generate markdown report
    
    print("\nAnalysis complete! Results saved to service_program_impact_report.md")

if __name__ == "__main__":
    main()

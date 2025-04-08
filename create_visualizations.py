import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as fm
import tempfile
import urllib.request
import matplotlib

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
plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
matplotlib.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Apply font settings specifically for seaborn plots
sns.set(font=chinese_font)

# Function to apply Chinese font to a plot
def apply_chinese_font(ax):
    """Apply Chinese font settings to plot elements"""
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))
    
    # Apply font to title and labels
    ax.title.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))
    ax.xaxis.label.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))
    ax.yaxis.label.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))
    
    if ax.legend_ is not None:
        for text in ax.legend_.get_texts():
            text.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def truncate_text(text, max_len=30):
    """Truncate text for better display in plots"""
    if pd.isna(text) or text == '':
        return 'Unknown'
    
    if len(text) > max_len:
        return text[:max_len] + '...'
    return text

# Load data
try:
    print("Loading data files...")
    service_programs = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
    program_metrics = pd.read_csv('results/program_metrics.csv')
    
    # In case we need to regenerate
    if not os.path.exists('results/program_metrics.csv'):
        print("Program metrics not found, please run analyze_statistics.py first")
        exit(1)
    
    # Prepare category data if needed
    if os.path.exists('results/category_stats.csv'):
        category_stats = pd.read_csv('results/category_stats.csv')
    else:
        print("Category stats not found, some visualizations may be skipped")
    
    print(f"Data loaded successfully. Found {len(service_programs)} service programs.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please run analyze_statistics.py first to generate the required data files.")
    exit(1)

# 1. Create a bar chart of positive sentiment by program
print("Creating positive sentiment bar chart...")
plt.figure(figsize=(16, 12))

# Sort by positive sentiment score
sorted_programs = program_metrics.sort_values('Positive_mean', ascending=False)

# Add truncated names
sorted_programs['short_name'] = sorted_programs['program_name'].apply(truncate_text)

# Plot with error bars if standard deviation is available
if 'Positive_std' in sorted_programs.columns:
    ax = sns.barplot(
        x='Positive_mean',
        y='short_name',
        data=sorted_programs,
        palette='viridis',
        xerr=sorted_programs['Positive_std']
    )
else:
    ax = sns.barplot(
        x='Positive_mean',
        y='short_name',
        data=sorted_programs,
        palette='viridis'
    )

# Add value labels
for i, v in enumerate(sorted_programs['Positive_mean']):
    t = ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    t.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))

plt.title('服务项目积极情感得分 (Positive Sentiment Score by Service Program)', fontsize=16)
plt.xlabel('Average Positive Sentiment Score')
plt.ylabel('Service Program')
plt.xlim(0, max(sorted_programs['Positive_mean']) * 1.1)
apply_chinese_font(ax)
plt.tight_layout()
plt.savefig('figures/positive_sentiment_by_program.png')
plt.close()

# 2. Create a comparative bar chart for sentiment categories
print("Creating comparative sentiment bar chart...")
plt.figure(figsize=(16, 12))

# Prepare data for grouped bar chart
top_10_programs = sorted_programs.head(10)
sentiment_data = pd.melt(
    top_10_programs,
    id_vars=['short_name'],
    value_vars=['Positive_mean', 'Neutral_mean', 'Negative_mean'],
    var_name='Sentiment Type',
    value_name='Score'
)

# Clean up sentiment type labels
sentiment_data['Sentiment Type'] = sentiment_data['Sentiment Type'].str.replace('_mean', '')

# Create grouped bar chart
plt.figure(figsize=(16, 12))
ax = sns.barplot(
    x='Score',
    y='short_name',
    hue='Sentiment Type',
    data=sentiment_data,
    palette=['green', 'gray', 'red']
)

plt.title('前10个服务项目的情感分布 (Sentiment Distribution for Top 10 Service Programs)', fontsize=16)
plt.xlabel('Average Sentiment Score')
plt.ylabel('Service Program')
plt.legend(title='Sentiment Type')
apply_chinese_font(ax)
plt.tight_layout()
plt.savefig('figures/sentiment_distribution_top_programs.png')
plt.close()

# 3. Create a horizontal stacked bar chart for sentiment distribution
print("Creating stacked sentiment distribution chart...")
plt.figure(figsize=(16, 12))

# Prepare data for stacked bar chart
stacked_data = sorted_programs.copy()
stacked_data['Total'] = 1.0  # Normalize to show percentages

# Create stacked bar chart
ax = stacked_data.plot(
    kind='barh',
    y=['Positive_mean', 'Neutral_mean', 'Negative_mean'],
    x='short_name',
    stacked=True,
    figsize=(16, 12),
    color=['green', 'gray', 'red'],
    width=0.8
)

# Format as percentage
ax.xaxis.set_major_formatter(PercentFormatter(1.0))

plt.title('各服务项目情感分布 (Sentiment Distribution Across Service Programs)', fontsize=16)
plt.xlabel('Proportion of Sentiment')
plt.ylabel('Service Program')
plt.legend(['Positive', 'Neutral', 'Negative'])
apply_chinese_font(ax)
plt.tight_layout()
plt.savefig('figures/sentiment_distribution_stacked.png')
plt.close()

# 4. Service category comparison (if data available)
if 'category_stats.csv' in os.listdir('results'):
    print("Creating service category comparison chart...")
    category_stats = pd.read_csv('results/category_stats.csv')
    
    # Remove empty categories
    category_stats = category_stats[category_stats['program_category'].notna() & (category_stats['program_category'] != '')]
    
    if len(category_stats) > 0:
        plt.figure(figsize=(14, 10))
        
        # Sort by positive sentiment
        category_stats = category_stats.sort_values('Positive_mean', ascending=False)
        
        # Clean category names
        category_stats['category_name'] = category_stats['program_category'].apply(truncate_text)
        
        # Create bar chart
        ax = sns.barplot(
            x='Positive_mean',
            y='category_name',
            data=category_stats,
            palette='viridis'
        )
        
        # Add value labels with Chinese font
        for i, v in enumerate(category_stats['Positive_mean']):
            t = ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            t.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))
            
        plt.title('各服务类别积极情感得分 (Positive Sentiment by Service Category)', fontsize=16)
        plt.xlabel('Average Positive Sentiment Score')
        plt.ylabel('Service Category')
        apply_chinese_font(ax)
        plt.tight_layout()
        plt.savefig('figures/positive_sentiment_by_category.png')
        plt.close()
    else:
        print("No valid category data found for visualization")

# 5. Create a scatter plot of positive sentiment vs. feedback volume
print("Creating sentiment vs. feedback volume scatter plot...")
plt.figure(figsize=(14, 10))

# Create scatter plot
ax = sns.scatterplot(
    x='Positive_mean',
    y='feedback_volume',
    data=program_metrics,
    size='normalized_sentiment',  # Size points by normalized sentiment ratio
    sizes=(50, 400),
    alpha=0.7,
    palette='viridis',
    hue='Positive_mean'  # Color by positive sentiment
)

# Add program labels with Chinese font
for i, row in program_metrics.iterrows():
    t = plt.text(
        row['Positive_mean'] + 0.01,
        row['feedback_volume'],
        truncate_text(row['program_name'], 20),
        fontsize=9
    )
    t.set_fontproperties(fm.FontProperties(fname=fm.findfont(chinese_font)))

plt.title('积极情感与反馈量对比 (Positive Sentiment vs. Feedback Volume)', fontsize=16)
plt.xlabel('Average Positive Sentiment Score')
plt.ylabel('Number of Feedback Entries')
apply_chinese_font(ax)
plt.tight_layout()
plt.savefig('figures/sentiment_vs_volume.png')
plt.close()

# Sentiment correlation heatmap removed as requested

print("All visualizations created successfully.")
print("Visualization files saved to the 'figures' directory.")

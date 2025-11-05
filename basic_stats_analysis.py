import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Setup for Chinese font display
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import platform

# Check if on macOS
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB']
else:  # Windows or other systems
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun']

# Universal settings
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs properly

def load_program_data():
    """Load service program metadata"""
    program_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
    return program_df

def analyze_program_stats(program_ids=None, processed_dir='processed_data'):
    """Perform basic statistical analysis for each service program
    
    Args:
        program_ids (list, optional): 要分析的服务项目ID列表。如果为None，则分析所有项目。
        processed_dir (str, optional): 处理后数据的目录路径。默认为'processed_data'。
    
    Returns:
        DataFrame: 分析结果摘要数据框
    """
    print("Performing basic statistical analysis...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Get all daily data files
    if program_ids is not None:
        # 只获取指定项目的文件
        daily_files = [f'{processed_dir}/program_{pid}_daily.csv' for pid in program_ids 
                        if os.path.exists(f'{processed_dir}/program_{pid}_daily.csv')]
        if not daily_files:
            print("Warning: No daily data files found for the specified program IDs")
    else:
        # 获取所有项目的文件
        daily_files = glob.glob(f'{processed_dir}/program_*_daily.csv')
    
    # Create summary dataframe for all program results
    summary_results = pd.DataFrame(columns=[
        'program_id', 'program_name', 'pre_mean', 'post_mean', 'mean_diff', 
        'pre_median', 'post_median', 'median_diff', 
        'pre_std', 'post_std', 't_stat', 'p_value', 'significant_0.05'
    ])
    
    # 收集所有项目的pre和post数据，用于创建总体density图
    all_program_data = []
    
    # 如果没有找到任何文件，提前返回空的摘要结果
    if not daily_files:
        print("No daily data files found for analysis")
        return summary_results
    
    for file_path in daily_files:
        # Extract program ID from filename
        program_id = int(file_path.split('_')[-2])
        
        # Load program data
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            print(f"Warning: No data found for program {program_id}")
            continue
        
        if 'post_intervention' not in df.columns or 'mean_sentiment' not in df.columns:
            print(f"Warning: Required columns missing for program {program_id}")
            continue
        
        # Separate pre and post intervention data
        pre_data = df[df['post_intervention'] == 0]['mean_sentiment']
        post_data = df[df['post_intervention'] == 1]['mean_sentiment']
        
        # Check if we have enough data for analysis
        if len(pre_data) < 2 or len(post_data) < 2:
            print(f"Warning: Insufficient data for statistical analysis for program {program_id}")
            continue
        
        # Extract program name
        # Define standard program name mapping
        program_names_map = {
            0: "Temperature Consistency",
            1: "Smart Map Display",
            4: "QR Code Payment",
            5: "Restroom Renovation",
            15: "Mobile Nursing Rooms",
            22: "Fare Reduction"
        }
        # Override program name with standard mapping
        program_name = program_names_map.get(program_id, f"Program {program_id}")
        
        # Calculate descriptive statistics
        pre_mean = pre_data.mean()
        post_mean = post_data.mean()
        mean_diff = post_mean - pre_mean
        
        pre_median = pre_data.median()
        post_median = post_data.median()
        median_diff = post_median - pre_median
        
        pre_std = pre_data.std()
        post_std = post_data.std()
        
        # Perform t-test for mean comparison
        t_stat, p_value = stats.ttest_ind(post_data, pre_data, equal_var=False)
        significant = p_value < 0.05
        
        # Add results to summary dataframe
        summary_results = pd.concat([summary_results, pd.DataFrame({
            'program_id': [program_id],
            'program_name': [program_name],
            'pre_mean': [pre_mean],
            'post_mean': [post_mean],
            'mean_diff': [mean_diff],
            'pre_median': [pre_median],
            'post_median': [post_median],
            'median_diff': [median_diff],
            'pre_std': [pre_std],
            'post_std': [post_std],
            't_stat': [t_stat],
            'p_value': [p_value],
            'significant_0.05': [significant]
        })], ignore_index=True)
        
        # 收集每个项目的密度图数据
        if len(pre_data) >= 2 and len(post_data) >= 2:
            all_program_data.append({
                'program_id': program_id,
                'program_name': program_name,
                'pre_data': pre_data,
                'post_data': post_data
            })
        
        # Categorize sentiment (for Fisher/Chi-square tests)
        # Create sentiment categories (Positive: > 0.2, Neutral: -0.2 to 0.2, Negative: < -0.2)
        df['sentiment_category'] = pd.cut(
            df['mean_sentiment'], 
            bins=[-1, -0.2, 0.2, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Create contingency table
        try:
            contingency = pd.crosstab(df['post_intervention'], df['sentiment_category'])
            # Save contingency table
            contingency.to_csv(f'results/program_{program_id}_contingency.csv')
            
            # Check if we have a valid contingency table for statistical tests
            # Special case: only one sentiment category
            if contingency.shape[1] == 1:
                # In this case, we can use a proportion test (pre vs post) for the single category
                category = contingency.columns[0]  # The only sentiment category present
                
                # Get pre and post counts
                pre_count = contingency.loc[0, category] if 0 in contingency.index else 0
                post_count = contingency.loc[1, category] if 1 in contingency.index else 0
                
                # Get total pre and post samples
                total_pre = len(df[df['post_intervention'] == 0])
                total_post = len(df[df['post_intervention'] == 1])
                
                # Calculate proportions
                pre_prop = pre_count / total_pre if total_pre > 0 else 0
                post_prop = post_count / total_post if total_post > 0 else 0
                
                # Perform proportion test
                from statsmodels.stats.proportion import proportions_ztest
                count = np.array([post_count, pre_count])
                nobs = np.array([total_post, total_pre])
                
                # Only perform test if we have sufficient data
                if min(nobs) > 0 and min(count) > 0:
                    stat, p = proportions_ztest(count, nobs)
                    test_type = "Proportion Z-test"
                    test_stat = stat
                    test_p = p
                else:
                    test_type = "No statistical test"
                    test_stat = np.nan
                    test_p = np.nan
                
                # Save test results
                with open(f'results/program_{program_id}_categorical_test.txt', 'w') as f:
                    f.write(f"Test type: {test_type}\n")
                    f.write(f"Note: Only one sentiment category ({category}) present in data\n")
                    f.write(f"Pre-intervention proportion: {pre_prop:.4f} ({pre_count}/{total_pre})\n")
                    f.write(f"Post-intervention proportion: {post_prop:.4f} ({post_count}/{total_post})\n")
                    if test_type != "No statistical test":
                        f.write(f"Statistic: {test_stat}\n")
                        f.write(f"p-value: {test_p}\n")
                        f.write(f"Significant at 0.05: {test_p < 0.05}\n")
                    f.write(f"\nContingency table:\n{contingency.to_string()}\n")
                
                # Continue to next program
                continue
                
            # We need at least two rows (pre and post) 
            if contingency.shape[0] < 2:
                raise ValueError(f"Contingency table must have at least 2 rows but has shape {contingency.shape}")
            
            # For Fisher's exact test, we need exactly a 2x2 table
            if contingency.shape == (2, 2):
                # Chi-square test if all expected frequencies are >= 5
                expected = stats.chi2_contingency(contingency)[3]
                if (expected >= 5).all():
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    test_type = "Chi-square"
                    test_stat = chi2
                    test_p = p
                else:
                    # Use Fisher's exact test for small expected frequencies
                    odds_ratio, p = stats.fisher_exact(contingency)
                    test_type = "Fisher's exact"
                    test_stat = odds_ratio
                    test_p = p
            else:
                # If table is not 2x2, we can only use Chi-square test
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                test_type = "Chi-square"
                test_stat = chi2
                test_p = p
                
            # Save test results
            with open(f'results/program_{program_id}_categorical_test.txt', 'w') as f:
                f.write(f"Test type: {test_type}\n")
                f.write(f"Statistic: {test_stat}\n")
                f.write(f"p-value: {test_p}\n")
                f.write(f"Significant at 0.05: {test_p < 0.05}\n")
                f.write(f"\nContingency table:\n{contingency.to_string()}\n")
            
        except Exception as e:
            print(f"Warning: Could not perform categorical test for program {program_id}: {e}")
            # Still save the contingency table if it was created
            if 'contingency' in locals():
                with open(f'results/program_{program_id}_categorical_test.txt', 'w') as f:
                    f.write(f"Categorical test could not be performed: {e}\n\n")
                    f.write(f"Contingency table:\n{contingency.to_string()}\n")
    
    # Sort summary results by significance and effect size
    summary_results = summary_results.sort_values(by=['significant_0.05', 'mean_diff'], ascending=[False, False])
    
    # Save summary results
    summary_results.to_csv('results/program_stats_summary.csv', index=False)
    
    # Create visualizations for overall results
    create_overall_visualizations(summary_results)
    
    # 创建总的密度图
    create_overall_density_plot(all_program_data)
    
    print("Basic statistical analysis completed")
    return summary_results

def create_program_visualizations(df, program_id, program_name):
    """Create visualizations for a single program"""
    # 不生成单个项目的可视化，按照用户要求，只需要生成总的可视化
    pass

def create_overall_density_plot(program_data_list):
    """创建所有项目的综合密度图"""
    if not program_data_list:
        print("Warning: No data for overall density plot")
        return
        
    # 确保figures目录存在
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 创建一个大图，包含所有项目的密度子图
    n_programs = len(program_data_list)
    n_cols = 3  # 每行3个子图
    n_rows = (n_programs + n_cols - 1) // n_cols  # 计算需要的行数
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, program in enumerate(program_data_list):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 绘制密度图
        sns.kdeplot(program['pre_data'], label='Pre-intervention', shade=True, alpha=0.5)
        sns.kdeplot(program['post_data'], label='Post-intervention', shade=True, alpha=0.5)
        
        plt.title(f"{program['program_name']}")
        plt.xlabel('Sentiment Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Make SVG text editable
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects
    plt.rcParams['font.family'] = 'sans-serif'  # Use generic font family
    plt.savefig('figures/overall_density_plots.svg')
    plt.close()

def create_overall_visualizations(summary_df):
    """创建整体可视化：融合均值差异条形图和前后对比散点图"""
    if len(summary_df) == 0:
        print("Warning: No data for overall visualizations")
        return
    
    # 确保figures目录存在
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # 创建一个1行2列的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 第一个子图：均值差异条形图
    # 按均值差异排序
    sorted_df = summary_df.sort_values('mean_diff')
    
    # 创建条形图
    bars = ax1.bar(sorted_df['program_name'], sorted_df['mean_diff'], 
             color=[('green' if x > 0 else 'red') for x in sorted_df['mean_diff']])
    
    # 添加统计显著性标记
    for i, significant in enumerate(sorted_df['significant_0.05']):
        if significant:
            ax1.text(i, sorted_df['mean_diff'].iloc[i], '*', 
                 ha='center', va='bottom' if sorted_df['mean_diff'].iloc[i] > 0 else 'top', 
                 fontsize=20)
    
    ax1.set_title('Sentiment Mean Changes by Program (After - Before)')
    ax1.set_xlabel('Service Program')
    ax1.set_ylabel('Sentiment Mean Difference')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(sorted_df['program_name'], rotation=90)
    
    # 第二个子图：前后散点图对比
    ax2.scatter(summary_df['pre_mean'], summary_df['post_mean'], 
              c=['green' if x else 'red' for x in summary_df['significant_0.05']], 
              alpha=0.7, s=100)
    
    # 添加项目标签
    for i, row in summary_df.iterrows():
        ax2.annotate(f"Program {row['program_id']}", 
                  (row['pre_mean'], row['post_mean']),
                  xytext=(5, 5), textcoords='offset points')
    
    # 添加对角线（表示无变化）
    min_val = min(summary_df['pre_mean'].min(), summary_df['post_mean'].min())
    max_val = max(summary_df['pre_mean'].max(), summary_df['post_mean'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax2.set_title('Pre vs Post Intervention Sentiment Mean Comparison')
    ax2.set_xlabel('Pre-Intervention Mean')
    ax2.set_ylabel('Post-Intervention Mean')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.close()

def perform_time_series_analysis(program_ids=None, processed_dir='processed_data'):
    """Perform time series analysis for service programs
    
    Args:
        program_ids (list, optional): List of program IDs to analyze. If None, analyze all programs.
        processed_dir (str, optional): Directory path for processed data. Default is 'processed_data'.
        
    Returns:
        DataFrame: TSA results summary
    """
    print("Performing time series analysis...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Get all daily data files
    if program_ids is not None:
        daily_files = [f'{processed_dir}/program_{pid}_daily.csv' for pid in program_ids 
                        if os.path.exists(f'{processed_dir}/program_{pid}_daily.csv')]
        if not daily_files:
            print("Warning: No daily data files found for the specified program IDs")
    else:
        daily_files = glob.glob(f'{processed_dir}/program_*_daily.csv')
    
    # TSA results summary
    tsa_results = pd.DataFrame(columns=[
        'program_id', 'program_name', 'pre_adf_stat', 'pre_adf_pvalue', 'pre_stationary',
        'post_adf_stat', 'post_adf_pvalue', 'post_stationary',
        'pre_trend_slope', 'pre_trend_pvalue', 'post_trend_slope', 'post_trend_pvalue',
        'pre_mean', 'post_mean', 'trend_change', 'volatility_change'
    ])
    
    # Collect all time series data for visualization
    all_ts_data = []
    
    if not daily_files:
        print("No daily data files found for TSA")
        return tsa_results
    
    for file_path in daily_files:
        # Extract program ID from filename
        program_id = int(file_path.split('_')[-2])
        
        # Load program data
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            print(f"Warning: No data found for program {program_id}")
            continue
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Define standard program name mapping
        program_names_map = {
            0: "Temperature Consistency",
            1: "Smart Map Display", 
            4: "QR Code Payment",
            5: "Restroom Renovation",
            15: "Mobile Nursing Rooms",
            22: "Fare Reduction"
        }
        program_name = program_names_map.get(program_id, f"Program {program_id}")
        
        # Separate pre and post intervention data
        pre_data = df[df['post_intervention'] == 0]['mean_sentiment'].values
        post_data = df[df['post_intervention'] == 1]['mean_sentiment'].values
        
        # Check if we have enough data for TSA
        if len(pre_data) < 10 or len(post_data) < 10:
            print(f"Warning: Insufficient data for TSA for program {program_id}")
            continue
        
        # ADF test for stationarity
        try:
            pre_adf_stat, pre_adf_pvalue = adfuller(pre_data)[:2]
            pre_stationary = pre_adf_pvalue < 0.05
        except:
            pre_adf_stat, pre_adf_pvalue, pre_stationary = np.nan, np.nan, False
            
        try:
            post_adf_stat, post_adf_pvalue = adfuller(post_data)[:2]
            post_stationary = post_adf_pvalue < 0.05
        except:
            post_adf_stat, post_adf_pvalue, post_stationary = np.nan, np.nan, False
        
        # Trend analysis using linear regression
        def analyze_trend(data):
            if len(data) < 3:
                return np.nan, np.nan
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            
            # Simple t-test for slope significance
            y_pred = model.predict(X)
            residuals = y - y_pred
            se_slope = np.sqrt(np.sum(residuals**2) / (len(data) - 2)) / np.sqrt(np.sum((X.flatten() - X.mean())**2))
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))
            
            return slope, p_value
        
        pre_trend_slope, pre_trend_pvalue = analyze_trend(pre_data)
        post_trend_slope, post_trend_pvalue = analyze_trend(post_data)
        
        # Calculate volatility change
        pre_volatility = np.std(pre_data)
        post_volatility = np.std(post_data)
        volatility_change = post_volatility - pre_volatility
        
        # Calculate trend change
        trend_change = post_trend_slope - pre_trend_slope if not (np.isnan(post_trend_slope) or np.isnan(pre_trend_slope)) else np.nan
        
        # Add results to summary
        tsa_results = pd.concat([tsa_results, pd.DataFrame({
            'program_id': [program_id],
            'program_name': [program_name],
            'pre_adf_stat': [pre_adf_stat],
            'pre_adf_pvalue': [pre_adf_pvalue],
            'pre_stationary': [pre_stationary],
            'post_adf_stat': [post_adf_stat],
            'post_adf_pvalue': [post_adf_pvalue],
            'post_stationary': [post_stationary],
            'pre_trend_slope': [pre_trend_slope],
            'pre_trend_pvalue': [pre_trend_pvalue],
            'post_trend_slope': [post_trend_slope],
            'post_trend_pvalue': [post_trend_pvalue],
            'pre_mean': [np.mean(pre_data)],
            'post_mean': [np.mean(post_data)],
            'trend_change': [trend_change],
            'volatility_change': [volatility_change]
        })], ignore_index=True)
        
        # Collect data for visualization
        all_ts_data.append({
            'program_id': program_id,
            'program_name': program_name,
            'dates': df['date'].values,
            'sentiment': df['mean_sentiment'].values,
            'intervention': df['post_intervention'].values,
            'intervention_date': df['intervention_date'].iloc[0] if 'intervention_date' in df.columns else None
        })
    
    # Save TSA results
    tsa_results.to_csv('results/tsa_summary.csv', index=False)
    
    # Create TSA visualizations
    create_tsa_visualizations(all_ts_data)
    
    # Create TSA summary table
    create_tsa_summary_table(tsa_results)
    
    print("Time series analysis completed")
    return tsa_results

def create_tsa_visualizations(ts_data_list):
    """Create time series analysis visualizations"""
    if not ts_data_list:
        print("Warning: No data for TSA visualizations")
        return
        
    # Ensure figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Create a comprehensive time series plot
    n_programs = len(ts_data_list)
    n_cols = 2
    n_rows = (n_programs + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 4 * n_rows))
    
    for i, program in enumerate(ts_data_list):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot time series
        dates = pd.to_datetime(program['dates'])
        sentiment = program['sentiment']
        intervention = program['intervention']
        
        # Pre and post intervention periods
        pre_mask = intervention == 0
        post_mask = intervention == 1
        
        plt.scatter(dates[pre_mask], sentiment[pre_mask], c='blue', alpha=0.7, s=20, label='Pre-intervention')
        plt.scatter(dates[post_mask], sentiment[post_mask], c='red', alpha=0.7, s=20, label='Post-intervention')
        
        # Add intervention line
        if program['intervention_date']:
            intervention_date = pd.to_datetime(program['intervention_date'])
            plt.axvline(x=intervention_date, color='green', linestyle='--', alpha=0.8, label='Intervention')
        
        plt.title(f"{program['program_name']}")
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    # Make SVG text editable
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.savefig('figures/tsa_time_series.svg')
    plt.close()

def create_tsa_summary_table(tsa_results):
    """Create TSA summary table in text format"""
    if len(tsa_results) == 0:
        print("Warning: No TSA results to summarize")
        return
    
    with open('results/tsa_summary_table.txt', 'w') as f:
        f.write("TIME SERIES ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STATIONARITY TESTS (ADF Test)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Program':<20} {'Pre-ADF':<10} {'Pre-p':<8} {'Pre-Stat':<10} {'Post-ADF':<10} {'Post-p':<8} {'Post-Stat':<10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in tsa_results.iterrows():
            f.write(f"{row['program_name']:<20} "
                   f"{row['pre_adf_stat']:<10.3f} "
                   f"{row['pre_adf_pvalue']:<8.3f} "
                   f"{str(row['pre_stationary']):<10} "
                   f"{row['post_adf_stat']:<10.3f} "
                   f"{row['post_adf_pvalue']:<8.3f} "
                   f"{str(row['post_stationary']):<10}\n")
        
        f.write("\nTREND ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Program':<20} {'Pre-Slope':<12} {'Pre-p':<8} {'Post-Slope':<12} {'Post-p':<8} {'Change':<10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in tsa_results.iterrows():
            trend_change = row['trend_change'] if not pd.isna(row['trend_change']) else 0
            f.write(f"{row['program_name']:<20} "
                   f"{row['pre_trend_slope']:<12.6f} "
                   f"{row['pre_trend_pvalue']:<8.3f} "
                   f"{row['post_trend_slope']:<12.6f} "
                   f"{row['post_trend_pvalue']:<8.3f} "
                   f"{trend_change:<10.6f}\n")
        
        f.write("\nSUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Program':<20} {'Pre-Mean':<10} {'Post-Mean':<10} {'Vol-Change':<12}\n")
        f.write("-" * 60 + "\n")
        
        for _, row in tsa_results.iterrows():
            f.write(f"{row['program_name']:<20} "
                   f"{row['pre_mean']:<10.4f} "
                   f"{row['post_mean']:<10.4f} "
                   f"{row['volatility_change']:<12.6f}\n")
        
        f.write("\nNOTES:\n")
        f.write("- ADF Test: Augmented Dickey-Fuller test for stationarity (p<0.05 = stationary)\n")
        f.write("- Trend Slope: Linear trend coefficient (positive = increasing trend)\n")
        f.write("- Vol-Change: Change in volatility (standard deviation) post-intervention\n")
        f.write("- Trend Change: Difference in trend slopes (post - pre intervention)\n")

if __name__ == "__main__":
    summary_results = analyze_program_stats()
    tsa_results = perform_time_series_analysis()
    print(f"分析完成。基本统计分析和时间序列分析结果已保存至'results'和'figures'目录。")

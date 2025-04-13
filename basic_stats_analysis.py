import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
from statsmodels.stats.contingency_tables import mcnemar
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

def load_program_data():
    """Load service program metadata"""
    program_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
    return program_df

def analyze_program_stats(processed_dir='processed_data'):
    """Perform basic statistical analysis for each service program"""
    print("Performing basic statistical analysis...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Get all weekly data files
    weekly_files = glob.glob(f'{processed_dir}/program_*_weekly.csv')
    
    # Create summary dataframe for all program results
    summary_results = pd.DataFrame(columns=[
        'program_id', 'program_name', 'pre_mean', 'post_mean', 'mean_diff', 
        'pre_median', 'post_median', 'median_diff', 
        'pre_std', 'post_std', 't_stat', 'p_value', 'significant_0.05'
    ])
    
    for file_path in weekly_files:
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
        program_name = df['program_name'].iloc[0] if 'program_name' in df.columns else f"Program {program_id}"
        
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
        
        # Create visualizations for this program
        create_program_visualizations(df, program_id, program_name)
        
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
    
    print("Basic statistical analysis completed")
    return summary_results

def create_program_visualizations(df, program_id, program_name):
    """Create visualizations for a single program"""
    # Ensure figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Time series plot with intervention point
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['mean_sentiment'], marker='o', linestyle='-')
    
    # Find intervention point
    intervention_idx = df['post_intervention'].idxmax() if 1 in df['post_intervention'].values else None
    if intervention_idx is not None:
        plt.axvline(x=df.loc[intervention_idx, 'time'], color='r', linestyle='--', label='Intervention Point')
    
    plt.title(f'Program {program_id}: {program_name} - Sentiment Score Over Time')
    plt.xlabel('Time Point')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/program_{program_id}_timeseries.png')
    plt.close()
    
    # Box plot comparing pre and post intervention
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='post_intervention', y='mean_sentiment', data=df, 
                palette=['#1f77b4', '#ff7f0e'],
                order=[0, 1])
    plt.title(f'Program {program_id}: {program_name} - Pre vs Post Intervention Sentiment')
    plt.xlabel('Post Intervention (0=No, 1=Yes)')
    plt.ylabel('Mean Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/program_{program_id}_boxplot.png')
    plt.close()
    
    # Density plot comparing pre and post distributions
    plt.figure(figsize=(10, 6))
    
    # Get pre and post data
    pre_data = df[df['post_intervention'] == 0]['mean_sentiment']
    post_data = df[df['post_intervention'] == 1]['mean_sentiment']
    
    # Only create density plot if we have enough data points
    if len(pre_data) >= 2 and len(post_data) >= 2:
        sns.kdeplot(pre_data, label='Pre-intervention', shade=True, alpha=0.5)
        sns.kdeplot(post_data, label='Post-intervention', shade=True, alpha=0.5)
        plt.title(f'Program {program_id}: {program_name} - Sentiment Distribution')
        plt.xlabel('Mean Sentiment Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/program_{program_id}_density.png')
    plt.close()

def create_overall_visualizations(summary_df):
    """Create visualizations for overall results across all programs"""
    if len(summary_df) == 0:
        print("Warning: No data for overall visualizations")
        return
    
    # Bar plot of mean sentiment differences (sorted)
    plt.figure(figsize=(14, 8))
    # Sort by mean difference
    sorted_df = summary_df.sort_values('mean_diff')
    # Create bar plot
    bars = plt.bar(sorted_df['program_name'], sorted_df['mean_diff'], 
                   color=[('green' if x > 0 else 'red') for x in sorted_df['mean_diff']])
    
    # Add significance markers
    for i, significant in enumerate(sorted_df['significant_0.05']):
        if significant:
            plt.text(i, sorted_df['mean_diff'].iloc[i], '*', 
                     ha='center', va='bottom' if sorted_df['mean_diff'].iloc[i] > 0 else 'top', 
                     fontsize=20)
    
    plt.title('Mean Sentiment Difference by Program (Post - Pre)')
    plt.xlabel('Service Program')
    plt.ylabel('Mean Sentiment Difference')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/overall_mean_diff_barplot.png')
    plt.close()
    
    # Plot pre vs post means as a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(summary_df['pre_mean'], summary_df['post_mean'], 
                c=['green' if x else 'red' for x in summary_df['significant_0.05']], 
                alpha=0.7, s=100)
    
    # Add program labels
    for i, row in summary_df.iterrows():
        plt.annotate(f"Program {row['program_id']}", 
                    (row['pre_mean'], row['post_mean']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line (no change)
    min_val = min(summary_df['pre_mean'].min(), summary_df['post_mean'].min())
    max_val = max(summary_df['pre_mean'].max(), summary_df['post_mean'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title('Pre vs Post Intervention Mean Sentiment')
    plt.xlabel('Pre-Intervention Mean')
    plt.ylabel('Post-Intervention Mean')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/pre_vs_post_mean_scatter.png')
    plt.close()
    
    # Plot p-values
    plt.figure(figsize=(12, 8))
    # Sort by p-value
    sorted_p = summary_df.sort_values('p_value')
    plt.bar(sorted_p['program_name'], sorted_p['p_value'], 
            color=['green' if x < 0.05 else 'gray' for x in sorted_p['p_value']])
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance threshold (p=0.05)')
    plt.title('P-values by Program')
    plt.xlabel('Service Program')
    plt.ylabel('P-value (t-test)')
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/overall_pvalues.png')
    plt.close()

if __name__ == "__main__":
    summary_results = analyze_program_stats()
    print(f"Analysis complete. Results saved to 'results' and 'figures' directories.")

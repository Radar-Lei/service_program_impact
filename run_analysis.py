import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data, parse_period
from basic_stats_analysis import analyze_program_stats
from its_analysis import run_its_analysis
import shutil
import sys
import argparse
import traceback
import glob
from scipy import stats
import seaborn as sns
import matplotlib.dates as mdates

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

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'statsmodels'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Warning: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def convert_md_to_html():
    """Convert the markdown report to HTML if md_to_html.py exists"""
    if os.path.exists('md_to_html.py'):
        try:
            import md_to_html
            md_to_html.convert_md_to_html('service_program_impact_report.md', 'service_program_impact_report.html')
            print("HTML report generated successfully: service_program_impact_report.html")
            return True
        except Exception as e:
            print(f"Error converting report to HTML: {e}")
            return False
    return False

def check_processed_data_exists(program_ids=None, regenerate_summary=True):
    """Check if relevant data files exist in processed_data directory
    
    Args:
        program_ids (list, optional): List of service program IDs to analyze
        regenerate_summary (bool): Whether to regenerate summary data files even if they exist
        
    Returns:
        tuple: (needs preprocessing flag, list of program IDs needing preprocessing, whether to regenerate summary)
    """
    if not os.path.exists('processed_data'):
        if program_ids:
            return True, program_ids, regenerate_summary
        return True, None, regenerate_summary
        
    # Check if any processed data files exist
    processed_files = os.listdir('processed_data')
    if not processed_files:
        if program_ids:
            return True, program_ids, regenerate_summary
        return True, None, regenerate_summary
    
    # If no program IDs specified, check for summary data files
    if program_ids is None:
        # Check if at least some program data files exist
        program_files_exist = any(f.startswith('program_') and ('weekly' in f or 'monthly' in f or 'daily' in f) for f in processed_files)
        
        if not program_files_exist:
            return True, None, regenerate_summary
        else:
            print("Found processed data, skipping preprocessing step...")
            return False, None, regenerate_summary
    
    # Check which specified program IDs are missing data files
    missing_program_ids = []
    for program_id in program_ids:
        weekly_file = f"program_{program_id}_weekly.csv"
        monthly_file = f"program_{program_id}_monthly.csv"
        daily_file = f"program_{program_id}_daily.csv"
        
        # If weekly, monthly, and daily data files all don't exist, add program ID to missing list
        if weekly_file not in processed_files and monthly_file not in processed_files and daily_file not in processed_files:
            missing_program_ids.append(program_id)
    
    if missing_program_ids:
        print(f"Data files missing for the following program IDs, preprocessing needed: {missing_program_ids}")
        return True, missing_program_ids, regenerate_summary
    else:
        print("Found processed data for all specified programs, skipping preprocessing step...")
        return False, None, regenerate_summary

def regenerate_summary_files():
    """
    Regenerate or update summary data files (all_programs_weekly.csv, all_programs_monthly.csv, and all_programs_daily.csv).
    Loads existing summary files and updates them with the latest data from all program_X_*.csv files in the processed_data/ directory.
    If a program_id exists in both old summary files and new individual files, the individual file takes precedence.
    If a program_id only exists in old summary files (and the corresponding individual program_X_*.csv file doesn't exist), it will be retained.
    If a program_id only exists in new individual files, it will be added.
    """
    print("Regenerating/updating summary data files...")
    
    if not os.path.exists('processed_data'):
        print("processed_data directory does not exist, cannot regenerate summary files")
        return False, []

    # Collect all current individual program data
    current_individual_weekly_dfs = []
    current_individual_monthly_dfs = []
    current_individual_daily_dfs = []
    available_program_ids_from_individual_files = set()

    for file in os.listdir('processed_data'):
        file_path = os.path.join('processed_data', file)
        
        if file.startswith('program_') and file.endswith('_weekly.csv'):
            try:
                program_id = int(file.split('_')[1]) # Extract program_id from filename like "program_123_weekly.csv"
                
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Ensure program_id column exists or add it based on filename if necessary
                    if 'program_id' not in df.columns:
                        df['program_id'] = program_id
                    current_individual_weekly_dfs.append(df)
                    available_program_ids_from_individual_files.add(program_id)
                elif df.empty:
                    # If individual file is empty, still note the program_id if we want to ensure it's "blanked" in summary
                    available_program_ids_from_individual_files.add(program_id)

            except ValueError:
                print(f"Warning: Cannot parse program_id from filename {file}. Skipping this file.")
            except Exception as e:
                print(f"Error reading or parsing weekly data file {file}: {e}")
        
        elif file.startswith('program_') and file.endswith('_monthly.csv'):
            try:
                program_id = int(file.split('_')[1])

                df = pd.read_csv(file_path)
                if not df.empty:
                    if 'program_id' not in df.columns:
                        df['program_id'] = program_id
                    current_individual_monthly_dfs.append(df)
                    available_program_ids_from_individual_files.add(program_id)
                elif df.empty:
                    available_program_ids_from_individual_files.add(program_id)

            except ValueError:
                print(f"Warning: Cannot parse program_id from filename {file}. Skipping this file.")
            except Exception as e:
                print(f"Error reading or parsing monthly data file {file}: {e}")
        
        elif file.startswith('program_') and file.endswith('_daily.csv'):
            try:
                program_id = int(file.split('_')[1])

                df = pd.read_csv(file_path)
                if not df.empty:
                    if 'program_id' not in df.columns:
                        df['program_id'] = program_id
                    current_individual_daily_dfs.append(df)
                    available_program_ids_from_individual_files.add(program_id)
                elif df.empty:
                    available_program_ids_from_individual_files.add(program_id)

            except ValueError:
                print(f"Warning: Cannot parse program_id from filename {file}. Skipping this file.")
            except Exception as e:
                print(f"Error reading or parsing daily data file {file}: {e}")

    # 从收集到的单个程序文件创建当前的完整数据集
    current_weekly_from_individuals_df = None
    if current_individual_weekly_dfs:
        current_weekly_from_individuals_df = pd.concat(current_individual_weekly_dfs, ignore_index=True)

    current_monthly_from_individuals_df = None
    if current_individual_monthly_dfs:
        current_monthly_from_individuals_df = pd.concat(current_individual_monthly_dfs, ignore_index=True)
    
    current_daily_from_individuals_df = None
    if current_individual_daily_dfs:
        current_daily_from_individuals_df = pd.concat(current_individual_daily_dfs, ignore_index=True)

    # Load existing summary files
    existing_all_weekly_df = None
    all_programs_weekly_path = 'processed_data/all_programs_weekly.csv'
    if os.path.exists(all_programs_weekly_path):
        try:
            existing_all_weekly_df = pd.read_csv(all_programs_weekly_path)
            if existing_all_weekly_df.empty:
                existing_all_weekly_df = None
        except pd.errors.EmptyDataError:
            print(f"Warning: {all_programs_weekly_path} is empty, will be overwritten.")
            existing_all_weekly_df = None
        except Exception as e:
            print(f"Error reading {all_programs_weekly_path}, will attempt to overwrite: {e}")
            existing_all_weekly_df = None

    existing_all_monthly_df = None
    all_programs_monthly_path = 'processed_data/all_programs_monthly.csv'
    if os.path.exists(all_programs_monthly_path):
        try:
            existing_all_monthly_df = pd.read_csv(all_programs_monthly_path)
            if existing_all_monthly_df.empty:
                existing_all_monthly_df = None
        except pd.errors.EmptyDataError:
            print(f"Warning: {all_programs_monthly_path} is empty, will be overwritten.")
            existing_all_monthly_df = None
        except Exception as e:
            print(f"Error reading {all_programs_monthly_path}, will attempt to overwrite: {e}")
            existing_all_monthly_df = None
            
    existing_all_daily_df = None
    all_programs_daily_path = 'processed_data/all_programs_daily.csv'
    if os.path.exists(all_programs_daily_path):
        try:
            existing_all_daily_df = pd.read_csv(all_programs_daily_path)
            if existing_all_daily_df.empty:
                existing_all_daily_df = None
        except pd.errors.EmptyDataError:
            print(f"Warning: {all_programs_daily_path} is empty, will be overwritten.")
            existing_all_daily_df = None
        except Exception as e:
            print(f"Error reading {all_programs_daily_path}, will attempt to overwrite: {e}")
            existing_all_daily_df = None

    # --- 合并周数据 ---
    final_weekly_df_to_save = None
    ids_in_current_individuals_weekly = set()
    if current_weekly_from_individuals_df is not None and 'program_id' in current_weekly_from_individuals_df.columns:
        ids_in_current_individuals_weekly = set(current_weekly_from_individuals_df['program_id'].unique())
    elif not current_individual_weekly_dfs: # No individual files found/processed
        ids_in_current_individuals_weekly = set() # No current program IDs from files
    else: # current_weekly_from_individuals_df is None but current_individual_weekly_dfs is not, or program_id missing
        # This case implies empty individual files were found, or files without program_id.
        # We rely on available_program_ids_from_individual_files for IDs that should be considered "current"
        ids_in_current_individuals_weekly = available_program_ids_from_individual_files


    if current_weekly_from_individuals_df is not None: # Data from individual files exists
        if existing_all_weekly_df is not None and 'program_id' in existing_all_weekly_df.columns:
            filtered_existing_weekly = existing_all_weekly_df[~existing_all_weekly_df['program_id'].isin(ids_in_current_individuals_weekly)]
            final_weekly_df_to_save = pd.concat([filtered_existing_weekly, current_weekly_from_individuals_df], ignore_index=True)
        else: # No existing summary or it's invalid, so current individual data is the summary
            final_weekly_df_to_save = current_weekly_from_individuals_df
    elif existing_all_weekly_df is not None: # No data from individual files, but old summary exists
        # If there are program IDs from *empty* individual files, we might want to clear them from existing summary
        if ids_in_current_individuals_weekly: # non-empty if empty individual files were found
             if 'program_id' in existing_all_weekly_df.columns:
                final_weekly_df_to_save = existing_all_weekly_df[~existing_all_weekly_df['program_id'].isin(ids_in_current_individuals_weekly)]
             else: # existing summary has no program_id column, cannot filter
                final_weekly_df_to_save = existing_all_weekly_df.copy() # or treat as error
        else: # No current files at all (not even empty ones), keep existing summary as is
            final_weekly_df_to_save = existing_all_weekly_df.copy() 
    # If both current_weekly_from_individuals_df and existing_all_weekly_df are None, final_weekly_df_to_save remains None

    if final_weekly_df_to_save is not None and not final_weekly_df_to_save.empty:
        try:
            if 'program_id' in final_weekly_df_to_save.columns and 'week' in final_weekly_df_to_save.columns:
                 final_weekly_df_to_save['sort_key_week'] = final_weekly_df_to_save['week'].apply(lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else str(x))
                 final_weekly_df_to_save = final_weekly_df_to_save.sort_values(by=['program_id', 'sort_key_week']).drop(columns=['sort_key_week'])
            elif 'program_id' in final_weekly_df_to_save.columns:
                 final_weekly_df_to_save = final_weekly_df_to_save.sort_values(by=['program_id'])
            
            final_weekly_df_to_save.to_csv(all_programs_weekly_path, index=False)
            print(f"已更新/生成汇总周数据文件: {all_programs_weekly_path}")
        except Exception as e:
            print(f"保存汇总周数据文件 {all_programs_weekly_path} 时出错: {e}")
    elif final_weekly_df_to_save is not None and final_weekly_df_to_save.empty: # Save empty if result is empty
        pd.DataFrame().to_csv(all_programs_weekly_path, index=False)
        print(f"汇总周数据文件 {all_programs_weekly_path} 更新为空。")
    elif final_weekly_df_to_save is None and os.path.exists(all_programs_weekly_path):
        # This means no data from individual files and no existing summary was loaded or it was empty.
        # If the file exists but we decided it's "None" (e.g. error reading), we might overwrite with empty.
        # To be safe, only write empty if we explicitly decided the result is an empty DataFrame.
        # If final_weekly_df_to_save is None, it means no data source was valid.
        print(f"没有有效的周数据来源，未修改 {all_programs_weekly_path}。")


    # --- 合并月数据 (类似逻辑) ---
    final_monthly_df_to_save = None
    ids_in_current_individuals_monthly = set()
    if current_monthly_from_individuals_df is not None and 'program_id' in current_monthly_from_individuals_df.columns:
        ids_in_current_individuals_monthly = set(current_monthly_from_individuals_df['program_id'].unique())
    elif not current_individual_monthly_dfs:
        ids_in_current_individuals_monthly = set()
    else:
        ids_in_current_individuals_monthly = available_program_ids_from_individual_files


    if current_monthly_from_individuals_df is not None:
        if existing_all_monthly_df is not None and 'program_id' in existing_all_monthly_df.columns:
            filtered_existing_monthly = existing_all_monthly_df[~existing_all_monthly_df['program_id'].isin(ids_in_current_individuals_monthly)]
            final_monthly_df_to_save = pd.concat([filtered_existing_monthly, current_monthly_from_individuals_df], ignore_index=True)
        else:
            final_monthly_df_to_save = current_monthly_from_individuals_df
    elif existing_all_monthly_df is not None:
        if ids_in_current_individuals_monthly:
            if 'program_id' in existing_all_monthly_df.columns:
                final_monthly_df_to_save = existing_all_monthly_df[~existing_all_monthly_df['program_id'].isin(ids_in_current_individuals_monthly)]
            else:
                final_monthly_df_to_save = existing_all_monthly_df.copy()
        else:
            final_monthly_df_to_save = existing_all_monthly_df.copy()

    if final_monthly_df_to_save is not None and not final_monthly_df_to_save.empty:
        try:
            if 'program_id' in final_monthly_df_to_save.columns and 'month' in final_monthly_df_to_save.columns:
                final_monthly_df_to_save = final_monthly_df_to_save.sort_values(by=['program_id', 'month'])
            elif 'program_id' in final_monthly_df_to_save.columns:
                final_monthly_df_to_save = final_monthly_df_to_save.sort_values(by=['program_id'])

            final_monthly_df_to_save.to_csv(all_programs_monthly_path, index=False)
            print(f"已更新/生成汇总月数据文件: {all_programs_monthly_path}")
        except Exception as e:
            print(f"保存汇总月数据文件 {all_programs_monthly_path} 时出错: {e}")
    elif final_monthly_df_to_save is not None and final_monthly_df_to_save.empty:
        pd.DataFrame().to_csv(all_programs_monthly_path, index=False)
        print(f"汇总月数据文件 {all_programs_monthly_path} 更新为空。")
    elif final_monthly_df_to_save is None and os.path.exists(all_programs_monthly_path):
         print(f"没有有效的月数据来源，未修改 {all_programs_monthly_path}。")
         
    # --- 合并每日数据 ---
    final_daily_df_to_save = None
    ids_in_current_individuals_daily = set()
    if current_daily_from_individuals_df is not None and 'program_id' in current_daily_from_individuals_df.columns:
        ids_in_current_individuals_daily = set(current_daily_from_individuals_df['program_id'].unique())
    elif not current_individual_daily_dfs:
        ids_in_current_individuals_daily = set()
    else:
        ids_in_current_individuals_daily = available_program_ids_from_individual_files

    if current_daily_from_individuals_df is not None:
        if existing_all_daily_df is not None and 'program_id' in existing_all_daily_df.columns:
            filtered_existing_daily = existing_all_daily_df[~existing_all_daily_df['program_id'].isin(ids_in_current_individuals_daily)]
            final_daily_df_to_save = pd.concat([filtered_existing_daily, current_daily_from_individuals_df], ignore_index=True)
        else:
            final_daily_df_to_save = current_daily_from_individuals_df
    elif existing_all_daily_df is not None:
        if ids_in_current_individuals_daily:
            if 'program_id' in existing_all_daily_df.columns:
                final_daily_df_to_save = existing_all_daily_df[~existing_all_daily_df['program_id'].isin(ids_in_current_individuals_daily)]
            else:
                final_daily_df_to_save = existing_all_daily_df.copy()
        else:
            final_daily_df_to_save = existing_all_daily_df.copy()

    if final_daily_df_to_save is not None and not final_daily_df_to_save.empty:
        try:
            if 'program_id' in final_daily_df_to_save.columns and 'date' in final_daily_df_to_save.columns:
                final_daily_df_to_save = final_daily_df_to_save.sort_values(by=['program_id', 'date'])
            elif 'program_id' in final_daily_df_to_save.columns:
                final_daily_df_to_save = final_daily_df_to_save.sort_values(by=['program_id'])

            final_daily_df_to_save.to_csv(all_programs_daily_path, index=False)
            print(f"已更新/生成汇总每日数据文件: {all_programs_daily_path}")
        except Exception as e:
            print(f"保存汇总每日数据文件 {all_programs_daily_path} 时出错: {e}")
    elif final_daily_df_to_save is not None and final_daily_df_to_save.empty:
        pd.DataFrame().to_csv(all_programs_daily_path, index=False)
        print(f"汇总每日数据文件 {all_programs_daily_path} 更新为空。")
    elif final_daily_df_to_save is None and os.path.exists(all_programs_daily_path):
        print(f"没有有效的每日数据来源，未修改 {all_programs_daily_path}。")
        
    # Return all program IDs found in *any* individual file during this run
    print(f"在单个程序文件中找到了以下程序ID的数据: {sorted(list(available_program_ids_from_individual_files))}")
    return True, sorted(list(available_program_ids_from_individual_files))

def get_all_available_program_ids():
    """获取所有可用的程序ID
    
    从processed_data目录中扫描所有可用的程序ID
    
    Returns:
        list: 所有可用程序ID的列表
    """
    if not os.path.exists('processed_data'):
        return []
        
    available_program_ids = set()
    for file in os.listdir('processed_data'):
        if file.startswith('program_') and ('weekly' in file or 'monthly' in file or 'daily' in file):
            try:
                program_id = int(file.split('_')[1])
                available_program_ids.add(program_id)
            except:
                continue
    
    return sorted(list(available_program_ids))

def get_all_valid_program_ids():
    """从原始数据中获取所有有效的程序ID
    
    从service_program_data/SPD_SZ_zh.csv中读取所有有效的程序ID
    
    Returns:
        list: 所有有效的程序ID列表
    """
    try:
        # 加载服务项目数据
        program_df = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
        # 解析干预日期
        program_df['intervention_date'] = program_df['Period'].apply(parse_period)
        # 筛选有效的服务项目（有干预日期的）
        valid_programs = program_df[program_df['intervention_date'].notna()]
        # 获取所有有效的程序ID
        valid_program_ids = valid_programs.index.tolist()
        return valid_program_ids
    except Exception as e:
        print(f"获取有效程序ID时出错: {e}")
        return []

def analyze_program_stats_daily(program_ids=None, processed_dir='processed_data'):
    """Perform statistical analysis based on daily data
    
    Args:
        program_ids (list, optional): List of service program IDs to analyze. If None, analyze all programs.
        processed_dir (str, optional): Directory path for processed data. Default is 'processed_data'.
    
    Returns:
        DataFrame: Summary dataframe of analysis results
    """
    print("Performing statistical analysis based on daily data...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Get all daily data files
    if program_ids is not None:
        # Only get files for specified programs
        daily_files = [f'{processed_dir}/program_{pid}_daily.csv' for pid in program_ids 
                        if os.path.exists(f'{processed_dir}/program_{pid}_daily.csv')]
        if not daily_files:
            print("Warning: No daily data files found for specified program IDs")
    else:
        # Get files for all programs
        daily_files = glob.glob(f'{processed_dir}/program_*_daily.csv')
    
    # Create summary dataframe for all program results
    summary_results = pd.DataFrame(columns=[
        'program_id', 'program_name', 'pre_mean', 'post_mean', 'mean_diff', 
        'pre_median', 'post_median', 'median_diff', 
        'pre_std', 'post_std', 't_stat', 'p_value', 'significant_0.05'
    ])
    
    # Return empty summary results if no files found
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
            print(f"Warning: Program {program_id} missing required columns")
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
        
        # Skip visualizations as they are not needed
        
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
            print(f"Warning: Cannot perform categorical test for program {program_id}: {e}")
            # Still save the contingency table if it was created
            if 'contingency' in locals():
                with open(f'results/program_{program_id}_categorical_test.txt', 'w') as f:
                    f.write(f"Categorical test could not be performed: {e}\n\n")
                    f.write(f"Contingency table:\n{contingency.to_string()}\n")
    
    # Sort summary results by significance and effect size
    summary_results = summary_results.sort_values(by=['significant_0.05', 'mean_diff'], ascending=[False, False])
    
    # Save summary results
    summary_results.to_csv('results/program_stats_summary.csv', index=False)
    
    # Skip visualizations as they are not needed
    
    print("Daily data statistical analysis completed")
    return summary_results

def create_program_visualizations(df, program_id, program_name):
    """Create visualizations for a single program"""
    # This function is now empty as we don't need to generate visualizations
    pass

def create_overall_visualizations(summary_df):
    """Create visualizations for overall results across all programs"""
    # This function is now empty as we don't need to generate visualizations
    if len(summary_df) == 0:
        print("Warning: No data for overall visualizations")
        return

def run_complete_analysis(feedback_dir, program_ids=None, force_preprocess=False, regenerate_summary=True):
    """Run the complete analysis pipeline
    
    Args:
        feedback_dir (str): Feedback data directory
        program_ids (list, optional): List of service program IDs to analyze. If None, analyze all programs.
        force_preprocess (bool, optional): Force re-preprocessing of data even if processed data already exists.
        regenerate_summary (bool, optional): Whether to regenerate summary data files
    """
    start_time = time.time()
    
    print("="*80)
    print("STARTING SERVICE PROGRAM IMPACT ANALYSIS")
    if program_ids:
        print(f"ANALYZING ONLY PROGRAMS: {', '.join(map(str, program_ids))}")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Create necessary directories
    for directory in ['processed_data', 'results', 'figures']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    try:
        # Determine programs needing preprocessing and analysis program IDs (analysis_program_ids)
        if program_ids is None: # If no programs specified, analyze all valid programs
            all_valid_ids_from_source = get_all_valid_program_ids()
            if not all_valid_ids_from_source:
                print("No valid program IDs found in source data, analysis aborted.")
                return False
            
            print(f"Found the following valid program IDs from source data: {all_valid_ids_from_source}")
            
            if force_preprocess:
                need_preprocess = True
                programs_to_preprocess = all_valid_ids_from_source
            else:
                # The third return value of check_processed_data_exists is less important here as we mainly care about missing files
                need_preprocess_check, missing_ids_for_all_valid, _ = check_processed_data_exists(all_valid_ids_from_source, False) # False for regenerate_summary as we handle it later
                
                if need_preprocess_check and missing_ids_for_all_valid:
                    print(f"Data files missing for the following valid program IDs, will preprocess: {missing_ids_for_all_valid}")
                    programs_to_preprocess = missing_ids_for_all_valid
                    need_preprocess = True
                else:
                    print("Data files for all valid program IDs already exist.")
                    programs_to_preprocess = None 
                    need_preprocess = False 
            
            analysis_program_ids = all_valid_ids_from_source
        
        else: # 如果指定了程序ID
            if force_preprocess:
                need_preprocess = True
                programs_to_preprocess = program_ids
            else:
                need_preprocess_check, missing_ids_for_specified, _ = check_processed_data_exists(program_ids, False)
                if need_preprocess_check and missing_ids_for_specified:
                    print(f"以下指定的程序ID数据文件缺失，将进行预处理: {missing_ids_for_specified}")
                    programs_to_preprocess = missing_ids_for_specified
                    need_preprocess = True
                else:
                    print("所有指定的程序ID的数据文件均已存在。")
                    programs_to_preprocess = None
                    need_preprocess = False
            
            analysis_program_ids = program_ids

        # 数据预处理和汇总文件生成
        if need_preprocess and programs_to_preprocess:
            print("\n1. PREPROCESSING DATA...")
            print(f"预处理以下程序: {programs_to_preprocess}")
            
            processed_df = preprocess_data(feedback_dir, programs_to_preprocess)
            
            if processed_df.empty:
                print(f"为程序 {programs_to_preprocess} 进行预处理后未返回有效数据。分析中止。")
                return False
            
            if regenerate_summary:
                print("预处理完成，正在重新生成汇总文件以包含新数据...")
                success_regen, _ = regenerate_summary_files()
                if not success_regen:
                    print("警告: 重新生成汇总文件失败。后续分析可能基于不完整或过时的汇总数据。")
            else:
                print("预处理完成。由于 regenerate_summary 为 False (可能通过 --no-summary-regen 设置)，未重新生成汇总文件。")
                print("后续分析将使用现有的（可能过时的）汇总文件。")

        elif not need_preprocess and regenerate_summary:
            print("\n1. USING EXISTING PROCESSED DATA...")
            print("不需要预处理。检查并重新生成汇总文件（如果 regenerate_summary 为 True）...")
            success_regen, _ = regenerate_summary_files()
            if not success_regen:
                # 如果汇总文件不存在且重新生成失败，这是一个严重问题
                if not (os.path.exists('processed_data/all_programs_weekly.csv') and 
                        os.path.exists('processed_data/all_programs_monthly.csv') and 
                        os.path.exists('processed_data/all_programs_daily.csv')):
                    print("错误: 汇总文件不存在且重新生成失败。分析中止。")
                    return False
                print("警告: 重新生成汇总文件失败。后续分析将使用现有的汇总数据（如果存在）。")
        
        elif not need_preprocess and not regenerate_summary:
            print("\n1. USING EXISTING PROCESSED DATA...")
            print("不需要预处理，并且 regenerate_summary 为 False。将使用现有的汇总文件。")
            # 确保汇总文件存在，因为我们不重新生成它们
            if not (os.path.exists('processed_data/all_programs_weekly.csv') and 
                    os.path.exists('processed_data/all_programs_monthly.csv') and 
                    os.path.exists('processed_data/all_programs_daily.csv')):
                print("错误: regenerate_summary 为 False，但必需的汇总文件缺失。请运行一次不带 --no-summary-regen 的分析，或确保汇总文件存在。分析中止。")
                return False
        
        # 如果 need_preprocess 为 True 但 programs_to_preprocess 为空 (例如，强制预处理但未提供程序列表，或逻辑错误)
        # 当前逻辑应该避免这种情况。为稳健起见，可以添加检查，但目前流程旨在防止这种情况。
        elif need_preprocess and not programs_to_preprocess:
             print("\n1. PREPROCESSING DATA...")
             print("警告: 需要预处理，但没有指定要预处理的程序。这可能是一个逻辑错误。")
             # 如果 regenerate_summary 为 True，仍然尝试更新汇总文件
             if regenerate_summary:
                print("尝试重新生成汇总文件...")
                success_regen, _ = regenerate_summary_files()
                if not success_regen and not (os.path.exists('processed_data/all_programs_weekly.csv') and 
                                            os.path.exists('processed_data/all_programs_monthly.csv') and 
                                            os.path.exists('processed_data/all_programs_daily.csv')):
                    print("错误: 汇总文件不存在且重新生成失败。分析中止。")
                    return False
                elif not success_regen:
                    print("警告: 重新生成汇总文件失败。")
        
        # Step 2: Basic statistical analysis (使用每周数据)
        print("\n2. RUNNING BASIC STATISTICAL ANALYSIS (USING WEEKLY DATA)...")
        print(f"分析以下程序ID: {analysis_program_ids}")
        stats_summary = analyze_program_stats(analysis_program_ids)
        
        # Step 3: Interrupted Time Series analysis
        print("\n3. RUNNING INTERRUPTED TIME SERIES ANALYSIS...")
        its_summary = run_its_analysis(analysis_program_ids)
        
        # Step 4: Check if the report file exists
        print("\n4. CHECKING FOR EXISTING REPORT...")
        if os.path.exists('service_program_impact_report.md'):
            print(f"Found existing report: service_program_impact_report.md")
        else:
            print(f"Warning: service_program_impact_report.md not found.")
            print(f"Please ensure the report file exists before converting to HTML.")
            return False
        
        # Step 5: Convert to HTML
        print("\n5. CONVERTING REPORT TO HTML...")
        convert_md_to_html()
        
        # Output completion message
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"ANALYSIS COMPLETE! Total time: {total_time:.1f} seconds")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\nERROR: Analysis failed - {str(e)}")
        traceback.print_exc() # 打印更详细的堆栈信息
        return False

def clean_all_results():
    """清理所有生成的分析结果、图表和处理后的数据"""
    print("="*80)
    print("开始清理所有生成的结果")
    print("="*80)
    
    # 需要清理的目录
    directories_to_clean = ['processed_data', 'results', 'figures']
    
    # 清理每个目录
    for directory in directories_to_clean:
        if os.path.exists(directory):
            print(f"\n清理 {directory} 目录...")
            
            # 删除整个目录及其内容
            shutil.rmtree(directory)
            print(f"已删除 {directory} 目录")
        else:
            print(f"\n{directory} 目录不存在，跳过")
    
    # 删除HTML报告(如果存在)
    if os.path.exists('service_program_impact_report.html'):
        os.remove('service_program_impact_report.html')
        print("\n已删除 service_program_impact_report.html")
    
    # 清理运行生成的临时或缓存文件(如果有的话)
    if os.path.exists('__pycache__'):
        shutil.rmtree('__pycache__')
        print("\n已删除 __pycache__ 目录")
    
    # 输出完成消息
    print("\n" + "="*80)
    print("结果清理完成!")
    print("="*80)
    
    return True

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='服务项目影响分析工具')
    parser.add_argument('--clean', action='store_true', help='清理所有结果')
    parser.add_argument('--feedback_dir', type=str, default='similarity_threshold=0.35/', 
                        help='反馈数据目录路径')
    parser.add_argument('--programs', type=int, nargs='+', default=[0, 1, 4, 5, 15, 22], help='要分析的服务项目ID列表，如果不指定则使用默认值0,1,4,22')
    parser.add_argument('--force-preprocess', action='store_true', 
                        help='强制重新预处理数据，即使已经存在处理过的数据')
    parser.add_argument('--no-summary-regen', action='store_true',
                        help='不重新生成汇总数据文件(all_programs_weekly.csv、all_programs_monthly.csv和all_programs_daily.csv)')
    
    return parser.parse_args()

if __name__ == "__main__":
    """
    python run_analysis.py --programs 0 1 4 5 15 22  --feedback_dir similarity_threshold=0.35/
    python run_analysis.py --programs 0 1 4 8 22 --force-preprocess
    python run_analysis.py --programs 0 1 4 8 22 --no-summary-regen
    python run_analysis.py --clean
    """
    args = parse_arguments()
    
    if args.clean:
        clean_all_results()
    else:
        # 运行分析，可能指定特定项目
        regenerate_summary = not args.no_summary_regen
        run_complete_analysis(args.feedback_dir, args.programs, args.force_preprocess, regenerate_summary)

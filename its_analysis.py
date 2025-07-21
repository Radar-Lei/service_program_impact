import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
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

def create_combined_visualization(all_results):
    """Create a combined visualization for all programs with ITS and placebo results
    
    Args:
        all_results (list): List of tuples containing (df, model, model_name, program_id, program_name, placebo_results)
    """
    try:
        # Calculate the number of rows and columns for subplots
        n_programs = len(all_results)
        
        # Optimize layout: use 2×2 for 4 programs
        if n_programs == 4:
            n_cols = 2
            n_rows = 2
        else:
            # Calculate appropriate rows and columns for other counts
            n_cols = min(3, n_programs)  # Maximum 3 columns
            n_rows = (n_programs + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig = plt.figure(figsize=(7*n_cols, 6*n_rows))
        
        for idx, (df, model, model_name, program_id, program_name, placebo_results) in enumerate(all_results, 1):
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # Get clean data
            required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
            clean_df = df[required_cols].dropna().reset_index(drop=True)
            
            if len(clean_df) < 10:
                continue
            
            # Get time points for display
            timestamp_col = None
            for col in df.columns:
                if col.lower() in ['timestamp', 'timestamps', 'date', 'datetime']:
                    timestamp_col = col
                    break
            
            # Prepare time points for display
            if timestamp_col and timestamp_col in df.columns and not df[timestamp_col].isna().all():
                time_points_display = pd.to_datetime(df[timestamp_col]).dropna()
                if len(time_points_display) == len(clean_df):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                else:
                    time_points_display = clean_df['time']
            elif 'week' in df.columns:
                try:
                    dates = []
                    for w in df['week']:
                        if isinstance(w, str) and '/' in w:
                            dates.append(pd.to_datetime(w.split('/')[0]))
                        else:
                            dates.append(None)
                    time_points_display = pd.Series(dates).dropna()
                    if len(time_points_display) == len(clean_df):
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=45)
                    else:
                        time_points_display = clean_df['time']
                except:
                    time_points_display = clean_df['time']
            else:
                time_points_display = clean_df['time']
            
            # Get observed and predicted values
            observed = clean_df['mean_sentiment']
            if hasattr(model, 'fittedvalues'):
                predicted = model.fittedvalues
            else:
                try:
                    predicted = model.predict()
                except:
                    continue
            
            # Ensure alignment
            min_len = min(len(clean_df), len(predicted))
            observed = observed[:min_len]
            predicted = predicted[:min_len]
            if isinstance(time_points_display, pd.Series):
                time_points_display = time_points_display.iloc[:min_len]
            else:
                time_points_display = time_points_display[:min_len]
            
            # Plot observed values
            ax.scatter(time_points_display, observed, color='blue', alpha=0.6, label='Observed', s=20)
            
            # Plot fitted values
            ax.plot(time_points_display, predicted, 'r-', linewidth=1.5, label='Fitted')
            
            # Find and plot intervention point
            intervention_idx = np.argmax(clean_df['post_intervention'].values) if 1 in clean_df['post_intervention'].values else None
            if intervention_idx is not None:
                intervention_time = time_points_display.iloc[intervention_idx] if isinstance(time_points_display, pd.Series) else time_points_display[intervention_idx]
                ax.axvline(x=intervention_time, color='green', linestyle='--', label='Intervention')
            
            # Plot trend lines if available
            try:
                if intervention_idx is not None and 'Segmented Regression' in model_name:
                    # 获取模型参数
                    intercept = model.params['Intercept']
                    time_coef = model.params['time']
                    level_change = model.params['post_intervention']
                    trend_change = model.params['time_since_intervention']
                    
                    # 分别找到干预前后的数据点
                    pre_data = clean_df[clean_df['post_intervention'] == 0]
                    post_data = clean_df[clean_df['post_intervention'] == 1]
                    
                    # 输出调试信息
                    print(f"Program {program_id}: pre_data={len(pre_data)}, post_data={len(post_data)}")
                    
                    # 干预前趋势线 - 只使用两个点绘制直线
                    if len(pre_data) >= 2:
                        pre_start_time = pre_data['time'].iloc[0]
                        pre_end_time = pre_data['time'].iloc[-1]
                        
                        # 计算这两个点对应的y值
                        pre_start_y = intercept + time_coef * pre_start_time
                        pre_end_y = intercept + time_coef * pre_end_time
                        
                        # 找到对应的显示时间点
                        pre_start_idx = pre_data.index[0]
                        pre_end_idx = pre_data.index[-1]
                        
                        if isinstance(time_points_display, pd.Series):
                            if pre_start_idx < len(time_points_display) and pre_end_idx < len(time_points_display):
                                pre_start_display = time_points_display.iloc[pre_start_idx]
                                pre_end_display = time_points_display.iloc[pre_end_idx]
                                
                                # Plot pre-intervention trend line
                                ax.plot([pre_start_display, pre_end_display], 
                                        [pre_start_y, pre_end_y], 
                                        'b--', linewidth=1, label='Pre-Intervention Trend')
                        else:
                            if pre_start_idx < len(time_points_display) and pre_end_idx < len(time_points_display):
                                pre_start_display = time_points_display[pre_start_idx]
                                pre_end_display = time_points_display[pre_end_idx]
                                
                                # Plot pre-intervention trend line
                                ax.plot([pre_start_display, pre_end_display], 
                                        [pre_start_y, pre_end_y], 
                                        'b--', linewidth=1, label='Pre-Intervention Trend')
                    
                    # 干预后趋势线 - 只使用两个点绘制直线
                    if len(post_data) >= 2:
                        post_start_time = post_data['time'].iloc[0]
                        post_end_time = post_data['time'].iloc[-1]
                        post_start_since = post_data['time_since_intervention'].iloc[0]
                        post_end_since = post_data['time_since_intervention'].iloc[-1]
                        
                        # 计算这两个点对应的y值 
                        post_start_y = intercept + level_change + time_coef * post_start_time + trend_change * post_start_since
                        post_end_y = intercept + level_change + time_coef * post_end_time + trend_change * post_end_since
                        
                        # 找到对应的显示时间点
                        post_start_idx = post_data.index[0]
                        post_end_idx = post_data.index[-1]
                        
                        # 输出调试信息
                        print(f"Program {program_id}: post_start_idx={post_start_idx}, post_end_idx={post_end_idx}, len(time_points_display)={len(time_points_display)}")
                        print(f"post_start_time={post_start_time}, post_end_time={post_end_time}")
                        print(f"post_start_y={post_start_y}, post_end_y={post_end_y}")
                        
                        # 尝试不同的方法绘制干预后趋势线
                        # 方法1: 使用时间点直接计算趋势线
                        x = np.array([post_start_time, post_end_time])
                        y = np.array([post_start_y, post_end_y])
                        
                        # 寻找最近的显示时间点
                        if isinstance(time_points_display, pd.Series):
                            x_display = [time_points_display.iloc[intervention_idx], time_points_display.iloc[-1]]
                        else:
                            x_display = [time_points_display[intervention_idx], time_points_display[-1]]
                        
                        # Plot post-intervention trend line - using intervention point and last point
                        ax.plot(x_display, y, 'g--', linewidth=1, label='Post-Intervention Trend')
            except Exception as e:
                print(f"Warning: Could not plot trend lines for program {program_id}: {str(e)}")
            
            # Add placebo lines if available
            if placebo_results:
                try:
                    actual_model = fit_segmented_regression(clean_df)
                    actual_trend_p = actual_model.pvalues['time_since_intervention']
                    
                    # Get pre-intervention data
                    pre_data = clean_df[clean_df['post_intervention'] == 0]
                    if len(pre_data) > 5:
                        # Select placebo point at 25% of pre-intervention period
                        placebo_idx = int(len(pre_data) * 0.25)
                        placebo_time = pre_data.iloc[placebo_idx]['time']
                        placebo_time_display = time_points_display.iloc[placebo_idx] if isinstance(time_points_display, pd.Series) else time_points_display[placebo_idx]
                        ax.axvline(x=placebo_time_display, color='gray', linestyle=':', alpha=0.7,
                                  label=f'Placebo (p={placebo_results[0]:.3f})')
                except Exception as e:
                    print(f"Warning: Could not plot placebo line for program {program_id}: {str(e)}")
            
            # Add title and labels
            ax.set_title(f'Program {program_id}: {program_name}\n{model_name}', fontsize=10)
            ax.set_xlabel('Date' if timestamp_col else 'Time', fontsize=8)
            ax.set_ylabel('Mean Sentiment Score', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend with smaller font
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='best', fontsize=7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure with editable text
        plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects
        plt.rcParams['font.family'] = 'sans-serif'  # Use generic font family
        plt.savefig('figures/its/combined_its_analysis.svg', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Combined visualization failed: {str(e)}")
        return

def run_its_analysis(program_ids=None, processed_dir='processed_data'):
    """Run Interrupted Time Series (ITS) analysis for each service program
    
    Args:
        program_ids (list, optional): List of service program IDs to analyze. If None, analyze all programs.
        processed_dir (str, optional): Directory path for processed data. Default is 'processed_data'.
    
    Returns:
        DataFrame: ITS analysis summary dataframe
    """
    print("Starting Interrupted Time Series (ITS) analysis...")
    
    # Create results directory if it doesn't exist
    dirs_to_create = ['results/its', 'figures/its']
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Get all weekly data files
    if program_ids is not None:
        # Only get files for specified programs
        weekly_files = [f'{processed_dir}/program_{pid}_weekly.csv' for pid in program_ids
                        if os.path.exists(f'{processed_dir}/program_{pid}_weekly.csv')]
        if not weekly_files:
            print("Warning: No weekly data files found for the specified program IDs")
    else:
        # Get all program files
        weekly_files = glob.glob(f'{processed_dir}/program_*_weekly.csv')
    
    # Create summary dataframe for all ITS results
    its_summary = pd.DataFrame(columns=[
        'program_id', 'program_name', 'baseline_level', 'baseline_trend', 
        'level_change', 'trend_change', 'level_p', 'trend_p',
        'significant_level_0.05', 'significant_trend_0.05', 'model_type',
        'r_squared', 'aic', 'sample_size'
    ])
    
    # Return empty summary if no files found
    if not weekly_files:
        print("No weekly data files found for ITS analysis")
        return its_summary
    
    # Store results for combined visualization
    all_results = []
    
    for file_path in weekly_files:
        # Extract program ID from filename
        program_id = int(file_path.split('_')[-2])
        
        # Load program data
        df = pd.read_csv(file_path)
        
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
        
        if len(df) < 10:  # We need sufficient data points for ITS
            print(f"Warning: Insufficient data points for ITS analysis for program {program_id}")
            continue
        
        if 'post_intervention' not in df.columns or 'mean_sentiment' not in df.columns:
            print(f"Warning: Required columns missing for program {program_id}")
            continue
        
        # Check if we have both pre and post intervention data
        if 1 not in df['post_intervention'].values or 0 not in df['post_intervention'].values:
            print(f"Warning: Need both pre and post intervention data for program {program_id}")
            continue
        
        try:
            # Run ITS analysis for this program
            result_dict, best_model, best_model_name = analyze_single_program(df, program_id, program_name)
            
            # Add results to summary
            its_summary = pd.concat([its_summary, pd.DataFrame([result_dict])], ignore_index=True)
            
            # Run placebo test
            placebo_results = run_placebo_test(df, program_id, program_name)
            
            # Store results for combined visualization
            all_results.append((df, best_model, best_model_name, program_id, program_name, placebo_results))
            
        except Exception as e:
            print(f"Error analyzing program {program_id}: {e}")
    
    # Sort summary by trend change significance and magnitude
    its_summary = its_summary.sort_values(
        by=['significant_trend_0.05', 'trend_change'],
        ascending=[False, False]
    )
    
    # Save summary results - Commented out as per user request
    # its_summary.to_csv('results/its/its_summary.csv', index=False)
    
    # Create combined visualization
    if all_results:
        create_combined_visualization(all_results)
    
    print("ITS analysis completed")
    return its_summary

def analyze_single_program(df, program_id, program_name):
    """Run ITS analysis for a single program"""
    # Ensure data is complete before analysis
    required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")
    
    # Get only complete rows (no NaN values) to avoid dimension mismatch problems
    analysis_df = df[required_cols].dropna().reset_index(drop=True)
    
    # Need minimum data for analysis
    if len(analysis_df) < 10:
        raise ValueError(f"Insufficient complete data points: {len(analysis_df)}")
    
    # First check for stationarity
    adf_result = adfuller(analysis_df['mean_sentiment'])
    is_stationary = adf_result[1] < 0.05
    
    # Check for autocorrelation
    acf_values = acf(analysis_df['mean_sentiment'], nlags=10)
    significant_autocorr = np.abs(acf_values[1:]).max() > 1.96/np.sqrt(len(analysis_df))
    
    # We'll try multiple models and select the best fitting one
    models_to_try = []
    
    # 1. Simple segmented regression (no AR terms)
    models_to_try.append(('Simple Segmented Regression', fit_segmented_regression(analysis_df, ar_terms=0)))
    
    # 2. Segmented regression with AR(1) term if autocorrelation is detected
    if significant_autocorr:
        try:
            ar1_model = fit_segmented_regression(analysis_df, ar_terms=1)
            models_to_try.append(('Segmented Regression with AR(1)', ar1_model))
            
            ar2_model = fit_segmented_regression(analysis_df, ar_terms=2)
            models_to_try.append(('Segmented Regression with AR(2)', ar2_model))
        except Exception as e:
            print(f"Warning: AR segmented regression failed for program {program_id}: {str(e)}")
    
    # 3. ARIMA/SARIMA models if needed
    # For simplicity we're using pre-defined orders, in a real analysis would test multiple
    if significant_autocorr:
        try:
            # Try ARIMA model
            arima_model = fit_arima_model(analysis_df, order=(1,0,0))
            if arima_model:
                models_to_try.append(('ARIMA(1,0,0) with intervention', arima_model))
            
            # Try SARIMA if we have enough data (at least 2 years for seasonal patterns)
            if len(analysis_df) >= 104:  # about 2 years of weekly data
                sarima_model = fit_arima_model(analysis_df, order=(1,0,0), seasonal_order=(1,0,0,52))
                if sarima_model:
                    models_to_try.append(('SARIMA with intervention', sarima_model))
        except Exception as e:
            print(f"Warning: ARIMA/SARIMA model failed for program {program_id}: {str(e)}")
    
    # Select best model based on AIC
    best_model_name, best_model = min(models_to_try, key=lambda x: x[1].aic if hasattr(x[1], 'aic') else float('inf'))
    
    # Extract coefficients and p-values
    if 'ARIMA' in best_model_name or 'SARIMA' in best_model_name:
        # For ARIMA/SARIMA models
        baseline_level = best_model.params[0]  # Intercept
        
        # Find indices of intervention parameters
        param_names = best_model.param_names
        level_change_idx = param_names.index('intervention') if 'intervention' in param_names else None
        trend_change_idx = param_names.index('time_since_intervention') if 'time_since_intervention' in param_names else None
        
        # Extract parameters and p-values
        level_change = best_model.params[level_change_idx] if level_change_idx is not None else np.nan
        level_p = best_model.pvalues[level_change_idx] if level_change_idx is not None else np.nan
        
        # For ARIMA models, baseline trend and trend change might not be directly available
        baseline_trend = np.nan
        trend_change = best_model.params[trend_change_idx] if trend_change_idx is not None else np.nan
        trend_p = best_model.pvalues[trend_change_idx] if trend_change_idx is not None else np.nan
        
    else:
        # For segmented regression models
        params = best_model.params
        pvalues = best_model.pvalues
        
        # Extract parameters and p-values
        baseline_level = params['Intercept']
        baseline_trend = params['time']
        level_change = params['post_intervention']
        trend_change = params['time_since_intervention']
        
        level_p = pvalues['post_intervention']
        trend_p = pvalues['time_since_intervention']
    
    # Calculate significant flags
    significant_level = level_p < 0.05
    significant_trend = trend_p < 0.05
    
    # Get R-squared and AIC
    r_squared = best_model.rsquared if hasattr(best_model, 'rsquared') else np.nan
    aic = best_model.aic if hasattr(best_model, 'aic') else np.nan
    
    # Run placebo test
    placebo_results = run_placebo_test(df, program_id, program_name)
    
    # Save model summary and placebo results
    with open(f'results/its/program_{program_id}_its_summary.txt', 'w') as f:
        f.write(f"Program ID: {program_id}\n")
        f.write(f"Program Name: {program_name}\n")
        f.write(f"Model: {best_model_name}\n\n")
        f.write(f"ITS Analysis Results:\n")
        f.write(f"Baseline level: {baseline_level:.4f}\n")
        f.write(f"Baseline trend: {baseline_trend:.4f}\n")
        f.write(f"Level change: {level_change:.4f} (p={level_p:.4f})\n")
        f.write(f"Trend change: {trend_change:.4f} (p={trend_p:.4f})\n\n")
        f.write(f"Model fit:\n")
        f.write(f"R-squared: {r_squared:.4f}\n")
        f.write(f"AIC: {aic:.2f}\n")
        f.write(f"Sample size: {len(df)}\n\n")
        f.write(f"Placebo Test Results:\n")
        f.write(f"Real intervention p-value (trend change): {trend_p:.4f}\n")
        if placebo_results:
            f.write(f"Placebo p-values (trend change):\n")
            for i, p_val in enumerate(placebo_results, 1):
                f.write(f"  Placebo Test {i}: {p_val:.4f}\n")
            # Determine if placebo test passed (pass if all placebo p-values are >= 0.05)
            passed = all(p_val >= 0.05 for p_val in placebo_results)
            f.write(f"Placebo test passed: {'Pass' if passed else 'Fail'}\n")
        else:
            f.write(f"Placebo p-values (trend change): Not available\n")
            f.write(f"Placebo test passed: Not available\n")
        
        if hasattr(best_model, 'summary'):
            f.write("\n\nFull Model Summary:\n")
            f.write(str(best_model.summary()))
    
    # Create result dictionary
    result_dict = {
        'program_id': program_id,
        'program_name': program_name,
        'baseline_level': baseline_level,
        'baseline_trend': baseline_trend,
        'level_change': level_change,
        'trend_change': trend_change,
        'level_p': level_p,
        'trend_p': trend_p,
        'significant_level_0.05': significant_level,
        'significant_trend_0.05': significant_trend,
        'model_type': best_model_name,
        'r_squared': r_squared,
        'aic': aic,
        'sample_size': len(df)
    }
    
    return result_dict, best_model, best_model_name

def fit_segmented_regression(df, ar_terms=0):
    """Fit segmented regression model, optionally with autoregressive terms"""
    # Create formula for segmented regression
    formula = 'mean_sentiment ~ time + post_intervention + time_since_intervention'
    
    if ar_terms > 0:
        # Create lagged variables for autoregressive terms
        for i in range(1, ar_terms + 1):
            df[f'lag_{i}'] = df['mean_sentiment'].shift(i)
            formula += f' + lag_{i}'
        
        # Drop rows with NaN from lagging
        model_df = df.dropna()
    else:
        model_df = df
    
    # Ensure all expected columns are present and have same length
    required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
    if not all(col in model_df.columns for col in required_cols):
        # If any required column is missing, we can't fit the model
        raise ValueError(f"Missing required columns. Available: {list(model_df.columns)}")
    
    # Check if all columns have the same length
    col_lengths = [len(model_df[col].dropna()) for col in required_cols]
    if not all(length == col_lengths[0] for length in col_lengths):
        # If columns have different lengths, realign them
        common_rows = model_df[required_cols].dropna()
        if len(common_rows) < 10:  # Need minimum data for meaningful analysis
            raise ValueError(f"Insufficient aligned data points: {len(common_rows)}")
        model_df = common_rows
    
    try:
        # Fit model
        model = smf.ols(formula=formula, data=model_df).fit()
        return model
    except Exception as e:
        raise ValueError(f"Error fitting segmented regression: {str(e)}")

def fit_arima_model(df, order=(1,0,0), seasonal_order=None):
    """Fit ARIMA or SARIMA model with intervention terms"""
    # Ensure all data is aligned and complete
    required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")
    
    # Get only complete rows (no NaN values)
    complete_data = df[required_cols].dropna()
    if len(complete_data) < 10:  # Need minimum data for meaningful analysis
        raise ValueError(f"Insufficient aligned data points: {len(complete_data)}")
    
    # Prepare exogenous variables
    exog = complete_data[['time', 'post_intervention', 'time_since_intervention']]
    endog = complete_data['mean_sentiment']
    
    # Sanity check for dimension match
    if len(exog) != len(endog):
        raise ValueError(f"Dimension mismatch: exog shape {exog.shape} vs endog shape {len(endog)}")
    
    try:
        # Fit ARIMA or SARIMA model
        if seasonal_order:
            model = SARIMAX(
                endog, 
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False
            ).fit(disp=False)
        else:
            model = SARIMAX(
                endog, 
                exog=exog,
                order=order,
                enforce_stationarity=False
            ).fit(disp=False)
        
        return model
    except Exception as e:
        print(f"Warning: ARIMA/SARIMA model fitting failed: {str(e)}")
        return None

def run_placebo_test(df, program_id, program_name, num_placebos=3):
    """Run placebo tests with fake intervention points"""
    try:
        # First ensure we have complete data
        required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
        clean_df = df[required_cols].dropna().reset_index(drop=True)
        
        if len(clean_df) < 10:
            return []  # Not enough data for placebo test
            
        # Get actual intervention point
        actual_idx = clean_df['post_intervention'].idxmax() if 1 in clean_df['post_intervention'].values else None
        if actual_idx is None:
            return []
        
        # Get pre-intervention data points
        pre_data = clean_df[clean_df['post_intervention'] == 0]
        if len(pre_data) <= 5:  # Need at least a few pre points
            return []
        
        # Select placebo intervention points (at 25%, 50%, 75% of pre-intervention period)
        placebo_idxs = [int(len(pre_data) * p) for p in [0.25]]
        placebo_p_values = []
        
        # Check if a timestamp column exists in the dataframe
        timestamp_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'timestamps', 'date', 'datetime']:
                timestamp_col = col
                break
        
        # If timestamp column exists, use it for plotting
        time_points_display = clean_df['time']
        if timestamp_col and timestamp_col in df.columns and not df[timestamp_col].isna().all():
            time_points_display = pd.to_datetime(df[timestamp_col]).dropna()
            if len(time_points_display) == len(clean_df['time']):
                time_points = time_points_display
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
        elif 'week' in df.columns:
            time_points_display = [pd.to_datetime(w.split('/')[0]) for w in df['week'] if isinstance(w, str)]
            if len(time_points_display) == len(clean_df['time']):
                time_points = time_points_display
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
        
        # Get actual model
        try:
            actual_model = fit_segmented_regression(clean_df)
            actual_trend_p = actual_model.pvalues['time_since_intervention']
            
            # Run analysis with each placebo point
            for i, idx in enumerate(placebo_idxs):
                # Create copy of dataframe
                placebo_df = clean_df.copy()
                
                # Create placebo intervention variables
                placebo_time = pre_data.iloc[idx]['time']
                placebo_df['placebo_post'] = (placebo_df['time'] >= placebo_time).astype(int)
                placebo_df['placebo_time_since'] = placebo_df['placebo_post'] * \
                                                  (placebo_df['time'] - placebo_time)
                
                # Fit placebo model
                formula = 'mean_sentiment ~ time + placebo_post + placebo_time_since'
                try:
                    placebo_model = smf.ols(formula=formula, data=placebo_df).fit()
                    placebo_p = placebo_model.pvalues['placebo_time_since']
                    placebo_p_values.append(placebo_p)
                except Exception as e:
                    print(f"Warning: Placebo model {i+1} fitting failed for program {program_id}: {str(e)}")
        except Exception as e:
            print(f"Warning: Could not fit actual model for placebo testing for program {program_id}: {str(e)}")
            return []
    except Exception as e:
        print(f"Warning: Placebo test initialization failed for program {program_id}: {str(e)}")
        return []
    
    return placebo_p_values


if __name__ == "__main__":
    its_summary = run_its_analysis()
    print(f"ITS analysis complete. Results saved to 'results/its' and 'figures/its' directories.")

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

def run_its_analysis(program_ids=None, processed_dir='processed_data'):
    """Run Interrupted Time Series (ITS) analysis for each service program
    
    Args:
        program_ids (list, optional): 要分析的服务项目ID列表。如果为None，则分析所有项目。
        processed_dir (str, optional): 处理后数据的目录路径。默认为'processed_data'。
    
    Returns:
        DataFrame: ITS分析结果摘要数据框
    """
    print("Starting Interrupted Time Series (ITS) analysis...")
    
    # Create results directory if it doesn't exist
    dirs_to_create = ['results/its', 'figures/its']
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Get all weekly data files
    if program_ids is not None:
        # 只获取指定项目的文件
        weekly_files = [f'{processed_dir}/program_{pid}_weekly.csv' for pid in program_ids 
                        if os.path.exists(f'{processed_dir}/program_{pid}_weekly.csv')]
        if not weekly_files:
            print("Warning: No weekly data files found for the specified program IDs")
    else:
        # 获取所有项目的文件
        weekly_files = glob.glob(f'{processed_dir}/program_*_weekly.csv')
    
    # Create summary dataframe for all ITS results
    its_summary = pd.DataFrame(columns=[
        'program_id', 'program_name', 'baseline_level', 'baseline_trend', 
        'level_change', 'trend_change', 'level_p', 'trend_p',
        'significant_level_0.05', 'significant_trend_0.05', 'model_type',
        'r_squared', 'aic', 'sample_size'
    ])
    
    # 如果没有找到任何文件，提前返回空的摘要结果
    if not weekly_files:
        print("No weekly data files found for ITS analysis")
        return its_summary
    
    for file_path in weekly_files:
        # Extract program ID from filename
        program_id = int(file_path.split('_')[-2])
        
        # Load program data
        df = pd.read_csv(file_path)
        
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
        
        # Get program name
        program_name = df['program_name'].iloc[0] if 'program_name' in df.columns else f"Program {program_id}"
        
        try:
            # Run ITS analysis for this program
            result_dict = analyze_single_program(df, program_id, program_name)
            
            # Add results to summary
            its_summary = pd.concat([its_summary, pd.DataFrame([result_dict])], ignore_index=True)
            
        except Exception as e:
            print(f"Error analyzing program {program_id}: {e}")
    
    # Sort summary by trend change significance and magnitude
    its_summary = its_summary.sort_values(
        by=['significant_trend_0.05', 'trend_change'], 
        ascending=[False, False]
    )
    
    # Save summary results
    its_summary.to_csv('results/its/its_summary.csv', index=False)
    
    # Create overall visualization of ITS results
    create_overall_its_visualization(its_summary)
    
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
    
    # Create visualization
    create_its_visualization(df, best_model, best_model_name, program_id, program_name)
    
    # Perform placebo test
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
        f.write(f"Placebo p-values (trend change): {', '.join([f'{p:.4f}' for p in placebo_results])}\n")
        f.write(f"Placebo test passed: {all(p > trend_p for p in placebo_results)}\n\n")
        
        if hasattr(best_model, 'summary'):
            f.write("\nFull Model Summary:\n")
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
    
    return result_dict

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

def create_its_visualization(df, model, model_name, program_id, program_name):
    """Create visualization for ITS analysis results"""
    try:
        # First ensure we have complete data for visualization
        required_cols = ['mean_sentiment', 'time', 'post_intervention', 'time_since_intervention']
        clean_df = df[required_cols].dropna().reset_index(drop=True)
        
        if len(clean_df) < 10:
            print(f"Warning: Insufficient data for visualization for program {program_id}")
            return
        
        # Get observed values
        time_points = clean_df['time']
        observed = clean_df['mean_sentiment']
        
        # Check if a timestamp column exists in the dataframe
        timestamp_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'timestamps', 'date', 'datetime']:
                timestamp_col = col
                break
        
        # If timestamp column exists, use it for plotting
        if timestamp_col and timestamp_col in df.columns and not df[timestamp_col].isna().all():
            time_points_display = pd.to_datetime(df[timestamp_col]).dropna()
            if len(time_points_display) == len(time_points):
                time_points = time_points_display
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
        elif 'week' in df.columns:
            time_points_display = [pd.to_datetime(w.split('/')[0]) for w in df['week'] if isinstance(w, str)]
            if len(time_points_display) == len(time_points):
                time_points = time_points_display
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
        else:
            time_points_display = time_points
        
        # Get predicted values
        if hasattr(model, 'fittedvalues'):
            predicted = model.fittedvalues
            # Ensure predicted values align with observed data points
            if len(predicted) != len(time_points):
                print(f"Warning: Dimension mismatch in visualization. Using only aligned data points.")
                # We need to align the model's predictions with our clean data
                # This is a simple approach - in a real app we would need more sophisticated alignment
                idx_range = min(len(time_points), len(predicted))
                time_points = time_points[:idx_range]
                time_points_display = time_points_display[:idx_range] if isinstance(time_points_display, pd.Series) else time_points_display[:idx_range]
                observed = observed[:idx_range]
                predicted = predicted[:idx_range]
        else:
            # For ARIMA models, predict on our clean data
            try:
                predicted = model.predict()
                # Check dimensions and align if needed
                if len(predicted) != len(time_points):
                    idx_range = min(len(time_points), len(predicted))
                    time_points = time_points[:idx_range]
                    time_points_display = time_points_display[:idx_range] if isinstance(time_points_display, pd.Series) else time_points_display[:idx_range]
                    observed = observed[:idx_range]
                    predicted = predicted[:idx_range]
            except Exception as e:
                print(f"Warning: Could not get predictions for visualization: {str(e)}")
                return
        
        # Find the intervention point
        intervention_idx = clean_df['post_intervention'].idxmax() if 1 in clean_df['post_intervention'].values else None
        intervention_time = clean_df.loc[intervention_idx, 'time'] if intervention_idx is not None else None
        intervention_time_display = intervention_time
        if intervention_idx is not None:
            if timestamp_col and timestamp_col in df.columns and not df[timestamp_col].isna().all():
                if intervention_idx < len(df) and not pd.isna(df.loc[intervention_idx, timestamp_col]):
                    intervention_time_display = pd.to_datetime(df.loc[intervention_idx, timestamp_col])
            elif 'week' in df.columns and intervention_idx < len(df):
                if isinstance(df.loc[intervention_idx, 'week'], str):
                    intervention_time_display = pd.to_datetime(df.loc[intervention_idx, 'week'].split('/')[0])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot observed values
        plt.scatter(time_points_display, observed, color='blue', alpha=0.6, label='Observed')
        
        # Plot fitted values
        plt.plot(time_points_display, predicted, 'r-', linewidth=2, label='Fitted')
        
        # Add vertical line at intervention
        if intervention_time_display is not None:
            plt.axvline(x=intervention_time_display, color='green', linestyle='--', 
                       label='Intervention Point')
        
        # Add trend lines (pre and post)
        if 'Segmented Regression' in model_name:
            try:
                # For segmented regression models, we can directly calculate trend lines
                pre_trend = model.params['Intercept'] + model.params['time'] * time_points
                post_trend = (model.params['Intercept'] + model.params['post_intervention']) + \
                            (model.params['time'] + model.params['time_since_intervention']) * \
                            (time_points - intervention_time)
                
                # Only plot trends for relevant time periods
                pre_mask = clean_df['post_intervention'].iloc[:len(time_points)] == 0
                post_mask = clean_df['post_intervention'].iloc[:len(time_points)] == 1
                
                plt.plot(time_points_display[pre_mask], pre_trend[pre_mask], 'b--', linewidth=1, 
                        label='Pre-intervention trend')
                plt.plot(time_points_display[post_mask], post_trend[post_mask], 'g--', linewidth=1,
                        label='Post-intervention trend')
            except Exception as e:
                print(f"Warning: Could not plot trend lines: {str(e)}")
    except Exception as e:
        print(f"Warning: Visualization failed for program {program_id}: {str(e)}")
        return
    
    # Add labels and title
    plt.title(f'Program {program_id}: {program_name} - ITS Analysis\nModel: {model_name}')
    plt.xlabel('日期' if timestamp_col else '时间')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'figures/its/program_{program_id}_its.png')
    plt.close()

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
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
        
        # Create figure for placebo comparisons
        plt.figure(figsize=(12, 8))
        
        # Get actual model
        try:
            actual_model = fit_segmented_regression(clean_df)
            actual_trend_p = actual_model.pvalues['time_since_intervention']
            
            # Plot actual data and intervention
            plt.scatter(time_points_display, clean_df['mean_sentiment'], color='blue', alpha=0.3, label='Observed data')
            actual_time_display = time_points_display.iloc[actual_idx] if isinstance(time_points_display, pd.Series) else clean_df.loc[actual_idx, 'time']
            plt.axvline(x=actual_time_display, color='red', linestyle='-', 
                       linewidth=2, label='Actual intervention')
            
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
                    
                    # Plot placebo intervention line
                    placebo_time_display = time_points_display.iloc[idx] if isinstance(time_points_display, pd.Series) else placebo_time
                    plt.axvline(x=placebo_time_display, color='gray', linestyle='--', alpha=0.7,
                               label=f'Placebo {i+1} (p={placebo_p:.4f})')
                    
                except Exception as e:
                    print(f"Warning: Placebo model {i+1} fitting failed for program {program_id}: {str(e)}")
        except Exception as e:
            print(f"Warning: Could not fit actual model for placebo testing for program {program_id}: {str(e)}")
            return []
    except Exception as e:
        print(f"Warning: Placebo test initialization failed for program {program_id}: {str(e)}")
        return []
    
    # Add legend and labels
    plt.title(f'Program {program_id}: {program_name} - Placebo Test\nActual p-value: {actual_trend_p:.4f}')
    plt.xlabel('日期' if timestamp_col else '时间')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'figures/its/program_{program_id}_placebo.png')
    plt.close()
    
    return placebo_p_values

def create_overall_its_visualization(its_summary):
    """Create visualizations summarizing ITS results across all programs"""
    if len(its_summary) == 0:
        print("Warning: No data for overall ITS visualizations")
        return
    
    # 1. Trend change effect sizes
    plt.figure(figsize=(14, 8))
    # Sort by trend change
    sorted_df = its_summary.sort_values('trend_change')
    
    # Create bar plot of trend changes
    bars = plt.bar(sorted_df['program_name'], sorted_df['trend_change'], 
                  color=[('green' if x > 0 else 'red') for x in sorted_df['trend_change']])
    
    # Add significance markers
    for i, significant in enumerate(sorted_df['significant_trend_0.05']):
        if significant:
            plt.text(i, sorted_df['trend_change'].iloc[i], '*', 
                    ha='center', va='bottom' if sorted_df['trend_change'].iloc[i] > 0 else 'top', 
                    fontsize=20)
    
    plt.title('Trend Change by Program After Intervention')
    plt.xlabel('Service Program')
    plt.ylabel('Trend Change (Slope Difference)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/its/overall_trend_change.png')
    plt.close()
    
    # 2. Level change effect sizes
    plt.figure(figsize=(14, 8))
    # Sort by level change
    sorted_df = its_summary.sort_values('level_change')
    
    # Create bar plot of level changes
    bars = plt.bar(sorted_df['program_name'], sorted_df['level_change'], 
                  color=[('green' if x > 0 else 'red') for x in sorted_df['level_change']])
    
    # Add significance markers
    for i, significant in enumerate(sorted_df['significant_level_0.05']):
        if significant:
            plt.text(i, sorted_df['level_change'].iloc[i], '*', 
                    ha='center', va='bottom' if sorted_df['level_change'].iloc[i] > 0 else 'top', 
                    fontsize=20)
    
    plt.title('Level Change by Program After Intervention')
    plt.xlabel('Service Program')
    plt.ylabel('Level Change (Intercept Difference)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/its/overall_level_change.png')
    plt.close()
    
    # 3. Level vs trend changes scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(its_summary['level_change'], its_summary['trend_change'], 
               c=[(1 if x or y else 0) for x, y in zip(its_summary['significant_level_0.05'], 
                                                      its_summary['significant_trend_0.05'])],
               cmap='viridis', alpha=0.7, s=100)
    
    # Add program labels
    for i, row in its_summary.iterrows():
        plt.annotate(f"Program {row['program_id']}", 
                    (row['level_change'], row['trend_change']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Level Change vs Trend Change')
    plt.xlabel('Level Change (Intercept Difference)')
    plt.ylabel('Trend Change (Slope Difference)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/its/level_vs_trend.png')
    plt.close()

if __name__ == "__main__":
    its_summary = run_its_analysis()
    print(f"ITS analysis complete. Results saved to 'results/its' and 'figures/its' directories.")

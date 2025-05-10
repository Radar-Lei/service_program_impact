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
    """检查processed_data目录中是否已存在相关数据文件
    
    Args:
        program_ids (list, optional): 要分析的服务项目ID列表
        regenerate_summary (bool): 是否重新生成汇总数据文件，即使它们已存在
        
    Returns:
        tuple: (需要预处理的标志, 需要预处理的程序ID列表, 是否需要重新生成汇总)
    """
    if not os.path.exists('processed_data'):
        if program_ids:
            return True, program_ids, regenerate_summary
        return True, None, regenerate_summary
        
    # 检查是否有任何处理过的数据文件
    processed_files = os.listdir('processed_data')
    if not processed_files:
        if program_ids:
            return True, program_ids, regenerate_summary
        return True, None, regenerate_summary
    
    # 如果没有指定程序ID，则检查是否有汇总数据文件
    if program_ids is None:
        # 检查是否至少有一些程序数据文件
        program_files_exist = any(f.startswith('program_') and ('weekly' in f or 'monthly' in f) for f in processed_files)
        
        if not program_files_exist:
            return True, None, regenerate_summary
        else:
            print("已找到处理过的数据，跳过预处理步骤...")
            return False, None, regenerate_summary
    
    # 检查哪些指定的程序ID缺少数据文件
    missing_program_ids = []
    for program_id in program_ids:
        weekly_file = f"program_{program_id}_weekly.csv"
        monthly_file = f"program_{program_id}_monthly.csv"
        
        # 如果周数据和月数据文件都不存在，则将该程序ID添加到缺失列表
        if weekly_file not in processed_files and monthly_file not in processed_files:
            missing_program_ids.append(program_id)
    
    if missing_program_ids:
        print(f"以下程序ID的数据文件缺失，需要预处理: {missing_program_ids}")
        return True, missing_program_ids, regenerate_summary
    else:
        print("已找到所有指定程序的处理过数据，跳过预处理步骤...")
        return False, None, regenerate_summary

def regenerate_summary_files():
    """重新生成汇总数据文件（all_programs_weekly.csv和all_programs_monthly.csv）
    
    从processed_data目录中读取各个程序的数据文件，重新生成汇总文件
    
    Returns:
        tuple: (是否成功重新生成汇总文件, 所有可用程序ID列表)
    """
    print("重新生成汇总数据文件...")
    
    if not os.path.exists('processed_data'):
        print("processed_data目录不存在，无法重新生成汇总文件")
        return False, []
    
    # 收集所有周数据和月数据文件
    weekly_dfs = []
    monthly_dfs = []
    available_program_ids = set()
    
    for file in os.listdir('processed_data'):
        file_path = os.path.join('processed_data', file)
        
        # 收集各个程序的周数据
        if file.startswith('program_') and 'weekly' in file:
            try:
                # 从文件名中提取程序ID
                program_id = int(file.split('_')[1])
                available_program_ids.add(program_id)
                
                df = pd.read_csv(file_path)
                weekly_dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        
        # 收集各个程序的月数据
        elif file.startswith('program_') and 'monthly' in file:
            try:
                # 从文件名中提取程序ID
                program_id = int(file.split('_')[1])
                available_program_ids.add(program_id)
                
                df = pd.read_csv(file_path)
                monthly_dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    # 生成汇总周数据文件
    if weekly_dfs:
        try:
            all_weekly_df = pd.concat(weekly_dfs, ignore_index=True)
            all_weekly_df.to_csv('processed_data/all_programs_weekly.csv', index=False)
            print("已重新生成汇总周数据文件: all_programs_weekly.csv")
        except Exception as e:
            print(f"生成汇总周数据文件时出错: {e}")
    
    # 生成汇总月数据文件
    if monthly_dfs:
        try:
            all_monthly_df = pd.concat(monthly_dfs, ignore_index=True)
            all_monthly_df.to_csv('processed_data/all_programs_monthly.csv', index=False)
            print("已重新生成汇总月数据文件: all_programs_monthly.csv")
        except Exception as e:
            print(f"生成汇总月数据文件时出错: {e}")
    
    print(f"找到了以下所有程序ID的数据: {sorted(list(available_program_ids))}")
    return True, sorted(list(available_program_ids))

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
        if file.startswith('program_') and ('weekly' in file or 'monthly' in file):
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

def run_complete_analysis(feedback_dir, program_ids=None, force_preprocess=False, regenerate_summary=True):
    """Run the complete analysis pipeline
    
    Args:
        feedback_dir (str): 反馈数据目录
        program_ids (list, optional): 要分析的服务项目ID列表。如果为None，则分析所有项目。
        force_preprocess (bool, optional): 强制重新预处理数据，即使已有处理过的数据。
        regenerate_summary (bool, optional): 是否重新生成汇总数据文件
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
        # 获取所有可用的程序ID，用于后续分析
        existing_program_ids = get_all_available_program_ids()
        
        # 如果未指定程序ID，则获取所有有效的程序ID
        if program_ids is None:
            valid_program_ids = get_all_valid_program_ids()
            if not valid_program_ids:
                print("未找到有效的程序ID，分析中止。")
                return False
            
            print(f"从原始数据中找到以下有效程序ID: {valid_program_ids}")
            
            # 找出缺失的程序ID（在有效列表中但不在现有数据中）
            missing_program_ids = [pid for pid in valid_program_ids if pid not in existing_program_ids]
            
            if missing_program_ids:
                print(f"以下程序ID缺少处理过的数据，将为它们生成数据: {missing_program_ids}")
                programs_to_preprocess = missing_program_ids
                need_preprocess = True
            else:
                print("所有有效程序ID的数据都已存在，无需预处理。")
                programs_to_preprocess = None
                need_preprocess = False
            
            # 设置分析使用的程序ID为所有有效程序ID
            analysis_program_ids = valid_program_ids
        else:
            # 如果指定了程序ID，检查哪些需要预处理
            if force_preprocess:
                # 强制预处理所有指定的程序
                need_preprocess = True
                programs_to_preprocess = program_ids
            else:
                # 检查哪些程序需要预处理
                need_preprocess, programs_to_preprocess, _ = check_processed_data_exists(program_ids, regenerate_summary)
            
            # 设置分析使用的程序ID为指定的程序ID
            analysis_program_ids = program_ids
        
        # 如果需要预处理
        if need_preprocess:
            print("\n1. PREPROCESSING DATA...")
            if programs_to_preprocess:
                print(f"仅预处理以下程序: {programs_to_preprocess}")
            
            valid_programs = preprocess_data(feedback_dir, programs_to_preprocess)
            
            if valid_programs.empty:
                print("未找到指定ID的有效程序。分析中止。")
                return False
        else:
            print("\n1. USING EXISTING PROCESSED DATA...")
            # 读取有效项目信息
            try:
                # 如果需要重新生成汇总文件
                if regenerate_summary:
                    success, available_program_ids = regenerate_summary_files()
                    if not success:
                        print("重新生成汇总文件失败，使用现有文件继续分析")
                
                # 尝试读取所有程序数据
                if os.path.exists('processed_data/all_programs_weekly.csv'):
                    valid_programs = pd.read_csv('processed_data/all_programs_weekly.csv')
                # 如果周数据不存在，尝试读取月数据
                elif os.path.exists('processed_data/all_programs_monthly.csv'):
                    valid_programs = pd.read_csv('processed_data/all_programs_monthly.csv')
                else:
                    # 如果只有单独的程序文件，则需要读取它们
                    program_files = []
                    for file in os.listdir('processed_data'):
                        if file.startswith('program_') and ('weekly' in file or 'monthly' in file):
                            program_files.append(os.path.join('processed_data', file))
                    
                    if not program_files:
                        print("无法在processed_data目录中找到有效的数据文件。分析中止。")
                        return False
                    
                    # 读取第一个文件以获取程序信息
                    valid_programs = pd.read_csv(program_files[0])
                
                # 筛选特定程序ID
                valid_programs = valid_programs[valid_programs['program_id'].isin(analysis_program_ids)]
                
                if valid_programs.empty:
                    print("在已有数据中未找到有效程序。分析中止。")
                    return False
                    
            except Exception as e:
                print(f"读取已有的处理数据时出错: {e}")
                return False
        
        # Step 2: Basic statistical analysis
        print("\n2. RUNNING BASIC STATISTICAL ANALYSIS...")
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
    parser.add_argument('--programs', type=int, nargs='+', default=[0, 1, 4, 22], help='要分析的服务项目ID列表，如果不指定则使用默认值0,1,4,22')
    parser.add_argument('--force-preprocess', action='store_true', 
                        help='强制重新预处理数据，即使已经存在处理过的数据')
    parser.add_argument('--no-summary-regen', action='store_true',
                        help='不重新生成汇总数据文件(all_programs_weekly.csv和all_programs_monthly.csv)')
    
    return parser.parse_args()

if __name__ == "__main__":
    """
    python run_analysis.py --programs 0 1 4 8 22 --feedback_dir similarity_threshold=0.35/
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

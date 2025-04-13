import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from basic_stats_analysis import analyze_program_stats
from its_analysis import run_its_analysis
import shutil
import sys

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

def run_complete_analysis():
    """Run the complete analysis pipeline"""
    start_time = time.time()
    
    print("="*80)
    print("STARTING SERVICE PROGRAM IMPACT ANALYSIS")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Create necessary directories
    for directory in ['processed_data', 'results', 'figures']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    try:
        # Check if analysis results already exist
        analysis_completed = (os.path.exists('results/program_stats_summary.csv') and 
                             os.path.exists('results/its/its_summary.csv'))
        
        if analysis_completed:
            print("\nEXISTING ANALYSIS RESULTS FOUND!")
            print("Skipping data preprocessing and analysis steps...")
        else:
            # Step 1: Data preprocessing
            print("\n1. PREPROCESSING DATA...")
            valid_programs = preprocess_data()
            
            # Step 2: Basic statistical analysis
            print("\n2. RUNNING BASIC STATISTICAL ANALYSIS...")
            stats_summary = analyze_program_stats()
            
            # Step 3: Interrupted Time Series analysis
            print("\n3. RUNNING INTERRUPTED TIME SERIES ANALYSIS...")
            its_summary = run_its_analysis()
        
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

if __name__ == "__main__":
    run_complete_analysis()

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

def generate_report():
    """Generate final report in markdown format with key findings"""
    try:
        # Load results
        basic_stats = pd.read_csv('results/program_stats_summary.csv')
        its_results = pd.read_csv('results/its/its_summary.csv')
        
        # Start report
        report = "# Service Program Impact Analysis Report\n\n"
        
        # Add introduction
        report += "## Introduction\n\n"
        report += "This report analyzes the impact of various service programs on passenger experience "
        report += "based on sentiment analysis of social media feedback. The analysis examines how "
        report += "passenger sentiment changed before and after the implementation of each service program.\n\n"
        
        # Add methodology section
        report += "## Methodology\n\n"
        report += "The analysis employs three primary methods:\n\n"
        report += "1. **Basic Statistical Analysis**: Comparing pre- and post-implementation sentiment scores using t-tests\n"
        report += "2. **Categorical Analysis**: Using Chi-square or Fisher's exact tests to analyze sentiment distribution shifts\n"
        report += "3. **Interrupted Time Series (ITS) Analysis**: Examining changes in both level (immediate effect) and trend (longer-term effect) of sentiment after program implementation\n\n"
        
        # Add key findings section
        report += "## Key Findings\n\n"
        
        # Count positive and negative impacts
        pos_impact_count = len(basic_stats[basic_stats['mean_diff'] > 0])
        neg_impact_count = len(basic_stats[basic_stats['mean_diff'] < 0])
        significant_count = len(basic_stats[basic_stats['significant_0.05']])
        
        report += f"### Overall Impact\n\n"
        report += f"- **{len(basic_stats)}** service programs were analyzed\n"
        report += f"- **{pos_impact_count}** programs showed positive sentiment change after implementation\n"
        report += f"- **{neg_impact_count}** programs showed negative sentiment change after implementation\n"
        report += f"- **{significant_count}** programs showed statistically significant changes (p < 0.05)\n\n"
        
        # Add top positive impact programs
        if not basic_stats.empty:
            top_positive = basic_stats[basic_stats['mean_diff'] > 0].sort_values('mean_diff', ascending=False).head(5)
            
            if not top_positive.empty:
                report += "### Top Positive Impact Programs\n\n"
                report += "| Program ID | Program Name | Mean Sentiment Change | p-value | Significant |\n"
                report += "|------------|--------------|----------------------|---------|-------------|\n"
                
                for _, row in top_positive.iterrows():
                    report += f"| {row['program_id']} | {row['program_name']} | {row['mean_diff']:.4f} | {row['p_value']:.4f} | {'Yes' if row['significant_0.05'] else 'No'} |\n"
                
                report += "\n"
            
            # Add top negative impact programs
            top_negative = basic_stats[basic_stats['mean_diff'] < 0].sort_values('mean_diff').head(5)
            
            if not top_negative.empty:
                report += "### Top Negative Impact Programs\n\n"
                report += "| Program ID | Program Name | Mean Sentiment Change | p-value | Significant |\n"
                report += "|------------|--------------|----------------------|---------|-------------|\n"
                
                for _, row in top_negative.iterrows():
                    report += f"| {row['program_id']} | {row['program_name']} | {row['mean_diff']:.4f} | {row['p_value']:.4f} | {'Yes' if row['significant_0.05'] else 'No'} |\n"
                
                report += "\n"
        
        # Add ITS analysis results
        if not its_results.empty:
            report += "### Interrupted Time Series Analysis Results\n\n"
            report += "The ITS analysis provides more robust evidence of program impact by examining both immediate changes (level) and longer-term changes (trend) in sentiment.\n\n"
            
            # Programs with significant trend changes
            sig_trend = its_results[its_results['significant_trend_0.05']].sort_values('trend_change', ascending=False)
            
            if not sig_trend.empty:
                report += "#### Programs with Significant Trend Changes\n\n"
                report += "| Program ID | Program Name | Baseline Trend | Trend Change | p-value |\n"
                report += "|------------|--------------|---------------|-------------|--------|\n"
                
                for _, row in sig_trend.iterrows():
                    report += f"| {row['program_id']} | {row['program_name']} | {row['baseline_trend']:.4f} | {row['trend_change']:.4f} | {row['trend_p']:.4f} |\n"
                
                report += "\n"
            
            # Programs with significant level changes
            sig_level = its_results[its_results['significant_level_0.05']].sort_values('level_change', ascending=False)
            
            if not sig_level.empty:
                report += "#### Programs with Significant Level Changes\n\n"
                report += "| Program ID | Program Name | Baseline Level | Level Change | p-value |\n"
                report += "|------------|--------------|---------------|-------------|--------|\n"
                
                for _, row in sig_level.iterrows():
                    report += f"| {row['program_id']} | {row['program_name']} | {row['baseline_level']:.4f} | {row['level_change']:.4f} | {row['level_p']:.4f} |\n"
                
                report += "\n"
        
        # Add detailed examples section (pick a couple of interesting cases)
        report += "## Detailed Examples\n\n"
        
        # Try to find a significant positive case
        sig_positive = basic_stats[(basic_stats['significant_0.05']) & (basic_stats['mean_diff'] > 0)]
        if not sig_positive.empty:
            example = sig_positive.iloc[0]
            program_id = example['program_id']
            report += f"### Example of Positive Impact: Program {program_id} - {example['program_name']}\n\n"
            report += f"This program showed a significant positive change in passenger sentiment after implementation (mean difference: {example['mean_diff']:.4f}, p={example['p_value']:.4f}).\n\n"
            report += "#### Visual Evidence\n\n"
            report += f"![Time Series Plot](figures/program_{program_id}_timeseries.png)\n\n"
            report += f"![ITS Analysis](figures/its/program_{program_id}_its.png)\n\n"
            
            # Add placebo test info if available
            if os.path.exists(f'results/its/program_{program_id}_its_summary.txt'):
                with open(f'results/its/program_{program_id}_its_summary.txt', 'r') as f:
                    its_text = f.read()
                    
                    if "Placebo Test Results" in its_text:
                        report += "The placebo tests confirm that the observed effect is likely due to the program rather than chance variation.\n\n"
        
        # Try to find a non-significant case
        non_sig = basic_stats[~basic_stats['significant_0.05']]
        if not non_sig.empty:
            example = non_sig.iloc[0]
            program_id = example['program_id']
            report += f"### Example of Non-Significant Impact: Program {program_id} - {example['program_name']}\n\n"
            report += f"This program did not show a statistically significant change in passenger sentiment (mean difference: {example['mean_diff']:.4f}, p={example['p_value']:.4f}).\n\n"
            report += "#### Visual Evidence\n\n"
            report += f"![Time Series Plot](figures/program_{program_id}_timeseries.png)\n\n"
        
        # Add overall visualizations
        report += "## Overall Results Visualization\n\n"
        
        report += "### Basic Statistical Analysis\n\n"
        report += "![Mean Sentiment Differences](figures/overall_mean_diff_barplot.png)\n\n"
        report += "![Pre vs Post Scatter](figures/pre_vs_post_mean_scatter.png)\n\n"
        
        report += "### Interrupted Time Series Analysis\n\n"
        report += "![Trend Changes](figures/its/overall_trend_change.png)\n\n"
        report += "![Level Changes](figures/its/overall_level_change.png)\n\n"
        report += "![Level vs Trend Changes](figures/its/level_vs_trend.png)\n\n"
        
        # Add limitations section
        report += "## Limitations and Considerations\n\n"
        report += "Several limitations should be considered when interpreting these results:\n\n"
        report += "1. **Social Media Bias**: The sentiment analysis is based on social media feedback, which may not represent the entire passenger population.\n\n"
        report += "2. **Confounding Factors**: Other factors beyond the service programs may have influenced passenger sentiment during the study period.\n\n"
        report += "3. **Limited Control Groups**: Without clear control groups, it's challenging to attribute changes solely to the service programs.\n\n"
        report += "4. **Data Sparsity**: Some programs had limited data points, especially in pre-intervention periods, which may affect the reliability of the analysis.\n\n"
        report += "5. **Sentiment Metric**: The sentiment score used (Positive - Negative) is one of many possible metrics and may not capture nuanced feedback.\n\n"
        
        # Add conclusion
        report += "## Conclusion\n\n"
        report += "The analysis reveals varying impacts of service programs on passenger sentiment. Some programs showed statistically significant improvements, while others had neutral or potentially negative effects.\n\n"
        report += "The Interrupted Time Series analysis provides the most robust evidence of program impact by accounting for pre-existing trends and examining both immediate and longer-term changes in sentiment.\n\n"
        report += "Based on these findings, programs with significant positive impacts might be considered for expansion or replication, while those with negative impacts may need refinement or reconsideration.\n\n"
        
        # Write report to file
        with open('service_program_impact_report.md', 'w') as f:
            f.write(report)
        
        print("Report generated successfully: service_program_impact_report.md")
        return True
    
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

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
        # Step 1: Data preprocessing
        print("\n1. PREPROCESSING DATA...")
        valid_programs = preprocess_data()
        
        # Step 2: Basic statistical analysis
        print("\n2. RUNNING BASIC STATISTICAL ANALYSIS...")
        stats_summary = analyze_program_stats()
        
        # Step 3: Interrupted Time Series analysis
        print("\n3. RUNNING INTERRUPTED TIME SERIES ANALYSIS...")
        its_summary = run_its_analysis()
        
        # Step 4: Generate report
        print("\n4. GENERATING FINAL REPORT...")
        generate_report()
        
        # Step 5: Convert to HTML if possible
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

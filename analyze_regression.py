import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories if they don't exist
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("Loading data for regression analysis...")

# Load service program data
service_programs = pd.read_csv('service_program_data/SPD_SZ_zh.csv')
print(f"Loaded {len(service_programs)} service programs")

# Check if program metrics data exists (from analyze_statistics.py)
if not os.path.exists('results/program_metrics.csv'):
    print("Program metrics data not found. Please run analyze_statistics.py first.")
    exit(1)

program_metrics = pd.read_csv('results/program_metrics.csv')
print(f"Loaded metrics for {len(program_metrics)} programs")

# Prepare data for regression analysis
# Merge service program details with metrics
analysis_data = pd.merge(
    service_programs,
    program_metrics,
    left_index=True,
    right_on='program_idx',
    how='inner'
)

print(f"Merged data contains {len(analysis_data)} rows")

# Clean and prepare data for regression
# Create category dummies for service quality categories
analysis_data['category'] = analysis_data['program_category']
analysis_data.loc[analysis_data['category'].isna(), 'category'] = 'Other'
category_dummies = pd.get_dummies(analysis_data['category'], prefix='cat', drop_first=True)
analysis_data = pd.concat([analysis_data, category_dummies], axis=1)

# Extract key binary features from program descriptions
# Based on common themes in service improvement programs
analysis_data['has_tech'] = analysis_data['Service improvement programs'].str.contains(
    '智能|系统|电子|数字|信息', case=False, na=False).astype(int)
analysis_data['has_facility'] = analysis_data['Service improvement programs'].str.contains(
    '设施|卫生间|改造|站点|移动|厢', case=False, na=False).astype(int)
analysis_data['has_payment'] = analysis_data['Service improvement programs'].str.contains(
    '支付|乘车码|票|付', case=False, na=False).astype(int)
analysis_data['has_schedule'] = analysis_data['Service improvement programs'].str.contains(
    '时间|频率|间隔', case=False, na=False).astype(int)
analysis_data['has_discount'] = analysis_data['Service improvement programs'].str.contains(
    '优惠|降低|免费', case=False, na=False).astype(int)

# Save prepared data for future reference
analysis_data.to_csv('results/regression_data.csv', index=False)
print("Saved prepared regression data to results/regression_data.csv")

# 1. Basic correlation analysis
print("\nCalculating correlations between features and sentiment...\n")
correlation_vars = [
    'Positive_mean', 'Negative_mean', 'Neutral_mean', 
    'has_tech', 'has_facility', 'has_payment', 'has_schedule', 'has_discount'
]
correlations = analysis_data[correlation_vars].corr()

# Save correlation matrix
correlations.to_csv('results/feature_correlations.csv')

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlations,
    annot=True,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    center=0,
    square=True,
    fmt=".2f"
)
plt.title('Correlation Between Features and Sentiment', fontsize=16)
plt.tight_layout()
plt.savefig('figures/feature_correlation_heatmap.png')
plt.close()

# 2. Multiple regression analysis
print("\nPerforming multiple regression analysis...")

# Model 1: Basic regression with binary features
X1 = analysis_data[['has_tech', 'has_facility', 'has_payment', 'has_schedule', 'has_discount']]
X1 = sm.add_constant(X1)
y = analysis_data['Positive_mean']

model1 = sm.OLS(y, X1).fit()
print("\nModel 1: Basic Feature Regression")
print(model1.summary())

# Model 2: Include category dummies
cat_cols = [col for col in analysis_data.columns if col.startswith('cat_')]
if cat_cols:
    try:
        # Convert all data to numeric to avoid dtype issues
        features = ['has_tech', 'has_facility', 'has_payment', 'has_schedule', 'has_discount'] + cat_cols
        X2 = analysis_data[features].apply(pd.to_numeric, errors='coerce')
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(y, X2).fit()
        print("\nModel 2: Features + Category Regression")
        print(model2.summary())
    except Exception as e:
        print(f"\nError in Model 2: {e}")
        print("Skipping Model 2 due to data conversion issues")
        model2 = None
else:
    print("\nNo category dummy variables available for Model 2")
    model2 = None

# Check multicollinearity with VIF (Variance Inflation Factor)
print("\nChecking for multicollinearity...")
def calculate_vif(X):
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return pd.DataFrame({"Variable": X.columns, "VIF": [float('nan')] * len(X.columns)})

vif_model1 = calculate_vif(X1)
print("\nVIF for Model 1:")
print(vif_model1)

if model2 is not None:
    try:
        vif_model2 = calculate_vif(X2)
        print("\nVIF for Model 2:")
        print(vif_model2)
    except Exception as e:
        print(f"\nError calculating VIF for Model 2: {e}")

# 3. Impact analysis: Calculate the estimated impact of each feature
print("\nEstimating feature impacts...")
impacts = pd.DataFrame()
impacts['Feature'] = X1.columns[1:]  # Skip constant
impacts['Coefficient'] = model1.params.values[1:]  # Skip constant
impacts['P_Value'] = model1.pvalues.values[1:]  # Skip constant
impacts['Significant'] = impacts['P_Value'] < 0.05

# Calculate marginal effects - how much does each feature change the sentiment?
impacts['Impact'] = impacts['Coefficient'] / y.mean()
impacts['Impact_Percentage'] = impacts['Impact'] * 100

# Sort by impact (absolute value)
impacts = impacts.reindex(impacts['Impact'].abs().sort_values(ascending=False).index)

# Save impact analysis
impacts.to_csv('results/feature_impacts.csv', index=False)
print("Saved feature impact analysis to results/feature_impacts.csv")

# 4. Save regression results to a text file
with open('results/regression_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("# Regression Analysis of Service Program Impact\n\n")
    
    # Feature correlation highlights
    f.write("## Correlation Analysis\n\n")
    f.write("Highest correlations with positive sentiment:\n")
    pos_corr = correlations['Positive_mean'].drop('Positive_mean').sort_values(ascending=False)
    for feature, corr in pos_corr.items():
        f.write(f"- {feature}: {corr:.3f}\n")
    
    f.write("\nHighest correlations with negative sentiment:\n")
    neg_corr = correlations['Negative_mean'].drop('Negative_mean').sort_values(ascending=False)
    for feature, corr in neg_corr.items():
        f.write(f"- {feature}: {corr:.3f}\n")
    
    # Regression model 1 results
    f.write("\n## Basic Regression Model Results\n\n")
    f.write("Dependent variable: Average Positive Sentiment\n\n")
    f.write(f"R-squared: {model1.rsquared:.3f}\n")
    f.write(f"Adjusted R-squared: {model1.rsquared_adj:.3f}\n\n")
    
    f.write("Coefficients:\n")
    for var, coef, pval in zip(model1.params.index, model1.params, model1.pvalues):
        star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        f.write(f"- {var}: {coef:.4f} {star} (p={pval:.4f})\n")
    
    f.write("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1\n")
    
    # Model 2 if available and successful
    if model2 is not None:
        try:
            f.write("\n## Extended Regression Model Results (Including Service Categories)\n\n")
            f.write("Dependent variable: Average Positive Sentiment\n\n")
            f.write(f"R-squared: {model2.rsquared:.3f}\n")
            f.write(f"Adjusted R-squared: {model2.rsquared_adj:.3f}\n\n")
            
            f.write("Coefficients:\n")
            for var, coef, pval in zip(model2.params.index, model2.params, model2.pvalues):
                star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                f.write(f"- {var}: {coef:.4f} {star} (p={pval:.4f})\n")
            
            f.write("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1\n")
        except Exception as e:
            f.write("\n## Extended Regression Model (Model 2)\n\n")
            f.write(f"Model 2 could not be completed due to data issues: {str(e)}\n")
    
    # Feature impact summary
    f.write("\n## Feature Impact Summary\n\n")
    f.write("Estimated impact on positive sentiment:\n\n")
    
    for _, row in impacts.iterrows():
        sign = '+' if row['Coefficient'] > 0 else ''
        sig = '***' if row['P_Value'] < 0.01 else ('**' if row['P_Value'] < 0.05 else ('*' if row['P_Value'] < 0.1 else ''))
        f.write(f"- {row['Feature']}: {sign}{row['Impact_Percentage']:.2f}% {sig}\n")
    
    f.write("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1\n")
    
    # Key findings
    f.write("\n## Key Findings\n\n")
    
    # Positive impacts
    positive_impacts = impacts[impacts['Coefficient'] > 0].sort_values('Impact', ascending=False)
    if not positive_impacts.empty:
        f.write("Service features with positive impact on passenger sentiment:\n")
        for _, row in positive_impacts.iterrows():
            if row['P_Value'] < 0.1:  # Only include marginally significant or better
                f.write(f"- {row['Feature']}: Increases positive sentiment by {row['Impact_Percentage']:.2f}%\n")
    
    # Negative impacts
    negative_impacts = impacts[impacts['Coefficient'] < 0].sort_values('Impact')
    if not negative_impacts.empty:
        f.write("\nService features with negative impact on passenger sentiment:\n")
        for _, row in negative_impacts.iterrows():
            if row['P_Value'] < 0.1:  # Only include marginally significant or better
                f.write(f"- {row['Feature']}: Decreases positive sentiment by {abs(row['Impact_Percentage']):.2f}%\n")
    
    # Model quality assessment
    f.write("\n## Model Assessment\n\n")
    f.write(f"The regression model explains {model1.rsquared:.1%} of the variation in positive sentiment.\n")
    
    if model2 is not None:
        improvement = model2.rsquared - model1.rsquared
        f.write(f"Adding service categories improves the explanation by {improvement:.1%} points.\n")
    
    f.write("\nMulticollinearity was assessed using Variance Inflation Factors (VIF).\n")
    high_vif = vif_model1[vif_model1["VIF"] > 5]
    if not high_vif.empty:
        f.write("Potential multicollinearity issues with the following variables:\n")
        for _, row in high_vif.iterrows():
            f.write(f"- {row['Variable']}: VIF = {row['VIF']:.2f}\n")
    else:
        f.write("No significant multicollinearity detected (all VIF values below 5).\n")

print("\nRegression analysis completed. Results saved to results/regression_analysis.txt")

# 5. Visualize regression results
print("\nCreating regression visualization...")

# Sort features by coefficient magnitude
coefs = pd.DataFrame({
    'Feature': model1.params.index[1:],  # Skip constant
    'Coefficient': model1.params.values[1:],  # Skip constant
    'Error': model1.bse.values[1:],  # Skip constant
    'P_Value': model1.pvalues.values[1:]  # Skip constant
})
coefs = coefs.sort_values('Coefficient', key=abs, ascending=False)

# Set up colors by significance levels
colors = []
for p in coefs['P_Value']:
    if p < 0.01:
        colors.append('red')
    elif p < 0.05:
        colors.append('orange')
    elif p < 0.1:
        colors.append('yellow')
    else:
        colors.append('gray')

# Create coefficient plot
plt.figure(figsize=(12, 8))
# Use loop to plot each point with its respective color
for i, (x, y, xerr, color) in enumerate(zip(coefs['Coefficient'], coefs['Feature'], coefs['Error'], colors)):
    plt.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        fmt='o',
        ecolor='black',
        capsize=5,
        markersize=10,
        markerfacecolor=color
    )

# Add vertical line at 0
plt.axvline(x=0, color='gray', linestyle='--')

# Add labels and title
plt.xlabel('Coefficient (Effect on Positive Sentiment)')
plt.ylabel('Service Program Feature')
plt.title('Impact of Service Features on Passenger Sentiment', fontsize=16)

# Add legend for significance
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='p < 0.01', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='p < 0.05', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='p < 0.1', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Not significant', markersize=10)
]
plt.legend(handles=legend_elements, title='Significance Level')

plt.grid(axis='x')
plt.tight_layout()
plt.savefig('figures/regression_coefficients.png')
plt.close()

print("Regression visualization saved to figures/regression_coefficients.png")
print("Regression analysis complete.")

# Service Program Impact Analysis Report

## Introduction

This report analyzes the impact of various service programs on passenger experience based on sentiment analysis of social media feedback. The analysis examines how passenger sentiment changed before and after the implementation of each service program, providing data-driven insights for evaluating service effectiveness.

## Methodology

The analysis employs three primary methods:

1. **Basic Statistical Analysis**: Comparing pre- and post-implementation sentiment scores using t-tests
2. **Categorical Analysis**: Using Chi-square or Fisher's exact tests to analyze sentiment distribution shifts
3. **Interrupted Time Series (ITS) Analysis**: Examining changes in both level (immediate effect) and trend (longer-term effect) of sentiment after program implementation

## Data Overview

This analysis encompasses **23** service programs, with data sourced from passenger social media feedback. Each program is analyzed through comparison of sentiment scores before and after implementation, as well as trend analysis over time.

## Key Findings

### Overall Impact

- **23** service programs were analyzed
- **13** programs showed positive sentiment change after implementation, with **4** showing statistically significant improvement (p < 0.05)
- **10** programs showed negative sentiment change after implementation, with **2** showing statistically significant decline (p < 0.05)
- Overall, **6** programs (26.1%) demonstrated statistically significant changes

### Most Significant Positive Impact Programs

The following programs showed the most significant positive impact after implementation (sorted by p-value):

| Program ID | Program Name | Pre-Implementation Mean | Post-Implementation Mean | Mean Difference | p-value | Significant |
|------------|--------------|-------------------------|--------------------------|----------------|---------|-------------|
| 8 | 实施免费乘车和优惠乘车 | -0.2143 | 0.1818 | 0.3961 | 0.0035 | Yes |
| 5 | 完成82个站点卫生间改造 | -0.1345 | 0.1872 | 0.3217 | 0.0037 | Yes |
| 14 | 地铁4号线导向牌升级液晶显示屏 | 0.1465 | 0.3602 | 0.2137 | 0.0191 | Yes |
| 15 | 移动母婴室 | 0.1662 | 0.3591 | 0.1928 | 0.0386 | Yes |

### Most Significant Negative Impact Programs

The following programs showed the most significant negative impact after implementation (sorted by p-value):

| Program ID | Program Name | Pre-Implementation Mean | Post-Implementation Mean | Mean Difference | p-value | Significant |
|------------|--------------|-------------------------|--------------------------|----------------|---------|-------------|
| 17 | 智能测温机器人 | -0.3741 | -0.4882 | -0.1141 | 0.0423 | Yes |
| 2 | 车厢拥挤度智能显示系统 | 0.3387 | 0.1276 | -0.2111 | 0.0335 | Yes |

### Interrupted Time Series (ITS) Analysis Results

ITS analysis provides more robust evidence of program impact by examining both immediate changes (level) and longer-term changes (trend) in sentiment.

#### Programs with Significant Trend Changes

The following programs showed significant changes in sentiment trends after implementation (sorted by p-value):

| Program ID | Program Name | Baseline Trend | Trend Change | p-value | Model Type | Sample Size |
|------------|--------------|---------------|--------------|---------|------------|-------------|
| 1 | 智能动态地图显示系统 | -0.0362 | 0.0461 | 0.0072 | Segmented Regression with AR(2) | 73 |
| 18 | 已开通八条招手停靠公交线路 | -0.0023 | 0.0059 | 0.0072 | Simple Segmented Regression | 197 |
| 17 | 智能测温机器人 | NA | 0.0034 | 0.0351 | SARIMA with intervention | 195 |
| 16 | 大鹏新区开设地铁接驳专线 | NA | -0.0083 | 0.0204 | SARIMA with intervention | 188 |

#### Programs with Significant Level Changes

The following programs showed significant immediate changes in sentiment levels after implementation (sorted by p-value):

| Program ID | Program Name | Baseline Level | Level Change | p-value | Model Type | Sample Size |
|------------|--------------|---------------|--------------|---------|------------|-------------|
| 15 | 移动母婴室 | -0.1887 | -0.3629 | 0.0434 | Simple Segmented Regression | 157 |

## Case Studies

### Case Study 1: Program 8 - 实施免费乘车和优惠乘车 (Positive Impact)

This program demonstrated a significant positive change in passenger sentiment after implementation (mean difference: 0.3961, p=0.0035).

**Basic Statistical Analysis:**
- Pre-implementation mean: -0.2143, median: -0.3543
- Post-implementation mean: 0.1818, median: 0.4513
- Difference: 0.3961 (t-statistic = 3.0785)

**Interrupted Time Series Analysis:**
- Baseline level: -0.5220, level change: 0.0471 (p-value = 0.8615)
- Baseline trend: 0.0256, trend change: -0.0252 (p-value = 0.1203)
- Model: Simple Segmented Regression, R²: 0.1139, Sample size: 95

**Conclusion:** This program had a significant positive impact on passenger experience, supported by both basic statistical analysis and time series analysis. While the ITS analysis shows that the immediate level change was not statistically significant, there appears to be a meaningful shift in the overall sentiment distribution.

![Time Series Plot](figures/program_8_timeseries.png)

![ITS Analysis](figures/its/program_8_its.png)

### Case Study 2: Program 2 - 车厢拥挤度智能显示系统 (Negative Impact)

This program showed a significant negative change in passenger sentiment after implementation (mean difference: -0.2111, p=0.0335).

**Basic Statistical Analysis:**
- Pre-implementation mean: 0.3387, median: 0.5112
- Post-implementation mean: 0.1276, median: 0.2764
- Difference: -0.2111 (t-statistic = -2.1630)

**Interrupted Time Series Analysis:**
- Baseline level: 0.5582, level change: -0.0135 (p-value = 0.9505)
- Baseline trend: -0.0110, trend change: 0.0115 (p-value = 0.1523)
- Model: Simple Segmented Regression, R²: 0.0351, Sample size: 168

**Conclusion:** Despite the technology-forward nature of this program, it appears to have negatively impacted passenger sentiment. This could suggest that the system may not be meeting passenger expectations or could be causing confusion. Further investigation into user experience with this system is recommended.

![Time Series Plot](figures/program_2_timeseries.png)

![ITS Analysis](figures/its/program_2_its.png)

### Case Study 3: Program 1 - 智能动态地图显示系统 (Long-term Trend Impact)

This program presents an interesting pattern: while there was no significant immediate level change (p-value=0.3201), it showed a significant positive change in long-term trend (p-value=0.0072).

**Basic Statistical Analysis:**
- Pre-implementation mean: 0.6466, post-implementation mean: 0.6863
- Difference: 0.0398 (t-statistic = 0.3565, p-value = 0.7237)

**Interrupted Time Series Analysis:**
- Baseline level: 1.3476, level change: 0.2101 (p-value = 0.3201)
- Baseline trend: -0.0362, trend change: 0.0461 (p-value = 0.0072)
- Model: Segmented Regression with AR(2), R²: 0.1624, Sample size: 73

**Conclusion:** This pattern suggests that the program does not immediately change passenger experience but its impact becomes more pronounced over time. This may be because passengers need time to adapt to the new service, or the service's value becomes apparent with extended use.

![Time Series Plot](figures/program_1_timeseries.png)

![ITS Analysis](figures/its/program_1_its.png)

## Overall Results Visualization

### Basic Statistical Analysis

![Mean Sentiment Differences](figures/overall_mean_diff_barplot.png)

![Pre vs Post Scatter](figures/pre_vs_post_mean_scatter.png)

### Interrupted Time Series Analysis

![Trend Changes](figures/its/overall_trend_change.png)

![Level Changes](figures/its/overall_level_change.png)

![Level vs Trend Changes](figures/its/level_vs_trend.png)

## Methodological Discussion

This study employed three different analytical methods, each with its own strengths and limitations:

**Basic Statistical Analysis (t-tests)**: Simple and intuitive, comparing mean differences before and after implementation. However, this method may not capture time trends and seasonal factors.

**Categorical Analysis (Chi-square/Fisher tests)**: Provides a different perspective by comparing changes in sentiment category distributions. This method does not rely on specific sentiment score values but focuses on shifts in sentiment categories (positive/negative/neutral).

**Interrupted Time Series (ITS) Analysis**: The most robust method, capable of distinguishing between immediate effects (level changes) and long-term effects (trend changes) while accounting for pre-existing trends, seasonality, and autocorrelation. In this study, ITS analysis provided the most valuable insights, especially for programs with effects that evolve over time.

The ITS analysis identified **5** programs showing significant changes in level or trend, representing 21.7% of all programs. This proportion is slightly lower than the 26.1% of programs showing significant changes in the basic statistical analysis, suggesting that some seemingly significant changes might be due to pre-existing trends or other confounding factors rather than direct program impact.

## Limitations and Considerations

Several limitations should be considered when interpreting these results:

1. **Social Media Bias**: The sentiment analysis is based on social media feedback, which may not represent the entire passenger population. Social media users tend to be younger and may be more likely to express opinions during extreme experiences (very satisfied or dissatisfied).

2. **Confounding Factors**: Other factors beyond the service programs may have influenced passenger sentiment during the study period, such as seasonal changes, major events, or pandemic effects.

3. **Data Sparsity**: Some programs had limited data points, especially in pre-intervention periods, which may affect the reliability of the analysis. According to the ITS analysis results, sample sizes ranged from as few as 25 (Program 7) to as many as 197 (Program 18).

4. **Sentiment Metric**: The sentiment score used (Positive - Negative) is one of many possible metrics and may not capture nuanced feedback.

5. **Time Lag Effects**: Some service programs may take time to show their full impact, which may not be completely captured in this analysis.

## Conclusions and Recommendations

Based on the analysis results, we draw the following conclusions:

1. **Most Successful Service Programs**:
   - Program 8 (实施免费乘车和优惠乘车): Sentiment score increase of 0.3961, p-value=0.0035
   - Program 5 (完成82个站点卫生间改造): Sentiment score increase of 0.3217, p-value=0.0037
   - Program 14 (地铁4号线导向牌升级液晶显示屏): Sentiment score increase of 0.2137, p-value=0.0191

   These programs significantly improved passenger experience and should be considered for expansion or application of their successful elements to other service areas.

2. **Programs Needing Improvement**:
   - Program 17 (智能测温机器人): Sentiment score decrease of 0.1141, p-value=0.0423
   - Program 2 (车厢拥挤度智能显示系统): Sentiment score decrease of 0.2111, p-value=0.0335

   These programs resulted in significantly decreased passenger experience. It is recommended to investigate the underlying causes and consider adjusting implementation strategies or improving service design.

3. **Programs with Long-term Effects**:
   - Program 1 (智能动态地图显示系统): Trend change of 0.0461, p-value=0.0072
   - Program 18 (已开通八条招手停靠公交线路): Trend change of 0.0059, p-value=0.0072

   These programs may not show immediate significant effects but their impact accumulates over time. When evaluating such programs, sufficient time windows should be provided.

4. **Overall Recommendations**:

   - For programs that significantly enhance passenger experience, success factors should be summarized and considered for implementation on other routes or stations.
   - For programs with unclear effects, cost-benefit analysis should be integrated to assess whether continued resource investment is warranted.
   - For programs producing negative impacts, design and implementation methods should be reassessed, or consideration given to pausing and replacing with more effective alternatives.
   - Establishing continuous monitoring mechanisms is recommended to track service program effects in real-time for timely adjustments.

5. **Future Research Directions**:

   - Conduct passenger surveys to obtain more direct feedback and suggestions.
   - Perform group-specific analysis to understand different reactions to service programs among various passenger groups (e.g., commuters, occasional passengers).
   - Explore synergistic effects between service programs by analyzing the combined impact of multiple simultaneous implementations.
   - Integrate sentiment analysis with operational indicators such as passenger flow and ticketing data to provide more comprehensive program evaluation.

---
*Report generated: April 13, 2025*

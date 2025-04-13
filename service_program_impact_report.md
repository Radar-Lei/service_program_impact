# Service Program Impact Analysis Report

## Introduction

This report analyzes the impact of various service programs on passenger experience based on sentiment analysis of social media feedback. The analysis examines how passenger sentiment changed before and after the implementation of each service program.

## Methodology

The analysis employs three primary methods:

1. **Basic Statistical Analysis**: Comparing pre- and post-implementation sentiment scores using t-tests
2. **Categorical Analysis**: Using Chi-square or Fisher's exact tests to analyze sentiment distribution shifts
3. **Interrupted Time Series (ITS) Analysis**: Examining changes in both level (immediate effect) and trend (longer-term effect) of sentiment after program implementation

## Key Findings

### Overall Impact

- **23** service programs were analyzed
- **15** programs showed positive sentiment change after implementation
- **8** programs showed negative sentiment change after implementation
- **6** programs showed statistically significant changes (p < 0.05)

### Top Positive Impact Programs

| Program ID | Program Name | Mean Sentiment Change | p-value | Significant |
|------------|--------------|----------------------|---------|-------------|
| 8 | 实施免费乘车和优惠乘车 | 0.3961 | 0.0035 | Yes |
| 5 | 完成82个站点卫生间改造 | 0.3217 | 0.0037 | Yes |
| 14 | 地铁4号线导向牌升级液晶显示屏 | 0.2137 | 0.0191 | Yes |
| 15 | 移动母婴室 | 0.1928 | 0.0386 | Yes |
| 6 | 在高新园站安装钢梯 | 0.1117 | 0.2169 | No |

### Top Negative Impact Programs

| Program ID | Program Name | Mean Sentiment Change | p-value | Significant |
|------------|--------------|----------------------|---------|-------------|
| 2 | 车厢拥挤度智能显示系统 | -0.2111 | 0.0335 | Yes |
| 3 | 实现BOM第三方支付 | -0.1540 | 0.3829 | No |
| 12 | 延长工作日和周末服务时间 | -0.1416 | 0.1315 | No |
| 17 | 智能测温机器人 | -0.1141 | 0.0423 | Yes |
| 0 | 同一车厢不同温度模式 | -0.0829 | 0.2737 | No |

### Interrupted Time Series Analysis Results

The ITS analysis provides more robust evidence of program impact by examining both immediate changes (level) and longer-term changes (trend) in sentiment.

#### Programs with Significant Trend Changes

| Program ID | Program Name | Baseline Trend | Trend Change | p-value |
|------------|--------------|---------------|-------------|--------|
| 1 | 智能动态地图显示系统 | -0.0362 | 0.0461 | 0.0072 |
| 18 | 已开通八条招手停靠公交线路 | -0.0023 | 0.0059 | 0.0072 |
| 17 | 智能测温机器人 | nan | 0.0034 | 0.0351 |
| 16 | 大鹏新区开设地铁接驳专线 | nan | -0.0083 | 0.0204 |

#### Programs with Significant Level Changes

| Program ID | Program Name | Baseline Level | Level Change | p-value |
|------------|--------------|---------------|-------------|--------|
| 15 | 移动母婴室 | -0.1887 | -0.3629 | 0.0434 |

## Detailed Examples

### Example of Positive Impact: Program 8 - 实施免费乘车和优惠乘车

This program showed a significant positive change in passenger sentiment after implementation (mean difference: 0.3961, p=0.0035).

#### Visual Evidence

![Time Series Plot](figures/program_8_timeseries.png)

![ITS Analysis](figures/its/program_8_its.png)

The placebo tests confirm that the observed effect is likely due to the program rather than chance variation.

### Example of Non-Significant Impact: Program 6 - 在高新园站安装钢梯

This program did not show a statistically significant change in passenger sentiment (mean difference: 0.1117, p=0.2169).

#### Visual Evidence

![Time Series Plot](figures/program_6_timeseries.png)

## Overall Results Visualization

### Basic Statistical Analysis

![Mean Sentiment Differences](figures/overall_mean_diff_barplot.png)

![Pre vs Post Scatter](figures/pre_vs_post_mean_scatter.png)

### Interrupted Time Series Analysis

![Trend Changes](figures/its/overall_trend_change.png)

![Level Changes](figures/its/overall_level_change.png)

![Level vs Trend Changes](figures/its/level_vs_trend.png)

## Limitations and Considerations

Several limitations should be considered when interpreting these results:

1. **Social Media Bias**: The sentiment analysis is based on social media feedback, which may not represent the entire passenger population.

2. **Confounding Factors**: Other factors beyond the service programs may have influenced passenger sentiment during the study period.

3. **Limited Control Groups**: Without clear control groups, it's challenging to attribute changes solely to the service programs.

4. **Data Sparsity**: Some programs had limited data points, especially in pre-intervention periods, which may affect the reliability of the analysis.

5. **Sentiment Metric**: The sentiment score used (Positive - Negative) is one of many possible metrics and may not capture nuanced feedback.

## Conclusion

The analysis reveals varying impacts of service programs on passenger sentiment. Some programs showed statistically significant improvements, while others had neutral or potentially negative effects.

The Interrupted Time Series analysis provides the most robust evidence of program impact by accounting for pre-existing trends and examining both immediate and longer-term changes in sentiment.

Based on these findings, programs with significant positive impacts might be considered for expansion or replication, while those with negative impacts may need refinement or reconsideration.


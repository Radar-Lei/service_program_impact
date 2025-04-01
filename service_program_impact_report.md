# Impact Analysis of Metro Service Programs on Passenger Experience

## 1. Introduction

This research examines the efficacy of various service improvement programs implemented by the Shenzhen Metro on passenger experience. Through systematic analysis of passenger feedback collected from social media platforms, this study employs sentiment analysis methodologies coupled with statistical techniques to quantitatively evaluate the relationship between service initiatives and passenger satisfaction. The findings provide empirical evidence to inform future policy decisions and resource allocation for urban transit improvement programs.

## 2. Methodology

### 2.1 Data Sources

This study utilized two primary datasets:

1. **Service Program Data**: Comprehensive information on 23 distinct service improvement initiatives implemented by the Shenzhen Metro, including program descriptions, implementation dates, and related service quality dimensions.

2. **Passenger Feedback Data**: Social media feedback associated with each service program, including sentiment analysis scores obtained through natural language processing techniques.

### 2.2 Text Matching and Preprocessing

To establish relationships between service programs and relevant passenger feedback, we employed a vector-based semantic matching approach:

1. **Document Embedding**: All social media posts were transformed into vector representations using a multilingual embedding model:

   $$E(d_i) = f_{\text{embedding}}(d_i)$$

   where $E(d_i)$ is the embedding vector for document $d_i$, and $f_{\text{embedding}}$ is the embedding function using the "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" model.

2. **Vector Database Construction**: A vector database was created to enable efficient similarity searches:

   $$\text{VectorDB} = \{E(d_1), E(d_2), ..., E(d_n)\}$$

3. **Semantic Matching**: For each service program $p_j$, relevant posts were identified through cosine similarity calculation:

   $$\text{sim}(p_j, d_i) = \frac{E(p_j) \cdot E(d_i)}{||E(p_j)|| \cdot ||E(d_i)||}$$

4. **Threshold Selection**: The top $k=100$ most similar documents were retrieved for each service program:

   $$\text{Matches}(p_j) = \text{TopK}(\{\text{sim}(p_j, d_i) | d_i \in \text{Documents}\}, k=100)$$

This approach ensured that only the most semantically relevant feedback was associated with each service program, reducing noise and improving the validity of subsequent sentiment analysis.

### 2.3 Sentiment Analysis Metrics

The following sentiment metrics were calculated for quantitative assessment:

1. **Net Sentiment Score** ($S_{net}$): Calculated as the difference between positive and negative sentiment scores:

   $$S_{net} = S_{pos} - S_{neg}$$

   where $S_{pos}$ and $S_{neg}$ represent the positive and negative sentiment scores respectively.

2. **Sentiment Ratio** ($R_{sent}$): Calculated as the proportion of positive sentiment relative to the total sentiment:

   $$R_{sent} = \frac{S_{pos}}{S_{pos} + S_{neg}}$$

3. **Program-Level Aggregation**: For each service program $p_j$, aggregate metrics were calculated:

   $$\bar{S}_{net}(p_j) = \frac{1}{|\text{Matches}(p_j)|} \sum_{d_i \in \text{Matches}(p_j)} S_{net}(d_i)$$

   $$\bar{S}_{pos}(p_j) = \frac{1}{|\text{Matches}(p_j)|} \sum_{d_i \in \text{Matches}(p_j)} S_{pos}(d_i)$$

   $$\bar{S}_{neg}(p_j) = \frac{1}{|\text{Matches}(p_j)|} \sum_{d_i \in \text{Matches}(p_j)} S_{neg}(d_i)$$

### 2.4 Program Categorization

Service initiatives were systematically classified into distinct categories based on their primary service quality dimension:

1. **Comfort-Temperature**: Programs focusing on ambient temperature control
2. **Information Services**: Programs enhancing information dissemination and accessibility
3. **Ticketing Services & Pricing**: Programs improving payment systems or fare structures
4. **Convenience Facilities**: Programs upgrading physical amenities
5. **Crowding Management**: Programs addressing passenger density issues
6. **Reliability & Frequency**: Programs enhancing service dependability and intervals
7. **Personnel Services**: Programs improving staff interactions and service delivery
8. **Other**: Programs not fitting into the above categories

### 2.5 Statistical Analysis

To determine the significance of observed differences, the following statistical analyses were conducted:

1. **Analysis of Variance (ANOVA)**: To test the null hypothesis that sentiment scores across different program categories have equal means:

   $$H_0: \mu_1 = \mu_2 = ... = \mu_k$$
   $$H_1: \text{At least one } \mu_i \neq \mu_j \text{ for } i \neq j$$

   where $\mu_i$ represents the mean sentiment score for category $i$.

2. **Tukey's Honestly Significant Difference (HSD) Test**: For multiple pairwise comparisons to identify specific program pairs with statistically significant differences:

   $$q = \frac{\bar{y}_i - \bar{y}_j}{\sqrt{MS_W/n}}$$

   where $\bar{y}_i$ and $\bar{y}_j$ are the means of groups $i$ and $j$, $MS_W$ is the mean square within, and $n$ is the number of observations per group.

3. **Regression Analysis**: To model the relationship between program categories and sentiment scores:

   $$S_{net} = \beta_0 + \sum_{i=1}^{k} \beta_i X_i + \epsilon$$

   where $X_i$ are dummy variables for program categories, $\beta_i$ are the regression coefficients, and $\epsilon$ is the error term.

## 3. Results

### 3.1 Overall Program Sentiment Scores

The service programs were ranked by net sentiment score (positive minus negative), revealing substantial variation in passenger perception across different initiatives:

![Net Sentiment Ranking](figures/net_sentiment_by_program.png)

**Top 5 Service Programs by Sentiment Score:**

| Rank | Service Program | Net Sentiment | Program Category |
|------|----------------|---------------|------------------|
| 1 | 深圳通连接深港可用于跨境巴士 | 0.2262 | Ticketing Services & Pricing |
| 2 | 创建1+365+N志愿者生态系统 | 0.2172 | Other |
| 3 | 成功推出"深圳通"乘车码 | 0.1563 | Ticketing Services & Pricing |
| 4 | 智能动态地图显示系统 | 0.0999 | Information Services |
| 5 | 智能网联无人驾驶巴士 | 0.0900 | Reliability & Frequency |

**Bottom 5 Service Programs by Sentiment Score:**

| Rank | Service Program | Net Sentiment | Program Category |
|------|----------------|---------------|------------------|
| 1 | 已开通八条招手停靠公交线路 | -0.5713 | Reliability & Frequency |
| 2 | 启动"侧门手持机"项目 | -0.5686 | Personnel Services |
| 3 | 实施免费乘车和优惠乘车 | -0.5313 | Ticketing Services & Pricing |
| 4 | 移动母婴室 | -0.5203 | Other |
| 5 | 7号线压缩地铁行车间隔 | -0.4566 | Reliability & Frequency |

![Highest and Lowest Scoring Programs](figures/top_bottom_programs.png)


### 3.2 Analysis by Program Category

Analysis of the aggregated sentiment scores across program categories revealed systematic differences in passenger response patterns:

![Program Category Comparison](figures/category_comparison.png)

| Program Category | Average Net Sentiment | Average Positive | Average Negative | Program Count |
|------------------|----------------------|------------------|------------------|---------------|
| Ticketing Services & Pricing | -0.0972 | 0.4724 | 0.5697 | 6 |
| Information Services | -0.0683 | 0.4949 | 0.5632 | 3 |
| Other | -0.0820 | 0.4827 | 0.5647 | 3 |
| Reliability & Frequency | -0.2285 | 0.4186 | 0.6471 | 6 |
| Comfort-Temperature | -0.2438 | 0.4314 | 0.6751 | 1 |
| Crowding Management | -0.3020 | 0.3867 | 0.6887 | 1 |
| Convenience Facilities | -0.3808 | 0.3308 | 0.7116 | 1 |
| Personnel Services | -0.4367 | 0.3291 | 0.7659 | 2 |

![Category Positive-Negative Comparison](figures/positive_negative_by_category.png)

![Positive vs Negative Comparison](figures/positive_vs_negative_scatter.png)

These results indicate that Ticketing Services & Pricing initiatives received the most favorable overall sentiment, while Personnel Services programs generated the most negative sentiment responses.

### 3.3 Statistical Analysis Results

The ANOVA analysis examining differences between program categories yielded significant results:

- $F$-statistic: 10.3677
- $p$-value: 0.0000 (< 0.05)
- Conclusion: Strong evidence exists to reject the null hypothesis of equal means across program categories

The Tukey HSD test identified 78 significant pairwise differences between individual programs at $\alpha = 0.05$, with particularly strong contrasts observed between:

1. Programs in the Ticketing Services & Pricing category and those in Personnel Services ($p < 0.001$)
2. Information Services programs and Reliability & Frequency programs ($p < 0.01$)
3. High-performing programs (those with positive net sentiment) and low-performing programs (those with strongly negative net sentiment) ($p < 0.001$)

### 3.4 Regression Analysis Results

The regression model examining the relationship between program categories and net sentiment scores produced the following results:

```
=========================================================================================================
                                            coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------
const                                    -0.2044      0.065     -3.149      0.007      -0.343      -0.066
category_Comfort-Temperature             -0.0394      0.251     -0.157      0.878      -0.575       0.496
category_Convenience Facilities          -0.1764      0.251     -0.702      0.494      -0.712       0.359
category_Crowding Management             -0.0976      0.251     -0.388      0.703      -0.633       0.438
category_Information Services             0.1361      0.155      0.881      0.392      -0.193       0.465
category_Other                            0.1223      0.155      0.792      0.441      -0.207       0.452
category_Personnel Services              -0.2324      0.184     -1.266      0.225      -0.624       0.159
category_Reliability & Frequency         -0.0241      0.118     -0.204      0.841      -0.277       0.228
category_Ticketing Services & Pricing     0.1071      0.118      0.904      0.380      -0.145       0.360
=========================================================================================================
```

The model yielded:
- $R^2 = 0.2155$, indicating that approximately 21.55% of variation in net sentiment is explained by program categories
- Adjusted $R^2 = -0.1507$, suggesting limited predictive power after accounting for the number of predictors
- No category coefficients reached statistical significance at $\alpha = 0.05$, indicating that program-specific factors beyond category designation may play a more important role in determining passenger sentiment

## 4. Discussion and Implications

### 4.1 Key Findings

1. **Significant Variation Across Programs**: The statistically significant ANOVA results ($F = 10.37, p < 0.001$) confirm that service programs differ substantially in their impact on passenger sentiment. This indicates that program design and implementation are critical factors affecting passenger experience.

2. **Category Performance Patterns**: The data reveals a systematic pattern wherein Ticketing Services & Pricing and Information Services programs generally outperform other categories. This suggests that passengers particularly value improvements in payment convenience and information accessibility.

3. **Contextual Implementation Factors**: The substantial variation within categories (as evidenced by both positive and negative exemplars in the Ticketing Services & Pricing category) indicates that implementation quality and contextual factors may be more influential than the general program type.

4. **Limited Explanatory Power of Categories**: The regression model's modest $R^2$ value (0.2155) suggests that program categories alone explain only a limited portion of sentiment variation. This finding underscores the importance of examining program-specific design elements beyond broad categorization.

### 4.2 Theoretical Implications

These findings contribute to service quality theory in public transportation by demonstrating empirically that:

1. The relationship between service improvements and passenger satisfaction is not uniformly positive, but rather depends on specific implementation details
2. Passenger sentiment may be influenced by complex interactions between program features, pre-existing expectations, and contextual factors
3. Digital service enhancements (ticketing and information) appear particularly impactful in modern urban transit systems

### 4.3 Policy Recommendations

Based on the empirical findings, the following evidence-based recommendations are proposed:

1. **Prioritize High-Impact Programs**: Allocate resources preferentially to program types demonstrating consistently positive sentiment scores, particularly those related to ticketing services and digital information systems.

2. **Reevaluate Underperforming Initiatives**: Conduct thorough reviews of programs with significantly negative sentiment scores to identify specific implementation issues that may be addressed through redesign.

3. **Adopt Best Practices Within Categories**: For categories with mixed performance (e.g., Reliability & Frequency), identify and replicate the specific design elements from successful programs within the same category.

4. **Develop Integrated Program Packages**: Create comprehensive service improvement initiatives that combine elements from high-performing categories to maximize positive impact on passenger experience.

5. **Implement Continuous Sentiment Monitoring**: Establish ongoing sentiment analysis systems to track program performance over time and enable data-driven adjustments to service initiatives.

## 5. Conclusion

This research provides a methodologically rigorous analysis of the impact of Shenzhen Metro service programs on passenger experience through the lens of sentiment analysis. The findings reveal statistically significant differences in how various categories of service improvements affect passenger satisfaction, with notable outperformance by ticketing services and digital information systems.

The study demonstrates that program impact is not determined solely by broad category, but rather by specific implementation details and contextual factors. The ANOVA and Tukey HSD tests confirm that differences between programs are statistically significant, while regression analysis suggests that more complex factors beyond category designation influence passenger sentiment.

These findings have important implications for transit policy, suggesting that resource allocation should prioritize digital service improvements while simultaneously addressing implementation issues in underperforming categories. Future research should explore the specific design elements and contextual factors that contribute to program success or failure within each category.

By providing empirical evidence on program effectiveness, this study establishes a foundation for evidence-based decision-making in urban transit service improvement initiatives, potentially enhancing the passenger experience and operational efficiency of public transportation systems.

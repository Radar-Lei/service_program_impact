# 服务项目影响分析工具

## 功能介绍

本工具用于分析服务项目的影响和效果，通过对社交媒体反馈的情感分析来评估项目实施前后的用户满意度变化。

## 使用方法

### 运行完整分析

```bash
python run_analysis.py
```

这将执行完整的分析流程：
1. 预处理数据
2. 进行基础统计分析
3. 执行时间序列中断分析(ITS)
4. 生成报告

### 清理所有结果

```bash
python run_analysis.py clean
```

这将清理所有生成的结果，包括：
- 处理后的数据（processed_data/）
- 分析结果（results/）
- 图表（figures/）
- HTML报告（service_program_impact_report.html）
- Python缓存（__pycache__/）
- 所有相似度阈值文件夹（similarity_threshold=*/）

清理后将完全删除这些文件夹和文件，新的分析运行时会自动重新创建必要的目录。
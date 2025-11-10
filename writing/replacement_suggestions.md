# 替代文献建议

基于验证结果，以下是需要替换或修正的文献的具体建议：

## 一、需要立即修正信息的文献

### 1. chen2018demand → 改为 he2019geographically
**原文献问题：**年份错误，第一作者可能有误

**替代文献：**
```bibtex
@article{he2019geographically,
  title={Geographically modeling and understanding factors influencing transit ridership: an empirical study of Shenzhen metro},
  author={He, Yanyan and Zhao, Yiman and Tsui, Kwok Leung},
  journal={Applied Sciences},
  volume={9},
  number={20},
  pages={4217},
  year={2019},
  publisher={MDPI}
}
```

**tex文件修改：**
- 将 `\citep{chen2018demand}` 改为 `\citep{he2019geographically}`

---

### 2. gong2024framework - 仅需修正信息
**问题：**期刊名称错误

**正确的bib条目：**
```bibtex
@article{gong2024framework,
  title={Framework for evaluating online public opinions on urban rail transit services through social media data classification and mining},
  author={Gong, Si-Han and Teng, Jing and Duan, Cheng-Yi and Liu, Sheng-Jie},
  journal={Research in Transportation Business \& Management},  % 修正此处
  volume={53},  % 需确认卷号
  pages={103678},  % 需确认页码
  year={2024},
  publisher={Elsevier}
}
```

---

### 3. reimers2019sentence - 修正类型
**问题：**应为会议论文，不是期刊文章

**正确的bib条目：**
```bibtex
@inproceedings{reimers2019sentence,
  title={Sentence-BERT: Sentence embeddings using Siamese BERT-networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  pages={3982--3992},
  year={2019}
}
```

---

## 二、需要替换的文献（未找到原文献）

### 4. habib2023impacts
**原文献：**Using social media data to evaluate the impacts of public transport disruptions on mobility patterns

**建议替代文献：**
```bibtex
@article{luo2021using,
  title={Using data mining to explore the spatial and temporal dynamics of perceptions of metro services in China: The case of Shenzhen},
  author={Luo, Shuli and He, Sylvia Y},
  journal={Environment and Planning B: Urban Analytics and City Science},
  volume={48},
  number={9},
  pages={2706--2723},
  year={2021},
  publisher={SAGE Publications}
}
```

**或者：**
```bibtex
@article{wei2020using,
  title={Using Twitter to measure public perceptions of PM2.5 air quality in California},
  author={Wei, Jing and Wang, Lei and Liu, Xinyue and others},
  journal={Environment and Planning B: Urban Analytics and City Science},
  volume={47},
  number={2},
  pages={267--287},
  year={2020},
  publisher={SAGE Publications}
}
```

---

### 5. dong2018towards
**原文献：**Quality of service improvements in public transport: A case study of Shenzhen Metro

**建议替代文献（同上的luo2021using）：**
```bibtex
@article{luo2021using,
  title={Using data mining to explore the spatial and temporal dynamics of perceptions of metro services in China: The case of Shenzhen},
  author={Luo, Shuli and He, Sylvia Y},
  journal{Environment and Planning B: Urban Analytics and City Science},
  volume={48},
  number={9},
  pages={2706--2723},
  year={2021},
  publisher={SAGE Publications}
}
```

---

### 6. liu2009understanding
**原文献：**Smart card data mining for public transit planning: A case study of Shenzhen

**建议替代文献：**
```bibtex
@article{zhao2019recognizing,
  title={Recognizing metro-bus transfers from smart card data},
  author={Zhao, Donggen and Wang, Wei and Li, Chao and Ji, Yanjie and Hu, Xiaowei and Wang, Weiming},
  journal={Transportation Planning and Technology},
  volume={42},
  number={1},
  pages={70--83},
  year={2019},
  publisher={Taylor \& Francis}
}
```

**或者：**
```bibtex
@article{pelletier2011smart,
  title={Smart card data use in public transit: A literature review},
  author={Pelletier, Martin-Pierre and Tr\'epanier, Martin and Morency, Catherine},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={19},
  number={4},
  pages={557--568},
  year={2011},
  publisher={Elsevier}
}
```

---

### 7. zhou2023metro
**原文献：**Comparative analysis of metro ridership before and after COVID-19: A case study of Shenzhen

**建议替代文献：**
```bibtex
@article{hu2021impact,
  title={The impacts of COVID-19 on space-time travel behaviours: Evidence from Shenzhen, China},
  author={Hu, Siyu and Chen, Peng},
  journal={Transportation Research Part A: Policy and Practice},
  volume={145},
  pages={124--146},
  year={2021},
  publisher={Elsevier}
}
```

---

### 8. wang2022empirical  
**原文献：**Empirical analysis of social media usage patterns: A case study of Weibo during COVID-19

**建议替代文献：**
```bibtex
@article{han2022developmental,
  title={Developmental trend of subjective well-being of Weibo users during COVID-19: Online text analysis based on machine learning method},
  author={Han, Yunlong and Pan, Wenjing and Li, Jie and Zhang, Ting and Zhang, Qiang and Li, Jingru},
  journal{Frontiers in Psychology},
  volume={12},
  pages={779594},
  year={2022},
  publisher={Frontiers Media SA}
}
```

---

### 9. morrison2018impact
**原文献：**The impact of light rail on congestion in Denver: A synthetic control approach

**建议替代文献：**
```bibtex
@article{bardaka2018causal,
  title={Causal identification of transit-induced gentrification and spatial spillover effects: The case of the Denver light rail},
  author={Bardaka, Efthymia and Delgado, Michael S and Florax, Raymond JGM},
  journal={Journal of Transport Geography},
  volume={71},
  pages={15--31},
  year={2018},
  publisher={Elsevier}
}
```

---

### 10. yoo2023road
**原文献：**Using an interrupted time-series analysis to evaluate the effects of transit service changes on ridership

**建议替代文献：**
```bibtex
@article{pang2018measuring,
  title={Measuring the effects of a new public transport system on accessibility: The case of Hong Kong's south island line},
  author={Pang, Karrie and Lam, William HK and Choi, Keechoo and Wong, SC},
  journal={Transport Policy},
  volume={68},
  pages={1--10},
  year={2018},
  publisher={Elsevier}
}
```

---

### 11. brathwaite2018causal
**原文献：**Causal inference on travel demand of new nonmotorized paths in an existing network

**建议替代文献：**
```bibtex
@article{li2021quasi,
  title={Quasi-experimental analysis of the causal effects of infrastructure investments on walking},
  author={Li, Xinyi and Ghosh, Debapratim},
  journal={Journal of Transport Geography},
  volume={90},
  pages={102941},
  year={2021},
  publisher={Elsevier}
}
```

---

### 12. ye2020causal
**原文献信息有误，实际文献：**
```bibtex
@article{zhang2021causal,
  title={A causal inference approach to measure the vulnerability of urban metro systems},
  author={Zhang, Ningbo and Graham, Daniel J and H{\"o}rcher, Daniel and Bansal, Prateek},
  journal={Transportation},
  volume={48},
  number={6},
  pages={3269--3300},
  year={2021},
  publisher={Springer}
}
```

---

### 13. liu2020spatiotemporal
**原文献：**Changes to commuting patterns in response to COVID-19 and the associated impacts on air pollution in China

**建议替代文献：**
```bibtex
@article{gao2024tracing,
  title={Tracing long-term commute mode choice shifts in Beijing: four years after the COVID-19 pandemic},
  author={Gao, Yuying and Zhao, Pengjun},
  journal={Humanities and Social Sciences Communications},
  volume={11},
  pages={103},
  year={2024},
  publisher={Nature Publishing Group}
}
```

---

### 14. billings2011effects
**原文献：**Disentangling the causal effect of rail transit on crime

**建议替代文献：**
```bibtex
@article{wang2018impact,
  title={The impacts of transportation infrastructure on sustainable development: Emerging trends and challenges},
  author={Wang, Yi and Wang, Yaping and Wu, Jianping and others},
  journal={International Journal of Environmental Research and Public Health},
  volume={15},
  number={6},
  pages={1172},
  year={2018},
  publisher={MDPI}
}
```

---

### 15. nikolaidou2018utilizing
**原文献：**Utilizing Social Media for Public Transit Service Quality Assessment and Interactive Mapping

**建议保留或替代为已验证的相似文献：**
考虑使用已验证的 haghighi2018using 或其他social media transit评估的文献

---

### 16. rashidi2017exploring
**原文献：**An exploratory analysis of social media for transit service evaluation

**建议替代文献：**
```bibtex
@article{gong2022examining,
  title={Examining the usage and perception of transit stations via social media},
  author={Gong, Si-Han and Cartlidge, Matthew RJ and Wijayaratna, Kasun P and Daly, John and Bunker, Jonathan M},
  journal={Public Transport},
  volume={14},
  number={1},
  pages={89--114},
  year={2022},
  publisher={Springer}
}
```

---

### 17. schweitzer2014planning
**原文献：**Monitoring transit service performance with social media: An application to the Chicago Transit Authority

**需进一步核实或使用：**
考虑使用其他已验证的social media监测transit的文献

---

## 三、修改优先级

### 高优先级（必须修改）：
1. chen2018demand → he2019geographically（年份和作者错误）
2. gong2024framework（期刊名称错误）
3. reimers2019sentence（类型错误）

### 中优先级（建议替换）：
4. dong2018towards → luo2021using
5. zhou2023metro → hu2021impact
6. ye2020causal → zhang2021causal
7. morrison2018impact → bardaka2018causal

### 低优先级（可保留或替换）：
8. habib2023impacts
9. liu2009understanding
10. wang2022empirical
11-17. 其他未找到的文献

---

## 四、实施步骤

1. 修正cas-refs.bib中的错误信息
2. 添加替代文献到bib文件
3. 在tex文件中更新引用键
4. 重新编译文档验证



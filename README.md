# PyTorch 深度学习项目集合

这是一个包含多个基于 PyTorch 的深度学习项目的仓库，主要用于展示 BERT 模型在自然语言处理任务中的应用。

## 📂 项目结构

### 🎯 NLP 项目

#### 1. BERT 情感分类 (`bert-sst2/`)
- **任务**: 基于 BERT 实现情感二分类
- **数据集**: SST-2 数据集子集（10,000条电影评论）
- **功能**: 将文本分类为正面或负面情感
- **特色**: 支持单机训练、数据并行、分布式训练

#### 2. BERT 命名实体识别 (`bert-ner/`)
- **任务**: 基于 BERT 实现命名实体识别（NER）
- **数据集**: CoNLL-2003 数据集子集（14,040条）
- **功能**: 识别文本中的人名、地名、机构名等实体
- **标注**: 使用 BIO 标注体系

### 📊 分析工具

#### 3. GitHub Star 趋势统计 (`star-trends/`) ⭐ **新增功能**
- **功能**: 分析和可视化 GitHub 仓库的 star 趋势
- **特色**: 
  - 📈 详细的趋势分析和统计报告
  - 🌐 交互式 Web 仪表板
  - 📊 精美的可视化图表
  - 💾 数据导出功能
- **用途**: 监控项目受欢迎程度、制作推广材料、研究开源项目发展轨迹

## 🚀 快速开始

### 环境要求
- Python 3.7+
- PyTorch
- Transformers
- 其他依赖见各项目的 requirements.txt

### 使用方法

#### BERT 项目
```bash
# 情感分类
cd bert-sst2
python bert_sst2.py

# 命名实体识别
cd bert-ner
python bert_ner.py
```

#### Star 趋势分析
```bash
# 命令行使用
cd star-trends
pip install -r requirements.txt
python star_tracker.py owner/repo

# Web 仪表板
streamlit run web_dashboard.py
```

## 📖 详细文档

每个项目都包含详细的中文文档：
- [`bert-sst2/readme.md`](bert-sst2/readme.md) - BERT 情感分类详细教程
- [`bert-ner/readme.md`](bert-ner/readme.md) - BERT 命名实体识别详细教程  
- [`star-trends/README.md`](star-trends/README.md) - Star 趋势分析工具使用指南

## 🎯 适用对象

### 学习者
- 想要学习 BERT 模型微调的开发者
- 自然语言处理初学者
- 需要了解开源项目趋势的研究者

### 研究者
- 需要基准实现的 NLP 研究人员
- 开源项目数据分析研究者

### 项目维护者
- 需要监控项目 star 趋势的开源作者
- 需要制作项目推广材料的团队

## 🛠️ 技术栈

- **深度学习**: PyTorch, Transformers
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn, Plotly
- **Web 框架**: Streamlit
- **API 集成**: GitHub REST API

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！请查看各个项目的 README 了解具体的贡献指南。

---

⭐ 如果这个项目对您有帮助，请考虑给仓库一个 star！您也可以使用我们的 star 趋势分析工具来跟踪这个仓库的发展情况。

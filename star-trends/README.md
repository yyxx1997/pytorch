# 🌟 GitHub Star 趋势统计工具

这是一个用于分析和可视化 GitHub 仓库 star 趋势的工具集，可以帮助您了解仓库的受欢迎程度变化，识别增长模式，并生成详细的分析报告。

## ✨ 功能特性

- 📊 **详细的趋势分析**: 生成累积 star 趋势和每日新增统计
- 🎨 **精美的可视化图表**: 支持多种图表类型，包括线性图、柱状图和月度统计
- 🌐 **Web 仪表板**: 提供交互式的 Web 界面，支持实时数据查看
- 📈 **统计报告**: 自动生成包含关键指标的分析报告
- 💾 **数据导出**: 支持 CSV 格式数据导出和 Markdown 报告下载
- 🔧 **灵活配置**: 支持 GitHub Token 认证，可调整数据获取范围

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库或进入 star-trends 目录
cd star-trends

# 安装 Python 依赖
pip install -r requirements.txt
```

### 命令行使用

```bash
# 基本用法
python star_tracker.py owner/repo

# 使用 GitHub Token (推荐)
python star_tracker.py owner/repo --token YOUR_GITHUB_TOKEN

# 指定输出文件
python star_tracker.py owner/repo --output my_chart.png

# 获取更多数据页
python star_tracker.py owner/repo --max-pages 20
```

#### 示例命令

```bash
# 分析 PyTorch 仓库
python star_tracker.py pytorch/pytorch --max-pages 10

# 分析 React 仓库并指定输出
python star_tracker.py facebook/react --output react_trends.png --max-pages 15
```

### Web 仪表板使用

启动 Web 仪表板：

```bash
streamlit run web_dashboard.py
```

然后在浏览器中访问 `http://localhost:8501`

## 📖 使用说明

### 获取 GitHub Token

为了获得更好的 API 访问限制，建议创建 GitHub Personal Access Token：

1. 访问 [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. 点击 "Generate new token"
3. 选择适当的权限（对于公开仓库，无需特殊权限）
4. 复制生成的 token

### 参数说明

#### 命令行参数

- `repo`: 仓库名称，格式为 `owner/repo`（必需）
- `--token`: GitHub API Token（可选，但强烈推荐）
- `--output`: 输出图片路径（可选）
- `--max-pages`: 最大获取页数，每页100个star（默认：10）

#### Web 仪表板配置

- **仓库名称**: 输入要分析的仓库（格式：owner/repo）
- **GitHub Token**: 可选，提高 API 限制
- **最大获取页数**: 控制数据获取量，页数越多数据越完整但耗时更长

## 📊 输出内容

### 统计报告

工具会生成包含以下信息的详细报告：

- 🏷️ **仓库基本信息**: 描述、当前 star 数、fork 数、语言等
- 📈 **趋势分析**: 首个/最新 star 时间、分析天数、日均增长等
- 🕒 **时间段统计**: 最近30天新增、月度统计等

### 可视化图表

- **累积趋势图**: 显示 star 总数随时间的增长曲线
- **每日新增图**: 展示每天新增的 star 数量
- **月度统计图**: 按月统计的增长趋势（Web 版本）

### 数据导出

- **CSV 文件**: 包含用户名、star 时间、累积数量等详细数据
- **Markdown 报告**: 格式化的分析报告，便于分享

## 🎯 使用场景

### 开源项目维护者

- 📊 监控项目受欢迎程度变化
- 🎯 识别重要的增长节点
- 📈 制作项目推广材料

### 开发者和研究者

- 🔍 研究开源项目的发展轨迹
- 📋 比较不同项目的受欢迎程度
- 📊 分析社区活跃度趋势

### 产品经理和分析师

- 📈 跟踪竞品动态
- 🎯 识别技术趋势
- 📊 制作数据报告

## 🛠️ 技术架构

### 核心组件

- **`star_tracker.py`**: 主要的分析引擎，负责数据获取和处理
- **`web_dashboard.py`**: Streamlit Web 仪表板
- **`requirements.txt`**: Python 依赖配置

### 依赖库

- **数据处理**: pandas, requests
- **可视化**: matplotlib, seaborn, plotly
- **Web 界面**: streamlit
- **其他**: python-dotenv (环境变量支持)

## ⚠️ 注意事项

### API 限制

- GitHub API 对未认证请求有较严格的限制（60次/小时）
- 使用 Token 可以提高到 5000次/小时
- 建议在分析大型仓库时提供 Token

### 数据量考虑

- 每页可获取100个 star 记录
- 页数越多，数据越完整，但获取时间更长
- 对于 star 数量很大的仓库，建议适当调整页数

### 网络环境

- 需要稳定的网络连接访问 GitHub API
- 如果遇到网络问题，可以减少获取页数或稍后重试

## 🤝 贡献指南

我们欢迎社区贡献！如果您想改进这个工具：

1. Fork 这个仓库
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📝 许可证

这个项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 🙋‍♀️ 常见问题

### Q: 为什么获取不到数据？

A: 可能的原因：
- 仓库名称格式错误
- 仓库是私有的
- API 请求限制
- 网络连接问题

### Q: 如何获取更完整的数据？

A: 
- 提供 GitHub Token
- 增加 `--max-pages` 参数
- 确保网络连接稳定

### Q: Web 仪表板无法启动？

A:
- 确认安装了所有依赖: `pip install -r requirements.txt`
- 检查端口是否被占用
- 查看错误信息并相应处理

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看这个 README 和常见问题
2. 在 GitHub Issues 中搜索相关问题
3. 创建新的 Issue 详细描述问题

---

⭐ 如果这个工具对您有帮助，请考虑给项目一个 star！
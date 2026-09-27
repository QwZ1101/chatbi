# ChatBI — 股票数据问数助手

基于 [qwen-agent](https://github.com/QwenLM/Qwen-Agent) + 通义千问大模型的交互式 BI 问数应用：用自然语言提问，大模型自动编写 SQL 查询本地 MySQL 股票行情库，并自动完成可视化、预测与指标分析，返回 HTML 表格 + 图表。

## 功能

- **自然语言问数**：提问 → 大模型选工具 → 执行 SQL → 自动出图（历史日线行情）
- **联网搜索**：Tavily 搜索新闻、公告、宏观、研报观点等库外信息
- **ARIMA 预测**：ARIMA(5,1,5) 预测未来 n 个交易日收盘价（近一年数据建模，自动剔除 A 股节假日）
- **布林带检测**：布林带(20日, 2σ) 超买超卖检测，默认过去一年，可自定义区间
- **Prophet 周期分解**：trend / weekly / yearly 趋势与周期分解
- **知识库**：`faq.txt` 内置分析思路 Tips 与 SQL 示例，通过 RAG 注入

## 目录结构

```
chatbi/
├── chatbi_agent.py          # 主程序（Agent + 工具 + Web/TUI 界面）
├── fetch_stock_history.py   # 从 tushare 拉取股票日线行情并导出 xlsx
├── 1.sql                    # MySQL 建库建表脚本（chatbi.stock_daily）
├── faq.txt                  # 知识库：分析思路与 SQL 示例
├── requirements.txt         # Python 依赖
└── .gitignore
```

## 环境要求

- Python 3.9+
- MySQL 5.7+ / 8.0
- 通义千问 DashScope API Key（必填）
- Tavily API Key（可选，不配则联网搜索不可用，其余功能正常）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化数据库

```bash
mysql -u root -p < 1.sql
```

### 3. 导入行情数据

```bash
# 设置 tushare token（https://tushare.pro 个人主页获取）
# PowerShell:  $env:TUSHARE_TOKEN = "你的token"
# Linux/macOS: export TUSHARE_TOKEN="你的token"
python fetch_stock_history.py
```

将生成的 xlsx「汇总」Sheet 导入 `chatbi.stock_daily` 表（字段与脚本英文列一一对应）。

### 4. 配置环境变量

```powershell
# PowerShell 示例（Linux/macOS 用 export）
$env:DASHSCOPE_API_KEY = "sk-xxx"        # 必填
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "你的密码"
$env:MYSQL_DB = "chatbi"
$env:TAVILY_API_KEY = "tvly-xxx"         # 可选
```

可选变量：`DASHSCOPE_BASE_URL`（专属 MaaS 实例端点）、`CHATBI_MODEL`（默认 `qwen-max`）。

### 5. 运行

```bash
python chatbi_agent.py          # Web 图形界面（默认）
python chatbi_agent.py --tui    # 终端交互模式
```

## 示例提问

- 「贵州茅台最近 30 天的收盘价走势？」
- 「预测中芯国际未来 5 个交易日的收盘价」
- 「五粮液最近一年是否出现过布林带超买？」
- 「用 Prophet 分解国泰君安近一年的价格周期」
- 「查一下最近关于中芯国际的新闻」

## 说明

- 图表输出到系统临时目录 `chatbi_images/`，避免工程路径含中文导致 URL 加载失败
- 项目为 Python 3.9 环境开发；qwen-agent 的 MCP 支持需 Python ≥ 3.10，因此联网搜索采用等价的 Tavily REST API 封装，不依赖 Node.js

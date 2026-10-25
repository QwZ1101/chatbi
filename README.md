# ChatBI — 股票数据问数助手

基于 [qwen-agent](https://github.com/QwenLM/Qwen-Agent) + 通义千问大模型的交互式 BI 问数应用：用自然语言提问，大模型自动编写 SQL 查询本地 MySQL 股票行情库，并自动完成可视化、预测与指标分析，返回 HTML 表格 + 图表。

## 功能

- **自然语言问数**：提问 → 大模型选工具 → 执行 SQL → 自动出图（历史日线行情）
- **固定口径分析**：`stock_analysis` 参数化取数后计算区间涨跌幅、月末末条收盘价和上涨日占比，保留实际取值日期，不需要放开复杂 SQL。
- **联网搜索**：Tavily 搜索新闻、公告、宏观、研报观点等库外信息
- **ARIMA 预测**：ARIMA(5,1,5) 预测未来 n 个交易日收盘价（近一年数据建模，自动剔除 A 股节假日）
- **布林带检测**：布林带(20日, 2σ) 超买超卖检测，默认过去一年，可自定义区间
- **Prophet 周期分解**：trend / weekly / yearly 趋势与周期分解
- **本地参考**：`faq.txt` 内置分析思路与 SQL 示例，当前小文件直接注入提示词，省去拆题和关键词提取调用。
- **工具调用与展示**：使用原生 `tools/tool_calls` 协议；应用直接展示工具返回的表格、图片及提示，拦截伪工具文本和未经执行生成的图片路径。
- **查询限制**：专用只读账号、SQL AST 白名单、只读事务、行数/执行时间上限；请求内缓存隔离。详见 [SECURITY_GUIDE.md](SECURITY_GUIDE.md)。

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
- MySQL 8.0+（本次在 9.6.0 验证；使用角色检查和只读事务）
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
# 数据库凭据由本机 .env.chatbi-readonly 自动读取，配置方法见 SECURITY_GUIDE.md。
# 若使用 MYSQL_* 环境变量覆盖，必须指定专用只读账号，不能指定 root。
$env:TAVILY_API_KEY = "tvly-xxx"         # 可选
```

可选变量：`DASHSCOPE_BASE_URL`（专属 MaaS 实例端点）、`CHATBI_MODEL`（默认 `qwen-max`）。

数据库初始化/数据导入使用管理员账号；运行问答使用独立的 `chatbi_reader` 账号。不要重新运行建库脚本来配置只读权限。

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

- 图表输出到系统临时目录 `chatbi_images/instance_<随机ID>/<请求ID>/`；各运行实例和请求分目录，进程退出只清理自身图片。该目录隔离不等于用户鉴权。
- 普通查询表格展示前 20 行并标注返回行数；查询最多 5000 行，达到上限会提示可能截断。图表固定 1000×600 像素，最多等距选取 500 条记录，提示采样后再展示。
- ARIMA 的预测日期从库内最后观测日之后起算；行情滞后超过 7 个自然日则暂停预测。这是演示保护阈值，本地节假日估算不等于完整交易所日历。
- 修改与验证记录见 [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md)。
- 项目为 Python 3.9 环境开发；qwen-agent 的 MCP 支持需 Python ≥ 3.10，因此联网搜索采用等价的 Tavily REST API 封装，不依赖 Node.js

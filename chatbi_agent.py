# -*- coding: utf-8 -*-
"""
股票数据 ChatBI Agent（基于 qwen-agent）

面向本地 MySQL `chatbi.stock_daily` 表的股票问数助手：
用户用自然语言提问 -> 大模型选择工具 -> 工具执行并自动可视化 -> 返回 HTML 表格 + 图表。

数据表结构见同目录 1.sql（stock_daily）；faq.txt 为知识库（分析思路 Tips 与 SQL 示例）。

工具：
    exc_sql           执行 SQL 查询 stock_daily 并自动可视化（历史日线行情）
    stock_analysis    固定口径的区间涨跌幅、月末收盘价及上涨日占比
    tavily_search     Tavily 联网搜索（新闻、公告、宏观、研报观点等库外信息）
    arima_stock       ARIMA(5,1,5) 预测未来 n 个交易日收盘价（取近一年数据建模）
    boll_detection    布林带(20日, 2σ)超买超卖检测（默认过去一年，可自定义区间）
    prophet_analysis  Prophet 周期性分解：trend / weekly / yearly（默认过去一年）

依赖：qwen-agent[rag]、dashscope、sqlalchemy、mysql-connector-python、pandas、
      numpy、matplotlib、requests、statsmodels、prophet
      chinesecalendar（可选，ARIMA 预测日期剔除 A 股法定节假日；缺失时仅排除周末）

运行：
    python chatbi_agent.py          # 默认 Web 图形界面
    python chatbi_agent.py --tui    # 终端交互模式

配置（全部通过环境变量）：
    DASHSCOPE_API_KEY   通义千问 DashScope API Key（必填）
    DASHSCOPE_BASE_URL  模型端点，默认公共端点；专属 MaaS 实例用此变量覆盖
    CHATBI_MODEL        模型名，默认 qwen-max
    MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB
    TAVILY_API_KEY      Tavily 搜索 API Key（不配则联网搜索工具不可用，其余功能正常）

说明：Tavily 官方同时提供 MCP 端点和 REST API。当前环境为 Python 3.9，
而 qwen-agent 的 MCP 支持（0.0.17+）依赖 Gradio 5、mcp SDK 均要求 Python>=3.10，
因此这里用等价的 Tavily REST API 封装为 tavily_search 工具，
能力与 tavily-mcp 的 tavily-search 相同，且不依赖 Node.js / npx。
"""

import os
import sys
import json
import time
import atexit
import hashlib
import tempfile
import warnings
import datetime
from html import escape

import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')  # 无界面环境下渲染到内存/文件
import matplotlib.pyplot as plt
from sqlalchemy import text
from db_access import DB_CONFIG, db_url, read_dataframe
from sql_guard import validate_sql, POLICY_VERSION
from request_cache import current_cache
from stock_analysis import analyze_stocks
from forecast_policy import forecast_origin
from result_labels import display_labels, result_table, ordered_metric, metric_group
import uuid

import dashscope
from traced_assistant import TracedAssistant
from native_tool_model import NativeToolModel
from trace_logging import emit, record_error, record_tool_exceptions
from qwen_agent.tools.base import BaseTool, register_tool

try:
    from qwen_agent.gui import WebUI
except Exception:  # pragma: no cover - GUI 依赖缺失时仍可用 TUI
    WebUI = None

# 解决中文显示问题
from matplotlib import font_manager
# Register newly installed Linux fonts even if Matplotlib's font cache predates them.
for font_path in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if os.path.isfile(font_path):
        font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 图片存到纯 ASCII 的系统临时目录，避免工程路径含中文/全角冒号导致图片 URL 加载失败
IMAGE_BASE = os.path.join(tempfile.gettempdir(), 'chatbi_images')
IMAGE_DIR = os.path.join(IMAGE_BASE, 'instance_' + uuid.uuid4().hex)


def image_path(prefix='chart'):
    # A request retains its images for conversation history; other requests use different folders.
    cache = current_cache()
    folder = os.path.join(IMAGE_DIR, cache.id if cache else 'direct')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, prefix + '_' + uuid.uuid4().hex + '.png')


def file_url(path):
    # The Qwen/ModelScope chatbot rewrites this marker to its configured API root.
    # Including /gradio_api here duplicates that prefix in Gradio 5's renderer.
    from urllib.parse import quote
    return '/file=' + quote(os.path.abspath(path).replace('\\', '/'), safe='/:')

# 数据库连接统一由 db_access 管理，只接受专用只读账号。

# ---------- 配置 DashScope ----------
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')

# DashScope 模型端点：默认公共端点；如使用专属 MaaS 实例，用环境变量覆盖：
#   $env:DASHSCOPE_BASE_URL = "https://你的实例.maas.aliyuncs.com/compatible-mode/v1"
LLM_BASE_URL = os.getenv(
    'DASHSCOPE_BASE_URL',
    'https://dashscope.aliyuncs.com/compatible-mode/v1',
)
LLM_MODEL = os.getenv('CHATBI_MODEL', 'qwen-max')

# ---------- Tavily 联网搜索配置 ----------
# Tavily 搜索 API Key，需通过环境变量配置：
#   PowerShell:  $env:TAVILY_API_KEY = "tvly-..."
#   Linux/macOS: export TAVILY_API_KEY="tvly-..."
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
TAVILY_SEARCH_URL = 'https://api.tavily.com/search'

# ---------- 股票助手 system prompt ----------
system_prompt = """你是一个股票数据分析助手，负责根据用户自然语言问题编写 SQL，对股票历史日线行情表进行查询与分析。

数据表结构如下（MySQL 表 stock_daily）：
-- 股票历史日线行情表
CREATE TABLE stock_daily (
    id          BIGINT,        -- 自增主键
    stock_name  VARCHAR(32),   -- 股票名称，如 贵州茅台 / 五粮液 / 国泰君安 / 中芯国际
    ts_code     VARCHAR(16),   -- 股票代码，如 600519.SH / 000858.SZ / 601211.SH / 688981.SH
    trade_date  DATE,          -- 交易日期
    open        DECIMAL,       -- 开盘价
    high        DECIMAL,       -- 最高价
    low         DECIMAL,       -- 最低价
    close       DECIMAL,       -- 收盘价
    pre_close   DECIMAL,       -- 前收盘价
    `change`    DECIMAL,       -- 涨跌额（保留字，SQL 中需用反引号）
    pct_chg     DECIMAL,       -- 涨跌幅(%)
    vol         DECIMAL,       -- 成交量(手)
    amount      DECIMAL,       -- 成交额(千元)
    created_at  DATETIME,      -- 记录创建时间
    updated_at  DATETIME       -- 记录更新时间
);

覆盖的股票：
贵州茅台(600519.SH)、五粮液(000858.SZ)、国泰君安(601211.SH)、中芯国际(688981.SH)
实际数据范围以查询结果为准，不保证行情更新到今天。

编写 SQL 须知：
1. 只查询 stock_daily 单表，只允许单条 SELECT，不要生成增删改语句。禁止注释、CTE、子查询、UNION、JOIN、锁和文件操作；最多返回 5000 行。
2. `change` 是 MySQL 保留字，出现时务必写成反引号 `change`。
3. 日期过滤用 trade_date，格式 'YYYY-MM-DD'，如 WHERE trade_date BETWEEN '2025-01-01' AND '2025-03-31'。
4. 按月份聚合可用 DATE_FORMAT(trade_date, '%Y-%m')；按周可用 YEARWEEK(trade_date, 1)。
5. 常用指标：涨跌幅 pct_chg、收盘价 close、成交量 vol、成交额 amount。
   示例：区间最高收盘价 MAX(close)、平均涨跌幅 AVG(pct_chg)、按股票分组 GROUP BY stock_name。

按问题性质选择以下工具，必要时可以协同使用：

工具一：exc_sql（本地行情数据）
- 数据全部来自 stock_daily 表；具体覆盖区间以库内实际记录为准。
- 凡是让你「查询/统计/对比/排行/走势/涨跌幅/成交量/成交额/收盘价」等
  可由本地行情表直接算出的普通量化问题，用 exc_sql 跑 SQL，不要联网搜索。
- 区间首末涨跌幅、月末收盘价、上涨/下跌天数优先用 stock_analysis，不要编写被禁止的窗口函数、子查询或 CTE。
- 外部网站给出的行情数字可能与库内口径不一致，行情数据以 exc_sql 查询结果为准。

工具二：tavily_search（联网搜索）
- 用于库内没有、或需要实时外部信息的场景：最新新闻、公司公告、宏观与市场环境、
  行业动态、政策消息、券商/研报观点、突发事件，以及库内数据截止日之后的情况。
- 搜索词用精简的关键词组合（实体 + 主题 + 时间），一次搜索把需要的关键词列全。
- 若问题需要「本地数据 + 外部信息」结合分析（例如：结合近期新闻解释股价异动、
  先用 exc_sql 找出成交量异动日再搜索当天消息面），可以两个工具各调用一次，
  但同一个问题 exc_sql 只允许调用一次。

工具三：arima_stock（ARIMA 价格预测）
- 当用户要求「预测/预估/ forecast 某只股票未来 N 天价格/走势」时使用本工具，
  不要自己用 SQL 或凭空推测未来价格。
- 入参 ts_code 必填（把股票名称换算成代码：贵州茅台 600519.SH、五粮液 000858.SZ、
  国泰君安 601211.SH、中芯国际 688981.SH），n 为预测交易日天数（用户未说时默认 5）。
- 工具内部会自动取近一年收盘价并用 ARIMA(5,1,5) 建模，返回内容同样包含
  HTML 表格和一张历史+预测图，由应用直接展示。
- 向用户解读时要说明这是基于历史价格的统计预测、不确定性高，不构成投资建议。

工具四：boll_detection（布林带超买超卖检测）
- 当用户要求「检测超买超卖/异常点/突破上下轨/布林带/BOLL」时使用本工具。
- 入参 ts_code 必填（股票名称换算同工具三）；start_date/end_date 选填（YYYY-MM-DD），
  用户没给时间范围时不传这两个参数（默认检测过去一年）；用户指定了区间就传入对应日期。
- 工具内部自动从 MySQL 取数并计算 20 日均线 ± 2σ 布林带，返回超买超卖日期表格和
  布林带走势图，由应用直接展示。
- 解读时说明：超买/超卖仅表示价格相对近期 20 日波动区间的统计偏离，不构成投资建议。

工具五：prophet_analysis（Prophet 周期性分析）
- 当用户要求「分析周期性/趋势/seasonality/trend/weekly/yearly/年内季节/周内效应」时使用本工具。
- 入参 ts_code 必填（股票名称换算同工具三）；start_date/end_date 选填（YYYY-MM-DD），
  用户没给时间范围时不传这两个参数（默认过去一年）；用户指定了区间就传入对应日期。
- 工具内部使用 Prophet 对历史收盘价做时间序列分解，输出长期趋势(trend)、周内效应(weekly)、
  年内季节效应(yearly) 三个分量的摘要表格和分量图，由应用直接展示。
- 解读时说明：三个分量是模型对历史数据的统计分解结果，不完全代表真实因果，不构成投资建议。

工具六：stock_analysis（固定口径的历史区间分析）
- 区间首末收盘价涨跌幅用 mode=interval_return；每月末条收盘价用 mode=monthly_close；上涨/下跌/平盘天数和上涨日占比用 mode=daily_stats。
- ts_codes 为 1～4 个完整股票代码的数组，start_date/end_date 必填 YYYY-MM-DD；时间范围不清楚时先澄清。
- 工具通过参数化只读查询取数，再按真实交易日期计算。不得用 MAX(close) 代替月末收盘价，不得把日涨跌幅相加作为区间涨跌幅。
- 应用展示工具返回的实际取值日期和缺失数据说明；仅一条观测时不能声称区间涨跌幅为 0。

输出要求：
- 所有股票数值、预测及图表必须来自实际工具调用；禁止模拟返回、编造文件路径或把工具协议作为普通文本输出。
- 应用直接展示工具返回的表格、图片、统计说明及搜索来源。不要在最终回复中复制或改写这些数据；最终回复只需简短说明执行状态。
- 工具返回失败、数据过期或没有数据时，不得宣称已成功完成分析。需要更多步骤时可以选择其他工具；不得用相同参数重复调用已完成的工具。
- 通过 API 原生 tool_calls 发出调用，arguments 必须是合法 JSON 对象；不要在 content 中输出调用代码。
"""

@register_tool('stock_analysis')
class StockAnalysisTool(BaseTool):
    description = '计算历史区间涨跌幅、每月最后一条收盘价、上涨下跌天数；无需生成复杂 SQL'
    parameters = [
        {'name': 'mode', 'type': 'string', 'enum': ['interval_return', 'monthly_close', 'daily_stats'],
         'description': '分析口径', 'required': True},
        {'name': 'ts_codes', 'type': 'array', 'items': {'type': 'string'},
         'description': '1～4 个完整股票代码，如 ["600519.SH","000858.SZ"]', 'required': True},
        {'name': 'start_date', 'type': 'string', 'description': '开始日期 YYYY-MM-DD', 'required': True},
        {'name': 'end_date', 'type': 'string', 'description': '结束日期 YYYY-MM-DD', 'required': True},
    ]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        try:
            result = analyze_stocks(args)
            output = escape(result.note)
            if result.table.empty:
                return output
            output += '\n\n' + result_table(result.table, na_rep='不可计算')
            os.makedirs(IMAGE_DIR, exist_ok=True)
            path = image_path()
            if generate_chart_png(result.chart, path):
                output += '\n\n![区间分析图](' + file_url(path) + ')'
            return output
        except Exception as error:
            record_error(error, handled=True)
            return '区间分析失败：' + escape(str(error))


# ---------- exc_sql 工具实现 ----------
# 缓存仅存在于一次 Agent.run 请求内，不跨用户或会话共享。


@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """执行传入的 SQL 查询 stock_daily，并自动可视化。"""
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True,
    }]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        sql_input = args['sql_input']
        try:
            sql_input = validate_sql(sql_input)
            cache = current_cache()
            sql_key = (POLICY_VERSION, DB_CONFIG['host'], DB_CONFIG['port'], DB_CONFIG['db'], DB_CONFIG['user'], sql_input)
            cache_entry = cache.get(sql_key) if cache else None
            emit('cache_policy', scope='request', request_cache_id=cache.id if cache else None, ttl_seconds=5)
            emit('cache_lookup', tool='exc_sql', hit=cache_entry is not None, enabled=cache is not None)
            if cache_entry is not None:
                return cache_entry
            df = read_dataframe(sql_input)
            if df.empty:
                return '查询执行成功，但没有返回任何数据。请检查查询条件（如日期范围、股票代码）。'

            # 用 HTML 表格而非 markdown 管道表：HTML 不依赖空行/换行，
            # 避免 Gradio 把未识别的表格当普通段落导致换行被折叠成一行
            table_html = (f'查询返回 {len(df)} 行，表格显示前 {min(20, len(df))} 行。'
                          + (' 已达到查询行数上限，结果可能截断，请缩小范围。' if len(df) >= 5000 else '')
                          + '\n' + result_table(df.head(20), sql_input))
            if len(df) > 500:
                table_html += '\n图表最多等距选取 500 条记录展示，不改变表格数据或查询计算结果。'
            table_html += '\n图表优先展示数值排序字段；其他情况只对比同单位指标，完整结果见表格。'

            # 生成图表并保存到本地，通过 Gradio 静态文件路由暴露
            # （会话期间保留所有图以便翻阅历史，关闭界面时由 cleanup_images 统一清理）
            os.makedirs(IMAGE_DIR, exist_ok=True)
            save_path = image_path()
            chart_ok = generate_chart_png(df, save_path, sql_input)

            result = table_html
            if chart_ok:
                result = f"{table_html}\n\n![查询结果图]({file_url(save_path)})"
            if cache is not None:
                cache.put(sql_key, result)
            return result
        except Exception as e:
            record_error(e, handled=True)
            return f"SQL执行或可视化出错: {str(e)}"


@register_tool('tavily_search')
class TavilySearchTool(BaseTool):
    """调用 Tavily Search REST API 做联网搜索（等价于 tavily-mcp 的 tavily-search）。"""
    description = '联网搜索实时外部信息：最新新闻、公司公告、宏观市场、行业动态、政策与研报观点等。本地 stock_daily 表里没有的信息才需要调用本工具。'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '搜索关键词，精简的关键词组合（实体 + 主题 + 时间），一次把关键词列全',
        'required': True,
    }, {
        'name': 'max_results',
        'type': 'number',
        'description': '返回结果条数，默认 5，一般不用修改',
        'required': False,
    }]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        query = args['query']
        try:
            max_results = int(args.get('max_results', 5) or 5)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 10))

        if not TAVILY_API_KEY:
            return '联网搜索不可用：未配置 TAVILY_API_KEY。'

        try:
            resp = requests.post(
                TAVILY_SEARCH_URL,
                headers={'Authorization': f'Bearer {TAVILY_API_KEY}',
                         'Content-Type': 'application/json'},
                json={
                    'query': query,
                    'max_results': max_results,
                    'search_depth': 'basic',
                    'topic': 'general',
                    'include_answer': 'basic',
                    'include_raw_content': False,
                },
                timeout=30,
            )
        except requests.RequestException as e:
            record_error(e, handled=True)
            return f'Tavily 搜索请求失败（网络异常）: {str(e)}'

        if resp.status_code != 200:
            return f'Tavily 搜索失败：HTTP {resp.status_code}，{resp.text[:300]}'

        try:
            data = resp.json()
        except ValueError:
            return f'Tavily 搜索失败：返回内容不是合法 JSON：{resp.text[:300]}'

        results = data.get('results') or []
        if not results:
            return f'未搜索到与「{query}」相关的结果。'

        lines = [f'联网搜索关键词：{query}']
        answer = data.get('answer')
        if answer:
            lines.append(f'搜索摘要：{answer}')
        for i, item in enumerate(results[:max_results], start=1):
            title = item.get('title') or '(无标题)'
            url = item.get('url') or ''
            content = (item.get('content') or '').strip().replace('\n', ' ')
            date = item.get('published_date') or ''
            head = f'[{i}] {title}' + (f'（{date[:10]}）' if date else '')
            lines.append(f'{head}\n    链接：{url}\n    摘要：{content[:400]}')
        return '\n'.join(lines)


@register_tool('arima_stock')
class ArimaStockTool(BaseTool):
    """取截止今天前一年的日线收盘价，用 ARIMA(5,1,5) 预测未来 n 个交易日价格。"""
    description = ('基于某只股票近一年的历史收盘价，使用 ARIMA(5,1,5) 时间序列模型预测未来 n 个交易日的价格，'
                   '返回预测表格和历史+预测走势图。参数 ts_code 为必填的股票代码（如 600519.SH）。')
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': "必填，股票代码，如 '600519.SH'(贵州茅台)、'000858.SZ'(五粮液)、'601211.SH'(国泰君安)、'688981.SH'(中芯国际)",
        'required': True,
    }, {
        'name': 'n',
        'type': 'number',
        'description': '预测未来的交易日天数，选填，默认 5，取值范围 1~60',
        'required': False,
    }]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        ts_code = str(args.get('ts_code', '')).strip()
        if not ts_code:
            return 'arima_stock 调用失败：ts_code（股票代码）为必填参数，例如 600519.SH。'
        try:
            n = int(float(args.get('n', 5) or 5))
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(n, 60))

        # 1. 从 MySQL 获取截止今天前一年、按交易日升序的收盘价序列
        query = text("""
            SELECT trade_date, stock_name, close
            FROM stock_daily
            WHERE ts_code = :code
              AND trade_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
              AND close IS NOT NULL
            ORDER BY trade_date ASC
        """)
        try:
            df = read_dataframe(query, params={'code': ts_code})
        except Exception as e:
            record_error(e, handled=True)
            return f'读取历史行情失败：{str(e)}'

        if df.empty:
            return (f'未在 stock_daily 表中查询到 ts_code={ts_code} 近一年的行情数据。'
                    "请检查股票代码是否正确，如 600519.SH / 000858.SZ / 601211.SH / 688981.SH。")
        if len(df) < 20:
            return f'ts_code={ts_code} 近一年仅有 {len(df)} 条行情，样本过少，无法稳定拟合 ARIMA(5,1,5)。'

        stock_name = str(df['stock_name'].iloc[0])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        hist_dates = df['trade_date'].dt.strftime('%Y-%m-%d').tolist()
        close = df['close'].astype(float)
        hist_y = close.values
        m = len(hist_y)
        last_close = float(hist_y[-1])
        last_date = df['trade_date'].iloc[-1]
        today = datetime.date.today()
        try:
            anchor_date = forecast_origin(last_date.date(), today)
            if (df['trade_date'].duplicated().any() or not np.isfinite(hist_y).all()
                    or np.any(hist_y <= 0)):
                raise ValueError('历史行情包含重复日期、无效价格或非正价格，暂停预测')
        except ValueError as error:
            return str(error)

        # 2. ARIMA(5,1,5) 建模并预测未来 n 步
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return 'ARIMA 预测不可用：缺少 statsmodels，请先执行 pip install statsmodels。'

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = ARIMA(hist_y, order=(5, 1, 5))
                try:
                    fit = model.fit(method_kwargs={'maxiter': 200})
                except TypeError:
                    fit = model.fit()
                # statsmodels 不同版本 forecast 可能返回 ndarray 或 Series，统一规范成 ndarray
                forecast = np.asarray(fit.forecast(steps=n)).ravel()
        except Exception as e:
            record_error(e, handled=True)
            return f'ARIMA(5,1,5) 建模或预测失败：{str(e)}'

        # Forecast step 1 always follows the last observation. Never relabel it as a later step.
        future_dates, cal_note = _next_trade_days(anchor_date, n)
        future_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
        # 库内数据滞后于今天时显式提示，避免用户误以为训练数据包含最新行情
        stale_note = ''
        if last_date.date() < today:
            stale_note = (f'注意：库内行情数据截止 {last_date.strftime("%Y-%m-%d")}，'
                          f'早于今天 {today.strftime("%Y-%m-%d")}；模型基于截至该日的收盘价训练，'
                          f'预测从该观测日之后起算，可能包含已过去日期，不代表基于今日行情的实时预测。')

        pred_df = pd.DataFrame({
            '预测步(交易日)': list(range(1, n + 1)),
            '预测交易日': future_labels,
            '预测收盘价(元)': np.round(forecast, 2),
            '相对最新收盘涨跌幅(%)': np.round((forecast - last_close) / last_close * 100, 2),
        })
        table_html = result_table(pred_df)

        # 4. 绘制历史走势 + 预测（预测段与历史最后一个点相连，注意 x/y 长度对齐）
        os.makedirs(IMAGE_DIR, exist_ok=True)
        save_path = image_path('arima')
        chart_ok = _plot_arima_forecast(
            hist_dates=hist_dates, hist_y=hist_y,
            future_labels=future_labels, forecast=forecast,
            save_path=save_path, title=f'{stock_name}({ts_code}) ARIMA(5,1,5) 价格预测')

        head = (f'ARIMA(5,1,5) 预测结果：{stock_name}({ts_code})<br>'
                f'训练区间：{hist_dates[0]} ~ {hist_dates[-1]}，共 {m} 个交易日；'
                f'库内末条收盘价：{last_close:.2f} 元；从末条观测后预测 {n} 步。')
        result = f'{head}\n\n{table_html}'
        if chart_ok:
            result += f"\n\n![{stock_name}ARIMA预测图]({file_url(save_path)})"
        result += ('\n\n注：日期由本地工作日/节假日规则估算，未核对交易所完整交易日历。')
        if stale_note:
            result += f'\n{stale_note}'
        if cal_note:
            result += f'\n{cal_note}'
        result += 'ARIMA 仅依据历史收盘价做统计外推，实际股价受消息面、政策、市场情绪等多重因素影响，预测结果不构成投资建议。'
        return result


def _next_trade_days(last_date, n):
    """从 last_date 的次日起推算未来 n 个 A 股交易日。

    A 股交易日规则：周一至周五，且不是法定节假日（周末即使因调休需要补班，股市也不开市，
    因此周末一律排除，不能用 chinese_calendar.is_workday，它会把补班的周末算作工作日）。
    优先用本地免联网的 chinese_calendar 判断节假日；若未安装或日期超出其数据年份范围
    （该库只覆盖已公布放假安排的年份），则退化为仅排除周末，并通过第二个返回值给出提示。

    返回：(日期列表[datetime.date], 日历提示语 str)
    """
    import datetime as _dt

    note = ''
    try:
        import chinese_calendar as cc
    except ImportError:
        cc = None
        note = ('未安装 chinese_calendar，未来交易日仅按周一至周五推算，未能剔除法定节假日，'
                '实际交易安排以交易所公告为准（可执行 pip install chinesecalendar 获得更准确结果）。')

    d = last_date
    if isinstance(d, pd.Timestamp):
        d = d.date()
    cur = d + datetime.timedelta(days=1)

    days = []
    while len(days) < n:
        is_trade_day = cur.weekday() < 5  # 周末（含调休补班的周末）一律休市
        if is_trade_day and cc is not None:
            try:
                if cc.is_holiday(cur):
                    is_trade_day = False
            except NotImplementedError:
                # 超出日历支持年份（官方放假安排尚未公布），退化为仅排除周末
                cc = None
                note = ('部分预测日期已超出本地节假日日历覆盖范围，该区间仅按周一至周五推算，'
                        '未能剔除尚未公布的法定节假日，实际交易安排以交易所公告为准。')
        if is_trade_day:
            days.append(cur)
        cur += datetime.timedelta(days=1)
    return days, note


def _plot_arima_forecast(hist_dates, hist_y, future_labels, forecast, save_path, title):
    """画近一年历史收盘价 + 未来预测折线；成功返回 True。"""
    try:
        m = len(hist_y)
        n = len(forecast)
        all_labels = list(hist_dates) + list(future_labels)

        hist_x = np.arange(m)
        # 预测段从历史最后一点开始连接：x、y 都必须是 n+1 个点，维度对齐
        pred_x = np.arange(m - 1, m + n)
        pred_y = np.concatenate([[hist_y[-1]], np.asarray(forecast, dtype=float).ravel()])

        # 固定 10 英寸宽（100dpi≈1000px）：聊天气泡没有给 markdown 图片做自适应约束，
        # 图过宽（如按点数放到 1500px）会撑破消息容器导致整段横向溢出
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hist_x, hist_y, color='#1f77b4', linewidth=1.4, label='历史收盘价')
        ax.plot(pred_x, pred_y, color='#d62728', linewidth=1.8, linestyle='--',
                marker='o', markersize=3, label=f'ARIMA预测({n}个交易日)')
        ax.axvline(m - 1, color='gray', linewidth=0.8, linestyle=':')

        total = m + n
        step = max(1, total // 12)
        idx = list(range(0, total, step))
        ax.set_xticks(idx)
        ax.set_xticklabels([all_labels[i] for i in idx], rotation=45, ha='right')
        ax.set_ylabel('收盘价(元)')
        ax.set_title(title)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        record_error(e, handled=True)
        print(f'ARIMA 预测图生成失败（仅返回表格）: {e}')
        return False


@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    """布林带（20 日均线 ± 2σ）超买超卖检测。"""
    description = ('基于布林带 BOLL（20 日周期 ± 2 倍标准差）检测某只股票的超买（收盘价突破上轨）和'
                   '超卖（收盘价跌破下轨）日期。默认检测过去一年，可通过 start_date/end_date 自定义检测范围。'
                   '返回超买超卖日期表格和布林带走势图。参数 ts_code 为必填的股票代码（如 600519.SH）。')
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': "必填，股票代码，如 '600519.SH'(贵州茅台)、'000858.SZ'(五粮液)、'601211.SH'(国泰君安)、'688981.SH'(中芯国际)",
        'required': True,
    }, {
        'name': 'start_date',
        'type': 'string',
        'description': "选填，检测区间起始日期，格式 YYYY-MM-DD，默认为一年前",
        'required': False,
    }, {
        'name': 'end_date',
        'type': 'string',
        'description': "选填，检测区间结束日期，格式 YYYY-MM-DD，默认为今天",
        'required': False,
    }]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        ts_code = str(args.get('ts_code', '')).strip()
        if not ts_code:
            return 'boll_detection 调用失败：ts_code（股票代码）为必填参数，例如 600519.SH。'

        today = datetime.date.today()
        try:
            end_date = pd.to_datetime(args['end_date']).date() if args.get('end_date') else today
            start_date = (pd.to_datetime(args['start_date']).date()
                          if args.get('start_date') else end_date - datetime.timedelta(days=365))
        except (ValueError, TypeError, KeyError):
            return '日期格式错误：start_date / end_date 需为 YYYY-MM-DD，例如 2025-10-01。'
        if start_date > end_date:
            return f'日期范围错误：start_date({start_date}) 晚于 end_date({end_date})。'

        # 1. 取数：检测区间第一天也需要其前 19 个交易日来计算 20 日均线/标准差，
        #    因此 SQL 下界往前多取 60 个自然日（约 40+ 个交易日），算完指标再过滤回检测区间
        fetch_from = start_date - datetime.timedelta(days=60)
        query = text("""
            SELECT trade_date, stock_name, close
            FROM stock_daily
            WHERE ts_code = :code
              AND trade_date >= :from_d
              AND trade_date <= :to_d
              AND close IS NOT NULL
            ORDER BY trade_date ASC
        """)
        try:
            df = read_dataframe(query,
                             params={'code': ts_code, 'from_d': str(fetch_from), 'to_d': str(end_date)})
        except Exception as e:
            record_error(e, handled=True)
            return f'读取历史行情失败：{str(e)}'

        if df.empty:
            return (f'未在 stock_daily 表中查询到 ts_code={ts_code} 的行情数据。'
                    "请检查股票代码是否正确，如 600519.SH / 000858.SZ / 601211.SH / 688981.SH。")

        stock_name = str(df['stock_name'].iloc[0])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['close'] = df['close'].astype(float)

        # 2. 布林带：中轨 = 20 日均线，上下轨 = 中轨 ± 2σ（20 日总体标准差 ddof=0）
        df['MB'] = df['close'].rolling(20).mean()
        df['STD'] = df['close'].rolling(20).std(ddof=0)
        df['UP'] = df['MB'] + 2 * df['STD']
        df['DN'] = df['MB'] - 2 * df['STD']

        # 过滤到用户检测区间，且要求窗口数据完整（前 19 个交易日指标为 NaN，自动剔除）
        det = df[(df['trade_date'] >= pd.Timestamp(start_date))
                 & (df['trade_date'] <= pd.Timestamp(end_date))
                 & df['UP'].notna()].copy()
        if det.empty:
            return (f'{stock_name}({ts_code}) 在 {start_date} ~ {end_date} 区间内的有效交易日不足 20 个，'
                    '无法计算布林带，请扩大检测范围。')

        det['type'] = ''
        det.loc[det['close'] > det['UP'], 'type'] = '超买'
        det.loc[det['close'] < det['DN'], 'type'] = '超卖'
        hits = det[det['type'] != ''].copy()
        n_ob = int((hits['type'] == '超买').sum())
        n_os = int((hits['type'] == '超卖').sum())

        # 3. 结果表：日期、收盘价、对应轨值、类型、相对轨线偏离度
        if hits.empty:
            table_html = f'<p>该区间内未检测到超买或超卖点（收盘价始终运行在布林带上下轨之间）。</p>'
        else:
            hits['偏离上轨/下轨(%)'] = np.where(
                hits['type'] == '超买',
                (hits['close'] - hits['UP']) / hits['UP'] * 100,
                (hits['close'] - hits['DN']) / hits['DN'] * 100).round(2)
            out = pd.DataFrame({
                '日期': hits['trade_date'].dt.strftime('%Y-%m-%d'),
                '收盘价(元)': hits['close'].round(2),
                '上轨(元)': hits['UP'].round(2),
                '下轨(元)': hits['DN'].round(2),
                '类型': hits['type'],
                '偏离轨线(%)': hits['偏离上轨/下轨(%)'],
            })
            table_html = result_table(out)

        # 4. 绘制布林带走势 + 超买超卖标记（仅画检测区间）
        os.makedirs(IMAGE_DIR, exist_ok=True)
        save_path = image_path('boll')
        chart_ok = _plot_boll(det, hits, save_path,
                              f'{stock_name}({ts_code}) BOLL(20,2) 超买超卖检测')

        head = (f'布林带 BOLL(20日, 2σ) 检测结果：{stock_name}({ts_code})<br>'
                f'检测区间：{det["trade_date"].iloc[0].strftime("%Y-%m-%d")} ~ '
                f'{det["trade_date"].iloc[-1].strftime("%Y-%m-%d")}，共 {len(det)} 个交易日；'
                f'超买 <b>{n_ob}</b> 天，超卖 <b>{n_os}</b> 天。')
        result = f'{head}\n\n{table_html}'
        if chart_ok:
            result += f"\n\n![{stock_name}BOLL超买超卖图]({file_url(save_path)})"
        result += ('\n\n注：布林带为 20 日均线 ± 2 倍标准差（总体标准差 ddof=0）；收盘价高于上轨记为超买、'
                   '低于下轨记为超卖，仅代表相对近期价格的统计偏离，不构成投资建议。')
        return result


def _plot_boll(det, hits, save_path, title):
    """绘制收盘价 + 布林带三轨 + 超买(红点)/超卖(绿点)标记；成功返回 True。"""
    try:
        x = np.arange(len(det))
        labels = det['trade_date'].dt.strftime('%Y-%m-%d').tolist()

        fig, ax = plt.subplots(figsize=(10, 6))  # 固定宽度，避免撑破聊天气泡
        ax.plot(x, det['close'].values, color='#1f77b4', linewidth=1.4, label='收盘价')
        ax.plot(x, det['UP'].values, color='#d62728', linewidth=1.0, linestyle='--', label='上轨 UP')
        ax.plot(x, det['MB'].values, color='#ff7f0e', linewidth=1.0, label='中轨 MB(20日均线)')
        ax.plot(x, det['DN'].values, color='#2ca02c', linewidth=1.0, linestyle='--', label='下轨 DN')
        ax.fill_between(x, det['DN'].values, det['UP'].values, color='gray', alpha=0.08)

        idx_map = {d: i for i, d in enumerate(det['trade_date'])}
        ob = hits[hits['type'] == '超买']
        ost = hits[hits['type'] == '超卖']
        if not ob.empty:
            ax.scatter([idx_map[d] for d in ob['trade_date']], ob['close'].values,
                       color='#d62728', marker='v', s=60, zorder=5, label=f'超买({len(ob)})')
        if not ost.empty:
            ax.scatter([idx_map[d] for d in ost['trade_date']], ost['close'].values,
                       color='#2ca02c', marker='^', s=60, zorder=5, label=f'超卖({len(ost)})')

        step = max(1, len(det) // 12)
        idx = list(range(0, len(det), step))
        ax.set_xticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha='right')
        ax.set_ylabel('收盘价(元)')
        ax.set_title(title)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        record_error(e, handled=True)
        print(f'BOLL 检测图生成失败（仅返回表格）: {e}')
        return False


@register_tool('prophet_analysis')
class ProphetAnalysisTool(BaseTool):
    """用 Prophet 对股票收盘价做周期性分解：趋势(trend) + 周度(weekly) + 年度(yearly)。"""
    description = ('基于 Prophet 时间序列模型分析某只股票收盘价的周期性规律，'
                   '分解出 trend(长期趋势)、weekly(周内效应，周一至周五)、yearly(年内季节效应) 三个分量，'
                   '返回分量摘要表格和分量可视化图。默认分析过去一年，可通过 start_date/end_date 自定义范围。'
                   '参数 ts_code 为必填的股票代码（如 600519.SH）。')
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': "必填，股票代码，如 '600519.SH'(贵州茅台)、'000858.SZ'(五粮液)、'601211.SH'(国泰君安)、'688981.SH'(中芯国际)",
        'required': True,
    }, {
        'name': 'start_date',
        'type': 'string',
        'description': "选填，分析区间起始日期，格式 YYYY-MM-DD，默认为一年前",
        'required': False,
    }, {
        'name': 'end_date',
        'type': 'string',
        'description': "选填，分析区间结束日期，格式 YYYY-MM-DD，默认为今天",
        'required': False,
    }]

    @record_tool_exceptions
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params) if isinstance(params, str) else params
        ts_code = str(args.get('ts_code', '')).strip()
        if not ts_code:
            return 'prophet_analysis 调用失败：ts_code（股票代码）为必填参数，例如 600519.SH。'

        today = datetime.date.today()
        try:
            end_date = pd.to_datetime(args['end_date']).date() if args.get('end_date') else today
            start_date = (pd.to_datetime(args['start_date']).date()
                          if args.get('start_date') else end_date - datetime.timedelta(days=365))
        except (ValueError, TypeError, KeyError):
            return '日期格式错误：start_date / end_date 需为 YYYY-MM-DD，例如 2025-10-01。'
        if start_date > end_date:
            return f'日期范围错误：start_date({start_date}) 晚于 end_date({end_date})。'

        # 1. 取数
        query = text("""
            SELECT trade_date, stock_name, close
            FROM stock_daily
            WHERE ts_code = :code
              AND trade_date >= :from_d
              AND trade_date <= :to_d
              AND close IS NOT NULL
            ORDER BY trade_date ASC
        """)
        try:
            df = read_dataframe(query,
                             params={'code': ts_code, 'from_d': str(start_date), 'to_d': str(end_date)})
        except Exception as e:
            record_error(e, handled=True)
            return f'读取历史行情失败：{str(e)}'

        if df.empty:
            return (f'未在 stock_daily 表中查询到 ts_code={ts_code} 的行情数据。'
                    "请检查股票代码是否正确，如 600519.SH / 000858.SZ / 601211.SH / 688981.SH。")
        if len(df) < 30:
            return f'{ts_code} 在该区间仅有 {len(df)} 条行情，样本过少，无法稳定做周期性分解。'

        stock_name = str(df['stock_name'].iloc[0])

        # 2. Prophet 建模（只要 trend + weekly + yearly）
        try:
            from prophet import Prophet
        except ImportError:
            return 'Prophet 分析不可用：缺少 prophet，请先执行 pip install prophet。'

        try:
            pf = pd.DataFrame({
                'ds': pd.to_datetime(df['trade_date']),
                'y': df['close'].astype(float),
            })
            model = Prophet(
                growth='linear',
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(pf)
            fc = model.predict(pf)  # 在历史区间上分解
        except Exception as e:
            record_error(e, handled=True)
            return f'Prophet 建模或分解失败：{str(e)}'

        # 3. 汇总分量统计表
        trend_change = fc['trend'].iloc[-1] - fc['trend'].iloc[0]
        trend_pct = trend_change / fc['trend'].iloc[0] * 100
        weekly = fc.groupby(fc['ds'].dt.dayofweek)['weekly'].mean()  # 0=周一
        wd_names = ['周一', '周二', '周三', '周四', '周五']
        weekly_rows = [(wd_names[i], weekly.get(i, np.nan)) for i in range(5) if i in weekly.index]
        weekly_best = max(weekly_rows, key=lambda x: x[1]) if weekly_rows else ('-', np.nan)
        weekly_worst = min(weekly_rows, key=lambda x: x[1]) if weekly_rows else ('-', np.nan)

        stats = pd.DataFrame({
            '分量': ['trend(长期趋势)', 'weekly(周内效应)', 'yearly(年内季节)'],
            '区间/说明': [
                f'区间变动 {trend_change:+.2f} 元 ({trend_pct:+.2f}%)',
                f'最强 {weekly_best[0]}({weekly_best[1]:+.3f})，最弱 {weekly_worst[0]}({weekly_worst[1]:+.3f})',
                '见可视化图（按年内日期展开）',
            ],
        })
        table_html = result_table(stats)

        # 4. 可视化：trend、weekly、yearly 三个分量子图
        os.makedirs(IMAGE_DIR, exist_ok=True)
        save_path = image_path('prophet')
        chart_ok = _plot_prophet_components(
            fc, save_path,
            f'{stock_name}({ts_code}) Prophet 周期性分解 '
            f'({pd.to_datetime(start_date).strftime("%Y-%m-%d")} ~ {pd.to_datetime(end_date).strftime("%Y-%m-%d")})')

        head = (f'Prophet 周期性分析：{stock_name}({ts_code})<br>'
                f'分析区间：{fc["ds"].iloc[0].strftime("%Y-%m-%d")} ~ {fc["ds"].iloc[-1].strftime("%Y-%m-%d")}，'
                f'共 {len(fc)} 个交易日；模型：线性趋势 + 周内效应 + 年内季节（加性模型）。')
        result = f'{head}\n\n{table_html}'
        if chart_ok:
            result += f"\n\n![{stock_name}Prophet周期性分解图]({file_url(save_path)})"
        result += ('\n\n注：trend 为长期价格趋势；weekly 反映周一~周五的系统性偏离（周内效应）；'
                   'yearly 反映一年内的季节性规律。分量是模型对历史数据的分解结果，不构成投资建议。')
        return result


def _plot_prophet_components(fc, save_path, title):
    """绘制 Prophet 分解出的 trend/weekly/yearly 分量；成功返回 True。"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

        # trend
        axes[0].plot(fc['ds'], fc['trend'], color='#1f77b4', linewidth=1.6)
        axes[0].set_ylabel('trend')
        axes[0].set_title('长期趋势 (trend)')

        # weekly：按星期几展开
        wd = fc.copy()
        wd['dow'] = wd['ds'].dt.dayofweek
        weekly_by_dow = wd.groupby('dow')['weekly'].mean().reindex(range(5))
        axes[1].bar(range(5), weekly_by_dow.values, color='#ff7f0e', width=0.5)
        axes[1].set_xticks(range(5))
        axes[1].set_xticklabels(['周一', '周二', '周三', '周四', '周五'])
        axes[1].axhline(0, color='gray', linewidth=0.8, linestyle=':')
        axes[1].set_ylabel('weekly')
        axes[1].set_title('周内效应 (weekly)')

        # yearly：按年内日期展开
        yr = fc.copy()
        yr['doy'] = yr['ds'].dt.dayofyear
        yearly_by_doy = yr.groupby('doy')['yearly'].mean().sort_index()
        axes[2].plot(yearly_by_doy.index, yearly_by_doy.values, color='#2ca02c', linewidth=1.2)
        axes[2].axhline(0, color='gray', linewidth=0.8, linestyle=':')
        axes[2].set_ylabel('yearly')
        axes[2].set_title('年内季节效应 (yearly)')
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        axes[2].set_xticks(month_starts)
        axes[2].set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                                 '7月', '8月', '9月', '10月', '11月', '12月'])

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        record_error(e, handled=True)
        print(f'Prophet 分量图生成失败（仅返回表格）: {e}')
        return False


def generate_chart_png(df: pd.DataFrame, save_path: str, sql: str = '') -> bool:
    """根据查询结果自适应出图；成功返回 True，失败返回 False（不影响表格返回）。

    规则：
    - 以第一列为 X 轴，其余数值列为数据系列。
    - 若 X 轴是日期/时间且点数 > 8，画折线图（适合行情走势）。
    - 否则画柱状图（适合分类对比）。
    """
    fig = None
    try:
        plot_df = df.copy()
        if len(plot_df) > 500:
            plot_df = plot_df.iloc[np.linspace(0, len(plot_df) - 1, 500, dtype=int)].copy()
        if plot_df.shape[1] < 2:
            return False  # 只有 1 列无法画图

        x_col = plot_df.columns[0]
        labels = dict(zip(plot_df.columns, display_labels(plot_df.columns, sql)))
        # 仅保留可转为数值的列作为 Y
        y_cols = []
        for c in plot_df.columns[1:]:
            if labels[c] in ('股票代码', '记录编号'):
                continue
            coerced = pd.to_numeric(plot_df[c], errors='coerce')
            if coerced.notna().any():
                plot_df[c] = coerced
                y_cols.append(c)
        if not y_cols:
            return False

        rank_metric = ordered_metric(y_cols, sql) if sql else None
        if rank_metric:
            # A volume leaderboard is not a chronological price chart.
            y_cols = [rank_metric]
        elif sql:
            # Keep only comparable units; never put money, volume and % on one scale.
            group = metric_group(labels[y_cols[0]])
            y_cols = [c for c in y_cols if metric_group(labels[c]) == group][:6]

        plot_df = plot_df.dropna(subset=y_cols, how='all')
        if plot_df.empty:
            return False

        x_labels = plot_df[x_col].astype(str).tolist()
        n = len(plot_df)
        is_date_like = x_col.lower() in ('trade_date', 'date', '交易日期', '月份')
        if (not is_date_like and not pd.api.types.is_numeric_dtype(plot_df[x_col])
                and plot_df[x_col].astype(str).str.match(r'^\d{4}[-/]\d{1,2}([-/]\d{1,2})?').all()):
            try:
                is_date_like = pd.to_datetime(plot_df[x_col], errors='raise').dtype == 'datetime64[ns]'
            except Exception:
                is_date_like = False

        fig, ax = plt.subplots(figsize=(10, 6))
        chronological = is_date_like and plot_df[x_col].is_monotonic_increasing
        if chronological and n > 8 and not rank_metric:
            # 折线图：行情走势
            for c in y_cols:
                ax.plot(range(n), plot_df[c].values, marker='o', linewidth=1.5, label=labels[c])
            _set_xticks(ax, x_labels, n)
            ax.set_ylabel('数值')
        else:
            # 柱状图：分类对比（系列过多时只取前 6 列）
            bar_cols = y_cols[:6]
            width = 0.8 / max(1, len(bar_cols))
            for i, c in enumerate(bar_cols):
                offsets = (np.arange(n) + i * width - (len(bar_cols) - 1) * width / 2)
                ax.bar(offsets, plot_df[c].values, width=width, label=labels[c])
            ax.set_xticks(np.arange(n))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_ylabel('数值')

        ax.set_xlabel(labels[x_col])
        ax.set_ylabel(labels[y_cols[0]] if len(y_cols) == 1 else '收盘价（元）' if x_col == '月份' else metric_group(labels[y_cols[0]]))
        ax.legend(loc='best')
        ax.set_title(labels[rank_metric] + '排名' if rank_metric else labels[x_col] + '对比')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        record_error(e, handled=True)
        print(f'图表生成失败（仅返回表格）: {e}')
        return False
    finally:
        if fig is not None:
            plt.close(fig)


def _set_xticks(ax, labels, n):
    # 最多显示约 10 个刻度，避免拥挤
    step = max(1, n // 10)
    idx = list(range(0, n, step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha='right')


# ---------- 初始化助手服务 ----------
def init_agent_service():
    if not dashscope.api_key:
        raise SystemExit('错误：未检测到环境变量 DASHSCOPE_API_KEY，请先设置通义千问 API Key。')

    llm_cfg = {
        'model': LLM_MODEL,
        'model_server': LLM_BASE_URL,
        'api_key': dashscope.api_key,
        'generate_cfg': {
            'max_input_tokens': 30000,
            'max_retries': 0,
        },
    }
    # FAQ 仅约 2k tokens，直接注入，避免每次提问先用模型拆题和提取关键词。
    faq_path = os.path.join(BASE_DIR, 'faq.txt')
    with open(faq_path, encoding='utf-8') as source:
        local_knowledge = source.read()
    bot = TracedAssistant(
        llm=NativeToolModel(llm_cfg),
        name='股票数据助手',
        description='基于 stock_daily 表的股票问数分析 + Tavily 联网资讯搜索',
        system_message=system_prompt + '\n\n本地分析参考：\n' + local_knowledge +
            '\n\n工具调用使用 API 原生 tool_calls 字段，不要输出工具协议标记、模拟工具返回或编造文件路径。'
            '股票数值和预测必须先调用工具。工具执行后由应用直接显示真实表格、图片及失败提示；'
            '最终回复只需简短说明执行状态，不要复制表格、图片或再次调用已完成的相同工具。',
        function_list=['exc_sql', 'stock_analysis', 'tavily_search', 'arima_stock', 'boll_detection', 'prophet_analysis'],
        files=[],
    )
    print('助手初始化成功！')
    return bot


def app_tui():
    """终端交互模式。"""
    bot = init_agent_service()
    messages = []
    print('进入终端对话模式（Ctrl+C 退出）')
    while True:
        try:
            query = input('\nuser question: ')
            if not query.strip():
                print('问题不能为空。')
                continue
            messages.append({'role': 'user', 'content': query})
            print('正在处理您的请求...')
            response = []
            for response in bot.run(messages):
                print('bot response:', response)
            messages.extend(response)
        except KeyboardInterrupt:
            print('\n已退出。')
            break
        except Exception as e:
            record_error(e, handled=True)
            print(f'处理请求时出错: {e}')


def cleanup_images():
    """Remove only this process's images; never scan other instance directories."""
    prefixes = ('chart_', 'arima_', 'boll_', 'prophet_')
    try:
        if not os.path.isdir(IMAGE_DIR):
            return
        removed = 0
        root = os.path.realpath(IMAGE_DIR)
        for directory, _, files in os.walk(root, followlinks=False):
            for f in files:
                candidate = os.path.realpath(os.path.join(directory, f))
                if (os.path.commonpath([root, candidate]) == root
                        and f.startswith(prefixes) and f.endswith('.png')):
                    try:
                        os.remove(candidate)
                        removed += 1
                    except OSError:
                        pass
        if removed:
            print(f'\n已清理临时图表文件 {removed} 张。')
    except OSError:
        pass


def app_gui():
    """Web 图形界面模式。"""
    if WebUI is None:
        print('未安装 qwen-agent 的 WebUI 组件，自动回退到终端模式。')
        return app_tui()
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            '对比贵州茅台和五粮液 2025 年每个月末的收盘价走势',
            '统计 2025 年 1 月四只股票的平均日涨跌幅，并从高到低排名',
            '查询中芯国际 2025 年成交量最大的前 10 个交易日',
        ]
    }
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # qwen-agent 的 WebUI.run 不会把 allowed_paths 传给 gradio.launch()，
    # 改用环境变量开放图片目录的静态文件访问
    os.environ['GRADIO_ALLOWED_PATHS'] = IMAGE_DIR
    # 会话期间保留所有图以便翻阅历史；关闭界面（Ctrl+C / 正常退出）时统一清理
    atexit.register(cleanup_images)
    print('正在启动 Web 界面...')
    # 只绑回环地址，对外入口统一走 Nginx（basic-auth + HTTPS）
    WebUI(bot, chatbot_config=chatbot_config).run(server_name='127.0.0.1')


if __name__ == '__main__':
    if '--tui' in sys.argv:
        app_tui()
    else:
        app_gui()

# 本地调用日志

这是新增的运行记录功能。新日志只能证明新增功能后的执行情况，不能充当早期项目记录；也不能据此恢复或证明简历中原先的百分比。

## 正常使用

按原方式启动 ChatBI Web/TUI 或 RAG Web/命令行问答。日志默认自动写入项目下 `logs/traces/UTC日期/请求ID.jsonl`，一行一个事件，一次请求一个文件。JSONL 可直接用文本编辑器打开。

ChatBI 也可执行单次问答：

```powershell
python trace_query.py "查询中芯国际最新的 5 条日线数据"
```

RAG 默认 Web、单问、批量问答入口均有请求记录；多公司比较会将同一请求上下文传入工作线程。未覆盖离线 PDF 解析/建库过程。ChatBI 记录已注册业务工具的调用边界；当前小规模 FAQ 直接加入系统上下文，不再单独调用模型拆词和检索。

## 日志包含什么

- `request_start`：实际输入、UTC 时间、请求 ID、执行模式、Git commit/dirty 和代码摘要。未提交代码也纳入摘要；摘要范围写在日志里。
- `call_start` / `call_return` / `call_failed`：调用名称、实际参数、返回、耗时、span ID。RAG 包括检索片段和答案生成参数；ChatBI 包括 SQL/搜索等业务工具。
- `cache_policy` / `cache_lookup`：ChatBI SQL 缓存策略及命中情况。
- `model_request` / `model_response`：ChatBI 原生工具协议、模型名、工具名、响应 ID、工具调用数和 token 用量；不记录密钥或模型思维链。模型请求失败时可能只有请求事件，具体异常见错误事件。
- `error` / `handled_error`：异常类型和脱敏消息；被业务代码捕获并转成字符串的异常也记录。
- `request_end`：最终返回、总耗时和完成/失败/取消状态。`completed` 只表示调用正常返回，不保证 SQL 结果、引用或模型答案正确；`call_return` 也不代表业务成功。参数校验返回的文字错误可能正常返回。

Ctrl+C、生成器明确关闭等会记录取消；强制杀进程可能留下没有结束事件的文件，导出时标为不完整。日志落盘失败会给出警告，不阻断原有业务请求，也不声称有完整记录。

## 导出已有日志

```powershell
New-Item -ItemType Directory -Force trace_exports
python trace_export.py "logs/traces/日期/请求ID.jsonl" --out "trace_exports/调用记录.json"
```

导出校验请求 ID、序号、调用开始/结束和终止事件，保留原始时间及模式。截断或缺少事件时 `complete_record=false`；失败但记录齐全的请求可以为 true，它只表示记录完整。

`AGENT_TRACE_DIR` 可指定日志目录。`AGENT_TRACE_MODE` 默认为 `live`；测试代码必须设为 `offline_test`。这个标签由运行环境设定，不是第三方认证：核验真实调用时仍需检查代码、事件和外部服务结果。

## 脱敏与保存边界

自动隐藏常见密钥字段、环境变量中的凭据值、Bearer、数据库 URL 口令、邮箱和手机号。不保存 `step_by_step_analysis` 和 `reasoning_content`。自动脱敏无法识别所有商业秘密，SQL 结果、公开资料片段和业务问题仍会保存在本机；对外发送前人工复核。不自动上传。

单事件超过 512 KiB 或一条请求已累计超过 8 MiB 时，会省略后续超限内容并记录摘要及截断标记。不会将截断记录标为完整。多请求日志未实现全局容量淘汰，请自行定期清理需要删除的日志。`logs/` 和 `trace_exports/` 已加入 Git 忽略。

后续已加入专用数据库只读授权、SQL 白名单校验和请求内缓存隔离，具体实现及限制见 `SECURITY_GUIDE.md`。完整 trace 是排查问题的依据，不代表整个系统已完成安全审计，也不证明模型答案正确。

"""Trace both WebUI and TUI through the same Qwen Agent run/tool hooks."""
from pathlib import Path
import copy
import re
from html import escape
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from native_tool_model import ToolProtocolError, message_text
from request_cache import request_scoped_stream
from trace_logging import Trace, observed_call, trace_stream, emit, record_error


def render_tool_result(result):
    # Pandas indents <tr>/<td> by four spaces: Markdown interprets these as code.
    return re.sub(r'<table\b.*?</table>',
                  lambda m: ''.join(line.strip() for line in m.group(0).splitlines()),
                  str(result), flags=re.S | re.I)


class TracedAssistant(Assistant):
    def _run(self, messages, lang='zh', **kwargs):
        messages = copy.deepcopy(messages)
        # Older UI sessions may contain fabricated tool markup. Do not reuse it.
        if any(m.role == 'assistant' and re.search(r'<tool_call>|</think>', message_text(m)) for m in messages):
            last_user = max(i for i, m in enumerate(messages) if m.role == 'user')
            messages = messages[last_user:]
            emit('history_reset', reason='legacy_text_tool_protocol')
        response, results = [], []

        def final_text():
            return '\n\n---\n\n'.join(render_tool_result(r) for r in results)

        try:
            for _ in range(4):
                output = []
                for output in self._call_llm(messages=messages,
                                             functions=[t.function for t in self.function_map.values()],
                                             extra_generate_cfg={'lang': lang}):
                    pass  # Validate complete native calls before displaying or executing.
                calls = [m for m in output if m.function_call]
                if not calls:
                    # Numerical tables and assets are rendered from actual tool returns.
                    # The model's final copy is not an authoritative data source.
                    answer = final_text() if results else '\n'.join(message_text(m) for m in output)
                    yield response + [Message(role='assistant', content=answer or '没有得到可展示的结果，请补充问题。')]
                    return
                response.extend(calls)
                messages.extend(calls)
                for call in calls:
                    yield response + [Message(role='assistant', content='正在执行工具：' + call.function_call.name)]
                    result = self._call_tool(call.function_call.name, call.function_call.arguments, messages=messages, **kwargs)
                    results.append(result)
                    reply = Message(role='function', name=call.function_call.name, content=result)
                    response.append(reply)
                    messages.append(reply)
                yield response + [Message(role='assistant', content=final_text())]
            record_error(RuntimeError('tool_round_limit'), handled=True)
            yield response + [Message(role='assistant', content=final_text() + '\n\n已达到本次分析步骤上限，请拆分剩余问题。')]
        except Exception as error:
            record_error(error, handled=True)
            if isinstance(error, ToolProtocolError):
                note = str(error)
            else:
                note = '模型请求未完成，请稍后重试；已执行的工具结果保留如下。'
            yield response + [Message(role='assistant', content=(final_text() + '\n\n' if results else '') + escape(note))]

    def run(self, messages, **kwargs):
        trace = Trace('chatbi', Path(__file__).resolve().parent, {'messages': messages},
                      {'entrypoint': 'Assistant.run', 'model': getattr(self.llm, 'model', None)})
        yield from request_scoped_stream(trace_stream(trace, super().run(messages, **kwargs)))

    def _call_tool(self, tool_name, tool_args='{}', **kwargs):
        # Log only tool arguments, never framework internals or full credential-bearing config.
        def invoke(arguments):
            return super(TracedAssistant, self)._call_tool(tool_name, arguments, **kwargs)
        return observed_call(tool_name, invoke, tool_args)

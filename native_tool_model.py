"""Native tools/tool_calls adapter for the project's Python 3.9 Qwen-Agent.

The installed legacy adapter injects textual tool markers and reads content only.
This adapter accepts tool calls exclusively from the API's structured field.
"""
import json
import re
import copy

from openai import OpenAI
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import Message, FunctionCall
from trace_logging import emit


class ToolProtocolError(ValueError):
    pass


def native_parameters(parameters):
    """Translate legacy Qwen parameter lists into native JSON Schema."""
    if isinstance(parameters, dict):
        return copy.deepcopy(parameters)
    properties, required = {}, []
    for parameter in parameters:
        item = copy.deepcopy(parameter)
        name = item.pop('name')
        if item.pop('required', False):
            required.append(name)
        properties[name] = item
    return {'type': 'object', 'properties': properties, 'required': required}


def message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return '\n'.join(item.text for item in (content or []) if item.text)


def native_messages(messages):
    """Keep call IDs and match every tool result, including same-name calls."""
    output, pending = [], []
    for index, msg in enumerate(messages):
        if msg.function_call:
            if output and output[-1]['role'] == 'assistant' and output[-1].get('tool_calls'):
                target = output[-1]
            else:
                if pending:
                    raise ToolProtocolError('工具调用历史缺少返回结果，请新建对话。')
                target = {'role': 'assistant', 'content': None, 'tool_calls': []}
                output.append(target)
            call_id = (msg.extra or {}).get('tool_call_id') or f'call_{index}'
            target['tool_calls'].append({'id': call_id, 'type': 'function',
                                        'function': {'name': msg.function_call.name,
                                                     'arguments': msg.function_call.arguments}})
            pending.append((msg.function_call.name, call_id))
        elif msg.role == 'function':
            match = next((i for i, (name, _) in enumerate(pending) if name == msg.name), None)
            if match is None:
                raise ToolProtocolError('工具返回缺少对应调用，请新建对话。')
            _, call_id = pending.pop(match)
            output.append({'role': 'tool', 'tool_call_id': call_id, 'content': message_text(msg)})
        else:
            if pending:
                raise ToolProtocolError('工具调用历史缺少返回结果，请新建对话。')
            output.append({'role': msg.role, 'content': message_text(msg)})
    if pending:
        raise ToolProtocolError('工具调用尚未执行，不能继续生成回答。')
    return output


class NativeToolModel(BaseChatModel):
    # The legacy text conversion recreates Message and drops extra/tool_call_id.
    # Preserve the envelopes; the adapter itself only sends text to this model.
    @property
    def support_multimodal_input(self):
        return True

    @property
    def support_multimodal_output(self):
        return True

    def _preprocess_messages(self, messages, lang, generate_cfg, functions=None, use_raw_api=False):
        for msg in messages:
            if isinstance(msg.content, list) and any(item.text is None for item in msg.content):
                raise ToolProtocolError('当前股票问数入口仅支持文字，请使用文字描述问题。')
        return messages

    def _postprocess_messages(self, messages, fncall_mode, generate_cfg):
        return messages

    def __init__(self, cfg, client=None):
        super().__init__(cfg)
        self.client = client or OpenAI(api_key=cfg['api_key'], base_url=cfg['model_server'],
                                       timeout=60, max_retries=0)

    def _complete(self, messages, generate_cfg, functions=None):
        payload = {'model': self.model, 'messages': native_messages(messages), 'stream': False,
                   'max_tokens': 4096, 'temperature': 0}
        for key in ('max_tokens', 'temperature', 'top_p', 'seed'):
            if key in generate_cfg:
                payload[key] = generate_cfg[key]
        if functions:
            payload['tools'] = [{'type': 'function', 'function': {
                'name': f.get('name', f.get('name_for_model')),
                'description': f.get('description', ''), 'parameters': native_parameters(f['parameters'])}}
                for f in functions]
            choice = generate_cfg.get('function_choice', 'auto')
            payload['tool_choice'] = choice if choice in ('auto', 'none') else {
                'type': 'function', 'function': {'name': choice}}
        emit('model_request', protocol='native_tool_calls', model=self.model,
             tool_names=[t['function']['name'] for t in payload.get('tools', [])])
        response = self.client.chat.completions.create(**payload)
        if not response.choices or response.choices[0].finish_reason not in ('stop', 'tool_calls', 'function_call'):
            raise ToolProtocolError('模型响应未完整结束，未执行其中的工具调用，请重试。')
        msg = response.choices[0].message
        calls = msg.tool_calls or []
        emit('model_response', response_id=response.id, protocol='native_tool_calls',
             tool_calls=len(calls), finish_reason=response.choices[0].finish_reason,
             usage=response.usage.model_dump() if response.usage else None)
        if calls:
            allowed = {f.get('name', f.get('name_for_model')) for f in (functions or [])}
            if len(calls) > 6:
                raise ToolProtocolError('单轮工具调用过多，已停止。请拆分问题。')
            result, ids = [], set()
            # Validate the complete batch before dispatching any member.
            for call in calls:
                if call.type != 'function' or call.function.name not in allowed or not call.id or call.id in ids:
                    raise ToolProtocolError('模型返回了无效的工具名称或调用 ID，未执行。')
                ids.add(call.id)
                try:
                    args = json.loads(call.function.arguments)
                except (TypeError, ValueError):
                    raise ToolProtocolError('工具参数不是完整 JSON，未执行。')
                if not isinstance(args, dict):
                    raise ToolProtocolError('工具参数必须是 JSON 对象，未执行。')
                result.append(Message(role='assistant', content='',
                                      function_call=FunctionCall(name=call.function.name,
                                                                arguments=call.function.arguments),
                                      extra={'tool_call_id': call.id}))
            return result
        content = msg.content or ''
        if re.search(r'<\s*/?\s*(?:tool_call|tool_response|think)\b|<\|(?:im_|tool)|✿FUNCTION✿', content, re.I):
            raise ToolProtocolError('模型将工具协议当成文本返回；本次没有执行这些文本中的调用或采用其结果。请重试。')
        # No model-created local asset paths or financial result tables without tools.
        last_user = max((i for i, m in enumerate(messages) if m.role == 'user'), default=-1)
        if not any(m.role == 'function' for m in messages[last_user + 1:]) and re.search(r'/file=|<table\b|!\[|\|[^\n]+\|', content, re.I):
            raise ToolProtocolError('模型返回了未经工具核实的表格或图片，已停止展示，请重试。')
        return [Message(role='assistant', content=content)]

    def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
        result = self._complete(messages, generate_cfg, functions)
        return iter([result]) if stream else result

    def _chat_no_stream(self, messages, generate_cfg):
        return self._complete(messages, generate_cfg)

    def _chat_stream(self, messages, delta_stream, generate_cfg):
        yield self._complete(messages, generate_cfg)

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from openai.types.chat import ChatCompletion
from qwen_agent.llm.schema import Message, FunctionCall

import chatbi_agent as app
from native_tool_model import NativeToolModel, ToolProtocolError, native_messages
from traced_assistant import TracedAssistant, render_tool_result


def completion(text='', calls=None, finish=None):
    return ChatCompletion.model_validate({
        'id': 'offline-fixture', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
        'choices': [{'index': 0, 'finish_reason': finish or ('tool_calls' if calls else 'stop'),
                     'message': {'role': 'assistant', 'content': text, 'tool_calls': calls}}]})


def call(name, args, id='call_1'):
    return {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': args}}


class NativeToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'AGENT_TRACE_MODE': 'offline_test', 'AGENT_TRACE_DIR': self.temp.name})
        self.env.start()
        self.client = MagicMock()
        self.model = NativeToolModel({'model': 'offline', 'generate_cfg': {'max_input_tokens': 30000}}, client=self.client)
        self.tool = app.StockAnalysisTool().function

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def chat(self):
        return self.model.chat([Message(role='user', content='月末收盘价')], functions=[self.tool], stream=False)

    def test_native_schema_and_call_id_are_preserved(self):
        self.client.chat.completions.create.return_value = completion(calls=[call('stock_analysis', '{}')])
        result = self.chat()
        self.assertEqual(result[0].function_call.name, 'stock_analysis')
        self.assertEqual(result[0].extra['tool_call_id'], 'call_1')
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload['tools'][0]['function']['name'], 'stock_analysis')
        schema = payload['tools'][0]['function']['parameters']
        self.assertEqual(schema['type'], 'object')
        self.assertEqual(schema['properties']['ts_codes']['type'], 'array')
        self.assertEqual(schema['properties']['ts_codes']['items'], {'type': 'string'})
        self.assertIn('ts_codes', schema['required'])
        self.assertIsInstance(self.tool['parameters'], list)  # No mutation of Qwen metadata.
        self.assertNotIn('✿FUNCTION✿', str(payload['messages']))

    def test_same_name_parallel_tool_results_keep_matching_ids(self):
        messages = [Message(role='user', content='compare')]
        for id in ['a', 'b']:
            messages.append(Message(role='assistant', content='', function_call=FunctionCall(name='stock_analysis', arguments='{}'),
                                    extra={'tool_call_id': id}))
        messages.extend([Message(role='function', name='stock_analysis', content=value) for value in ['one', 'two']])
        result = native_messages(messages)
        self.assertEqual([t['id'] for t in result[1]['tool_calls']], ['a', 'b'])
        self.assertEqual([r['tool_call_id'] for r in result[2:]], ['a', 'b'])
        self.assertEqual([r['content'] for r in result[2:]], ['one', 'two'])

    def test_truncated_unknown_and_invalid_calls_are_not_dispatched(self):
        for response in [completion(calls=[call('stock_analysis', '{}')], finish='length'),
                         completion(calls=[call('stock_analysis', '{}'), call('delete_all', '{}', 'b')]),
                         completion(calls=[call('stock_analysis', '{')]),
                         completion(calls=[call('stock_analysis', '[]')])]:
            with self.subTest(response=response):
                self.client.chat.completions.create.return_value = response
                with self.assertRaises(ToolProtocolError):
                    self.chat()

    def test_recorded_text_protocol_and_invented_artifacts_fail_closed(self):
        for text in ['<tool_call>@result fabricated result', '</think>假结果',
                     '<table><tr><td>999</td></tr></table>', '![图](/file=E:/fake.png)']:
            self.client.chat.completions.create.return_value = completion(text)
            with self.assertRaises(ToolProtocolError):
                self.chat()

    def test_real_agent_dispatches_real_tool_and_renders_actual_result(self):
        args = {'mode': 'monthly_close', 'ts_codes': ['600519.SH'], 'start_date': '2025-01-01', 'end_date': '2025-01-31'}
        self.client.chat.completions.create.side_effect = [completion(calls=[call('stock_analysis', json.dumps(args))]),
                                                          completion('虚构价格 999999，请相信此数值')]
        bot = TracedAssistant(llm=self.model, function_list=['stock_analysis'])
        frame = pd.DataFrame({'ts_code': ['600519.SH'] * 2, 'stock_name': ['测试'] * 2,
                              'trade_date': ['2025-01-02', '2025-01-31'], 'close': [10, 12], 'pct_chg': [0, 20]})
        with patch('stock_analysis.read_dataframe', return_value=frame) as db, \
                patch.object(app, 'generate_chart_png', return_value=False):
            result = list(bot.run([{'role': 'user', 'content': '测试月末收盘价'}]))[-1]
        final = result[-1]['content']
        self.assertIn('12', final)
        self.assertNotIn('999999', final)
        self.assertIn('<table', final)
        self.assertNotIn('\n    <tr>', final)
        db.assert_called_once()
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertTrue(any(m.get('tool_call_id') == 'call_1' for m in payload['messages']))
        events = [json.loads(line) for p in Path(self.temp.name).rglob('*.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(e['event'] == 'call_return' and e['data']['name'] == 'stock_analysis' for e in events))

    def test_protocol_failure_is_visible_and_never_executes_tool(self):
        self.client.chat.completions.create.return_value = completion('<tool_call>@result <table>fake</table>')
        bot = TracedAssistant(llm=self.model, function_list=['stock_analysis'])
        with patch.object(bot, '_call_tool') as dispatch:
            final = list(bot.run([{'role': 'user', 'content': '月末价格'}]))[-1][-1]['content']
        dispatch.assert_not_called()
        self.assertIn('没有执行', final)
        self.assertNotIn('<table>', final)

    def test_tool_error_cannot_be_replaced_by_model_success(self):
        self.client.chat.completions.create.side_effect = [completion(calls=[call('stock_analysis', '{}')]), completion('已成功分析！')]
        bot = TracedAssistant(llm=self.model, function_list=['stock_analysis'])
        with patch.object(bot, '_call_tool', return_value='区间分析失败：缺少日期'):
            final = list(bot.run([{'role': 'user', 'content': '价格'}]))[-1][-1]['content']
        self.assertIn('区间分析失败', final)
        self.assertNotIn('成功', final)

    def test_table_rendering_preserves_cells_and_image_line(self):
        raw = '<table>\n    <tr><td>&lt;安全&gt;</td></tr>\n</table>\n\n![图](/file=C:/temp/chart.png)'
        rendered = render_tool_result(raw)
        self.assertEqual(rendered, '<table><tr><td>&lt;安全&gt;</td></tr></table>\n\n![图](/file=C:/temp/chart.png)')


if __name__ == '__main__':
    unittest.main()

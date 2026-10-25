"""Offline integration: real tool/agent code, mocked DB and model; no network."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import pandas as pd
from qwen_agent.llm.schema import Message
import chatbi_agent as app
from traced_assistant import TracedAssistant
from trace_logging import Trace, observed_call


class OfflineAssistant(TracedAssistant):
    def __init__(self, args):
        # Only the model generation is replaced; use real Agent.run/_call_tool.
        self.args = args
        self.llm = types.SimpleNamespace(model='offline_test_no_model')
        self.name = 'offline-test'
        self.function_map = {'exc_sql': app.ExcSQLTool()}

    def _run(self, messages, **kwargs):
        result = self._call_tool('exc_sql', self.args)
        result = self._call_tool('exc_sql', self.args)
        yield [Message(role='assistant', content=result)]


class ChatBIIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'AGENT_TRACE_DIR': str(self.root), 'AGENT_TRACE_MODE': 'offline_test'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def events(self):
        events = [json.loads(line) for p in self.root.rglob('*.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
        return sorted(events, key=lambda event: event['time_utc'])

    def test_real_agent_hook_and_cache_hit(self):
        sql = 'SELECT close FROM stock_daily LIMIT 1'
        bot = OfflineAssistant(json.dumps({'sql_input': sql}))
        with patch.object(app, 'read_dataframe', return_value=pd.DataFrame({'close': [12.3]})) as db, patch.object(app, 'generate_chart_png', return_value=False), patch.object(app, 'IMAGE_DIR', str(self.root / 'images')):
            result = list(bot.run([{'role': 'user', 'content': 'test query'}]))
            list(bot.run([{'role': 'user', 'content': 'same query'}]))
        self.assertIn('12.3', result[-1][0]['content'])
        self.assertEqual(db.call_count, 2)
        self.assertEqual([e['data']['hit'] for e in self.events() if e['event'] == 'cache_lookup'], [False, True, False, True])
        self.assertTrue(any(e['event'] == 'call_start' and e['data']['name'] == 'exc_sql' for e in self.events()))

    def test_caught_db_error_is_recorded_and_not_cached(self):
        bot = OfflineAssistant('{"sql_input":"SELECT close FROM stock_daily LIMIT 1"}')
        with patch.object(app, 'read_dataframe', side_effect=RuntimeError('offline injected DB failure')) as db:
            list(bot.run([{'role': 'user', 'content': 'test failure'}]))
        events = self.events()
        self.assertTrue(any(e['event'] == 'handled_error' for e in events))
        self.assertEqual(events[-1]['data']['status'], 'completed_with_errors')
        self.assertEqual(db.call_count, 2)

    def test_exception_before_tool_try_block_is_preserved(self):
        bot = OfflineAssistant('invalid-json')
        list(bot.run([{'role': 'user', 'content': 'invalid args test'}]))
        events = self.events()
        self.assertTrue(any(e['event'] == 'error' and e['data']['error_type'] == 'JSONDecodeError' for e in events))
        self.assertEqual(events[-1]['data']['status'], 'completed_with_errors')


if __name__ == '__main__':
    unittest.main()

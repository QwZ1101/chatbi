import json
import os
import types
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import stock_analysis as analytics
from sql_guard import validate_sql


class StockAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.args = dict(mode='interval_return', ts_codes=['600519.SH'], start_date='2025-01-01', end_date='2025-02-28')
        # Unsorted rows: monthly maximum differs from month-end close.
        self.frame = pd.DataFrame([
            ['600519.SH', '测试股票', '2025-02-28', 150, -2],
            ['600519.SH', '测试股票', '2025-01-02', 100, 0],
            ['600519.SH', '测试股票', '2025-01-15', 200, 100],
            ['600519.SH', '测试股票', '2025-01-27', 110, -45],
        ], columns=['ts_code', 'stock_name', 'trade_date', 'close', 'pct_chg'])

    def run_analysis(self, frame=None, **kwargs):
        with patch.object(analytics, 'read_dataframe', return_value=self.frame if frame is None else frame) as read:
            result = analytics.analyze_stocks(dict(self.args, **kwargs))
            query = read.call_args.args[0]
            validate_sql(query, allow_parameters=True)
            self.assertNotIn('600519.SH', query)
            self.assertEqual(read.call_args.kwargs['params']['code'], '600519.SH')
            return result

    def test_interval_uses_sorted_endpoint_prices_not_sum_of_daily_returns(self):
        result = self.run_analysis()
        self.assertEqual(result.table.iloc[0]['区间涨跌幅(%)'], 50)
        self.assertEqual(result.table.iloc[0]['首个日期'], '2025-01-02')

    def test_month_end_uses_last_observation_not_maximum(self):
        result = self.run_analysis(mode='monthly_close')
        row = result.table.iloc[0]
        self.assertEqual(row['月内末条收盘价(元)'], 110)
        self.assertEqual(row['实际取值日期'], '2025-01-27')
        self.assertEqual(list(result.chart['600519.SH']), [110, 150])

    def test_up_day_ratio_includes_flat_days(self):
        row = self.run_analysis(mode='daily_stats').table.iloc[0]
        self.assertEqual((row['上涨天数'], row['下跌天数'], row['平盘天数']), (1, 2, 1))
        self.assertEqual(row['上涨日占比(%)'], 25)

    def test_single_observation_is_not_zero_return(self):
        row = self.run_analysis(self.frame.iloc[:1]).table.iloc[0]
        self.assertTrue(pd.isna(row['区间涨跌幅(%)']))
        self.assertIn('仅一条', row['状态'])

    def test_invalid_arguments_never_query(self):
        for invalid in [{'ts_codes': ["600519.SH' OR 1=1"]}, {'start_date': '2025-02-30'},
                        {'end_date': '2024-12-31'}, {'mode': 'raw_sql'}, {'ts_codes': '600519.SH'}]:
            with self.subTest(invalid=invalid), patch.object(analytics, 'read_dataframe') as read:
                with self.assertRaises(ValueError):
                    analytics.analyze_stocks(dict(self.args, **invalid))
                read.assert_not_called()

    def test_duplicates_invalid_prices_and_truncation_fail_closed(self):
        bad = self.frame.copy()
        bad.loc[0, 'close'] = None
        zero = self.frame.copy()
        zero.loc[0, 'close'] = 0
        for frame in [pd.concat([self.frame, self.frame]), bad, zero, pd.concat([self.frame] * 1250)]:
            with self.subTest(size=len(frame)), self.assertRaises(ValueError):
                self.run_analysis(frame)

    def test_missing_stock_is_reported_without_fabricating_rows(self):
        with patch.object(analytics, 'read_dataframe', side_effect=[self.frame, pd.DataFrame()]):
            result = analytics.analyze_stocks(dict(self.args, ts_codes=['600519.SH', '000858.SZ']))
        self.assertEqual(len(result.table), 1)
        self.assertIn('000858.SZ', result.note)

    def test_remaining_faq_sql_examples_pass_current_guard(self):
        faq = (Path(__file__).resolve().parents[1] / 'faq.txt').read_text(encoding='utf-8')
        examples = re.findall(r'^SELECT\b.*?;', faq, re.M | re.S)
        self.assertEqual(len(examples), 4)
        for query in examples:
            validate_sql(query)

    def test_registered_tool_formats_result_via_real_agent_hook(self):
        import chatbi_agent as app
        from traced_assistant import TracedAssistant
        from qwen_agent.llm.schema import Message

        class OfflineAnalysisAgent(TracedAssistant):
            def __init__(self):
                self.llm = types.SimpleNamespace(model='offline_test')
                self.name = 'offline-analysis'
                self.function_map = {'stock_analysis': app.StockAnalysisTool()}

            def _run(self, messages, **kwargs):
                yield [Message(role='assistant', content=self._call_tool('stock_analysis', messages[-1]['content']))]

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'AGENT_TRACE_DIR': temp, 'AGENT_TRACE_MODE': 'offline_test'}), \
                patch.object(analytics, 'read_dataframe', return_value=self.frame), \
                patch.object(app, 'generate_chart_png', return_value=False), patch.object(app, 'IMAGE_DIR', temp):
            result = list(OfflineAnalysisAgent().run([{'role': 'user', 'content': json.dumps(self.args)}]))
            self.assertIn('50.0', result[-1][0]['content'])
            events = [json.loads(line) for p in Path(temp).rglob('*.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(e['event'] == 'call_start' and e['data']['name'] == 'stock_analysis' for e in events))
            self.assertEqual(events[-1]['data']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import pandas as pd
import chatbi_agent as app
from result_labels import display_labels, result_table


class PresentationTests(unittest.TestCase):
    def test_market_fields_and_units_preserve_values(self):
        frame = pd.DataFrame({'trade_date': ['2025-08-28'], 'stock_name': ['中芯国际'],
                              'vol': [2420325.11], 'amount': [27118768.322],
                              'close': [119.22], 'pct_chg': [17.4466]})
        before = frame.copy(deep=True)
        html = result_table(frame)
        for label in ['交易日期', '股票名称', '成交量（手）', '成交额（千元）', '收盘价（元）', '日涨跌幅（%）']:
            self.assertIn(label, html)
        self.assertIn('translate="no"', html)
        self.assertIn('27118768.322', html)
        pd.testing.assert_frame_equal(frame, before)

    def test_aliases_cannot_mislabel_source_or_aggregate(self):
        sql = 'SELECT close AS 关闭, vol AS 卷, ROUND(AVG(pct_chg), 2) AS avg_pct, MAX(trade_date) AS latest_date, COUNT(*) AS n FROM stock_daily'
        self.assertEqual(display_labels(['关闭', '卷', 'avg_pct', 'latest_date', 'n'], sql),
                         ['收盘价（元）', '成交量（手）', '平均日涨跌幅（%）', '最新交易日期', '记录数'])

    def test_computed_unit_is_not_guessed_from_alias(self):
        self.assertEqual(display_labels(['成交额（元）'], 'SELECT amount * 1000 AS `成交额（元）` FROM stock_daily'), ['成交额（元）'])
        self.assertEqual(display_labels(['ratio'], 'SELECT close / pre_close AS ratio FROM stock_daily'), ['ratio'])
        self.assertIn('&lt;script&gt;', result_table(pd.DataFrame({'stock_name': ['<script>']})))

    def test_volume_ranking_is_bar_chart_with_no_mixed_units(self):
        df = pd.DataFrame({'trade_date': ['2025-08-28', '2025-08-22'] * 5,
                           'vol': list(range(100, 90, -1)), 'amount': [10000] * 10,
                           'close': [119.22] * 10, 'pct_chg': [17.4466] * 10})
        fig, ax = app.plt.subplots()
        with tempfile.TemporaryDirectory() as folder, patch.object(app.plt, 'subplots', return_value=(fig, ax)):
            self.assertTrue(app.generate_chart_png(df, str(Path(folder) / 'chart.png'),
                            'SELECT trade_date, vol, amount, close, pct_chg FROM stock_daily ORDER BY vol DESC LIMIT 10'))
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(len(ax.patches), 10)
            self.assertEqual(ax.get_ylabel(), '成交量（手）')
            self.assertEqual(ax.get_title(), '成交量（手）排名')

    def test_file_marker_is_not_prefixed_twice_and_is_url_encoded(self):
        path = str(Path(tempfile.gettempdir()) / '中文 a#b.png')
        marker = app.file_url(path)
        self.assertTrue(marker.startswith('/file='))
        self.assertNotIn('/gradio_api', marker)
        self.assertEqual(unquote(marker[len('/file='):]), str(Path(path).absolute()).replace('\\', '/'))
        # ModelScope chatbot supplies the Gradio API prefix when rendering this marker.
        self.assertEqual(marker.replace('/file=', 'http://example/gradio_api/file=').count('/gradio_api'), 1)

    def test_real_query_tool_uses_chinese_display_without_changing_sql(self):
        with patch.object(app, 'read_dataframe', return_value=pd.DataFrame({'close': [119.22], 'vol': [100.0]})) as read, \
                patch.object(app, 'generate_chart_png', return_value=False):
            html = app.ExcSQLTool().call(json.dumps({'sql_input': 'SELECT close, vol FROM stock_daily LIMIT 1'}))
        self.assertIn('收盘价（元）', html)
        self.assertIn('成交量（手）', html)
        self.assertIn('SELECT close, vol', read.call_args.args[0])


if __name__ == '__main__':
    unittest.main()

import datetime
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image

import chatbi_agent as app
from forecast_policy import forecast_origin
from request_cache import request_scoped_stream


class RuntimeFixTests(unittest.TestCase):
    def test_forecast_origin_rejects_stale_or_future_data(self):
        today = datetime.date(2026, 10, 24)
        self.assertEqual(forecast_origin(datetime.date(2026, 10, 22), today), datetime.date(2026, 10, 22))
        for last in [datetime.date(2026, 10, 1), datetime.date(2026, 10, 25)]:
            with self.assertRaises(ValueError):
                forecast_origin(last, today)

    def test_arima_forecast_dates_follow_last_observation_not_today(self):
        class Today(datetime.date):
            @classmethod
            def today(cls):
                return cls(2026, 10, 24)

        dates = pd.bdate_range(end='2026-10-22', periods=25)
        frame = pd.DataFrame({'trade_date': dates, 'stock_name': ['测试'] * 25, 'close': np.arange(25) + 100})
        model = MagicMock()
        model.fit.return_value.forecast.return_value = [125, 126]
        with tempfile.TemporaryDirectory() as temp, patch.object(app, 'IMAGE_DIR', temp), \
                patch.object(app, 'read_dataframe', return_value=frame), \
                patch.object(app, 'datetime', types.SimpleNamespace(date=Today, timedelta=datetime.timedelta)), \
                patch('statsmodels.tsa.arima.model.ARIMA', return_value=model), \
                patch.object(app, '_plot_arima_forecast', return_value=False), \
                patch.object(app, '_next_trade_days', wraps=app._next_trade_days) as calendar:
            output = app.ArimaStockTool().call(json.dumps({'ts_code': '600519.SH', 'n': 2}))
        self.assertEqual(calendar.call_args.args[0], datetime.date(2026, 10, 22))
        self.assertIn('2026-10-23', output)
        self.assertIn('不代表基于今日行情', output)

    def test_long_query_plot_is_bounded_and_closes_figure(self):
        frame = pd.DataFrame({'trade_date': pd.date_range('2000-01-01', periods=5000), 'close': np.arange(5000)})
        before = app.plt.get_fignums()
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'chart.png')
            self.assertTrue(app.generate_chart_png(frame, path))
            with Image.open(path) as image:
                self.assertEqual(image.size, (1000, 600))
        self.assertEqual(app.plt.get_fignums(), before)

    def test_query_reports_display_truncation(self):
        frame = pd.DataFrame({'close': np.arange(30)})
        with tempfile.TemporaryDirectory() as temp, patch.object(app, 'IMAGE_DIR', temp), \
                patch.object(app, 'read_dataframe', return_value=frame), \
                patch.object(app, 'generate_chart_png', return_value=False):
            output = app.ExcSQLTool().call('{"sql_input":"SELECT close FROM stock_daily"}')
        self.assertIn('查询返回 30 行，表格显示前 20 行', output)

    def test_request_images_and_instance_cleanup_are_isolated(self):
        def image_stream():
            yield app.image_path()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / 'instance_first'
            second = root / 'instance_second'
            second.mkdir()
            other = second / 'chart_other.png'
            other.write_bytes(b'fixture')
            with patch.object(app, 'IMAGE_DIR', str(first)):
                a = list(request_scoped_stream(image_stream()))[0]
                b = list(request_scoped_stream(image_stream()))[0]
                self.assertNotEqual(Path(a).parent, Path(b).parent)
                Path(a).write_bytes(b'fixture')
                Path(b).write_bytes(b'fixture')
                app.cleanup_images()
            self.assertFalse(Path(a).exists())
            self.assertFalse(Path(b).exists())
            self.assertEqual(other.read_bytes(), b'fixture')


if __name__ == '__main__':
    unittest.main()

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from sqlalchemy import text
from sql_guard import validate_sql, UnsafeSQL
from request_cache import RequestCache, current_cache, request_scoped_stream
import db_access


class SQLGuardTests(unittest.TestCase):
    def test_common_analytics(self):
        queries = [
            'SELECT * FROM stock_daily',
            'SELECT stock_name,AVG(close) AS avg_close FROM stock_daily GROUP BY stock_name ORDER BY avg_close DESC',
            "SELECT DATE_FORMAT(trade_date,'%Y-%m') AS month,MAX(close) FROM stock_daily GROUP BY month",
            'SELECT YEARWEEK(trade_date,1),COUNT(*) FROM stock_daily GROUP BY YEARWEEK(trade_date,1)',
            'SELECT ts_code, ROUND(SUM(amount),2) FROM stock_daily WHERE close BETWEEN 1 AND 20 GROUP BY ts_code HAVING COUNT(*)>2',
            "SELECT 'DROP TABLE' AS label FROM stock_daily",
            'SELECT d.close FROM stock_daily AS d LIMIT 10',
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertIn('LIMIT', validate_sql(query))

    def test_dangerous_queries_never_connect(self):
        queries = [
            'DELETE FROM stock_daily', 'UPDATE stock_daily SET close=0',
            'INSERT INTO stock_daily(id) VALUES(1)', 'DROP TABLE stock_daily',
            'SELECT close FROM stock_daily; DELETE FROM stock_daily',
            "SELECT * FROM stock_daily INTO OUTFILE '/tmp/data'",
            "SELECT LOAD_FILE('/etc/passwd') FROM stock_daily",
            'SELECT SLEEP(2) FROM stock_daily', 'SELECT BENCHMARK(2,MD5(1)) FROM stock_daily',
            "SELECT GET_LOCK('x',1) FROM stock_daily", 'SELECT my_udf(close) FROM stock_daily',
            'SELECT * FROM mysql.user', 'SELECT * FROM other_db.stock_daily',
            'SELECT * FROM stock_daily FOR UPDATE', 'SELECT * FROM stock_daily LOCK IN SHARE MODE',
            'SELECT @x:=close FROM stock_daily', 'SELECT @@version FROM stock_daily',
            'SELECT * FROM stock_daily /*!50000 INTO OUTFILE "/tmp/x" */',
            'WITH x AS (SELECT * FROM stock_daily) SELECT * FROM x',
            'SELECT close FROM stock_daily UNION SELECT 1',
            'SELECT (SELECT password FROM users) FROM stock_daily',
            'SELECT * FROM stock_daily a JOIN stock_daily b ON a.id=b.id',
            'CALL dangerous()', 'SELECT close FROM stock_daily -- comment',
            'SELECT close FROM stock_daily LIMIT :n',
            'SELECT close FROM stock_daily LIMIT 1 OFFSET 999999',
            'SELECT * FROM stock_daily WHERE ts_code=:code',
        ]
        with patch.object(db_access, 'create_engine') as engine:
            for query in queries:
                with self.subTest(query=query), self.assertRaises(UnsafeSQL):
                    db_access.read_dataframe(query)
            engine.assert_not_called()

    def test_row_limit_and_parameter_binding(self):
        self.assertTrue(validate_sql('SELECT close FROM stock_daily LIMIT 999999').endswith('LIMIT 5000'))
        sql = validate_sql('SELECT close FROM stock_daily WHERE ts_code=:code', allow_parameters=True)
        self.assertIn(':code', sql)

    def test_all_existing_fixed_queries_parse(self):
        source = (Path(__file__).resolve().parents[1] / 'chatbi_agent.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        queries = [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'text' and node.args and isinstance(node.args[0], ast.Constant)]
        self.assertEqual(len(queries), 3)
        for query in queries:
            with self.subTest(query=query):
                self.assertIn('LIMIT 5000', validate_sql(query, allow_parameters=True))


class CacheIsolationTests(unittest.TestCase):
    def test_completion_ttl_and_capacity(self):
        now = [0]
        cache = RequestCache(ttl=5, max_entries=2, clock=lambda: now[0])
        now[0] = 10  # Simulate slow query completing at t=10.
        cache.put('a', 'answer')
        now[0] = 14
        self.assertEqual(cache.get('a'), 'answer')
        now[0] = 15
        self.assertIsNone(cache.get('a'))
        cache.put('a', '1')
        cache.put('b', '2')
        cache.put('c', '3')
        self.assertIsNone(cache.get('a'))
        self.assertEqual(len(cache.entries), 2)

    def test_interleaved_requests_do_not_share(self):
        def run(value):
            cache = current_cache()
            self.assertIsNone(cache.get('same-sql'))
            cache.put('same-sql', value)
            yield cache.id
            yield cache.get('same-sql')
        a = request_scoped_stream(run('user-a'))
        b = request_scoped_stream(run('user-b'))
        self.assertNotEqual(next(a), next(b))
        self.assertIsNone(current_cache())
        self.assertEqual(next(a), 'user-a')
        self.assertEqual(next(b), 'user-b')
        a.close()
        b.close()
        self.assertIsNone(current_cache())

    def test_threaded_requests_do_not_share(self):
        barrier = threading.Barrier(4)
        def run(value):
            def inner():
                cache = current_cache()
                cache.put('sql', str(value))
                barrier.wait(timeout=5)
                yield cache.get('sql')
            return list(request_scoped_stream(inner()))[0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(run, range(4))), ['0', '1', '2', '3'])

    def test_cancel_releases_cache(self):
        saved = []
        def inner():
            saved.append(current_cache())
            current_cache().put('key', 'result')
            yield 'partial'
        stream = request_scoped_stream(inner())
        next(stream)
        stream.close()
        self.assertEqual(len(saved[0].entries), 0)


class DatabaseBoundaryTests(unittest.TestCase):
    def test_grants_fail_closed(self):
        usage = 'GRANT USAGE ON *.* TO `reader`@`localhost`'
        select = 'GRANT SELECT ON `chatbi`.`stock_daily` TO `reader`@`localhost`'
        db_access.validate_grants([usage, select], 'chatbi')
        bad = [
            'GRANT ALL PRIVILEGES ON *.* TO `reader`@`localhost`',
            'GRANT SELECT, UPDATE ON `chatbi`.`stock_daily` TO `reader`@`localhost`',
            'GRANT SELECT ON `chatbi`.* TO `reader`@`localhost`',
            'GRANT `admin_role`@`%` TO `reader`@`localhost`',
            select + ' WITH GRANT OPTION',
        ]
        for grant in bad:
            with self.subTest(grant=grant), self.assertRaises(PermissionError):
                db_access.validate_grants([usage, select, grant], 'chatbi')

    def test_readonly_transaction_and_cleanup(self):
        engine = MagicMock()
        conn = engine.connect.return_value.__enter__.return_value
        def execute(sql):
            if sql == 'SELECT CURRENT_ROLE()':
                return MagicMock(scalar=lambda: 'NONE')
            if sql == 'SHOW GRANTS':
                return [('GRANT SELECT ON `chatbi`.`stock_daily` TO `reader`@`localhost`',)]
            return MagicMock()
        conn.exec_driver_sql.side_effect = execute
        params = {'code': "x'; DROP TABLE stock_daily; --"}
        with patch.object(db_access, 'db_url', return_value='unused'), patch.object(db_access, 'create_engine', return_value=engine), patch.object(db_access.pd, 'read_sql', side_effect=RuntimeError('query failure')) as read:
            with self.assertRaisesRegex(RuntimeError, 'query failure'):
                db_access.read_dataframe(text('SELECT close FROM stock_daily WHERE ts_code=:code'), params=params)
        commands = [call.args[0] for call in conn.exec_driver_sql.call_args_list]
        self.assertLess(commands.index('SHOW GRANTS'), commands.index('START TRANSACTION READ ONLY'))
        self.assertIn('SET SESSION MAX_EXECUTION_TIME = 5000', commands)
        self.assertEqual(read.call_args.kwargs['params'], params)
        self.assertIn(':code', str(read.call_args.args[0]))
        self.assertGreaterEqual(conn.rollback.call_count, 2)
        engine.dispose.assert_called_once()


if __name__ == '__main__':
    unittest.main()

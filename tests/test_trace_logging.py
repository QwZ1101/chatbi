import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from trace_logging import Trace, emit, observed_call, sanitize, trace_stream, traced_request
from trace_export import export_trace


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'AGENT_TRACE_DIR': str(self.root / 'logs'), 'AGENT_TRACE_MODE': 'offline_test'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def events(self, trace):
        return [json.loads(line) for line in trace.path.read_text(encoding='utf-8').splitlines()]

    def test_redaction_nested_strings_and_env_values(self):
        with patch.dict(os.environ, {'DEMO_API_KEY': 'private-example-value'}):
            data = sanitize({'params': '{"password":"keep-out","sql":"SELECT 1"}',
                             'error': 'mysql://root:db-secret@localhost/db Bearer xyz123 private-example-value',
                             'text': 'someone@example.com 13812345678',
                             'step_by_step_analysis': 'hidden thinking', 'output_tokens': 23})
        serialized = json.dumps(data)
        for secret in ('keep-out', 'db-secret', 'xyz123', 'private-example-value', 'someone@', '13812345678', 'hidden thinking'):
            self.assertNotIn(secret, serialized)
        self.assertEqual(data['output_tokens'], 23)
        self.assertEqual(data['params']['sql'], 'SELECT 1')
        self.assertNotIn('hidden-secret', sanitize('server error {"password": "hidden-secret"}'))

    def test_full_trace_export_and_failed_call(self):
        trace = Trace('test', self.root, {'question': 'input'})
        with trace.activate():
            result = observed_call('tool', lambda n: {'value': n + 1}, 2)
            trace.finish('completed', result)
        report = export_trace(trace.path, self.root / 'export.json')
        self.assertTrue(report['complete_record'])
        self.assertEqual(report['mode'], 'offline_test')
        self.assertEqual(report['events'][-1]['data']['output'], {'value': 3})
        self.assertEqual(report['events'][1]['data']['args'], [2])
        bad = Trace('test', self.root, {})
        with bad.activate():
            with self.assertRaises(ValueError):
                observed_call('bad', lambda: int('invalid'))
            bad.finish('failed')
        self.assertIn('call_failed', [e['event'] for e in self.events(bad)])

    def test_interleaved_streams_do_not_cross_contaminate(self):
        def stream(value):
            for i in range(2):
                emit('step', owner=value)
                yield [value, i]
        a, b = Trace('t', self.root, 'a'), Trace('t', self.root, 'b')
        ga, gb = trace_stream(a, stream('a')), trace_stream(b, stream('b'))
        next(ga)
        next(gb)
        emit('must_not_attach_to_either')
        list(ga)
        list(gb)
        for trace, owner in ((a, 'a'), (b, 'b')):
            events = self.events(trace)
            self.assertNotIn('must_not_attach_to_either', [e['event'] for e in events])
            self.assertEqual([e['data']['owner'] for e in events if e['event'] == 'step'], [owner, owner])

    def test_cancellation_is_not_success(self):
        trace = Trace('t', self.root, {})
        def stream():
            yield 'partial'
            yield 'later'
        gen = trace_stream(trace, stream())
        next(gen)
        gen.close()
        self.assertEqual(self.events(trace)[-1]['data']['status'], 'cancelled')

    def test_concurrent_requests_have_independent_ids(self):
        @traced_request('parallel', self.root)
        def run(n):
            return observed_call('square', lambda: n * n)
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(run, range(8))), [n*n for n in range(8)])
        paths = list((self.root / 'logs').rglob('*.jsonl'))
        self.assertEqual(len(paths), 8)
        for i, path in enumerate(paths):
            self.assertTrue(export_trace(path, self.root / ('export%d.json' % i))['complete_record'])

    def test_truncation_and_incomplete_trace_are_explicit(self):
        trace = Trace('t', self.root, {})
        trace.record('large_result', result='x' * 600000)
        trace.finish('completed')
        self.assertFalse(export_trace(trace.path, self.root / 'large.json')['complete_record'])
        missing = Trace('t', self.root, {})
        self.assertFalse(export_trace(missing.path, self.root / 'missing.json')['complete_record'])

    def test_writer_failure_does_not_break_business_call(self):
        @traced_request('test', self.root)
        def run():
            return 42
        with patch.object(Path, 'open', side_effect=PermissionError('no write')), self.assertWarns(RuntimeWarning):
            self.assertEqual(run(), 42)

    def test_decorator_keeps_exception(self):
        @traced_request('test', self.root)
        def run(question):
            raise RuntimeError('expected failure')
        with self.assertRaisesRegex(RuntimeError, 'expected failure'):
            run('question')
        events = [json.loads(l) for l in next((self.root / 'logs').rglob('*.jsonl')).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(events[-1]['data']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

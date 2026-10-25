"""Local, redacted execution traces. Python 3.9+, standard library only.

These records describe observed execution, not answer correctness. Never use
offline_test traces as evidence of real model/database calls.
"""
import contextvars
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

_current = contextvars.ContextVar('agent_trace', default=None)
_span = contextvars.ContextVar('agent_span', default=None)
_secret_key = re.compile(r'api.?key|password|passwd|secret|authorization|cookie|access.?token|refresh.?token', re.I)
_omit_key = {'step_by_step_analysis', 'reasoning_content'}


def _safe_text(value):
    # Read at emission time: .env may have been loaded after module import.
    for key, secret in os.environ.items():
        if _secret_key.search(key) and len(secret) >= 4:
            value = value.replace(secret, '[REDACTED]')
    value = re.sub(r'(?i)(://)[^\s/@:]+:[^\s/@]+@', r'\1[REDACTED]@', value)
    value = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+', 'Bearer [REDACTED]', value)
    value = re.sub(r'''(?i)((?:["']?(?:api[_-]?key|password|passwd|secret|token)["']?)\s*:\s*)(["'])(.*?)\2''', r'\1"[REDACTED]"', value)
    value = re.sub(r'\b(?:sk-|tvly-)[A-Za-z0-9_-]{8,}', '[REDACTED]', value)
    value = re.sub(r'(?i)((?:api[_-]?key|password|passwd|secret|token)\s*[=:]\s*)[^\s,;&]+', r'\1[REDACTED]', value)
    if '@' in value:
        value = re.sub(r'[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}', '[EMAIL]', value)
    return re.sub(r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)', '[PHONE]', value)


def sanitize(value, depth=0):
    if depth > 24:
        return {'truncated': True, 'reason': 'nesting_limit'}
    if hasattr(value, 'model_dump'):
        value = value.model_dump()
    if isinstance(value, dict):
        return {str(k): ('[OMITTED]' if str(k) in _omit_key else
                         '[REDACTED]' if _secret_key.search(str(k)) or str(k).lower() == 'token' else
                         sanitize(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, depth + 1) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    # Tools commonly receive JSON encoded as a string; redact inside it too.
    if text.lstrip().startswith(('{', '[')):
        try:
            return sanitize(json.loads(text), depth + 1)
        except (ValueError, TypeError):
            pass
    return _safe_text(text)


def code_version(root):
    root = Path(root)
    result = {'commit': None, 'dirty': None}
    try:
        args = ['git', '-C', str(root)]
        result['commit'] = subprocess.check_output(args + ['rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        result['dirty'] = bool(subprocess.check_output(args + ['status', '--porcelain'], stderr=subprocess.DEVNULL, timeout=3).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    # This digest includes uncommitted Python changes, unlike commit alone.
    digest = hashlib.sha256()
    files = list(root.glob('*.py')) + list((root / 'src').glob('*.py'))
    for name in ('requirements.txt', 'faq.txt'):
        if (root / name).is_file():
            files.append(root / name)
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode('utf-8'))
        digest.update(path.read_bytes())
    result['source_sha256'] = digest.hexdigest()
    result['digest_scope'] = 'root/*.py, src/*.py, requirements.txt, faq.txt'
    return result


class Trace:
    def __init__(self, project, root, input_data, metadata=None):
        self.id = uuid.uuid4().hex
        self.project = project
        self.mode = os.getenv('AGENT_TRACE_MODE', 'live')
        if self.mode not in ('live', 'offline_test'):
            raise ValueError('AGENT_TRACE_MODE must be live or offline_test')
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        base = Path(os.getenv('AGENT_TRACE_DIR') or Path(root) / 'logs' / 'traces')
        self.path = base / day / (self.id + '.jsonl')
        self.lock = threading.RLock()
        self.seq = 0
        self.start_time = time.monotonic()
        self.failed = False
        self.truncated = False
        self.errors = 0
        self.closed = False
        self.size = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=False)
            version = code_version(root)
        except Exception:
            version = {'unavailable': True}
        self.record('request_start', input=input_data, metadata=metadata or {}, code_version=version)

    @contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def record(self, event, **data):
        with self.lock:
            if self.failed or self.closed:
                return
            try:
                safe = sanitize(data)
                payload = json.dumps(safe, ensure_ascii=False, default=str)
                if len(payload.encode('utf-8')) > 512 * 1024 or (self.size > 8 * 1024 * 1024 and event != 'request_end'):
                    self.truncated = True
                    safe = {'truncated': True, 'reason': 'trace_size_limit',
                            'redacted_payload_sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
                            'redacted_payload_characters': len(payload)}
                self.seq += 1
                entry = {'schema_version': 1, 'trace_id': self.id, 'project': self.project,
                         'mode': self.mode, 'seq': self.seq,
                         'time_utc': datetime.now(timezone.utc).isoformat(),
                         'event': event, 'span_id': _span.get(), 'data': safe}
                line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n'
                with self.path.open('a', encoding='utf-8') as out:
                    out.write(line)
                self.size += len(line.encode('utf-8'))
            except Exception:
                self.failed = True
                warnings.warn('Trace write failed; this request has no complete trace. Check log directory access.', RuntimeWarning)

    def finish(self, status, output=None):
        with self.lock:
            if self.closed:
                return
            self.record('request_end', status=status, output=output,
                        handled_errors=self.errors, truncated=self.truncated,
                        elapsed_ms=round((time.monotonic() - self.start_time) * 1000, 2),
                        interpretation='execution record only; answer correctness not evaluated')
            self.closed = True


def emit(event, **data):
    trace = _current.get()
    if trace:
        trace.record(event, **data)


def record_error(error, handled=False):
    trace = _current.get()
    if trace:
        trace.errors += 1
        # Do not persist stack locals, credentials or chain-of-thought.
        trace.record('handled_error' if handled else 'error',
                     error_type=type(error).__name__, message=str(error))


def record_tool_exceptions(function):
    """Capture errors before Qwen Agent converts exceptions to tool-result text."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            record_error(error)
            raise
    return wrapper


def observed_call(name, function, *args, **kwargs):
    trace = _current.get()
    if trace is None:
        return function(*args, **kwargs)
    parent = _span.get()
    token = _span.set(uuid.uuid4().hex)
    started = time.monotonic()
    try:
        emit('call_start', name=name, parent_span_id=parent, args=args, kwargs=kwargs)
        result = function(*args, **kwargs)
        emit('call_return', name=name, result=result,
             elapsed_ms=round((time.monotonic() - started) * 1000, 2),
             interpretation='function returned; business success not inferred')
        return result
    except BaseException as error:
        record_error(error)
        emit('call_failed', name=name, elapsed_ms=round((time.monotonic() - started) * 1000, 2))
        raise
    finally:
        _span.reset(token)


def traced_request(project, root):
    def decorate(function):
        signature = inspect.signature(function)
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if _current.get() is not None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            inputs = {k: v for k, v in bound.arguments.items() if k not in ('self', 'cls')}
            trace = Trace(project, root, inputs, {'entrypoint': function.__qualname__})
            with trace.activate():
                try:
                    result = function(*args, **kwargs)
                    output = result
                    status = 'completed_with_errors' if trace.errors else 'completed'
                    if hasattr(result, 'status_code') and hasattr(result, 'body'):
                        try:
                            output = {'status_code': result.status_code, 'body': json.loads(result.body)}
                        except (ValueError, TypeError):
                            output = {'status_code': result.status_code, 'body': '[non-JSON response]'}
                        if result.status_code >= 400:
                            status = 'failed'
                    trace.finish(status, output)
                    return result
                except BaseException as error:
                    record_error(error)
                    trace.finish('cancelled' if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else 'failed')
                    raise
        return wrapper
    return decorate


def trace_stream(trace, iterator):
    """Reset ContextVar before each yield, preventing interleaved GUI sessions leaking."""
    final = None
    try:
        while True:
            with trace.activate():
                try:
                    final = next(iterator)
                except StopIteration:
                    trace.finish('completed_with_errors' if trace.errors else 'completed', final)
                    return
            yield final
    except BaseException as error:
        with trace.activate():
            record_error(error)
            trace.finish('cancelled' if isinstance(error, (GeneratorExit, KeyboardInterrupt)) else 'failed')
        raise
    finally:
        with trace.activate():
            close = getattr(iterator, 'close', None)
            if close:
                close()

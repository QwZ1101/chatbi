"""Request-local caching. No cross-user/session authentication assumptions."""
from collections import OrderedDict
from contextvars import ContextVar
import threading
import time
import uuid

_current = ContextVar('chatbi_request_cache', default=None)


class RequestCache:
    def __init__(self, ttl=5, max_entries=32, clock=time.monotonic):
        self.id = uuid.uuid4().hex
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            now = self.clock()
            for old in list(self.entries):
                if self.entries[old][0] <= now:
                    del self.entries[old]
            entry = self.entries.get(key)
            if entry is None:
                return None
            self.entries.move_to_end(key)
            return entry[1]

    def put(self, key, result):
        if len(result.encode('utf-8')) > 256 * 1024:
            return
        with self.lock:
            self.entries[key] = (self.clock() + self.ttl, result)
            self.entries.move_to_end(key)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)


def current_cache():
    return _current.get()


def request_scoped_stream(iterator):
    cache = RequestCache()
    try:
        while True:
            token = _current.set(cache)
            try:
                item = next(iterator)
            except StopIteration:
                return
            finally:
                _current.reset(token)
            yield item
    finally:
        cache.entries.clear()
        close = getattr(iterator, 'close', None)
        if close:
            close()

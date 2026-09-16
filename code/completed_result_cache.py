"""Bounded, process-local storage of completed server results for export."""

from collections import OrderedDict
from hashlib import sha256
import json
from threading import Lock
from time import monotonic
import zlib


def completed_result_key(inputs: dict, model: dict) -> str:
    payload = json.dumps(
        {"inputs": inputs, "model": model}, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class CompletedResultCache:
    def __init__(self, *, max_bytes=32 * 1024**2, max_entries=4,
                 max_result_bytes=128 * 1024**2, ttl_seconds=1800, clock=monotonic):
        if min(max_bytes, max_entries, max_result_bytes, ttl_seconds) <= 0:
            raise ValueError("completed-result cache limits must be positive")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.max_result_bytes = max_result_bytes
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def _remove(self, key):
        _, payload = self._entries.pop(key)
        self._bytes -= len(payload)

    def _expire(self, now):
        for key, (deadline, _) in list(self._entries.items()):
            if now >= deadline:
                self._remove(key)

    def get(self, key):
        with self._lock:
            self._expire(self.clock())
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            payload = entry[1]
        # Decode separately for each caller: workbook construction cannot mutate
        # another user's stored result. Only server-produced bytes are decoded.
        return json.loads(zlib.decompress(payload))

    def put(self, key, result):
        compressor = zlib.compressobj(level=1)
        compressed = bytearray()
        parts, chars, total = [], 0, 0
        encoder = json.JSONEncoder(separators=(",", ":"), allow_nan=False)
        # Stream encoding avoids a second full-size JSON string for dense fields.
        for token in encoder.iterencode(result):
            parts.append(token)
            chars += len(token)
            if chars < 65536:
                continue
            block = "".join(parts).encode("utf-8")
            total += len(block)
            if total > self.max_result_bytes:
                return False
            compressed.extend(compressor.compress(block))
            if len(compressed) > self.max_bytes:
                return False
            parts, chars = [], 0
        block = "".join(parts).encode("utf-8")
        if total + len(block) > self.max_result_bytes:
            return False
        compressed.extend(compressor.compress(block))
        compressed.extend(compressor.flush())
        if len(compressed) > self.max_bytes:
            return False
        payload = bytes(compressed)
        with self._lock:
            now = self.clock()
            self._expire(now)
            if key in self._entries:
                self._remove(key)
            while self._entries and (
                self._bytes + len(payload) > self.max_bytes
                or len(self._entries) >= self.max_entries
            ):
                self._remove(next(iter(self._entries)))
            self._entries[key] = (now + self.ttl_seconds, payload)
            self._bytes += len(payload)
        return True

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

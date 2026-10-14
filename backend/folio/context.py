"""Process-wide clients, created on first use and reused across Function invocations."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Settings, get_settings
from .search_index import ChunkIndex
from .storage import Store


@dataclass
class Context:
    settings: Settings
    store: Store
    index: ChunkIndex


_lock = threading.Lock()
_context: Context | None = None


def get_context() -> Context:
    """Build the clients and make sure containers, tables, and the index exist (once per process)."""
    global _context
    if _context is None:
        with _lock:
            if _context is None:
                settings = get_settings()
                store = Store(settings)
                index = ChunkIndex(settings)
                store.ensure_resources()
                index.ensure_index()
                _context = Context(settings, store, index)
    return _context

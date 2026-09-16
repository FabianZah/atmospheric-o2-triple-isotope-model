"""Pytest/path bootstrap for the publication snapshot.

The archive keeps the importable model core in ``code/`` and the validation,
audit, and scoring scripts in ``validation/``. Several validation scripts import
core modules, and a few reporting scripts in ``code/`` import scoring helpers
from ``validation/``. To keep every entry point runnable without installing the
package, both directories are placed on ``sys.path`` here. Pytest imports this
file automatically; standalone scripts can do the same by running from the
snapshot root or by adding ``code/`` (and ``validation/``) to ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("code", "validation"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

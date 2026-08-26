"""Small cross-version shims.

`SLOTS` lets dataclasses opt into ``slots=True`` where the runtime supports it
(Python 3.10+) and degrade gracefully on 3.9.  Slots are a memory optimisation
-- this package builds hundreds of thousands of `Match` and `Bet` instances --
not a semantic requirement, so losing them on an older interpreter is strictly
better than refusing to run there.

Usage::

    from wcq._compat import SLOTS

    @dataclass(frozen=True, **SLOTS)
    class Thing: ...
"""
from __future__ import annotations

import sys

SLOTS: dict = {"slots": True} if sys.version_info >= (3, 10) else {}

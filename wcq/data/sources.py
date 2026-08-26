"""Dataset registry, download and on-disk cache.

Design notes
------------
* Every source is declared once in `SOURCES` with a URL and a SHA-256 that is
  recorded on first download.  Re-running a backtest months later against a
  silently-updated CSV is one of the easiest ways to produce a number you
  cannot reproduce, so the cache is content-addressed and the digest is
  written next to the file.
* Downloads are atomic (temp file + rename) so an interrupted run can never
  leave a half-written CSV that looks valid to the loader.
* Nothing here parses.  Fetching and parsing are separate so the parsers can
  be unit-tested against small fixtures with no network at all.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("WCQ_CACHE", Path.home() / ".cache" / "wcq"))


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    kind: str          # 'file' | 'git'
    description: str
    licence: str

    @property
    def path(self) -> Path:
        return DEFAULT_CACHE / self.filename


SOURCES: dict[str, Source] = {
    "international": Source(
        name="international",
        url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        filename="international_results.csv",
        kind="file",
        description=(
            "Every men's international football result since 1872-11-30. "
            "~49,500 rows: date, teams, score, tournament, venue, neutral flag."
        ),
        licence="CC BY 4.0 (martj42/international_results)",
    ),
    "club_matches": Source(
        name="club_matches",
        url="https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
        filename="club_matches.csv",
        kind="file",
        description=(
            "~230,000 club matches, 38 divisions, 2000-2025, mirroring "
            "football-data.co.uk.  Carries Bet365 1X2 prices and the best "
            "price across ~17 European books.  This is the only table in the "
            "project with real bookmaker prices attached, so it is where the "
            "betting backtest runs."
        ),
        licence="CC0 (xgabora/Club-Football-Match-Data-2000-2025)",
    ),
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(name: str, *, force: bool = False, cache: Path | None = None) -> Path:
    """Return a local path to the named dataset, downloading it if needed.

    Idempotent: a second call is a hash check, not a download.  A cache hit
    is verified against the digest recorded at download time -- a
    content-addressed cache that never re-reads the content is only a naming
    convention, and a corrupted or half-written file would otherwise be
    served forever.
    """
    src = SOURCES[name]
    root = cache or DEFAULT_CACHE
    root.mkdir(parents=True, exist_ok=True)
    dest = root / src.filename
    sidecar = root / (src.filename + ".sha256")

    if dest.exists() and not force:
        if not sidecar.exists() or _sha256(dest) == sidecar.read_text().strip():
            return dest
        raise RuntimeError(
            f"{dest} does not match its recorded SHA-256; the cached file is "
            "corrupt. Re-download with fetch(name, force=True)."
        )

    tmp_fd, tmp_name = tempfile.mkstemp(dir=root, suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    side_tmp = Path(tmp_name + ".sha256")
    try:
        req = urllib.request.Request(src.url, headers={"User-Agent": "wcq/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        digest = _sha256(tmp)
        # Sidecar goes live first: at every instant the visible (file, digest)
        # pair is either the complete old one or the complete new one.
        side_tmp.write_text(digest + "\n")
        side_tmp.replace(sidecar)              # atomic within one filesystem
        tmp.replace(dest)
    finally:
        for leftover in (tmp, side_tmp):
            if leftover.exists():
                leftover.unlink()
    return dest


def verify(name: str, cache: Path | None = None) -> bool:
    """Check the cached file still matches the digest recorded at download."""
    src = SOURCES[name]
    root = cache or DEFAULT_CACHE
    dest, sidecar = root / src.filename, root / (src.filename + ".sha256")
    if not dest.exists() or not sidecar.exists():
        return False
    return _sha256(dest) == sidecar.read_text().strip()


def fetch_all(force: bool = False) -> dict[str, Path]:
    return {n: fetch(n, force=force) for n in SOURCES}

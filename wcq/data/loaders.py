"""CSV -> `Match` parsers.

Each loader is a pure function from a path to a chronologically sorted list of
`Match`.  They are streamed with `csv.reader` rather than pandas: the club file
is ~44 MB and 50 columns wide, we need 12 of them, and a generator keeps peak
memory to a single row instead of a 230k x 50 frame.  It also means the package
has no hard pandas dependency.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Iterator, Optional

from wcq.schema import Match, Odds, sort_chronologically

# Competitions treated as "major tournament" for the World Cup pipeline.
TOURNAMENT_KEYWORDS = ("FIFA World Cup", "UEFA Euro", "Copa América", "Copa America")


def _f(x: str) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None       # NaN check


def _i(x: str) -> Optional[int]:
    v = _f(x)
    return None if v is None else int(v)


def _date(x: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(x.strip()[:10])
    except (ValueError, AttributeError):
        return None


def iter_international(path: Path) -> Iterator[Match]:
    """martj42 international results: date,home_team,away_team,home_score,
    away_score,tournament,city,country,neutral"""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = _date(row["date"])
            hg, ag = _i(row["home_score"]), _i(row["away_score"])
            if d is None or hg is None or ag is None:
                continue
            yield Match(
                date=d,
                home=row["home_team"].strip(),
                away=row["away_team"].strip(),
                competition=row["tournament"].strip(),
                home_goals=hg,
                away_goals=ag,
                neutral=str(row.get("neutral", "")).strip().upper() in ("TRUE", "1"),
                odds=None,                       # no price history for these
                source="martj42/international_results",
            )


def iter_club(path: Path, *, price: str = "b365", divisions: set[str] | None = None) -> Iterator[Match]:
    """Club matches with real bookmaker prices.

    `price` selects which column set becomes the tradeable line:
      'b365'  -> Bet365's own price (a single book; what you could actually bet)
      'best'  -> best price across ~17 books (an upper bound you would only get
                 with accounts everywhere, and the first place a naive backtest
                 manufactures fake edge)
    """
    cols = {
        "b365": ("OddHome", "OddDraw", "OddAway", "Bet365"),
        "best": ("MaxHome", "MaxDraw", "MaxAway", "BestOf17"),
    }[price]
    h_col, d_col, a_col, book = cols

    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = _date(row["MatchDate"])
            hg, ag = _i(row["FTHome"]), _i(row["FTAway"])
            if d is None or hg is None or ag is None:
                continue
            div = row["Division"].strip()
            if divisions and div not in divisions:
                continue
            oh, od, oa = _f(row[h_col]), _f(row[d_col]), _f(row[a_col])
            odds = None
            if None not in (oh, od, oa):
                cand = Odds(oh, od, oa, book=book, taken="prematch")
                odds = cand if cand.is_valid() else None
            yield Match(
                date=d,
                home=row["HomeTeam"].strip(),
                away=row["AwayTeam"].strip(),
                competition=div,
                home_goals=hg,
                away_goals=ag,
                neutral=False,
                odds=odds,
                source="football-data.co.uk (via xgabora mirror)",
            )


def iter_football_data_native(path: Path, *, prefer_closing: bool = True) -> Iterator[Match]:
    """Parser for a raw football-data.co.uk season CSV downloaded directly.

    Those files carry both an opening and a closing Pinnacle line
    (PSH/PSD/PSA vs PSCH/PSCD/PSCA).  When the closing columns are present we
    take them -- see the note in `schema.Odds` on why that matters.  Kept
    separate from `iter_club` so the mirror can disappear without breaking the
    path to the primary source.
    """
    close = ("PSCH", "PSCD", "PSCA")
    open_ = ("PSH", "PSD", "PSA")
    fallback = ("B365H", "B365D", "B365A")

    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or ())
        if prefer_closing and close[0] in fields:
            cols, book, taken = close, "Pinnacle", "close"
        elif open_[0] in fields:
            cols, book, taken = open_, "Pinnacle", "open"
        else:
            cols, book, taken = fallback, "Bet365", "prematch"

        for row in reader:
            raw = row.get("Date", "")
            d = None
            for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
                try:
                    d = dt.datetime.strptime(raw.strip(), fmt).date()
                    break
                except ValueError:
                    continue
            hg, ag = _i(row.get("FTHG", "")), _i(row.get("FTAG", ""))
            if d is None or hg is None or ag is None:
                continue
            vals = [_f(row.get(c, "")) for c in cols]
            odds = None
            if None not in vals:
                cand = Odds(*vals, book=book, taken=taken)
                odds = cand if cand.is_valid() else None
            yield Match(
                date=d, home=row["HomeTeam"].strip(), away=row["AwayTeam"].strip(),
                competition=row.get("Div", "").strip(), home_goals=hg, away_goals=ag,
                odds=odds, source="football-data.co.uk",
            )


def load_international(path: Path, tournaments_only: bool = False) -> list[Match]:
    ms = list(iter_international(path))
    if tournaments_only:
        ms = [m for m in ms if any(k in m.competition for k in TOURNAMENT_KEYWORDS)]
    return sort_chronologically(ms)


def load_club(path: Path, **kw) -> list[Match]:
    return sort_chronologically(list(iter_club(path, **kw)))

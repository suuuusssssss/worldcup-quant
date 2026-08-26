"""Parser tests against tiny inline fixtures -- no network, no cached files."""
import datetime as dt

import pytest

from wcq.data.loaders import (iter_club, iter_football_data_native, iter_international,
                              load_international)

INTL = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE
2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE
2023-01-01,A,B,,,Friendly,X,Y,FALSE
bad-date,C,D,1,0,Friendly,X,Y,FALSE
"""

CLUB = """Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,FTHome,FTAway,FTResult,OddHome,OddDraw,OddAway,MaxHome,MaxDraw,MaxAway
E0,2020-08-15,14:00:00,Arsenal,Chelsea,1800,1790,2,1,H,2.40,3.30,3.00,2.55,3.45,3.20
SP1,2020-08-16,16:00:00,Barcelona,Sevilla,1900,1750,0,0,D,1.50,4.20,6.00,1.58,4.50,6.60
E0,2020-08-17,14:00:00,Spurs,Everton,1700,1690,,,,2.10,3.40,3.60,2.20,3.50,3.80
E0,2020-08-18,14:00:00,Leeds,Burnley,1600,1580,1,2,A,,,,,,
E0,2020-08-19,14:00:00,Fulham,Brentford,1550,1560,3,1,H,0.90,3.00,4.00,1.00,3.10,4.10
"""

NATIVE_CLOSING = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,PSCH,PSCD,PSCA
E0,12/08/2023,Burnley,Man City,0,3,A,7.50,4.50,1.44,7.60,4.60,1.45,8.00,4.80,1.40
"""

NATIVE_OPENING = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA
E0,12/08/2023,Burnley,Man City,0,3,A,7.50,4.50,1.44,7.60,4.60,1.45
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_international_parses_and_skips_bad_rows(tmp_path):
    ms = list(iter_international(write(tmp_path, "i.csv", INTL)))
    assert len(ms) == 2                       # blank score and bad date dropped
    assert ms[0].date == dt.date(1872, 11, 30)
    assert ms[0].result == "D"
    assert ms[1].neutral is True
    assert ms[1].competition == "FIFA World Cup"
    assert all(m.odds is None for m in ms)


def test_tournament_filter(tmp_path):
    p = write(tmp_path, "i.csv", INTL)
    assert len(load_international(p, tournaments_only=True)) == 1


def test_club_parses_prices_and_drops_unplayed(tmp_path):
    ms = list(iter_club(write(tmp_path, "c.csv", CLUB)))
    assert len(ms) == 4                       # unplayed fixture dropped
    assert ms[0].odds.as_tuple() == (2.40, 3.30, 3.00)
    assert ms[0].odds.book == "Bet365"
    assert ms[0].odds.overround > 1.0


def test_club_best_price_selection(tmp_path):
    ms = list(iter_club(write(tmp_path, "c.csv", CLUB), price="best"))
    assert ms[0].odds.as_tuple() == (2.55, 3.45, 3.20)
    assert ms[0].odds.book == "BestOf17"


def test_sub_hundred_percent_best_price_book_is_kept(tmp_path):
    """Regression test for a silent sampling bug.

    Best-of-market prices across ~17 books can sum to under 100%. Rejecting
    those rows as malformed deleted exactly the matches where the books
    disagreed most -- a selection bias with no error message attached.
    """
    ms = list(iter_club(write(tmp_path, "c.csv", CLUB), price="best"))
    o = ms[0].odds
    assert o.overround < 1.0
    assert o.is_valid()                 # structurally fine
    assert not o.has_positive_margin    # but flagged as a sub-100% book


def test_single_book_prices_carry_a_real_margin(tmp_path):
    ms = list(iter_club(write(tmp_path, "c.csv", CLUB), price="b365"))
    assert ms[0].odds.has_positive_margin


def test_club_division_filter(tmp_path):
    ms = list(iter_club(write(tmp_path, "c.csv", CLUB), divisions={"SP1"}))
    assert [m.home for m in ms] == ["Barcelona"]


def test_missing_odds_yield_a_match_with_no_price(tmp_path):
    ms = {m.home: m for m in iter_club(write(tmp_path, "c.csv", CLUB))}
    assert ms["Leeds"].odds is None
    assert ms["Leeds"].result == "A"


def test_impossible_odds_are_rejected_not_propagated(tmp_path):
    """A decimal odd of 0.90 implies a probability above 1. Letting that
    through would produce a negative-probability 'edge' downstream, so the
    price is discarded while the result is kept."""
    ms = {m.home: m for m in iter_club(write(tmp_path, "c.csv", CLUB))}
    assert ms["Fulham"].odds is None


def test_native_loader_prefers_the_closing_line(tmp_path):
    m = next(iter_football_data_native(write(tmp_path, "n.csv", NATIVE_CLOSING)))
    assert m.odds.taken == "close"
    assert m.odds.as_tuple() == (8.00, 4.80, 1.40)
    assert m.date == dt.date(2023, 8, 12)     # dd/mm/yyyy parsed correctly


def test_native_loader_falls_back_to_the_opening_line(tmp_path):
    m = next(iter_football_data_native(write(tmp_path, "n.csv", NATIVE_OPENING)))
    assert m.odds.taken == "open"
    assert m.odds.as_tuple() == (7.60, 4.60, 1.45)


def test_native_loader_can_be_told_not_to_prefer_closing(tmp_path):
    m = next(iter_football_data_native(write(tmp_path, "n.csv", NATIVE_CLOSING),
                                       prefer_closing=False))
    assert m.odds.taken == "open"

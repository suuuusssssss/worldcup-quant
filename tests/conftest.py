import datetime as dt
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcq.schema import Match, Odds


def make_match(day: int, home: str, away: str, hg: int, ag: int, *,
               comp: str = "T", neutral: bool = False, odds=None) -> Match:
    return Match(
        date=dt.date(2020, 1, 1) + dt.timedelta(days=day),
        home=home, away=away, competition=comp,
        home_goals=hg, away_goals=ag, neutral=neutral,
        odds=Odds(*odds) if odds else None,
    )


@pytest.fixture(scope="session")
def synthetic_season():
    """A deterministic multi-season league: 12 teams, six double round-robins,
    one fixture per day.  Results are drawn from fixed latent strengths, so
    there is real signal for the model to find, and the ~2.2-year span means
    the walk-forward harness actually crosses several refit boundaries."""
    rng = random.Random(1234)
    teams = [f"T{i:02d}" for i in range(12)]
    strength = {t: rng.gauss(0, 0.35) for t in teams}
    matches, day = [], 0
    for _ in range(6):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                for home, away in ((h, a), (a, h)):
                    lam_h = 1.4 * math.exp(strength[home] - strength[away] + 0.2)
                    lam_a = 1.4 * math.exp(strength[away] - strength[home])
                    hg = sum(1 for _ in range(12) if rng.random() < lam_h / 12)
                    ag = sum(1 for _ in range(12) if rng.random() < lam_a / 12)
                    matches.append(make_match(day, home, away, hg, ag))
                    day += 1
    return matches

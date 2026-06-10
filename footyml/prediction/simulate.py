"""Monte Carlo tournament simulator for FIFA World Cup 2026.

Loads the per-match probability predictions produced by predict_tournament.py,
simulates the full 48-team bracket (group stage + knockout) N times, and
returns win/finalist/semi-finalist probabilities for every team.

WC 2026 format
--------------
* 48 teams, 12 groups (A–L) of 4 teams each
* Top-2 from each group + 8 best 3rd-place teams = 32 advance
* Bracket is split into 4 paths of 8 teams each:
    Path 1: groups A, B, C   |   Path 3: groups G, H, I
    Path 2: groups D, E, F   |   Path 4: groups J, K, L
  Within each path:
    R32: (G1 vs G2') (G2 vs G1') (G3 vs 3rd) (G3' vs 3rd)
    R16: (R32-1w vs R32-2w) (R32-3w vs R32-4w)
    QF:  R16-1w vs R16-2w   → SF participant
* SF1: Path-1-QF vs Path-2-QF   SF2: Path-3-QF vs Path-4-QF
* Final + 3rd-place playoff
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from footyml.config import ELO_INITIAL

# ---------------------------------------------------------------------------
# Bracket definition
# ---------------------------------------------------------------------------

WC2026_PATHS: list[tuple[str, str, str]] = [
    ("GROUP_A", "GROUP_B", "GROUP_C"),
    ("GROUP_D", "GROUP_E", "GROUP_F"),
    ("GROUP_G", "GROUP_H", "GROUP_I"),
    ("GROUP_J", "GROUP_K", "GROUP_L"),
]

ROUNDS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamStats:
    name: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


class SimResult(NamedTuple):
    champion: str
    finalist: str
    third: str
    fourth: str
    semifinalists: list[str]


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------

def _derive_team_elo(group_matches: dict[str, list[dict]]) -> dict[str, float]:
    """Back-calculate team Elo ratings from group-stage win probabilities.

    Iteratively solves the system: elo_h - elo_a = 400 * log10(p_h / p_a)
    where p_h, p_a are the draw-adjusted win probabilities from predictions.
    Converges in ~20 iterations.
    """
    from collections import defaultdict

    elo: dict[str, float] = defaultdict(lambda: ELO_INITIAL)

    for _ in range(25):
        updates: dict[str, list[float]] = defaultdict(list)
        for matches in group_matches.values():
            for m in matches:
                h = float(m.get("home_win_prob") or 37.5) / 100.0
                a = float(m.get("away_win_prob") or 37.5) / 100.0
                if h + a < 0.01:
                    continue
                p_h = h / (h + a)
                home, away = m["home_team"], m["away_team"]
                if not home or not away:
                    continue
                implied = 400.0 * np.log10(max(p_h, 1e-4) / max(1.0 - p_h, 1e-4))
                updates[home].append(elo[away] + implied)
                updates[away].append(elo[home] - implied)
        for team, vals in updates.items():
            elo[team] = float(np.mean(vals))

    return dict(elo)


def _ko_p_win(team_a: str, team_b: str, elo: dict[str, float]) -> float:
    """Return P(team_a wins) in a knockout match (no draw)."""
    ea = elo.get(team_a, ELO_INITIAL)
    eb = elo.get(team_b, ELO_INITIAL)
    return 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))


# ---------------------------------------------------------------------------
# Group-stage simulation helpers
# ---------------------------------------------------------------------------

def _goals(outcome: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a plausible scoreline for an outcome (0=away, 1=draw, 2=home)."""
    if outcome == 2:
        hg = int(rng.integers(1, 4))
        ag = int(rng.integers(0, hg))
    elif outcome == 0:
        ag = int(rng.integers(1, 4))
        hg = int(rng.integers(0, ag))
    else:
        g = int(rng.integers(0, 3))
        hg, ag = g, g
    return hg, ag


def _simulate_group(
    matches: list[dict],
    outcomes: np.ndarray,
    ko_rng: np.random.Generator,
) -> list[TeamStats]:
    """Simulate one group from pre-sampled outcomes; return sorted standings."""
    teams: dict[str, TeamStats] = {}

    for match, outcome in zip(matches, outcomes):
        home, away = match["home_team"], match["away_team"]
        if home not in teams:
            teams[home] = TeamStats(home)
        if away not in teams:
            teams[away] = TeamStats(away)

        hg, ag = _goals(int(outcome), ko_rng)

        st_h, st_a = teams[home], teams[away]
        st_h.gf += hg
        st_h.ga += ag
        st_a.gf += ag
        st_a.ga += hg

        if outcome == 2:
            st_h.pts += 3
            st_h.wins += 1
            st_a.losses += 1
        elif outcome == 0:
            st_a.pts += 3
            st_a.wins += 1
            st_h.losses += 1
        else:
            st_h.pts += 1
            st_a.pts += 1
            st_h.draws += 1
            st_a.draws += 1

    tiebreak = ko_rng.random(len(teams))
    return sorted(
        teams.values(),
        key=lambda t: (t.pts, t.gd, t.gf, tiebreak[list(teams.keys()).index(t.name)]),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Knockout simulation
# ---------------------------------------------------------------------------

def _ko(team_a: str, team_b: str, elo: dict[str, float], rand: float) -> str:
    return team_a if rand < _ko_p_win(team_a, team_b, elo) else team_b


def _simulate_knockout(
    standings: dict[str, list[TeamStats]],
    best_thirds: list[TeamStats],
    elo: dict[str, float],
    ko_randoms: np.ndarray,
) -> SimResult:
    """Simulate knockout bracket from pre-sampled random values."""
    thirds = list(best_thirds)
    ri = 0  # index into ko_randoms

    path_qf_winners: list[str] = []
    sf_losers: list[str] = []

    for g1, g2, g3 in WC2026_PATHS:
        s1, s2, s3 = standings[g1], standings[g2], standings[g3]
        t3a = thirds.pop(0).name if thirds else "Unknown"
        t3b = thirds.pop(0).name if thirds else "Unknown"

        # Round of 32 (4 matches per path)
        r32_1 = _ko(s1[0].name, s2[1].name, elo, ko_randoms[ri]); ri += 1
        r32_2 = _ko(s2[0].name, s1[1].name, elo, ko_randoms[ri]); ri += 1
        r32_3 = _ko(s3[0].name, t3a,         elo, ko_randoms[ri]); ri += 1
        r32_4 = _ko(s3[1].name, t3b,         elo, ko_randoms[ri]); ri += 1

        # Round of 16 (2 matches per path)
        r16_1 = _ko(r32_1, r32_2, elo, ko_randoms[ri]); ri += 1
        r16_2 = _ko(r32_3, r32_4, elo, ko_randoms[ri]); ri += 1

        # Quarter-final (1 match per path)
        qf_w = _ko(r16_1, r16_2, elo, ko_randoms[ri]); ri += 1
        path_qf_winners.append(qf_w)

    # Semi-finals
    sf1_w = _ko(path_qf_winners[0], path_qf_winners[1], elo, ko_randoms[ri]); ri += 1
    sf1_l = path_qf_winners[0] if sf1_w == path_qf_winners[1] else path_qf_winners[1]
    sf2_w = _ko(path_qf_winners[2], path_qf_winners[3], elo, ko_randoms[ri]); ri += 1
    sf2_l = path_qf_winners[2] if sf2_w == path_qf_winners[3] else path_qf_winners[3]

    # 3rd-place playoff
    third  = _ko(sf1_l, sf2_l, elo, ko_randoms[ri]); ri += 1
    fourth = sf1_l if third == sf2_l else sf2_l

    # Final
    champion = _ko(sf1_w, sf2_w, elo, ko_randoms[ri])
    finalist = sf1_w if champion == sf2_w else sf2_w

    return SimResult(
        champion=champion,
        finalist=finalist,
        third=third,
        fourth=fourth,
        semifinalists=[sf1_l, sf2_l, path_qf_winners[0], path_qf_winners[1],
                       path_qf_winners[2], path_qf_winners[3]],
    )


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class TournamentSimulator:
    """Monte Carlo simulator for a 48-team World Cup bracket.

    Parameters
    ----------
    predictions_path:
        Path to the JSON produced by predict_tournament.py.
    """

    def __init__(self, predictions_path: Path | None = None):
        from footyml.config import DATA_DIR
        default = DATA_DIR / "predictions" / "wc2026_upcoming.json"
        self._path = predictions_path or default

    def _load(self) -> tuple[dict[str, list[dict]], dict[str, float]]:
        """Load group-stage matches and derive team Elo from probabilities."""
        with open(self._path) as f:
            all_preds: list[dict] = json.load(f)

        group_matches: dict[str, list[dict]] = {}
        for m in all_preds:
            if m.get("stage") != "GROUP_STAGE":
                continue
            gid = m.get("group_id", "")
            group_matches.setdefault(gid, []).append(m)

        elo = _derive_team_elo(group_matches)
        return group_matches, elo

    def _presample_group(
        self,
        group_matches: dict[str, list[dict]],
        n: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Pre-sample outcome arrays of shape (n,) per match per group."""
        group_outcomes: dict[str, np.ndarray] = {}
        for gid in sorted(group_matches.keys()):
            matches = group_matches[gid]
            # outcomes[sim_idx, match_idx] ∈ {0=away, 1=draw, 2=home}
            arr = np.empty((n, len(matches)), dtype=np.int8)
            for mi, m in enumerate(matches):
                h = float(m.get("home_win_prob") or 37.5) / 100.0
                d = float(m.get("draw_prob") or 25.0) / 100.0
                a = float(m.get("away_win_prob") or 37.5) / 100.0
                total = h + d + a
                probs = np.array([a / total, d / total, h / total])
                arr[:, mi] = rng.choice([0, 1, 2], size=n, p=probs)
            group_outcomes[gid] = arr
        return group_outcomes

    def run(self, n: int = 50_000, seed: int | None = 42) -> dict[str, dict[str, float]]:
        """Run N Monte Carlo simulations.

        Returns
        -------
        dict mapping team_name → {
            "champion": float,    # probability of winning the tournament
            "finalist": float,    # probability of reaching the final
            "top4": float,        # probability of reaching the semi-finals
            "top8": float,        # probability of reaching the quarter-finals
            "group_advance": float,  # probability of advancing from group stage
        }
        """
        rng = np.random.default_rng(seed)
        group_matches, elo = self._load()

        # Pre-sample group-stage outcomes
        group_outcomes = self._presample_group(group_matches, n, rng)

        # Pre-sample knockout randomness: at most 28 ko matches per sim (4×7=28 within paths + 3 SF/3rd/Final)
        ko_randoms = rng.random(size=(n, 35))

        # Counters
        counts: dict[str, dict[str, int]] = {}

        def _inc(team: str, key: str) -> None:
            if team not in counts:
                counts[team] = {k: 0 for k in ("champion", "finalist", "top4", "top8", "group_advance")}
            counts[team][key] += 1

        inner_rng = np.random.default_rng(seed + 1 if seed is not None else None)

        for sim in range(n):
            # --- Group stage ---
            standings: dict[str, list[TeamStats]] = {}
            for gid in sorted(group_matches.keys()):
                outs = group_outcomes[gid][sim]
                standings[gid] = _simulate_group(group_matches[gid], outs, inner_rng)

            # Track group advancement (top 2 per group + best 8 3rd place)
            thirds: list[TeamStats] = []
            for gid, s in standings.items():
                _inc(s[0].name, "group_advance")
                _inc(s[1].name, "group_advance")
                if len(s) >= 3:
                    thirds.append(s[2])

            thirds.sort(key=lambda t: (t.pts, t.gd, t.gf), reverse=True)
            best_thirds = thirds[:8]
            for t in best_thirds:
                _inc(t.name, "group_advance")

            # --- Knockout ---
            result = _simulate_knockout(standings, best_thirds, elo, ko_randoms[sim])

            _inc(result.champion, "champion")
            _inc(result.finalist, "finalist")
            for sf in {result.champion, result.finalist, result.third, result.fourth}:
                _inc(sf, "top4")
            for sf_team in result.semifinalists:
                _inc(sf_team, "top8")

        # Normalise to probabilities
        probs: dict[str, dict[str, float]] = {}
        for team, c in counts.items():
            probs[team] = {k: round(v / n * 100, 2) for k, v in c.items()}
        return probs

    def most_likely_bracket(self) -> dict:
        """Return the deterministic "most likely" bracket (greedy highest-prob winner).

        Useful for drawing a single bracket tree rather than probability distributions.
        """
        group_matches, elo = self._load()
        rng = np.random.default_rng(0)

        # Simulate group stage using expected values (pick highest-prob outcome per match)
        standings: dict[str, list[TeamStats]] = {}
        for gid in sorted(group_matches.keys()):
            matches = group_matches[gid]
            outcomes = np.array([
                np.argmax([
                    float(m.get("away_win_prob") or 37.5),
                    float(m.get("draw_prob") or 25.0),
                    float(m.get("home_win_prob") or 37.5),
                ])
                for m in matches
            ], dtype=np.int8)
            standings[gid] = _simulate_group(matches, outcomes, rng)

        thirds = sorted(
            [s[2] for s in standings.values() if len(s) >= 3],
            key=lambda t: (t.pts, t.gd, t.gf),
            reverse=True,
        )
        best_thirds = thirds[:8]

        # Build bracket deterministically
        bracket: dict = {"paths": [], "sf1": {}, "sf2": {}, "third_place": {}, "final": {}}

        path_qf_winners: list[str] = []
        for path_idx, (g1, g2, g3) in enumerate(WC2026_PATHS):
            s1, s2, s3 = standings[g1], standings[g2], standings[g3]
            t3a_name = best_thirds[path_idx * 2].name if path_idx * 2 < len(best_thirds) else "TBD"
            t3b_name = best_thirds[path_idx * 2 + 1].name if path_idx * 2 + 1 < len(best_thirds) else "TBD"

            def det_winner(a: str, b: str) -> str:
                return a if _ko_p_win(a, b, elo) >= 0.5 else b

            r32_1 = det_winner(s1[0].name, s2[1].name)
            r32_2 = det_winner(s2[0].name, s1[1].name)
            r32_3 = det_winner(s3[0].name, t3a_name)
            r32_4 = det_winner(s3[1].name, t3b_name)
            r16_1 = det_winner(r32_1, r32_2)
            r16_2 = det_winner(r32_3, r32_4)
            qf_w  = det_winner(r16_1, r16_2)
            path_qf_winners.append(qf_w)

            bracket["paths"].append({
                "groups": [g1, g2, g3],
                "group_winners": [s1[0].name, s2[0].name, s3[0].name],
                "group_runners_up": [s1[1].name, s2[1].name, s3[1].name],
                "third_place_slots": [t3a_name, t3b_name],
                "r32": [
                    {"home": s1[0].name, "away": s2[1].name, "winner": r32_1},
                    {"home": s2[0].name, "away": s1[1].name, "winner": r32_2},
                    {"home": s3[0].name, "away": t3a_name,    "winner": r32_3},
                    {"home": s3[1].name, "away": t3b_name,    "winner": r32_4},
                ],
                "r16": [
                    {"home": r32_1, "away": r32_2, "winner": r16_1},
                    {"home": r32_3, "away": r32_4, "winner": r16_2},
                ],
                "qf": {"home": r16_1, "away": r16_2, "winner": qf_w},
            })

        sf1_w = path_qf_winners[0] if _ko_p_win(path_qf_winners[0], path_qf_winners[1], elo) >= 0.5 else path_qf_winners[1]
        sf1_l = path_qf_winners[0] if sf1_w == path_qf_winners[1] else path_qf_winners[1]
        sf2_w = path_qf_winners[2] if _ko_p_win(path_qf_winners[2], path_qf_winners[3], elo) >= 0.5 else path_qf_winners[3]
        sf2_l = path_qf_winners[2] if sf2_w == path_qf_winners[3] else path_qf_winners[3]

        third_winner = sf1_l if _ko_p_win(sf1_l, sf2_l, elo) >= 0.5 else sf2_l
        champion = sf1_w if _ko_p_win(sf1_w, sf2_w, elo) >= 0.5 else sf2_w
        finalist = sf1_w if champion == sf2_w else sf2_w

        bracket["sf1"] = {"home": path_qf_winners[0], "away": path_qf_winners[1], "winner": sf1_w}
        bracket["sf2"] = {"home": path_qf_winners[2], "away": path_qf_winners[3], "winner": sf2_w}
        bracket["third_place"] = {"home": sf1_l, "away": sf2_l, "winner": third_winner}
        bracket["final"] = {"home": sf1_w, "away": sf2_w, "winner": champion}
        bracket["champion"] = champion
        bracket["finalist"] = finalist
        bracket["third"] = third_winner
        bracket["group_standings"] = {
            gid: [{"rank": i+1, "team": t.name, "pts": t.pts, "gd": t.gd, "gf": t.gf,
                   "wins": t.wins, "draws": t.draws, "losses": t.losses}
                  for i, t in enumerate(s)]
            for gid, s in standings.items()
        }
        return bracket

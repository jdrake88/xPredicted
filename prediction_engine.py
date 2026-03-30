"""
Prediction Engine v2 — Poisson model using understat xG + football-data.org results.

Pipeline:
1. Generate features for upcoming fixture (Elo, PPG, xG rolling, diffs)
2. XGBoost regressors predict home_xg and away_xg
3. Poisson simulation (100k) → win/draw/loss %, clean sheet %
"""
import asyncio
import os
import csv
import pickle
import time
from collections import defaultdict
from datetime import datetime

import aiohttp
import numpy as np
import requests
from scipy.stats import poisson
from dotenv import load_dotenv
from understat import Understat

load_dotenv()

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
API_BASE = 'https://api.football-data.org/v4'
HEADERS = {'X-Auth-Token': API_KEY}

FEATURE_COLS = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_ppg', 'away_ppg', 'home_ppg_home', 'away_ppg_away',
    'home_xg_for_pg', 'home_xg_against_pg',
    'away_xg_for_pg', 'away_xg_against_pg',
    'home_form_points', 'away_form_points',
    'ppg_diff', 'xg_diff', 'defensive_diff',
    'attack_vs_defence_home', 'attack_vs_defence_away',
    'days_rest_home', 'days_rest_away',
    'league_position_diff',
]

N_SIMS = 100_000

# Mapping from football-data.org names to understat names
FD_TO_US = {
    'Arsenal FC': 'Arsenal',
    'Aston Villa FC': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Brentford FC': 'Brentford',
    'Brighton & Hove Albion FC': 'Brighton',
    'Burnley FC': 'Burnley',
    'Chelsea FC': 'Chelsea',
    'Crystal Palace FC': 'Crystal Palace',
    'Everton FC': 'Everton',
    'Fulham FC': 'Fulham',
    'Ipswich Town FC': 'Ipswich',
    'Leeds United FC': 'Leeds',
    'Leicester City FC': 'Leicester',
    'Liverpool FC': 'Liverpool',
    'Luton Town FC': 'Luton',
    'Manchester City FC': 'Manchester City',
    'Manchester United FC': 'Manchester United',
    'Newcastle United FC': 'Newcastle United',
    'Nottingham Forest FC': 'Nottingham Forest',
    'Sheffield United FC': 'Sheffield United',
    'Southampton FC': 'Southampton',
    'Sunderland AFC': 'Sunderland',
    'Tottenham Hotspur FC': 'Tottenham',
    'West Ham United FC': 'West Ham',
    'Wolverhampton Wanderers FC': 'Wolverhampton Wanderers',
}
US_TO_FD = {v: k for k, v in FD_TO_US.items()}


class PredictionEngine:
    def __init__(self):
        with open('home_regressor.pkl', 'rb') as f:
            self.home_reg = pickle.load(f)
        with open('away_regressor.pkl', 'rb') as f:
            self.away_reg = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # State — populated by initialise()
        self.elo = defaultdict(lambda: 1500.0)
        self.history = defaultdict(list)  # team -> [{date_obj, points, xg_for, xg_against, is_home}]
        self.season_points = defaultdict(int)
        self.season_played = defaultdict(int)
        self.current_gd = defaultdict(int)
        self.remaining_fixtures = []

        self._initialised = False

    def initialise(self):
        if self._initialised:
            return

        # Rebuild Elo from all historical matches (training_data.csv)
        self._rebuild_elo_from_history()

        # Fetch current season data from both sources
        self._process_current_season()

        self._initialised = True

    def _rebuild_elo_from_history(self):
        """Rebuild Elo from training_data.csv (all historical matches)."""
        if not os.path.exists('training_data.csv'):
            return

        with open('training_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            matches = sorted(reader, key=lambda m: m['date'])

        for match in matches:
            home = match['home_team']
            away = match['away_team']
            hg = int(match['home_goals'])
            ag = int(match['away_goals'])
            home_xg = float(match['home_xg'])
            away_xg = float(match['away_xg'])
            match_date = datetime.strptime(match['date'], '%Y-%m-%d')

            h_elo = self.elo[home]
            a_elo = self.elo[away]

            h_exp = 1.0 / (1.0 + 10 ** ((a_elo - (h_elo + 100)) / 400))
            a_exp = 1.0 - h_exp

            if hg > ag:
                h_act, a_act = 1.0, 0.0
                hp, ap = 3, 0
            elif hg == ag:
                h_act, a_act = 0.5, 0.5
                hp, ap = 1, 1
            else:
                h_act, a_act = 0.0, 1.0
                hp, ap = 0, 3

            self.elo[home] = h_elo + 20 * (h_act - h_exp)
            self.elo[away] = a_elo + 20 * (a_act - a_exp)

            # Build rolling history with xG
            self.history[home].append({
                'date_obj': match_date,
                'points': hp,
                'xg_for': home_xg,
                'xg_against': away_xg,
                'is_home': True,
            })
            self.history[away].append({
                'date_obj': match_date,
                'points': ap,
                'xg_for': away_xg,
                'xg_against': home_xg,
                'is_home': False,
            })

    def _process_current_season(self):
        """Fetch current season from football-data.org for standings + remaining fixtures."""
        url = f'{API_BASE}/competitions/PL/matches?season=2025'
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f'Warning: Could not fetch current season matches ({resp.status_code})')
            return

        data = resp.json()

        # Build season stats from finished matches
        finished = [m for m in data['matches'] if m['status'] == 'FINISHED']
        finished.sort(key=lambda m: m['utcDate'])

        for match in finished:
            home = match['homeTeam']['name']
            away = match['awayTeam']['name']
            hg = match['score']['fullTime']['home']
            ag = match['score']['fullTime']['away']

            if hg > ag:
                hp, ap = 3, 0
            elif hg == ag:
                hp, ap = 1, 1
            else:
                hp, ap = 0, 3

            self.season_points[home] += hp
            self.season_points[away] += ap
            self.season_played[home] += 1
            self.season_played[away] += 1
            self.current_gd[home] += hg - ag
            self.current_gd[away] += ag - hg

        # Remaining fixtures
        self.remaining_fixtures = []
        for m in data['matches']:
            if m['status'] != 'FINISHED':
                self.remaining_fixtures.append({
                    'home_team': m['homeTeam']['name'],
                    'away_team': m['awayTeam']['name'],
                    'date': m['utcDate'][:10],
                })

    def get_features(self, home_team, away_team, match_date=None):
        """Calculate all features for an upcoming fixture."""
        if match_date is None:
            match_date = datetime.now()

        h_elo = self.elo[home_team]
        a_elo = self.elo[away_team]
        elo_diff = h_elo - a_elo

        home_hist = self.history[home_team]
        away_hist = self.history[away_team]

        # Rolling PPG (last 10)
        home_last10 = home_hist[-10:] if home_hist else []
        away_last10 = away_hist[-10:] if away_hist else []
        home_ppg = np.mean([h['points'] for h in home_last10]) if home_last10 else 1.0
        away_ppg = np.mean([h['points'] for h in away_last10]) if away_last10 else 1.0

        # Home PPG at home / Away PPG away (last 10 venue-specific)
        home_home = [h for h in home_hist if h['is_home']][-10:]
        away_away = [h for h in away_hist if not h['is_home']][-10:]
        home_ppg_home = np.mean([h['points'] for h in home_home]) if home_home else home_ppg
        away_ppg_away = np.mean([h['points'] for h in away_away]) if away_away else away_ppg

        # xG rolling (last 10)
        home_xg_for = np.mean([h['xg_for'] for h in home_last10]) if home_last10 else 1.5
        home_xg_against = np.mean([h['xg_against'] for h in home_last10]) if home_last10 else 1.5
        away_xg_for = np.mean([h['xg_for'] for h in away_last10]) if away_last10 else 1.5
        away_xg_against = np.mean([h['xg_against'] for h in away_last10]) if away_last10 else 1.5

        # Form (last 5)
        home_form = sum(h['points'] for h in home_hist[-5:]) if home_hist else 5
        away_form = sum(h['points'] for h in away_hist[-5:]) if away_hist else 5

        # Diffs
        ppg_diff = home_ppg - away_ppg
        xg_diff = home_xg_for - away_xg_for
        defensive_diff = away_xg_against - home_xg_against

        # Attack vs defence matchups
        attack_vs_defence_home = home_xg_for - away_xg_against
        attack_vs_defence_away = away_xg_for - home_xg_against

        # Days rest
        home_rest = (match_date - home_hist[-1]['date_obj']).days if home_hist else 7
        away_rest = (match_date - away_hist[-1]['date_obj']).days if away_hist else 7

        # League position diff
        all_teams = set(self.season_played.keys())
        if all_teams and home_team in all_teams and away_team in all_teams:
            sorted_teams = sorted(all_teams, key=lambda t: -self.season_points[t])
            pos_map = {t: i + 1 for i, t in enumerate(sorted_teams)}
            league_pos_diff = pos_map[home_team] - pos_map[away_team]
        else:
            league_pos_diff = 0

        return np.array([[
            h_elo, a_elo, elo_diff,
            home_ppg, away_ppg, home_ppg_home, away_ppg_away,
            home_xg_for, home_xg_against,
            away_xg_for, away_xg_against,
            home_form, away_form,
            ppg_diff, xg_diff, defensive_diff,
            attack_vs_defence_home, attack_vs_defence_away,
            home_rest, away_rest,
            league_pos_diff,
        ]])

    def predict(self, home_team, away_team, match_date=None):
        """Full ensemble prediction for a fixture."""
        self.initialise()

        if match_date and isinstance(match_date, str):
            match_date = datetime.strptime(match_date, '%Y-%m-%d')

        features = self.get_features(home_team, away_team, match_date)
        features_scaled = self.scaler.transform(features)

        # Step 1: XGBoost xG predictions
        home_xg = max(0.2, float(self.home_reg.predict(features_scaled)[0]))
        away_xg = max(0.2, float(self.away_reg.predict(features_scaled)[0]))

        # Step 2: Poisson simulation (fixed seed for consistent results)
        np.random.seed(42)
        home_goals_sim = np.random.poisson(home_xg, N_SIMS)
        away_goals_sim = np.random.poisson(away_xg, N_SIMS)

        hw_poisson = np.sum(home_goals_sim > away_goals_sim) / N_SIMS
        dr_poisson = np.sum(home_goals_sim == away_goals_sim) / N_SIMS
        aw_poisson = np.sum(home_goals_sim < away_goals_sim) / N_SIMS

        home_cs = np.sum(away_goals_sim == 0) / N_SIMS
        away_cs = np.sum(home_goals_sim == 0) / N_SIMS

        home_win_pct = hw_poisson
        draw_pct = dr_poisson
        away_win_pct = aw_poisson

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'home_win_pct': round(home_win_pct * 100, 1),
            'draw_pct': round(draw_pct * 100, 1),
            'away_win_pct': round(away_win_pct * 100, 1),
            'home_cs_pct': round(home_cs * 100, 1),
            'away_cs_pct': round(away_cs * 100, 1),
        }

    def _get_match_xg(self, home_team, away_team, match_date=None):
        """Get xG predictions for a fixture (used by season simulator)."""
        features = self.get_features(home_team, away_team, match_date)
        features_scaled = self.scaler.transform(features)
        home_xg = max(0.2, float(self.home_reg.predict(features_scaled)[0]))
        away_xg = max(0.2, float(self.away_reg.predict(features_scaled)[0]))
        return home_xg, away_xg

    def simulate_seasons(self, n_simulations=10000):
        """Simulate remainder of season n_simulations times."""
        self.initialise()

        teams = list(self.season_played.keys())
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        actual_points = np.array([self.season_points[t] for t in teams])
        actual_gd = np.array([self.current_gd[t] for t in teams])

        # Pre-compute xG for all remaining fixtures
        fixture_xgs = []
        for fix in self.remaining_fixtures:
            home = fix['home_team']
            away = fix['away_team']
            match_date = datetime.strptime(fix['date'], '%Y-%m-%d') if fix['date'] else None
            home_xg, away_xg = self._get_match_xg(home, away, match_date)
            fixture_xgs.append({
                'home_idx': team_idx[home],
                'away_idx': team_idx[away],
                'home_xg': home_xg,
                'away_xg': away_xg,
            })

        position_counts = np.zeros((n_teams, n_teams), dtype=int)
        total_sim_points = np.zeros(n_teams)
        total_sim_gd = np.zeros(n_teams)

        for sim in range(n_simulations):
            sim_points = actual_points.copy()
            sim_gd = actual_gd.copy()

            for fix in fixture_xgs:
                hg = np.random.poisson(fix['home_xg'])
                ag = np.random.poisson(fix['away_xg'])
                hi = fix['home_idx']
                ai = fix['away_idx']

                sim_gd[hi] += hg - ag
                sim_gd[ai] += ag - hg

                if hg > ag:
                    sim_points[hi] += 3
                elif hg == ag:
                    sim_points[hi] += 1
                    sim_points[ai] += 1
                else:
                    sim_points[ai] += 3

            order = np.lexsort((-sim_gd, -sim_points))
            for pos, team_i in enumerate(order):
                position_counts[team_i][pos] += 1

            total_sim_points += sim_points
            total_sim_gd += sim_gd

        avg_points = total_sim_points / n_simulations
        avg_gd = total_sim_gd / n_simulations

        projected = []
        for i, team in enumerate(teams):
            projected.append({
                'team': team,
                'projected_points': round(avg_points[i]),
                'projected_gd': round(avg_gd[i]),
                'current_points': int(actual_points[i]),
                'current_gd': int(actual_gd[i]),
            })

        projected.sort(key=lambda x: (-x['projected_points'], -x['projected_gd']))
        for i, row in enumerate(projected):
            row['position'] = i + 1

        pos_probs = {}
        for i, team in enumerate(teams):
            pos_probs[team] = (position_counts[i] / n_simulations * 100).tolist()

        return projected, pos_probs

    def get_elo_table(self):
        """Return all Elo ratings sorted highest to lowest."""
        self.initialise()
        sorted_elo = sorted(self.elo.items(), key=lambda x: -x[1])
        return [(team, round(rating, 1)) for team, rating in sorted_elo]


# === Sanity checks ===
if __name__ == '__main__':
    print('Initialising prediction engine (Poisson only)...')
    engine = PredictionEngine()
    engine.initialise()

    print('\n=== SANITY CHECKS — POISSON ONLY ===\n')

    tests = [
        ('Arsenal FC', 'AFC Bournemouth'),
        ('Manchester City FC', 'Chelsea FC'),
        ('Liverpool FC', 'Manchester United FC'),
        ('Sunderland AFC', 'Tottenham Hotspur FC'),
        ('Crystal Palace FC', 'Newcastle United FC'),
        ('Nottingham Forest FC', 'Aston Villa FC'),
    ]

    results = {}
    all_consistent = True
    for i, (home, away) in enumerate(tests, 1):
        p = engine.predict(home, away)
        results[(home, away)] = p
        print(f'Fixture {i}: {p["home_team"]} vs {p["away_team"]}')
        print(f'  Home Win: {p["home_win_pct"]}%  |  Draw: {p["draw_pct"]}%  |  Away Win: {p["away_win_pct"]}%')
        print(f'  Home xG:  {p["home_xg"]}       |  Away xG:  {p["away_xg"]}')
        print(f'  Home CS:  {p["home_cs_pct"]}%      |  Away CS:  {p["away_cs_pct"]}%')

        # xG consistency check
        if p['home_xg'] > p['away_xg']:
            consistent = p['home_win_pct'] > p['away_win_pct']
        elif p['away_xg'] > p['home_xg']:
            consistent = p['away_win_pct'] > p['home_win_pct']
        else:
            consistent = True  # equal xG, any result is fine

        if not consistent:
            all_consistent = False
        print(f'  Higher xG = higher win%: {"YES" if consistent else "NO"}')
        print()

    # Named checks
    p1 = results[('Arsenal FC', 'AFC Bournemouth')]
    p2 = results[('Manchester City FC', 'Chelsea FC')]
    p3 = results[('Liverpool FC', 'Manchester United FC')]

    checks = []

    c = p1['home_win_pct'] >= 40
    checks.append(c)
    print(f'Fixture 1 — Arsenal favoured:         {p1["home_win_pct"]}% -> {"PASS" if c else "FAIL"}')

    c = p1['home_xg'] > p1['away_xg']
    checks.append(c)
    print(f'Fixture 1 — Arsenal xG > Bournemouth: {p1["home_xg"]} vs {p1["away_xg"]} -> {"PASS" if c else "FAIL"}')

    c = p2['home_xg'] > p2['away_xg']
    checks.append(c)
    print(f'Fixture 2 — Man City xG > Chelsea:    {p2["home_xg"]} vs {p2["away_xg"]} -> {"PASS" if c else "FAIL"}')

    c = p3['home_win_pct'] > p3['away_win_pct']
    checks.append(c)
    print(f'Fixture 3 — Liverpool favoured:       {p3["home_win_pct"]}% vs {p3["away_win_pct"]}% -> {"PASS" if c else "FAIL"}')

    checks.append(all_consistent)
    print(f'\nAll 6 fixtures xG-consistent:          {"PASS" if all_consistent else "FAIL"}')
    print(f'All checks: {"PASS" if all(checks) else "FAIL"}')

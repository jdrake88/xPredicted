"""
Step 1 — Fetch training data from both sources and build feature dataset.
Fetches match results from football-data.org and xG from understat.
Processes all matches in strict chronological order.
"""
import asyncio
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import aiohttp
import numpy as np
import requests
from dotenv import load_dotenv
from understat import Understat

load_dotenv()

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
API_BASE = 'https://api.football-data.org/v4'
HEADERS = {'X-Auth-Token': API_KEY}

SEASONS_FD = [2023, 2024, 2025]
SEASONS_US = [2023, 2024, 2025]

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


def fetch_fd_matches():
    """Fetch match results from football-data.org for all seasons."""
    all_matches = []
    for season in SEASONS_FD:
        label = f'{season}/{season + 1}'
        print(f'  Fetching football-data.org {label}...')
        url = f'{API_BASE}/competitions/PL/matches?season={season}'
        resp = requests.get(url, headers=HEADERS)

        if resp.status_code == 429:
            print('    Rate limited — waiting 60s...')
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS)

        if resp.status_code != 200:
            print(f'    ERROR: {resp.status_code} — {resp.text[:100]}')
            continue

        data = resp.json()
        count = 0
        for m in data['matches']:
            if m['status'] != 'FINISHED':
                continue
            all_matches.append({
                'date': m['utcDate'][:10],
                'home_team': m['homeTeam']['name'],
                'away_team': m['awayTeam']['name'],
                'home_goals': m['score']['fullTime']['home'],
                'away_goals': m['score']['fullTime']['away'],
                'matchday': m['matchday'],
                'season': season,
            })
            count += 1
        print(f'    {count} finished matches')
        if season != SEASONS_FD[-1]:
            time.sleep(10)

    return all_matches


async def fetch_understat_matches():
    """Fetch xG data from understat for all seasons."""
    all_matches = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for season in SEASONS_US:
            label = f'{season}/{season + 1}'
            print(f'  Fetching understat {label}...')
            matches = await understat.get_league_results("epl", season)
            count = 0
            for m in matches:
                h_title = m['h']['title']
                a_title = m['a']['title']
                # Map to FD names
                h_fd = US_TO_FD.get(h_title, h_title)
                a_fd = US_TO_FD.get(a_title, a_title)
                all_matches.append({
                    'date': m['datetime'][:10],
                    'home_team': h_fd,
                    'away_team': a_fd,
                    'home_xg': float(m['xG']['h']),
                    'away_xg': float(m['xG']['a']),
                    'home_goals': int(m['goals']['h']),
                    'away_goals': int(m['goals']['a']),
                })
                count += 1
            print(f'    {count} matches')
    return all_matches


def merge_matches(fd_matches, us_matches):
    """Merge football-data.org results with understat xG by date + teams."""
    # Build lookup from understat: (date, home, away) -> xG
    us_lookup = {}
    for m in us_matches:
        key = (m['date'], m['home_team'], m['away_team'])
        us_lookup[key] = (m['home_xg'], m['away_xg'])

    merged = []
    matched = 0
    unmatched = 0

    for m in fd_matches:
        key = (m['date'], m['home_team'], m['away_team'])
        if key in us_lookup:
            m['home_xg'] = us_lookup[key][0]
            m['away_xg'] = us_lookup[key][1]
            merged.append(m)
            matched += 1
        else:
            unmatched += 1

    print(f'\n  Merge results: {matched} matched, {unmatched} unmatched')
    return merged


def compute_features(matches):
    """
    Process matches in strict chronological order.
    Calculate features using ONLY data available before each match.
    """
    # Sort chronologically
    matches.sort(key=lambda m: m['date'])

    # --- State trackers ---
    elo = defaultdict(lambda: 1500.0)

    # Rolling match history per team (all matches)
    history = defaultdict(list)  # team -> list of {date, points, xg_for, xg_against, is_home}

    # Season-specific stats for league position
    season_points = defaultdict(lambda: defaultdict(int))
    season_played = defaultdict(lambda: defaultdict(int))

    dataset = []
    skipped = 0

    for match in matches:
        home = match['home_team']
        away = match['away_team']
        hg = match['home_goals']
        ag = match['away_goals']
        match_date = datetime.strptime(match['date'], '%Y-%m-%d')
        season = match['season']
        home_xg_actual = match['home_xg']
        away_xg_actual = match['away_xg']

        # --- Calculate pre-match features ---
        home_hist = history[home]
        away_hist = history[away]

        # Need at least 5 matches for each team to have meaningful features
        if len(home_hist) < 5 or len(away_hist) < 5:
            # Still update state below
            pass
        else:
            # Elo features
            h_elo = elo[home]
            a_elo = elo[away]
            elo_diff = h_elo - a_elo

            # Rolling PPG (last 10 matches)
            home_last10 = home_hist[-10:]
            away_last10 = away_hist[-10:]
            home_ppg = np.mean([h['points'] for h in home_last10])
            away_ppg = np.mean([h['points'] for h in away_last10])

            # Home PPG at home (last 10 home games)
            home_home_hist = [h for h in home_hist if h['is_home']][-10:]
            away_away_hist = [h for h in away_hist if not h['is_home']][-10:]
            home_ppg_home = np.mean([h['points'] for h in home_home_hist]) if home_home_hist else home_ppg
            away_ppg_away = np.mean([h['points'] for h in away_away_hist]) if away_away_hist else away_ppg

            # xG features (rolling last 10)
            home_xg_for = np.mean([h['xg_for'] for h in home_last10])
            home_xg_against = np.mean([h['xg_against'] for h in home_last10])
            away_xg_for = np.mean([h['xg_for'] for h in away_last10])
            away_xg_against = np.mean([h['xg_against'] for h in away_last10])

            # Form (last 5)
            home_form = sum(h['points'] for h in home_hist[-5:])
            away_form = sum(h['points'] for h in away_hist[-5:])

            # Difference features
            ppg_diff = home_ppg - away_ppg
            xg_diff = home_xg_for - away_xg_for
            defensive_diff = away_xg_against - home_xg_against

            # Attack vs defence matchups
            attack_vs_defence_home = home_xg_for - away_xg_against
            attack_vs_defence_away = away_xg_for - home_xg_against

            # Days rest
            home_rest = (match_date - home_hist[-1]['date_obj']).days if home_hist else 7
            away_rest = (match_date - away_hist[-1]['date_obj']).days if away_hist else 7

            # League position diff (using current season standings)
            sp = season_points[season]
            spl = season_played[season]
            all_teams_in_season = set(spl.keys())
            if all_teams_in_season and home in all_teams_in_season and away in all_teams_in_season:
                sorted_teams = sorted(all_teams_in_season, key=lambda t: -sp[t])
                pos_map = {t: i + 1 for i, t in enumerate(sorted_teams)}
                league_pos_diff = pos_map[home] - pos_map[away]
            else:
                league_pos_diff = 0

            # Match outcome
            if hg > ag:
                outcome = 2  # home win
            elif hg == ag:
                outcome = 1  # draw
            else:
                outcome = 0  # away win

            dataset.append({
                'home_elo': h_elo,
                'away_elo': a_elo,
                'elo_diff': elo_diff,
                'home_ppg': home_ppg,
                'away_ppg': away_ppg,
                'home_ppg_home': home_ppg_home,
                'away_ppg_away': away_ppg_away,
                'home_xg_for_pg': home_xg_for,
                'home_xg_against_pg': home_xg_against,
                'away_xg_for_pg': away_xg_for,
                'away_xg_against_pg': away_xg_against,
                'home_form_points': home_form,
                'away_form_points': away_form,
                'ppg_diff': ppg_diff,
                'xg_diff': xg_diff,
                'defensive_diff': defensive_diff,
                'attack_vs_defence_home': attack_vs_defence_home,
                'attack_vs_defence_away': attack_vs_defence_away,
                'days_rest_home': home_rest,
                'days_rest_away': away_rest,
                'league_position_diff': league_pos_diff,
                # Targets
                'home_xg_actual': home_xg_actual,
                'away_xg_actual': away_xg_actual,
                'match_outcome': outcome,
                # Metadata (not used as features)
                '_date': match['date'],
                '_home': home,
                '_away': away,
            })

        # --- Update state AFTER feature extraction ---

        # Update Elo
        h_elo_pre = elo[home]
        a_elo_pre = elo[away]
        h_exp = 1.0 / (1.0 + 10 ** ((a_elo_pre - (h_elo_pre + 100)) / 400))
        a_exp = 1.0 - h_exp
        if hg > ag:
            h_act, a_act = 1.0, 0.0
        elif hg == ag:
            h_act, a_act = 0.5, 0.5
        else:
            h_act, a_act = 0.0, 1.0
        elo[home] = h_elo_pre + 20 * (h_act - h_exp)
        elo[away] = a_elo_pre + 20 * (a_act - a_exp)

        # Points
        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        # Update rolling history
        history[home].append({
            'date_obj': match_date,
            'points': hp,
            'xg_for': home_xg_actual,
            'xg_against': away_xg_actual,
            'is_home': True,
        })
        history[away].append({
            'date_obj': match_date,
            'points': ap,
            'xg_for': away_xg_actual,
            'xg_against': home_xg_actual,
            'is_home': False,
        })

        # Update season standings
        season_points[season][home] += hp
        season_points[season][away] += ap
        season_played[season][home] += 1
        season_played[season][away] += 1

    return dataset, elo


def save_dataset(dataset, filepath='features.csv'):
    """Save feature dataset to CSV."""
    if not dataset:
        print('No features to save!')
        return

    fieldnames = list(dataset[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f'  Saved {len(dataset)} rows to {filepath}')


async def main():
    print('=== STEP 1: BUILD TRAINING DATASET ===\n')

    # Fetch from both sources
    print('Fetching football-data.org matches...')
    fd_matches = fetch_fd_matches()
    print(f'  Total: {len(fd_matches)} matches\n')

    print('Fetching understat xG data...')
    us_matches = await fetch_understat_matches()
    print(f'  Total: {len(us_matches)} matches\n')

    print('Merging datasets...')
    merged = merge_matches(fd_matches, us_matches)
    print(f'  Merged dataset: {len(merged)} matches\n')

    if len(merged) < 500:
        print(f'STOPPING: Only {len(merged)} merged matches — need at least 500.')
        return

    print('Computing features...')
    dataset, elo = compute_features(merged)

    print(f'\n=== DATASET SUMMARY ===')
    print(f'  Total feature rows: {len(dataset)}')

    if len(dataset) < 500:
        print(f'  STOPPING: Only {len(dataset)} rows after feature computation — need 500+.')
        return

    # Stats
    home_xgs = [r['home_xg_actual'] for r in dataset]
    away_xgs = [r['away_xg_actual'] for r in dataset]
    print(f'  Average home xG: {np.mean(home_xgs):.3f}')
    print(f'  Average away xG: {np.mean(away_xgs):.3f}')

    # Chronological check
    dates = [r['_date'] for r in dataset]
    is_sorted = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
    print(f'  Chronologically sorted: {is_sorted}')

    # No future data check
    print(f'  Elo calculated progressively: True (by construction — updated after each match)')

    # First 3 and last 3 rows
    print(f'\n  --- First 3 rows ---')
    for r in dataset[:3]:
        print(f'    {r["_date"]} {r["_home"]} vs {r["_away"]}')
        print(f'      elo_diff={r["elo_diff"]:.1f}  ppg_diff={r["ppg_diff"]:.2f}  xg_diff={r["xg_diff"]:.2f}')
        print(f'      home_xg_actual={r["home_xg_actual"]:.2f}  away_xg_actual={r["away_xg_actual"]:.2f}  outcome={r["match_outcome"]}')

    print(f'\n  --- Last 3 rows ---')
    for r in dataset[-3:]:
        print(f'    {r["_date"]} {r["_home"]} vs {r["_away"]}')
        print(f'      elo_diff={r["elo_diff"]:.1f}  ppg_diff={r["ppg_diff"]:.2f}  xg_diff={r["xg_diff"]:.2f}')
        print(f'      home_xg_actual={r["home_xg_actual"]:.2f}  away_xg_actual={r["away_xg_actual"]:.2f}  outcome={r["match_outcome"]}')

    save_dataset(dataset)

    # Save merged raw data too for reference
    with open('training_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'season', 'date', 'matchday', 'home_team', 'away_team',
            'home_goals', 'away_goals', 'home_xg', 'away_xg',
        ])
        writer.writeheader()
        for m in sorted(merged, key=lambda x: x['date']):
            writer.writerow({
                'season': f'{m["season"]}/{m["season"]+1}',
                'date': m['date'],
                'matchday': m['matchday'],
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'home_goals': m['home_goals'],
                'away_goals': m['away_goals'],
                'home_xg': m['home_xg'],
                'away_xg': m['away_xg'],
            })
    print(f'  Saved raw training data to training_data.csv')


if __name__ == '__main__':
    asyncio.run(main())

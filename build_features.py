"""
Step 2.1 — Build features for each historical match.
Processes matches chronologically, calculates progressive Elo ratings,
and generates all features needed for XGBoost training.
"""
import csv
from datetime import datetime, timedelta
from collections import defaultdict


def load_matches(filepath='training_data.csv'):
    """Load matches from CSV and sort chronologically."""
    matches = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['home_goals'] = int(row['home_goals'])
            row['away_goals'] = int(row['away_goals'])
            row['matchday'] = int(row['matchday'])
            row['date_obj'] = datetime.strptime(row['date'], '%Y-%m-%d')
            matches.append(row)

    matches.sort(key=lambda m: m['date_obj'])
    return matches


def expected_score(rating_a, rating_b):
    """Calculate expected score for player A given both ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(goals_for, goals_against):
    """Convert match result to score: 1 = win, 0.5 = draw, 0 = loss."""
    if goals_for > goals_against:
        return 1.0
    elif goals_for == goals_against:
        return 0.5
    else:
        return 0.0


def update_elo(rating, expected, actual, k=20):
    """Update Elo rating after a match."""
    return rating + k * (actual - expected)


def get_outcome(home_goals, away_goals):
    """Return match outcome as H, D, or A."""
    if home_goals > away_goals:
        return 'H'
    elif home_goals == away_goals:
        return 'D'
    else:
        return 'A'


def compute_form(results, n=5):
    """Compute form from last n results. Win=3, Draw=1, Loss=0. Returns avg points."""
    recent = results[-n:]
    if not recent:
        return 0.0
    return sum(recent) / len(recent)


def build_feature_dataset(matches):
    """
    Process all matches chronologically and build features for each.
    Returns list of feature dicts ready for training.
    """
    # --- State trackers ---
    elo = defaultdict(lambda: 1500.0)

    # Season-specific stats (reset each season)
    season_stats = {}

    def get_season_stats(season):
        if season not in season_stats:
            season_stats[season] = {
                'played': defaultdict(int),
                'points': defaultdict(int),
                'home_played': defaultdict(int),
                'home_points': defaultdict(int),
                'away_played': defaultdict(int),
                'away_points': defaultdict(int),
                'goals_for': defaultdict(int),
                'goals_against': defaultdict(int),
                'results': defaultdict(list),  # list of points per match
                'last_match_date': {},
            }
        return season_stats[season]

    # --- Build features for each match ---
    dataset = []
    skipped = 0

    for match in matches:
        season = match['season']
        stats = get_season_stats(season)

        home = match['home_team']
        away = match['away_team']
        hg = match['home_goals']
        ag = match['away_goals']
        match_date = match['date_obj']

        # --- FEATURES (calculated BEFORE updating stats) ---

        home_elo = elo[home]
        away_elo = elo[away]

        home_played = stats['played'][home]
        away_played = stats['played'][away]

        # Skip matches where either team has played fewer than 3 games
        # (not enough data for meaningful per-game averages)
        if home_played < 3 or away_played < 3:
            # Still update stats after skipping feature extraction
            pass
        else:
            # Points per game
            home_ppg = stats['points'][home] / home_played
            away_ppg = stats['points'][away] / away_played

            # Home PPG at home / Away PPG away
            home_ppg_home = (stats['home_points'][home] / stats['home_played'][home]
                            if stats['home_played'][home] > 0 else 0.0)
            away_ppg_away = (stats['away_points'][away] / stats['away_played'][away]
                            if stats['away_played'][away] > 0 else 0.0)

            # Goals per game
            home_gf_per_game = stats['goals_for'][home] / home_played
            home_ga_per_game = stats['goals_against'][home] / home_played
            away_gf_per_game = stats['goals_for'][away] / away_played
            away_ga_per_game = stats['goals_against'][away] / away_played

            # Last 5 form
            home_form = compute_form(stats['results'][home], 5)
            away_form = compute_form(stats['results'][away], 5)

            # Attack vs defence matchup
            home_attack_vs_away_defence = home_gf_per_game - away_ga_per_game
            away_attack_vs_home_defence = away_gf_per_game - home_ga_per_game

            # Days rest
            home_rest = (match_date - stats['last_match_date'][home]).days if home in stats['last_match_date'] else 7
            away_rest = (match_date - stats['last_match_date'][away]).days if away in stats['last_match_date'] else 7

            # League position gap (approximate via PPG ranking)
            # Build current table by points
            all_teams_in_season = set(stats['played'].keys())
            team_points_list = [(t, stats['points'][t]) for t in all_teams_in_season if stats['played'][t] > 0]
            team_points_list.sort(key=lambda x: (-x[1], x[0]))
            position_map = {t: i + 1 for i, (t, _) in enumerate(team_points_list)}

            home_pos = position_map.get(home, 10)
            away_pos = position_map.get(away, 10)
            position_gap = away_pos - home_pos  # positive = home team higher in table

            outcome = get_outcome(hg, ag)

            feature_row = {
                'home_elo': home_elo,
                'away_elo': away_elo,
                'home_ppg': home_ppg,
                'away_ppg': away_ppg,
                'home_ppg_home': home_ppg_home,
                'away_ppg_away': away_ppg_away,
                'home_gf_per_game': home_gf_per_game,
                'home_ga_per_game': home_ga_per_game,
                'away_gf_per_game': away_gf_per_game,
                'away_ga_per_game': away_ga_per_game,
                'home_form': home_form,
                'away_form': away_form,
                'home_attack_vs_away_defence': home_attack_vs_away_defence,
                'away_attack_vs_home_defence': away_attack_vs_home_defence,
                'home_rest': home_rest,
                'away_rest': away_rest,
                'position_gap': position_gap,
                # Targets
                'home_goals': hg,
                'away_goals': ag,
                'outcome': outcome,
            }
            dataset.append(feature_row)

        # --- UPDATE STATE (always, even for skipped matches) ---

        # Update Elo (home advantage = +100 for expected score calc only)
        home_expected = expected_score(home_elo + 100, away_elo)
        away_expected = 1.0 - home_expected

        home_actual = actual_score(hg, ag)
        away_actual = actual_score(ag, hg)

        elo[home] = update_elo(home_elo, home_expected, home_actual)
        elo[away] = update_elo(away_elo, away_expected, away_actual)

        # Update season stats
        stats['played'][home] += 1
        stats['played'][away] += 1

        # Points
        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        stats['points'][home] += hp
        stats['points'][away] += ap
        stats['home_played'][home] += 1
        stats['home_points'][home] += hp
        stats['away_played'][away] += 1
        stats['away_points'][away] += ap

        stats['goals_for'][home] += hg
        stats['goals_against'][home] += ag
        stats['goals_for'][away] += ag
        stats['goals_against'][away] += hg

        stats['results'][home].append(hp)
        stats['results'][away].append(ap)

        stats['last_match_date'][home] = match_date
        stats['last_match_date'][away] = match_date

    return dataset, elo


def save_features(dataset, filepath='features.csv'):
    """Save feature dataset to CSV."""
    if not dataset:
        print('No features to save!')
        return

    fieldnames = list(dataset[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f'Saved {len(dataset)} feature rows to {filepath}')


def main():
    print('Loading matches...')
    matches = load_matches()
    print(f'Loaded {len(matches)} matches')

    print('\nBuilding features...')
    dataset, elo_ratings = build_feature_dataset(matches)

    print(f'\nFeature dataset: {len(dataset)} rows')
    print(f'(Skipped early-season matches where teams had < 3 games played)')

    # Show sample
    if dataset:
        print(f'\nSample features (first row):')
        for k, v in dataset[0].items():
            print(f'  {k}: {v}')

    save_features(dataset)

    # Print Elo summary
    print(f'\n=== CURRENT ELO RATINGS (top 10) ===')
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: -x[1])
    for team, rating in sorted_elo[:10]:
        print(f'  {team}: {rating:.1f}')


if __name__ == '__main__':
    main()

import os
import requests
from flask import Flask, jsonify, send_from_directory, request
from dotenv import load_dotenv
from prediction_engine import PredictionEngine
from datetime import datetime

load_dotenv()

app = Flask(__name__, static_folder='static')

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
API_BASE = 'https://api.football-data.org/v4'
HEADERS = {'X-Auth-Token': API_KEY}

PL_ID = 'PL'

# Initialise prediction engine once at startup
engine = PredictionEngine()


# === Page routes ===

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/fixtures')
def fixtures_page():
    return send_from_directory('static', 'fixtures.html')


@app.route('/predictions')
def predictions_page():
    return send_from_directory('static', 'predictions.html')


@app.route('/probabilities')
def probabilities_page():
    return send_from_directory('static', 'probabilities.html')


@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('static', 'sitemap.xml')


# === API routes ===

@app.route('/api/standings')
def standings():
    url = f'{API_BASE}/competitions/{PL_ID}/standings'
    resp = requests.get(url, headers=HEADERS)

    if resp.status_code != 200:
        return jsonify({'error': f'API returned {resp.status_code}'}), resp.status_code

    data = resp.json()
    table = data['standings'][0]['table']

    teams = []
    for row in table:
        teams.append({
            'position': row['position'],
            'name': row['team']['name'],
            'crest': row['team']['crest'],
            'played': row['playedGames'],
            'won': row['won'],
            'drawn': row['draw'],
            'lost': row['lost'],
            'gf': row['goalsFor'],
            'ga': row['goalsAgainst'],
            'gd': row['goalDifference'],
            'points': row['points'],
        })

    season = data['season']
    competition = data['competition']['name']

    return jsonify({
        'competition': competition,
        'season': f"{season['startDate'][:4]}/{season['endDate'][:4]}",
        'teams': teams,
    })


@app.route('/api/fixtures')
def fixtures():
    matchday = request.args.get('matchday', type=int)

    # Fetch all matches for current season
    url = f'{API_BASE}/competitions/{PL_ID}/matches?season=2025'
    resp = requests.get(url, headers=HEADERS)

    if resp.status_code != 200:
        return jsonify({'error': f'API returned {resp.status_code}'}), resp.status_code

    data = resp.json()
    all_matches = data['matches']

    # Find total matchdays
    matchdays = sorted(set(m['matchday'] for m in all_matches))
    total_matchdays = max(matchdays) if matchdays else 38

    # Find current/next matchday if none specified
    if matchday is None:
        now = datetime.utcnow()
        # Find the first matchday that has unfinished matches
        for md in matchdays:
            md_matches = [m for m in all_matches if m['matchday'] == md]
            if any(m['status'] != 'FINISHED' for m in md_matches):
                matchday = md
                break
        if matchday is None:
            matchday = total_matchdays

    # Filter to requested matchday
    md_matches = [m for m in all_matches if m['matchday'] == matchday]
    md_matches.sort(key=lambda m: m['utcDate'])

    fixtures_out = []
    for match in md_matches:
        home_name = match['homeTeam']['name']
        away_name = match['awayTeam']['name']
        home_crest = match['homeTeam']['crest']
        away_crest = match['awayTeam']['crest']

        fixture = {
            'date': match['utcDate'],
            'status': match['status'],
            'home_team': home_name,
            'away_team': away_name,
            'home_crest': home_crest,
            'away_crest': away_crest,
            'home_goals': None,
            'away_goals': None,
            'prediction': None,
        }

        # Add score for finished matches
        if match['status'] == 'FINISHED':
            fixture['home_goals'] = match['score']['fullTime']['home']
            fixture['away_goals'] = match['score']['fullTime']['away']

        # Add prediction for all matches
        try:
            match_date = datetime.strptime(match['utcDate'][:10], '%Y-%m-%d')
            pred = engine.predict(home_name, away_name, match_date)
            fixture['prediction'] = pred
        except Exception as e:
            print(f'Prediction error for {home_name} vs {away_name}: {e}')

        fixtures_out.append(fixture)

    return jsonify({
        'matchday': matchday,
        'total_matchdays': total_matchdays,
        'fixtures': fixtures_out,
    })


@app.route('/api/season-simulation')
def season_simulation():
    try:
        projected, pos_probs = engine.simulate_seasons(n_simulations=10000)
        return jsonify({
            'projected_table': projected,
            'position_probabilities': pos_probs,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

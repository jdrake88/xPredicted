"""
Investigation script — diagnose Arsenal vs Bournemouth prediction.
Prints diagnostic info at every step without changing any code.
"""
import csv
import numpy as np
from datetime import datetime
from collections import defaultdict
from prediction_engine import PredictionEngine

engine = PredictionEngine()
engine.initialise()

HOME = 'Arsenal FC'
AWAY = 'AFC Bournemouth'

print("=" * 70)
print("STEP 1 — ELO RATINGS (all teams, highest to lowest)")
print("=" * 70)
elo_table = engine.get_elo_table()
for team, rating in elo_table:
    marker = " <---" if team in (HOME, AWAY) else ""
    print(f"  {rating:7.1f}  {team}{marker}")

home_elo = engine.elo[HOME]
away_elo = engine.elo[AWAY]
print(f"\n  Elo gap (Arsenal - Bournemouth): {home_elo - away_elo:.1f}")

print("\n" + "=" * 70)
print("STEP 2 — RAW FEATURES for Arsenal vs Bournemouth")
print("=" * 70)

features = engine.get_features(HOME, AWAY)
feature_names = [
    'home_elo', 'away_elo', 'home_ppg', 'away_ppg',
    'home_ppg_home', 'away_ppg_away',
    'home_gf_per_game', 'home_ga_per_game',
    'away_gf_per_game', 'away_ga_per_game',
    'home_form', 'away_form',
    'home_attack_vs_away_defence', 'away_attack_vs_home_defence',
    'home_rest', 'away_rest', 'position_gap',
]
for name, val in zip(feature_names, features[0]):
    print(f"  {name:40s} = {val:.4f}")

# Also print the raw stats used to compute features
s = engine.stats
print(f"\n  --- Raw season stats ---")
print(f"  Arsenal played: {s['played'][HOME]}, points: {s['points'][HOME]}")
print(f"  Arsenal home played: {s['home_played'][HOME]}, home points: {s['home_points'][HOME]}")
print(f"  Arsenal GF: {s['goals_for'][HOME]}, GA: {s['goals_against'][HOME]}")
print(f"  Bournemouth played: {s['played'][AWAY]}, points: {s['points'][AWAY]}")
print(f"  Bournemouth away played: {s['away_played'][AWAY]}, away points: {s['away_points'][AWAY]}")
print(f"  Bournemouth GF: {s['goals_for'][AWAY]}, GA: {s['goals_against'][AWAY]}")
print(f"  Arsenal last 5 results: {s['results'][HOME][-5:]}")
print(f"  Bournemouth last 5 results: {s['results'][AWAY][-5:]}")

print("\n" + "=" * 70)
print("STEP 3 — XGBOOST REGRESSOR RAW OUTPUT (xG)")
print("=" * 70)

home_xg_raw = float(engine.home_reg.predict(features)[0])
away_xg_raw = float(engine.away_reg.predict(features)[0])
home_xg_clamped = max(0.2, home_xg_raw)
away_xg_clamped = max(0.2, away_xg_raw)

print(f"  Home goals regressor raw output:    {home_xg_raw:.4f}")
print(f"  Away goals regressor raw output:    {away_xg_raw:.4f}")
print(f"  After clamping (min 0.2):           {home_xg_clamped:.4f} vs {away_xg_clamped:.4f}")

print("\n" + "=" * 70)
print("STEP 4 — XGBOOST CLASSIFIER RAW OUTPUT")
print("=" * 70)

clf_probs = engine.classifier.predict_proba(features)[0]
print(f"  Home Win (class 0): {clf_probs[0]*100:.1f}%")
print(f"  Draw     (class 1): {clf_probs[1]*100:.1f}%")
print(f"  Away Win (class 2): {clf_probs[2]*100:.1f}%")

print("\n" + "=" * 70)
print("STEP 5 — POISSON SIMULATION OUTPUT")
print("=" * 70)

np.random.seed(42)
N = 100_000
home_goals_sim = np.random.poisson(home_xg_clamped, N)
away_goals_sim = np.random.poisson(away_xg_clamped, N)

hw_poisson = np.sum(home_goals_sim > away_goals_sim) / N
dr_poisson = np.sum(home_goals_sim == away_goals_sim) / N
aw_poisson = np.sum(home_goals_sim < away_goals_sim) / N
hcs_poisson = np.sum(away_goals_sim == 0) / N
acs_poisson = np.sum(home_goals_sim == 0) / N

print(f"  Using home_xg={home_xg_clamped:.4f}, away_xg={away_xg_clamped:.4f}")
print(f"  Home Win:  {hw_poisson*100:.1f}%")
print(f"  Draw:      {dr_poisson*100:.1f}%")
print(f"  Away Win:  {aw_poisson*100:.1f}%")
print(f"  Home CS:   {hcs_poisson*100:.1f}%")
print(f"  Away CS:   {acs_poisson*100:.1f}%")

print("\n" + "=" * 70)
print("STEP 6 — TRAINING DATA CHECK")
print("=" * 70)

with open('training_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    matches = list(reader)

print(f"  Total matches: {len(matches)}")
dates = [m['date'] for m in matches]
home_goals_all = [int(m['home_goals']) for m in matches]
away_goals_all = [int(m['away_goals']) for m in matches]
print(f"  Avg home goals: {np.mean(home_goals_all):.3f}")
print(f"  Avg away goals: {np.mean(away_goals_all):.3f}")
print(f"  First match date: {dates[0]}")
print(f"  Last match date:  {dates[-1]}")

# Check chronological order
is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
print(f"  Chronologically sorted: {is_sorted}")

# Check Elo was progressive: rebuild and show Arsenal Elo at 3 points
elo_check = defaultdict(lambda: 1500.0)
arsenal_elo_history = []
for i, m in enumerate(matches):
    home = m['home_team']
    away = m['away_team']
    hg = int(m['home_goals'])
    ag = int(m['away_goals'])

    h_elo = elo_check[home]
    a_elo = elo_check[away]

    h_exp = 1.0 / (1.0 + 10 ** ((a_elo - (h_elo + 100)) / 400))
    a_exp = 1.0 - h_exp

    if hg > ag:
        h_act, a_act = 1.0, 0.0
    elif hg == ag:
        h_act, a_act = 0.5, 0.5
    else:
        h_act, a_act = 0.0, 1.0

    elo_check[home] = h_elo + 20 * (h_act - h_exp)
    elo_check[away] = a_elo + 20 * (a_act - a_exp)

    if home == HOME or away == HOME:
        arsenal_elo_history.append((m['date'], round(elo_check[HOME], 1)))

print(f"\n  Arsenal Elo progression (sample points):")
sample_indices = [0, len(arsenal_elo_history)//4, len(arsenal_elo_history)//2,
                  3*len(arsenal_elo_history)//4, len(arsenal_elo_history)-1]
for idx in sample_indices:
    date, elo_val = arsenal_elo_history[idx]
    print(f"    {date}: {elo_val}")

print(f"\n  Final Elo from rebuild: Arsenal={elo_check[HOME]:.1f}, Bournemouth={elo_check[AWAY]:.1f}")
print(f"  Engine Elo:             Arsenal={engine.elo[HOME]:.1f}, Bournemouth={engine.elo[AWAY]:.1f}")
elo_match = abs(elo_check[HOME] - engine.elo[HOME]) < 0.1
print(f"  Elo values match engine: {elo_match}")

# Check features.csv for the features that were used in training
print("\n  --- Training feature ranges ---")
with open('features.csv', 'r') as f:
    reader = csv.DictReader(f)
    feat_rows = list(reader)

for col in ['home_elo', 'away_elo', 'home_ppg', 'away_ppg', 'home_gf_per_game', 'away_gf_per_game']:
    vals = [float(r[col]) for r in feat_rows]
    print(f"    {col:25s}  min={min(vals):.2f}  max={max(vals):.2f}  mean={np.mean(vals):.2f}")

print("\n" + "=" * 70)
print("STEP 7 — BLENDING COMPARISON")
print("=" * 70)

# Poisson outputs
print(f"  {'':30s} {'Poisson':>10s}  {'Classifier':>10s}  {'Blended':>10s}")
hw_blend = hw_poisson * 0.6 + clf_probs[0] * 0.4
dr_blend = dr_poisson * 0.6 + clf_probs[1] * 0.4
aw_blend = aw_poisson * 0.6 + clf_probs[2] * 0.4
total = hw_blend + dr_blend + aw_blend
hw_final = hw_blend / total
dr_final = dr_blend / total
aw_final = aw_blend / total

print(f"  {'Home Win':30s} {hw_poisson*100:>9.1f}%  {clf_probs[0]*100:>9.1f}%  {hw_final*100:>9.1f}%")
print(f"  {'Draw':30s} {dr_poisson*100:>9.1f}%  {clf_probs[1]*100:>9.1f}%  {dr_final*100:>9.1f}%")
print(f"  {'Away Win':30s} {aw_poisson*100:>9.1f}%  {clf_probs[2]*100:>9.1f}%  {aw_final*100:>9.1f}%")
print(f"  {'Home xG':30s} {home_xg_clamped:>9.2f}   {'(Poisson only)':>10s}  {home_xg_clamped:>9.2f}")
print(f"  {'Away xG':30s} {away_xg_clamped:>9.2f}   {'(Poisson only)':>10s}  {away_xg_clamped:>9.2f}")
print(f"  {'Home CS':30s} {hcs_poisson*100:>9.1f}%  {'(Poisson only)':>10s}  {hcs_poisson*100:>9.1f}%")
print(f"  {'Away CS':30s} {acs_poisson*100:>9.1f}%  {'(Poisson only)':>10s}  {acs_poisson*100:>9.1f}%")

print("\n" + "=" * 70)
print("STEP 8 — SANITY: What does the model predict for a BIG Elo gap?")
print("=" * 70)
# Manually create features with a large Elo gap to see if the model responds
import copy
big_gap_features = features.copy()
# Set home_elo to 1720 and away_elo to 1450 (huge gap)
big_gap_features[0][0] = 1720.0  # home_elo
big_gap_features[0][1] = 1450.0  # away_elo
big_hxg = float(engine.home_reg.predict(big_gap_features)[0])
big_axg = float(engine.away_reg.predict(big_gap_features)[0])
print(f"  With home_elo=1720, away_elo=1450 (all else same):")
print(f"  Home xG: {big_hxg:.4f}, Away xG: {big_axg:.4f}")

# Also check feature importances
print("\n  --- Home goals regressor feature importances ---")
importances_home = engine.home_reg.feature_importances_
for name, imp in sorted(zip(feature_names, importances_home), key=lambda x: -x[1]):
    print(f"    {name:40s} = {imp:.4f}")

print("\n  --- Away goals regressor feature importances ---")
importances_away = engine.away_reg.feature_importances_
for name, imp in sorted(zip(feature_names, importances_away), key=lambda x: -x[1]):
    print(f"    {name:40s} = {imp:.4f}")

print("\n  --- Classifier feature importances ---")
importances_clf = engine.classifier.feature_importances_
for name, imp in sorted(zip(feature_names, importances_clf), key=lambda x: -x[1]):
    print(f"    {name:40s} = {imp:.4f}")

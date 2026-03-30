"""
Steps 2 & 3 — Split data chronologically, scale, train three XGBoost models.
"""
import csv
import pickle
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

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


def load_features(filepath='features.csv'):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X = np.array([[float(row[col]) for col in FEATURE_COLS] for row in rows])
    y_home_xg = np.array([float(row['home_xg_actual']) for row in rows])
    y_away_xg = np.array([float(row['away_xg_actual']) for row in rows])
    y_outcome = np.array([int(row['match_outcome']) for row in rows])
    dates = [row['_date'] for row in rows]

    return X, y_home_xg, y_away_xg, y_outcome, dates


def main():
    print('Loading features...')
    X, y_home_xg, y_away_xg, y_outcome, dates = load_features()
    n = len(X)
    print(f'Dataset: {n} samples, {X.shape[1]} features\n')

    # === STEP 2 — Chronological split ===
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_home_xg_train, y_home_xg_test = y_home_xg[:split_idx], y_home_xg[split_idx:]
    y_away_xg_train, y_away_xg_test = y_away_xg[:split_idx], y_away_xg[split_idx:]
    y_outcome_train, y_outcome_test = y_outcome[:split_idx], y_outcome[split_idx:]

    print(f'=== STEP 2 — SPLIT & SCALE ===')
    print(f'  Training set: {len(X_train)} matches ({dates[0]} to {dates[split_idx-1]})')
    print(f'  Test set:     {len(X_test)} matches ({dates[split_idx]} to {dates[-1]})')

    # Scale — fit on training only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Scaler saved to scaler.pkl\n')

    # === STEP 3 — Train models ===
    xgb_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    # --- Model A: Home xG Regressor ---
    print('=== Model A — Home xG Regressor ===')
    home_reg = XGBRegressor(**xgb_params)
    home_reg.fit(X_train_scaled, y_home_xg_train)

    home_pred_test = home_reg.predict(X_test_scaled)
    home_mae = mean_absolute_error(y_home_xg_test, home_pred_test)
    print(f'  Test MAE: {home_mae:.4f}')
    print(f'  Test avg actual home xG: {y_home_xg_test.mean():.3f}')
    print(f'  Test avg predicted home xG: {home_pred_test.mean():.3f}')

    print(f'\n  Feature importances (top 10):')
    imp = home_reg.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])
    for i, (name, val) in enumerate(ranked[:10]):
        print(f'    {i+1:2d}. {name:30s} {val:.4f}')

    # --- Model B: Away xG Regressor ---
    print(f'\n=== Model B — Away xG Regressor ===')
    away_reg = XGBRegressor(**xgb_params)
    away_reg.fit(X_train_scaled, y_away_xg_train)

    away_pred_test = away_reg.predict(X_test_scaled)
    away_mae = mean_absolute_error(y_away_xg_test, away_pred_test)
    print(f'  Test MAE: {away_mae:.4f}')
    print(f'  Test avg actual away xG: {y_away_xg_test.mean():.3f}')
    print(f'  Test avg predicted away xG: {away_pred_test.mean():.3f}')

    print(f'\n  Feature importances (top 10):')
    imp = away_reg.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])
    for i, (name, val) in enumerate(ranked[:10]):
        print(f'    {i+1:2d}. {name:30s} {val:.4f}')

    # --- Model C: Outcome Classifier ---
    print(f'\n=== Model C — Outcome Classifier ===')
    clf = XGBClassifier(
        **xgb_params,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
    )
    clf.fit(X_train_scaled, y_outcome_train)

    outcome_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_outcome_test, outcome_pred)
    print(f'  Test accuracy: {test_acc:.4f}')
    print(f'\n  Classification report (test set):')
    print(classification_report(y_outcome_test, outcome_pred,
                                target_names=['Away Win', 'Draw', 'Home Win']))

    print(f'  Feature importances (top 10):')
    imp = clf.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])
    for i, (name, val) in enumerate(ranked[:10]):
        print(f'    {i+1:2d}. {name:30s} {val:.4f}')

    # === Check: Elo/diff features in top 5 for at least 2 of 3 models ===
    print(f'\n=== FEATURE IMPORTANCE CHECK ===')
    target_features = {'elo_diff', 'ppg_diff', 'xg_diff', 'defensive_diff',
                       'home_elo', 'away_elo'}
    models = [
        ('Home xG Reg', home_reg.feature_importances_),
        ('Away xG Reg', away_reg.feature_importances_),
        ('Classifier', clf.feature_importances_),
    ]
    models_passing = 0
    for name, importances in models:
        ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
        top5_names = {r[0] for r in ranked[:5]}
        overlap = top5_names & target_features
        passed = len(overlap) >= 1
        if passed:
            models_passing += 1
        print(f'  {name}: top 5 = {[r[0] for r in ranked[:5]]}')
        print(f'    Elo/diff features in top 5: {overlap} -> {"PASS" if passed else "FAIL"}')

    if models_passing >= 2:
        print(f'\n  Overall: {models_passing}/3 models have Elo/diff in top 5 -> PASS')
    else:
        print(f'\n  Overall: {models_passing}/3 models have Elo/diff in top 5 -> FAIL')
        print(f'  STOPPING — review before continuing.')
        return

    # === Save models ===
    print(f'\nSaving models...')
    with open('home_regressor.pkl', 'wb') as f:
        pickle.dump(home_reg, f)
    with open('away_regressor.pkl', 'wb') as f:
        pickle.dump(away_reg, f)
    with open('outcome_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print(f'  Saved: home_regressor.pkl, away_regressor.pkl, outcome_classifier.pkl, scaler.pkl')


if __name__ == '__main__':
    main()

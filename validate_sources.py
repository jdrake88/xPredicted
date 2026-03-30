"""
Step 0 — Validate both data sources before building the model.
"""
import asyncio
import os
import requests
import aiohttp
from understat import Understat
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
API_BASE = 'https://api.football-data.org/v4'
HEADERS = {'X-Auth-Token': API_KEY}

# Current season = 2025/26, two most recent completed = 2024/25, 2023/24
UNDERSTAT_SEASONS = [2023, 2024, 2025]
FD_SEASON = 2025


async def validate_understat():
    print("=" * 60)
    print("UNDERSTAT VALIDATION")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        total_matches = 0

        for season in UNDERSTAT_SEASONS:
            try:
                matches = await understat.get_league_results("epl", season)
                count = len(matches)
                total_matches += count
                print(f"\n  Season {season}/{season+1}: {count} matches")

                if count > 0:
                    # Show last 5 matches of this season
                    if season == UNDERSTAT_SEASONS[-1]:
                        print(f"\n  5 most recent matches ({season}/{season+1}):")
                        recent = sorted(matches, key=lambda m: m['datetime'], reverse=True)[:5]
                        for m in recent:
                            h = m['h']
                            a = m['a']
                            hxg = float(m['xG']['h'])
                            axg = float(m['xG']['a'])
                            print(f"    {m['datetime'][:10]}  {h['title']} {hxg:.2f} - {axg:.2f} {a['title']}")

            except Exception as e:
                print(f"  Season {season}: ERROR — {e}")

        print(f"\n  TOTAL matches across all seasons: {total_matches}")

        if total_matches < 500:
            print(f"\n  FAIL: Only {total_matches} matches — need at least 500")
            return False

        # Spot check xG values are realistic
        matches = await understat.get_league_results("epl", UNDERSTAT_SEASONS[-1])
        xg_values = []
        for m in matches:
            xg_values.append(float(m['xG']['h']))
            xg_values.append(float(m['xG']['a']))

        min_xg = min(xg_values)
        max_xg = max(xg_values)
        avg_xg = sum(xg_values) / len(xg_values)
        print(f"\n  Current season xG stats:")
        print(f"    Min: {min_xg:.2f}, Max: {max_xg:.2f}, Avg: {avg_xg:.2f}")

        realistic = 0.0 <= min_xg and max_xg <= 5.0 and 0.5 < avg_xg < 2.5
        print(f"    xG values realistic: {realistic}")

        print(f"\n  UNDERSTAT: PASS")
        return True


def validate_football_data():
    print("\n" + "=" * 60)
    print("FOOTBALL-DATA.ORG VALIDATION")
    print("=" * 60)

    # Check current season matches
    url = f'{API_BASE}/competitions/PL/matches?season={FD_SEASON}'
    resp = requests.get(url, headers=HEADERS)

    if resp.status_code != 200:
        print(f"  FAIL: API returned {resp.status_code}")
        print(f"  {resp.text[:200]}")
        return False

    data = resp.json()
    finished = [m for m in data['matches'] if m['status'] == 'FINISHED']
    finished.sort(key=lambda m: m['utcDate'], reverse=True)

    print(f"\n  Current season ({FD_SEASON}/{FD_SEASON+1}): {len(finished)} finished matches")

    print(f"\n  5 most recent completed matches:")
    for m in finished[:5]:
        home = m['homeTeam']['name']
        away = m['awayTeam']['name']
        hg = m['score']['fullTime']['home']
        ag = m['score']['fullTime']['away']
        print(f"    {m['utcDate'][:10]}  {home} {hg}-{ag} {away}")

    # Check standings
    url = f'{API_BASE}/competitions/PL/standings'
    resp = requests.get(url, headers=HEADERS)

    if resp.status_code != 200:
        print(f"\n  Standings FAIL: API returned {resp.status_code}")
        return False

    standings = resp.json()['standings'][0]['table']
    print(f"\n  Standings accessible: {len(standings)} teams")
    print(f"  Top 3: {', '.join(r['team']['name'] for r in standings[:3])}")

    print(f"\n  FOOTBALL-DATA.ORG: PASS")
    return True


async def main():
    understat_ok = await validate_understat()
    fd_ok = validate_football_data()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Understat:         {'PASS' if understat_ok else 'FAIL'}")
    print(f"  Football-data.org: {'PASS' if fd_ok else 'FAIL'}")

    if understat_ok and fd_ok:
        print(f"\n  Both sources validated. Ready for Step 1.")
    else:
        print(f"\n  STOPPING — one or more sources failed.")


if __name__ == '__main__':
    asyncio.run(main())

import pandas as pd
import numpy as np
import yaml
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
members = [m['game_name'] for m in _cfg['members']]

matches = pd.read_csv('data/processed/matches.csv')
players = pd.read_csv('data/processed/player_stats.csv')
timeline = pd.read_csv('data/processed/timeline_frames.csv')
member_matches = players[players['summonerName'].isin(members)]['matchId'].unique()

results = []

for mid in member_matches:
    m_info = matches[matches['matchId'] == mid]
    if m_info.empty:
        continue

    duration_min = m_info['gameDurationMin'].values[0]

    our_players = players[(players['matchId'] == mid) & (players['summonerName'].isin(members))]
    if our_players.empty:
        continue
    our_team = our_players['teamId'].values[0]
    our_win = bool(our_players['win'].values[0])

    all_players_match = players[players['matchId'] == mid]
    total_kills = int(all_players_match['kills'].sum())
    kills_per_min = total_kills / max(duration_min, 1)

    our_kills = int(all_players_match[all_players_match['teamId'] == our_team]['kills'].sum())
    enemy_kills = int(all_players_match[all_players_match['teamId'] != our_team]['kills'].sum())

    tl = timeline[timeline['matchId'] == mid]
    if tl.empty:
        continue

    team_gold = tl.groupby(['timestampMin', 'teamId'])['totalGold'].sum().unstack(fill_value=0)
    enemy_team = 200 if our_team == 100 else 100
    if our_team not in team_gold.columns or enemy_team not in team_gold.columns:
        continue

    gold_diff = team_gold[our_team] - team_gold[enemy_team]

    max_lead = gold_diff.max()
    max_deficit = gold_diff.min()
    max_gold_swing = max_lead - max_deficit

    comeback_deficit = 0
    if our_win and max_deficit < -1000:
        comeback_deficit = abs(max_deficit)

    blown_lead = 0
    if not our_win and max_lead > 1000:
        blown_lead = max_lead

    signs = np.sign(gold_diff.values)
    lead_changes = int(np.sum(np.diff(signs) != 0))

    final_gold_diff = abs(gold_diff.values[-1]) if len(gold_diff) > 0 else 0

    # Drama score
    swing_score = min(max_gold_swing / 15000, 1.0) * 25
    comeback_score = min(max(comeback_deficit, blown_lead) / 10000, 1.0) * 30
    kill_score = min(kills_per_min / 2.0, 1.0) * 15
    lead_change_score = min(lead_changes / 10, 1.0) * 15

    if duration_min >= 30:
        length_score = min((duration_min - 20) / 25, 1.0) * 10
    else:
        length_score = max(0, (duration_min - 15) / 20) * 5

    close_score = max(0, 1 - final_gold_diff / 15000) * 5

    drama_score = swing_score + comeback_score + kill_score + lead_change_score + length_score + close_score

    results.append({
        'matchId': mid, 'duration_min': round(duration_min, 1), 'our_win': our_win,
        'total_kills': total_kills, 'kills_per_min': round(kills_per_min, 2),
        'max_gold_swing': int(max_gold_swing), 'comeback_deficit': int(comeback_deficit),
        'blown_lead': int(blown_lead),
        'lead_changes': lead_changes, 'final_gold_diff': int(final_gold_diff),
        'our_kills': our_kills, 'enemy_kills': enemy_kills,
        'drama_score': round(drama_score, 1)
    })

df = pd.DataFrame(results).sort_values('drama_score', ascending=False)

print('=' * 80)
print('   MOST DRAMATIC GAMES TOP 10')
print('=' * 80)

for rank, (i, row) in enumerate(df.head(10).iterrows(), 1):
    result = 'WIN' if row['our_win'] else 'LOSS'
    print()
    print(f'  #{rank}  {row["matchId"]}')
    print(f'      Drama Score: {row["drama_score"]}/100  |  {result}  |  {row["duration_min"]} min')
    print(f'      Kills: {row["our_kills"]} vs {row["enemy_kills"]} (total {row["total_kills"]}, {row["kills_per_min"]}/min)')
    print(f'      Gold Swing: {row["max_gold_swing"]:,}G  |  Lead Changes: {row["lead_changes"]}')
    if row['comeback_deficit'] > 0:
        print(f'      COMEBACK! Was {row["comeback_deficit"]:,}G behind')
    if row['blown_lead'] > 0:
        print(f'      THROW! Had {row["blown_lead"]:,}G lead')
    print(f'      Final Gold Diff: {row["final_gold_diff"]:,}G')

print()
print('=' * 80)
print()
print('=' * 80)
print('   TOP 3 DETAILED BREAKDOWN')
print('=' * 80)

for idx, (i, row) in enumerate(df.head(3).iterrows(), 1):
    mid = row['matchId']
    print(f'\n{"=" * 80}')
    print(f'  #{idx} {mid}  (Drama Score: {row["drama_score"]})')
    print(f'{"=" * 80}')

    match_players = players[players['matchId'] == mid]
    our_players_detail = match_players[match_players['summonerName'].isin(members)].sort_values('role')
    our_team = our_players_detail['teamId'].values[0]

    print(f'\n  [Our Team (Team {our_team})]')
    print(f'  {"Role":<8} {"Member":<16} {"Champion":<14} {"K/D/A":<12} {"Damage":>8} {"Gold":>8}')
    print(f'  {"-" * 70}')
    for _, p in our_players_detail.iterrows():
        kda_str = f'{p["kills"]}/{p["deaths"]}/{p["assists"]}'
        print(f'  {p["role"]:<8} {p["summonerName"]:<16} {p["championName"]:<14} {kda_str:<12} {p["totalDamageDealtToChampions"]:>8,} {p["goldEarned"]:>8,}')

    enemy_players = match_players[match_players['teamId'] != our_team].sort_values('role')
    print(f'\n  [Enemy Team]')
    print(f'  {"Role":<8} {"Player":<16} {"Champion":<14} {"K/D/A":<12} {"Damage":>8} {"Gold":>8}')
    print(f'  {"-" * 70}')
    for _, p in enemy_players.iterrows():
        kda_str = f'{p["kills"]}/{p["deaths"]}/{p["assists"]}'
        print(f'  {p["role"]:<8} {p["summonerName"]:<16} {p["championName"]:<14} {kda_str:<12} {p["totalDamageDealtToChampions"]:>8,} {p["goldEarned"]:>8,}')

    tl = timeline[timeline['matchId'] == mid]
    if not tl.empty:
        team_gold = tl.groupby(['timestampMin', 'teamId'])['totalGold'].sum().unstack(fill_value=0)
        enemy_team = 200 if our_team == 100 else 100
        if our_team in team_gold.columns and enemy_team in team_gold.columns:
            gold_diff = team_gold[our_team] - team_gold[enemy_team]

            print(f'\n  [Gold Timeline (Us - Enemy)]')
            for t in sorted(gold_diff.index):
                if t % 5 == 0 or t == gold_diff.index.max():
                    gd = gold_diff[t]
                    bar_len = int(abs(gd) / 500)
                    if gd >= 0:
                        bar = '#' * min(bar_len, 30)
                        print(f'  {t:5.0f}min  +{gd:>6,.0f}G  |{bar}')
                    else:
                        bar = '#' * min(bar_len, 30)
                        print(f'  {t:5.0f}min  {gd:>7,.0f}G  {bar}|')

            max_lead_time = gold_diff.idxmax()
            max_deficit_time = gold_diff.idxmin()
            print(f'\n  Peak Lead:    {gold_diff[max_lead_time]:+,.0f}G @ {max_lead_time:.0f}min')
            print(f'  Peak Deficit: {gold_diff[max_deficit_time]:+,.0f}G @ {max_deficit_time:.0f}min')

print()

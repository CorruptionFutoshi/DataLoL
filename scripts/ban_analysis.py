"""Analyze enemy champions to determine optimal ban priorities."""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

df = pd.read_csv(DATA / 'player_stats.csv')

with open(CONFIG, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
members = [m['game_name'] for m in cfg.get('members', [])]

our_rows = df[df['summonerName'].isin(members)]
our_matches = our_rows[['matchId', 'teamId']].drop_duplicates()

match_team = our_matches.groupby('matchId')['teamId'].first().reset_index()
match_team.columns = ['matchId', 'ourTeamId']

df2 = df.merge(match_team, on='matchId', how='inner')

enemy = df2[df2['teamId'] != df2['ourTeamId']].copy()

total_our_matches = match_team['matchId'].nunique()
print(f'=== 分析対象: {total_our_matches} 試合 ===')
print()

champ_stats = enemy.groupby('championName').agg(
    appearances=('matchId', 'nunique'),
    wins=('win', 'sum'),
    total_kills=('kills', 'sum'),
    total_deaths=('deaths', 'sum'),
    total_assists=('assists', 'sum'),
    avg_damage=('totalDamageDealtToChampions', 'mean'),
    avg_gold=('goldEarned', 'mean'),
).reset_index()

champ_stats['win_rate'] = (champ_stats['wins'] / champ_stats['appearances'] * 100).round(1)
champ_stats['pick_rate'] = (champ_stats['appearances'] / total_our_matches * 100).round(1)
champ_stats['avg_kda'] = ((champ_stats['total_kills'] + champ_stats['total_assists']) / champ_stats['total_deaths'].clip(lower=1)).round(2)

MIN_GAMES = 3
qualified = champ_stats[champ_stats['appearances'] >= MIN_GAMES].copy()

wr = qualified['win_rate']
pr = qualified['pick_rate']
qualified['wr_norm'] = ((wr - wr.min()) / (wr.max() - wr.min()) * 100).round(1)
qualified['pr_norm'] = ((pr - pr.min()) / (pr.max() - pr.min()) * 100).round(1)

# Ban Score = 60% win rate + 40% pick rate
qualified['ban_score'] = (qualified['wr_norm'] * 0.6 + qualified['pr_norm'] * 0.4).round(1)

top_bans = qualified.sort_values('ban_score', ascending=False).head(20)

print('■ 真にBANすべきチャンピオン TOP20')
print('  (BAN優先度 = 勝率重視60% + ピック率重視40%)')
print(f'  ※ {MIN_GAMES}試合以上出場のチャンピオンのみ')
print()

header = f"{'#':<3} {'Champion':<16} {'BAN':>5} {'WinRate':>8} {'Games':>6} {'PickR':>6} {'KDA':>6} {'AvgDmg':>8}"
print(header)
print('-' * len(header))
for i, (_, r) in enumerate(top_bans.iterrows(), 1):
    print(f"{i:<3} {r['championName']:<16} {r['ban_score']:>5.1f} {r['win_rate']:>7.1f}% {int(r['appearances']):>6} {r['pick_rate']:>5.1f}% {r['avg_kda']:>6.2f} {r['avg_damage']:>8.0f}")

print()
print('■ 超高勝率チャンピオン（勝率75%以上 & 3試合以上）')
high_wr = qualified[qualified['win_rate'] >= 75].sort_values(['win_rate', 'appearances'], ascending=[False, False])
if len(high_wr) > 0:
    for _, r in high_wr.iterrows():
        print(f"  {r['championName']:<16} 勝率{r['win_rate']}% ({int(r['wins'])}/{int(r['appearances'])}試合)  KDA:{r['avg_kda']}")
else:
    print('  該当なし')

print()
print('■ 頻出チャンピオン TOP10（メタの中心）')
top_picked = qualified.sort_values('appearances', ascending=False).head(10)
for _, r in top_picked.iterrows():
    wr_mark = " ★危険" if r['win_rate'] >= 55 else ""
    print(f"  {r['championName']:<16} {int(r['appearances'])}試合 (ピック率{r['pick_rate']}%)  勝率{r['win_rate']}%  KDA:{r['avg_kda']}{wr_mark}")

print()
print('■ ロール別BAN候補（各ロール最も脅威的なチャンピオン）')
role_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
enemy_roles = df2[df2['teamId'] != df2['ourTeamId']].copy()
for role in role_order:
    role_data = enemy_roles[enemy_roles['role'] == role]
    role_champ = role_data.groupby('championName').agg(
        appearances=('matchId', 'nunique'),
        wins=('win', 'sum'),
    ).reset_index()
    role_champ['win_rate'] = (role_champ['wins'] / role_champ['appearances'] * 100).round(1)
    role_q = role_champ[role_champ['appearances'] >= MIN_GAMES].copy()
    if len(role_q) > 0:
        role_q['threat'] = role_q['win_rate'] * np.log1p(role_q['appearances'])
        best = role_q.sort_values('threat', ascending=False).head(3)
        champs = ", ".join([f"{r['championName']}({r['win_rate']}%/{int(r['appearances'])}試合)" for _, r in best.iterrows()])
        print(f"  {role:<8} → {champs}")

"""Gold share vs win rate analysis: who benefits most from team gold concentration."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

with open(CONFIG, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
members_set = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get('members', [])}

ps = pd.read_csv(DATA / 'player_stats.csv')
ps['riotId'] = ps['summonerName'].astype(str) + '#' + ps['tagLine'].astype(str)
ps['is_member'] = ps['riotId'].isin(members_set)

mem = ps[ps['is_member']].copy()
team_gold = mem.groupby('matchId')['goldEarned'].transform('sum').clip(lower=1)
mem['gold_share'] = mem['goldEarned'] / team_gold

member_names = sorted(mem['summonerName'].unique())

print("=" * 78)
print("  ゴールドシェアと勝率の関係 — 「誰にゴールドが集まると勝てるか」")
print("=" * 78)
print()
print("  上位50%WR = そのメンバーのGシェアが自分の中央値以上の試合の勝率")
print("  下位50%WR = Gシェアが中央値未満の試合の勝率")
print("  差(+) = ゴールドを多く持つほど勝ちやすい → キャリーにゴールドを回す価値あり")
print()

header = f"  {'メンバー':14s}  {'平均Gシェア':>9s}  {'上位50%WR':>10s}  {'下位50%WR':>10s}  {'差':>9s}  {'Gシェア-WR相関':>14s}  {'p値':>8s}"
print(header)
print("  " + "-" * 90)

results = []
for name in member_names:
    sub = mem[mem['summonerName'] == name]
    n = len(sub)
    if n < 30:
        continue
    gs_mean = sub['gold_share'].mean()
    gs_med = sub['gold_share'].median()

    above = sub[sub['gold_share'] >= gs_med]
    below = sub[sub['gold_share'] < gs_med]

    wr_above = above['win'].mean()
    wr_below = below['win'].mean()
    diff = wr_above - wr_below

    r, p = stats.pointbiserialr(sub['win'].astype(int), sub['gold_share'])
    results.append((name, n, gs_mean, wr_above, len(above), wr_below, len(below), diff, r, p))

results.sort(key=lambda x: -x[7])
for name, n, gs_mean, wr_a, na, wr_b, nb, diff, r, p in results:
    sig_mark = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    print(f"  {name:14s}  {gs_mean*100:8.1f}%  {wr_a*100:5.1f}%({na:3d})  {wr_b*100:5.1f}%({nb:3d})  {diff*100:+8.1f}pp  r={r:+.3f}         {p:.4f} {sig_mark}")

print()

# Role-specific analysis
print("  ── ロール別: ゴールドシェアと勝率の相関 ──")
print()
print(f"  {'メンバー':14s}  {'ロール':>8s}  {'試合':>5s}  {'Gシェア→WR相関r':>16s}  {'p値':>8s}")
print("  " + "-" * 65)

for name in member_names:
    sub = mem[mem['summonerName'] == name]
    role_results = []
    for role in sub['role'].unique():
        role_sub = sub[sub['role'] == role]
        if len(role_sub) < 20:
            continue
        r, p = stats.pointbiserialr(role_sub['win'].astype(int), role_sub['gold_share'])
        role_results.append((role, len(role_sub), r, p))
    role_results.sort(key=lambda x: -x[2])
    for role, rn, r, p in role_results:
        sig_mark = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        print(f"  {name:14s}  {role:>8s}  {rn:5d}  r={r:+.3f}            {p:.4f} {sig_mark}")

print()
print("=" * 78)

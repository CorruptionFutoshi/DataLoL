"""ファーストタワー勝率: 味方チーム vs エメラルド帯ベンチマーク比較"""

import json
import os
import pandas as pd
import yaml
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'D:\データLoL')

# ── Load member data ──────────────────────────────────
events = pd.read_csv(ROOT / 'data/processed/timeline_events.csv')
ps     = pd.read_csv(ROOT / 'data/processed/player_stats.csv')

with open(ROOT / 'config/settings.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
members = [m['game_name'] for m in cfg['members']]

our_team = ps[ps['summonerName'].isin(members)].groupby('matchId')['teamId'].first().to_dict()
our_win  = ps[ps['summonerName'].isin(members)].groupby('matchId')['win'].first().to_dict()

# ── Helper ────────────────────────────────────────────
LANE_MAP  = {'TOP_LANE': 'TOP', 'MID_LANE': 'MID', 'BOT_LANE': 'BOT'}
TOWER_MAP = {'OUTER_TURRET': '外(T1)', 'INNER_TURRET': '内(T2)',
             'BASE_TURRET': 'インヒビ(T3)', 'NEXUS_TURRET': 'ネクサス'}

def classify_tower(lane_type, tower_type):
    lane  = LANE_MAP.get(lane_type, lane_type)
    tower = TOWER_MAP.get(tower_type, tower_type)
    if lane and tower:
        return f'{lane} {tower}'
    return None


# ══════════════════════════════════════════════════════
#  味方チーム分析 (既存CSVから)
# ══════════════════════════════════════════════════════
towers = events[
    (events['buildingType'] == 'TOWER_BUILDING') & (events['towerType'].notna())
].copy()
first_towers = towers.sort_values('timestampMin').groupby('matchId').first().reset_index()

us_rows = []
for _, row in first_towers.iterrows():
    mid = row['matchId']
    if mid not in our_team:
        continue
    label  = classify_tower(row['laneType'], row['towerType'])
    victim = row['teamId']
    side   = '取得' if victim != our_team[mid] else '喪失'
    us_rows.append({
        'matchId': mid, 'tower_label': label,
        'side': side, 'win': our_win[mid],
        'timestampMin': row['timestampMin'],
    })
us_df = pd.DataFrame(us_rows)


# ══════════════════════════════════════════════════════
#  ベンチマーク分析 (raw JSONをパース)
# ══════════════════════════════════════════════════════
BM_TL_DIR  = ROOT / 'data/raw/benchmark/timelines'
BM_MATCH_DIR = ROOT / 'data/raw/benchmark/matches'
bm_stats   = pd.read_csv(ROOT / 'data/processed/benchmark_stats.csv')

# matchId → {teamId: win}  from benchmark_stats
bm_wins = {}
for _, r in bm_stats[['matchId', 'teamId', 'win']].drop_duplicates().iterrows():
    bm_wins.setdefault(r['matchId'], {})[int(r['teamId'])] = bool(r['win'])

print('ベンチマークタイムライン解析中...', flush=True)

bm_rows = []
bm_files = list(BM_TL_DIR.glob('*.json'))

for fp in bm_files:
    match_id = fp.stem
    if match_id not in bm_wins:
        continue
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            tl = json.load(f)
    except Exception:
        continue

    frames = tl.get('info', {}).get('frames', [])
    first_tower = None
    for frame in frames:
        for ev in frame.get('events', []):
            if ev.get('type') != 'BUILDING_KILL':
                continue
            if ev.get('buildingType') != 'TOWER_BUILDING':
                continue
            tt = ev.get('towerType')
            lt = ev.get('laneType')
            if not tt:
                continue
            first_tower = {
                'towerType': tt,
                'laneType': lt,
                'teamId': ev.get('teamId'),        # tower owner (victim)
                'timestamp': ev.get('timestamp', 0),
            }
            break
        if first_tower:
            break

    if not first_tower:
        continue

    victim_team = first_tower['teamId']
    label = classify_tower(first_tower['laneType'], first_tower['towerType'])
    ts_min = first_tower['timestamp'] / 60000.0

    # Analyze from both teams' perspectives
    for team_id in [100, 200]:
        if team_id not in bm_wins[match_id]:
            continue
        win  = bm_wins[match_id][team_id]
        side = '取得' if victim_team != team_id else '喪失'
        bm_rows.append({
            'matchId': match_id, 'teamId': team_id,
            'tower_label': label, 'side': side,
            'win': win, 'timestampMin': ts_min,
        })

bm_df = pd.DataFrame(bm_rows)
bm_matches = bm_df['matchId'].nunique()


# ══════════════════════════════════════════════════════
#  比較表示
# ══════════════════════════════════════════════════════
us_total = us_df['matchId'].nunique()
print()
print('=' * 72)
print(f'  ファーストタワー勝率: 味方 vs エメラルド帯ベンチマーク')
print(f'  味方: {us_total}試合 / ベンチマーク: {bm_matches}試合')
print('=' * 72)

# ── PART 1: Overall ──
print()
print('━' * 60)
print('■ PART 1: ファーストタワー全体勝率')
print('━' * 60)
print(f'  {"":12s}  {"味方チーム":>14s}  {"エメラルド帯":>14s}  {"差分":>8s}')
print(f'  {"─" * 12}  {"─" * 14}  {"─" * 14}  {"─" * 8}')

for side in ['取得', '喪失']:
    us_sub = us_df[us_df['side'] == side]
    bm_sub = bm_df[bm_df['side'] == side]
    us_wr  = us_sub['win'].mean() * 100 if len(us_sub) else 0
    bm_wr  = bm_sub['win'].mean() * 100 if len(bm_sub) else 0
    us_n   = len(us_sub)
    bm_n   = len(bm_sub)
    diff   = us_wr - bm_wr
    sign   = '+' if diff >= 0 else ''
    label  = 'FT取得時' if side == '取得' else 'FT喪失時'
    print(f'  {label:12s}  {us_wr:5.1f}% ({us_n:3d}G)  {bm_wr:5.1f}% ({bm_n:4d}G)  {sign}{diff:5.1f}pp')

us_ft_diff = us_df[us_df['side'] == '取得']['win'].mean() * 100 - us_df[us_df['side'] == '喪失']['win'].mean() * 100
bm_ft_diff = bm_df[bm_df['side'] == '取得']['win'].mean() * 100 - bm_df[bm_df['side'] == '喪失']['win'].mean() * 100
print(f'  {"影響度(差)":12s}  {us_ft_diff:5.1f}pp       {bm_ft_diff:5.1f}pp')

# ── PART 2: Per tower ──
print()
print('━' * 60)
print('■ PART 2: タワー別ファーストタワー勝率 (T1)')
print('━' * 60)

for lane in ['TOP', 'MID', 'BOT']:
    tl = f'{lane} 外(T1)'
    print(f'\n  【{tl}】')
    print(f'  {"":12s}  {"味方チーム":>14s}  {"エメラルド帯":>14s}  {"差分":>8s}')
    print(f'  {"─" * 12}  {"─" * 14}  {"─" * 14}  {"─" * 8}')

    for side in ['取得', '喪失']:
        us_sub = us_df[(us_df['tower_label'] == tl) & (us_df['side'] == side)]
        bm_sub = bm_df[(bm_df['tower_label'] == tl) & (bm_df['side'] == side)]
        us_wr  = us_sub['win'].mean() * 100 if len(us_sub) else 0
        bm_wr  = bm_sub['win'].mean() * 100 if len(bm_sub) else 0
        us_n   = len(us_sub)
        bm_n   = len(bm_sub)
        diff   = us_wr - bm_wr
        sign   = '+' if diff >= 0 else ''
        label  = 'FT取得時' if side == '取得' else 'FT喪失時'
        print(f'  {label:12s}  {us_wr:5.1f}% ({us_n:3d}G)  {bm_wr:5.1f}% ({bm_n:4d}G)  {sign}{diff:5.1f}pp')

# ── PART 3: Frequency comparison ──
print()
print('━' * 60)
print('■ PART 3: どのタワーが最初に折れるか (頻度比較)')
print('━' * 60)
print(f'  {"タワー":20s}  {"味方":>10s}  {"エメラルド":>10s}')
print(f'  {"─" * 20}  {"─" * 10}  {"─" * 10}')

for tl in ['TOP 外(T1)', 'MID 外(T1)', 'BOT 外(T1)']:
    us_cnt = len(us_df[us_df['tower_label'] == tl])
    bm_cnt = len(bm_df[bm_df['tower_label'] == tl])
    us_pct = us_cnt / len(us_df) * 100 if len(us_df) else 0
    bm_pct = bm_cnt / len(bm_df) * 100 if len(bm_df) else 0
    print(f'  {tl:20s}  {us_pct:5.1f}%     {bm_pct:5.1f}%')

# ── PART 4: First-tower grab rate ──
print()
print('━' * 60)
print('■ PART 4: ファーストタワー取得率 (味方が先に折る率)')
print('━' * 60)
print(f'  {"タワー":20s}  {"味方":>10s}  {"エメラルド":>10s}')
print(f'  {"─" * 20}  {"─" * 10}  {"─" * 10}')

# Overall
us_grab = len(us_df[us_df['side'] == '取得']) / len(us_df) * 100
bm_grab = len(bm_df[bm_df['side'] == '取得']) / len(bm_df) * 100
print(f'  {"全体":20s}  {us_grab:5.1f}%     {bm_grab:5.1f}%')

for tl in ['TOP 外(T1)', 'MID 外(T1)', 'BOT 外(T1)']:
    us_sub = us_df[us_df['tower_label'] == tl]
    bm_sub = bm_df[bm_df['tower_label'] == tl]
    us_r   = len(us_sub[us_sub['side'] == '取得']) / len(us_sub) * 100 if len(us_sub) else 0
    bm_r   = len(bm_sub[bm_sub['side'] == '取得']) / len(bm_sub) * 100 if len(bm_sub) else 0
    print(f'  {tl:20s}  {us_r:5.1f}%     {bm_r:5.1f}%')

# ── PART 5: Timing comparison ──
print()
print('━' * 60)
print('■ PART 5: ファーストタワー時間 比較')
print('━' * 60)
print(f'  {"":12s}  {"味方(中央値)":>14s}  {"エメラルド(中央値)":>18s}')
print(f'  {"─" * 12}  {"─" * 14}  {"─" * 18}')

for side in ['取得', '喪失']:
    us_sub = us_df[us_df['side'] == side]
    bm_sub = bm_df[bm_df['side'] == side]
    us_med = us_sub['timestampMin'].median() if len(us_sub) else 0
    bm_med = bm_sub['timestampMin'].median() if len(bm_sub) else 0
    label  = 'FT取得時' if side == '取得' else 'FT喪失時'
    print(f'  {label:12s}  {us_med:6.1f}分       {bm_med:6.1f}分')

# ── PART 6: Key insight ──
print()
print('━' * 60)
print('■ PART 6: 味方チームの特徴 (vs エメラルド帯)')
print('━' * 60)

# Compare key metrics
insights = []

# FT grab rate
if us_grab > bm_grab + 2:
    insights.append(f'✓ ファーストタワー取得率が高い ({us_grab:.1f}% vs {bm_grab:.1f}%)')
elif us_grab < bm_grab - 2:
    insights.append(f'✗ ファーストタワー取得率が低い ({us_grab:.1f}% vs {bm_grab:.1f}%)')
else:
    insights.append(f'≈ ファーストタワー取得率は平均並み ({us_grab:.1f}% vs {bm_grab:.1f}%)')

# FT impact comparison
if us_ft_diff > bm_ft_diff + 3:
    insights.append(f'! ファーストタワーへの依存度が平均より高い (影響度 {us_ft_diff:.1f}pp vs {bm_ft_diff:.1f}pp)')
elif us_ft_diff < bm_ft_diff - 3:
    insights.append(f'✓ ファーストタワーを取られても平均より巻き返せている')

# BOT weakness check
us_bot_loss = us_df[(us_df['tower_label'] == 'BOT 外(T1)') & (us_df['side'] == '喪失')]['win'].mean() * 100
bm_bot_loss = bm_df[(bm_df['tower_label'] == 'BOT 外(T1)') & (bm_df['side'] == '喪失')]['win'].mean() * 100
if us_bot_loss < bm_bot_loss - 3:
    insights.append(f'✗ BOT T1を折られた時の勝率が平均より低い ({us_bot_loss:.1f}% vs {bm_bot_loss:.1f}%)')

# TOP resilience check
us_top_loss = us_df[(us_df['tower_label'] == 'TOP 外(T1)') & (us_df['side'] == '喪失')]['win'].mean() * 100
bm_top_loss = bm_df[(bm_df['tower_label'] == 'TOP 外(T1)') & (bm_df['side'] == '喪失')]['win'].mean() * 100
if us_top_loss > bm_top_loss + 3:
    insights.append(f'✓ TOP T1を折られても平均より巻き返せている ({us_top_loss:.1f}% vs {bm_top_loss:.1f}%)')

for ins in insights:
    print(f'  {ins}')

print()

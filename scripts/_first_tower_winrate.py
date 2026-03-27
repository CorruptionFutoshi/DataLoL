"""最初に破壊した/されたタワー別の勝率分析"""

import pandas as pd
import yaml
import sys
sys.stdout.reconfigure(encoding='utf-8')

events = pd.read_csv(r'D:\データLoL\data\processed\timeline_events.csv')
ps = pd.read_csv(r'D:\データLoL\data\processed\player_stats.csv')

with open(r'D:\データLoL\config\settings.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
members = [m['game_name'] for m in cfg['members']]

our_team = ps[ps['summonerName'].isin(members)].groupby('matchId')['teamId'].first().to_dict()
our_win  = ps[ps['summonerName'].isin(members)].groupby('matchId')['win'].first().to_dict()

towers = events[
    (events['buildingType'] == 'TOWER_BUILDING') & (events['towerType'].notna())
].copy()

first_towers = towers.sort_values('timestampMin').groupby('matchId').first().reset_index()

lane_map  = {'TOP_LANE': 'TOP', 'MID_LANE': 'MID', 'BOT_LANE': 'BOT'}
tower_map = {'OUTER_TURRET': '外(T1)', 'INNER_TURRET': '内(T2)',
             'BASE_TURRET': 'インヒビ(T3)', 'NEXUS_TURRET': 'ネクサス'}

first_towers['lane']  = first_towers['laneType'].map(lane_map)
first_towers['tower'] = first_towers['towerType'].map(tower_map)
first_towers['tower_label'] = first_towers['lane'] + ' ' + first_towers['tower']

rows = []
for _, row in first_towers.iterrows():
    mid = row['matchId']
    if mid not in our_team:
        continue
    our_tid = our_team[mid]
    win     = our_win[mid]
    victim  = row['teamId']  # teamId = tower owner (victim)

    side = '味方が破壊' if victim != our_tid else '敵が破壊'
    rows.append({
        'matchId': mid,
        'tower_label': row['tower_label'],
        'side': side,
        'win': win,
        'timestampMin': row['timestampMin'],
    })

rdf = pd.DataFrame(rows)
total = rdf['matchId'].nunique()

print('=' * 70)
print(f'  最初に破壊した/された タワー別 勝率分析  (全{total}試合)')
print('=' * 70)

# ── PART 1 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 1: ファーストタワー 全体勝率')
print('━' * 55)
for side in ['味方が破壊', '敵が破壊']:
    sub  = rdf[rdf['side'] == side]
    n    = len(sub)
    wins = int(sub['win'].sum())
    wr   = sub['win'].mean() * 100 if n else 0
    print(f'  {side}: {wr:.1f}%  ({wins}勝 {n - wins}敗 / {n}試合)')

# ── PART 2 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 2: タワー × 味方/敵 別勝率')
print('━' * 55)

tower_order = [
    'TOP 外(T1)', 'MID 外(T1)', 'BOT 外(T1)',
    'TOP 内(T2)', 'MID 内(T2)', 'BOT 内(T2)',
    'TOP インヒビ(T3)', 'MID インヒビ(T3)', 'BOT インヒビ(T3)',
]

for tl in tower_order:
    sub = rdf[rdf['tower_label'] == tl]
    if len(sub) == 0:
        continue
    print(f'\n  【{tl}】 計{len(sub)}試合')
    for side in ['味方が破壊', '敵が破壊']:
        ss   = sub[sub['side'] == side]
        n    = len(ss)
        if n == 0:
            continue
        wins = int(ss['win'].sum())
        wr   = ss['win'].mean() * 100
        bar  = '█' * int(wr // 5) + '░' * (20 - int(wr // 5))
        print(f'    {side}: {bar} {wr:.1f}%  ({wins}W {n - wins}L / {n}G)')

# ── PART 3 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 3: どのタワーが最初に折れるか (頻度順)')
print('━' * 55)
freq = rdf['tower_label'].value_counts()
for tl, cnt in freq.items():
    pct = cnt / len(rdf) * 100
    sub = rdf[rdf['tower_label'] == tl]
    wr  = sub['win'].mean() * 100
    our_rate = len(sub[sub['side'] == '味方が破壊']) / len(sub) * 100
    print(f'  {tl:20s}  {cnt:3d}回 ({pct:5.1f}%)  '
          f'勝率{wr:5.1f}%  味方先折率{our_rate:5.1f}%')

# ── PART 4 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 4: ファーストタワー平均時間')
print('━' * 55)
for side in ['味方が破壊', '敵が破壊']:
    sub = rdf[rdf['side'] == side]
    avg = sub['timestampMin'].mean()
    med = sub['timestampMin'].median()
    print(f'  {side}: 平均 {avg:.1f}分  中央値 {med:.1f}分')

# ── PART 5 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 5: レーン別ファーストタワー勝率 (外タワー T1 のみ)')
print('━' * 55)
t1 = rdf[rdf['tower_label'].str.contains('外')]

for lane in ['TOP', 'MID', 'BOT']:
    sub = t1[t1['tower_label'].str.startswith(lane)]
    if len(sub) == 0:
        continue
    print(f'\n  【{lane}レーン T1】 計{len(sub)}試合')
    for side in ['味方が破壊', '敵が破壊']:
        ss   = sub[sub['side'] == side]
        n    = len(ss)
        if n == 0:
            continue
        wins = int(ss['win'].sum())
        wr   = ss['win'].mean() * 100
        bar  = '█' * int(wr // 5) + '░' * (20 - int(wr // 5))
        print(f'    {side}: {bar} {wr:.1f}%  ({wins}W {n - wins}L / {n}G)')
    our_first = len(sub[sub['side'] == '味方が破壊'])
    print(f'    → 味方が先に折る率: {our_first / len(sub) * 100:.1f}%')

# ── PART 6 ────────────────────────────────────────
print()
print('━' * 55)
print('■ PART 6: ファーストタワー → 勝敗への影響度 (参考)')
print('━' * 55)
rdf['ft_us'] = (rdf['side'] == '味方が破壊').astype(int)
overall_wr = rdf['win'].mean() * 100
ft_wr  = rdf[rdf['ft_us'] == 1]['win'].mean() * 100
nft_wr = rdf[rdf['ft_us'] == 0]['win'].mean() * 100
diff   = ft_wr - nft_wr
print(f'  チーム全体勝率:           {overall_wr:.1f}%')
print(f'  ファーストタワー取得時勝率: {ft_wr:.1f}%')
print(f'  ファーストタワー喪失時勝率: {nft_wr:.1f}%')
print(f'  差分:                     +{diff:.1f}pp')
print()

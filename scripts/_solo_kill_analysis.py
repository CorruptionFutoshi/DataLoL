import pandas as pd
import yaml
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
members = [m['game_name'] for m in _cfg['members']]

events = pd.read_csv('data/processed/timeline_events.csv')
ps = pd.read_csv('data/processed/player_stats.csv')

kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()
kills['assist_list'] = kills['assistingParticipantIds'].apply(lambda x: eval(x) if pd.notna(x) else [])
kills['is_solo_kill'] = kills['assist_list'].apply(lambda x: len(x) == 0)

solo_kills = kills[kills['is_solo_kill']]

solo_kill_counts = solo_kills[solo_kills['killerName'].isin(members)].groupby('killerName').size().reset_index(name='solo_kills_made')
solo_death_counts = solo_kills[solo_kills['victimName'].isin(members)].groupby('victimName').size().reset_index(name='solo_deaths')
match_counts = ps[ps['summonerName'].isin(members)].groupby('summonerName')['matchId'].nunique().reset_index(name='total_matches')

result = match_counts.copy()
result = result.merge(solo_kill_counts, left_on='summonerName', right_on='killerName', how='left').drop(columns=['killerName'], errors='ignore')
result = result.merge(solo_death_counts, left_on='summonerName', right_on='victimName', how='left').drop(columns=['victimName'], errors='ignore')
result = result.fillna(0)
result['solo_kills_made'] = result['solo_kills_made'].astype(int)
result['solo_deaths'] = result['solo_deaths'].astype(int)
result['solo_kills_per_game'] = (result['solo_kills_made'] / result['total_matches']).round(3)
result['solo_deaths_per_game'] = (result['solo_deaths'] / result['total_matches']).round(3)
result['solo_kd'] = result.apply(lambda r: round(r['solo_kills_made'] / r['solo_deaths'], 2) if r['solo_deaths'] > 0 else float('inf'), axis=1)
result['net_solo'] = result['solo_kills_made'] - result['solo_deaths']
result = result.sort_values('solo_kills_made', ascending=False)

print('=== ソロキルランキング（総数） ===')
for _, r in result.iterrows():
    name = r['summonerName']
    m = int(r['total_matches'])
    sk = int(r['solo_kills_made'])
    sd = int(r['solo_deaths'])
    skpg = r['solo_kills_per_game']
    sdpg = r['solo_deaths_per_game']
    skd = r['solo_kd']
    ns = int(r['net_solo'])
    print(f"  {name:12s}  {m:3d}試合  ソロキル:{sk:3d}  ソロデス:{sd:3d}  /試合:{skpg:.2f}K/{sdpg:.2f}D  ソロKD:{skd:.2f}  差引:{ns:+d}")

print()
print('=== ネットソロキルランキング（ソロキル - ソロデス） ===')
result_net = result.sort_values('net_solo', ascending=False)
for _, r in result_net.iterrows():
    name = r['summonerName']
    sk = int(r['solo_kills_made'])
    sd = int(r['solo_deaths'])
    ns = int(r['net_solo'])
    print(f"  {name:12s}  {sk:3d} - {sd:3d} = {ns:+d}")

target = sys.argv[1] if len(sys.argv) > 1 else members[0]
target_ps = ps[ps['summonerName'] == target][['matchId', 'championName']]

print()
print(f'=== {target}のチャンピオン別ソロキル（上位10） ===')
target_solo = solo_kills[solo_kills['killerName'] == target].merge(target_ps, on='matchId', how='left')
champ_solo = target_solo.groupby('championName').size().reset_index(name='solo_kills').sort_values('solo_kills', ascending=False)
for _, r in champ_solo.head(10).iterrows():
    print(f"  {r['championName']:16s}: {int(r['solo_kills']):3d}回")

print()
print(f'=== {target}のチャンピオン別ソロデス（上位10） ===')
target_solo_d = solo_kills[solo_kills['victimName'] == target].merge(target_ps, on='matchId', how='left')
champ_solo_d = target_solo_d.groupby('championName').size().reset_index(name='solo_deaths').sort_values('solo_deaths', ascending=False)
for _, r in champ_solo_d.head(10).iterrows():
    print(f"  {r['championName']:16s}: {int(r['solo_deaths']):3d}回")

# 15分以前のソロキル（レーン戦ソロキル）
print()
print('=== レーン戦ソロキル（15分以前）ランキング ===')
early_solo = solo_kills[solo_kills['timestampMin'] <= 15]
early_sk = early_solo[early_solo['killerName'].isin(members)].groupby('killerName').size().reset_index(name='early_solo_kills')
early_sd = early_solo[early_solo['victimName'].isin(members)].groupby('victimName').size().reset_index(name='early_solo_deaths')

early_result = match_counts.copy()
early_result = early_result.merge(early_sk, left_on='summonerName', right_on='killerName', how='left').drop(columns=['killerName'], errors='ignore')
early_result = early_result.merge(early_sd, left_on='summonerName', right_on='victimName', how='left').drop(columns=['victimName'], errors='ignore')
early_result = early_result.fillna(0)
early_result['early_solo_kills'] = early_result['early_solo_kills'].astype(int)
early_result['early_solo_deaths'] = early_result['early_solo_deaths'].astype(int)
early_result['early_net'] = early_result['early_solo_kills'] - early_result['early_solo_deaths']
early_result['early_sk_pg'] = (early_result['early_solo_kills'] / early_result['total_matches']).round(3)
early_result = early_result.sort_values('early_solo_kills', ascending=False)

for _, r in early_result.iterrows():
    name = r['summonerName']
    esk = int(r['early_solo_kills'])
    esd = int(r['early_solo_deaths'])
    enet = int(r['early_net'])
    epg = r['early_sk_pg']
    print(f"  {name:12s}  レーンソロキル:{esk:3d}  レーンソロデス:{esd:3d}  差引:{enet:+d}  /試合:{epg:.3f}")

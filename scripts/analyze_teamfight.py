# -*- coding: utf-8 -*-
"""中盤の集団戦への寄りの意識を分析するスクリプト"""

import pandas as pd
import numpy as np
import ast
import yaml
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

events = pd.read_csv(DATA / 'timeline_events.csv')
ps = pd.read_csv(DATA / 'player_stats.csv')
frames = pd.read_csv(DATA / 'timeline_frames.csv')

with open(CONFIG, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in cfg.get('members', [])]

kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()

def parse_assists(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []

kills['assist_list'] = kills['assistingParticipantIds'].apply(parse_assists)
kills['num_assists'] = kills['assist_list'].apply(len)

MID_START = 14
MID_END = 25

mid_kills = kills[(kills['timestampMin'] >= MID_START) & (kills['timestampMin'] <= MID_END)].copy()

print(f'=== 中盤の集団戦への寄り分析 ({MID_START}~{MID_END}分) ===')
print(f'全キルイベント数: {len(kills)}  |  中盤キルイベント数: {len(mid_kills)}')
print()

# ---------------------------------------------------------------------------
# Build per-member mid-game participation records
# ---------------------------------------------------------------------------
records = []
for match_id in mid_kills['matchId'].unique():
    match_kills = mid_kills[mid_kills['matchId'] == match_id]
    match_ps = ps[ps['matchId'] == match_id]

    our_members = match_ps[match_ps['summonerName'].isin(MEMBERS)]
    if len(our_members) == 0:
        continue

    our_team_id = our_members['teamId'].iloc[0]
    team_kills_mid = match_kills[match_kills['killerTeamId'] == our_team_id]
    total_team_kills_mid = len(team_kills_mid)

    if total_team_kills_mid == 0:
        continue

    # Identify teamfights: clusters of kills within 30 seconds
    sorted_kills = team_kills_mid.sort_values('timestampMin')
    teamfight_kills = []
    cluster = [sorted_kills.iloc[0]]
    for i in range(1, len(sorted_kills)):
        curr = sorted_kills.iloc[i]
        prev = sorted_kills.iloc[i - 1]
        if (curr['timestampMin'] - prev['timestampMin']) <= 0.5:  # 30sec
            cluster.append(curr)
        else:
            if len(cluster) >= 2:
                teamfight_kills.extend(cluster)
            cluster = [curr]
    if len(cluster) >= 2:
        teamfight_kills.extend(cluster)

    num_teamfight_kills = len(teamfight_kills)

    for _, member in our_members.iterrows():
        pid = member['participantId']
        name = member['summonerName']
        role = member['role']
        win = member['win']

        involved_count = 0
        tf_involved = 0
        for _, kill in team_kills_mid.iterrows():
            is_involved = (kill['killerId'] == pid) or (pid in kill['assist_list'])
            if is_involved:
                involved_count += 1

        for kill_data in teamfight_kills:
            is_involved = (kill_data['killerId'] == pid) or (pid in kill_data['assist_list'])
            if is_involved:
                tf_involved += 1

        kp_mid = involved_count / total_team_kills_mid
        tf_kp = tf_involved / num_teamfight_kills if num_teamfight_kills > 0 else np.nan

        records.append({
            'matchId': match_id,
            'summonerName': name,
            'role': role,
            'win': win,
            'team_kills_mid': total_team_kills_mid,
            'participation_count': involved_count,
            'kp_mid': kp_mid,
            'teamfight_kills': num_teamfight_kills,
            'tf_involved': tf_involved,
            'tf_kp': tf_kp,
        })

df = pd.DataFrame(records)
n_matches = df['matchId'].nunique()
n_members = df['summonerName'].nunique()
print(f'分析対象: {n_matches} 試合, {n_members} メンバー')
print()

# ---------------------------------------------------------------------------
# 【1】メンバー別 中盤キル関与率 (KP%)
# ---------------------------------------------------------------------------
print('=' * 65)
print('【1】メンバー別  中盤キル関与率 (KP%)')
print('=' * 65)
member_kp = df.groupby('summonerName').agg(
    試合数=('matchId', 'nunique'),
    平均KP=('kp_mid', 'mean'),
    中央値KP=('kp_mid', 'median'),
    平均関与キル=('participation_count', 'mean'),
    平均チームキル=('team_kills_mid', 'mean'),
).sort_values('平均KP', ascending=False)
member_kp['平均KP'] = (member_kp['平均KP'] * 100).round(1).astype(str) + '%'
member_kp['中央値KP'] = (member_kp['中央値KP'] * 100).round(1).astype(str) + '%'
member_kp['平均関与キル'] = member_kp['平均関与キル'].round(1)
member_kp['平均チームキル'] = member_kp['平均チームキル'].round(1)
print(member_kp.to_string())
print()

# ---------------------------------------------------------------------------
# 【2】ロール別 中盤KP
# ---------------------------------------------------------------------------
print('=' * 65)
print('【2】ロール別  中盤キル関与率')
print('=' * 65)
role_kp = df.groupby('role').agg(
    試合数=('matchId', 'count'),
    平均KP=('kp_mid', 'mean'),
).sort_values('平均KP', ascending=False)
role_kp['平均KP'] = (role_kp['平均KP'] * 100).round(1).astype(str) + '%'
print(role_kp.to_string())
print()

# ---------------------------------------------------------------------------
# 【3】集団戦(2キル以上の連続キル)への関与率
# ---------------------------------------------------------------------------
print('=' * 65)
print('【3】メンバー別  集団戦への関与率 (30秒以内に2キル以上)')
print('=' * 65)
tf_df = df.dropna(subset=['tf_kp'])
if len(tf_df) > 0:
    tf_kp_table = tf_df.groupby('summonerName').agg(
        試合数=('matchId', 'nunique'),
        平均集団戦KP=('tf_kp', 'mean'),
        平均集団戦関与=('tf_involved', 'mean'),
    ).sort_values('平均集団戦KP', ascending=False)
    tf_kp_table['平均集団戦KP'] = (tf_kp_table['平均集団戦KP'] * 100).round(1).astype(str) + '%'
    tf_kp_table['平均集団戦関与'] = tf_kp_table['平均集団戦関与'].round(1)
    print(tf_kp_table.to_string())
else:
    print('集団戦データなし')
print()

# ---------------------------------------------------------------------------
# 【4】中盤KP と勝率の関係
# ---------------------------------------------------------------------------
print('=' * 65)
print('【4】中盤KP と勝率の関係')
print('=' * 65)
df['kp_bucket'] = pd.cut(
    df['kp_mid'],
    bins=[0, 0.3, 0.5, 0.7, 1.01],
    labels=['0-30%', '30-50%', '50-70%', '70-100%'],
    include_lowest=True,
)
bucket_wr = df.groupby('kp_bucket', observed=True).agg(
    試合数=('win', 'count'),
    勝利数=('win', 'sum'),
).reset_index()
bucket_wr['勝率'] = (bucket_wr['勝利数'] / bucket_wr['試合数'] * 100).round(1).astype(str) + '%'
bucket_wr.columns = ['中盤KP帯', '試合数', '勝利数', '勝率']
print(bucket_wr.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 【5】メンバー別: 中盤KP高い時 vs 低い時の勝率差
# ---------------------------------------------------------------------------
print('=' * 65)
print('【5】メンバー別  中盤KP高い時 vs 低い時の勝率')
print('=' * 65)
results = []
for name in sorted(df['summonerName'].unique()):
    mdf = df[df['summonerName'] == name]
    if len(mdf) < 5:
        continue
    median_kp = mdf['kp_mid'].median()
    high = mdf[mdf['kp_mid'] >= median_kp]
    low = mdf[mdf['kp_mid'] < median_kp]
    high_wr = high['win'].mean() * 100 if len(high) > 0 else 0
    low_wr = low['win'].mean() * 100 if len(low) > 0 else 0
    diff = high_wr - low_wr
    results.append({
        'メンバー': name,
        '高KP勝率': f'{high_wr:.1f}% ({len(high)}試合)',
        '低KP勝率': f'{low_wr:.1f}% ({len(low)}試合)',
        '勝率差': f'{diff:+.1f}%',
    })
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 【6】メンバー×ロール別: 中盤の寄り傾向
# ---------------------------------------------------------------------------
print('=' * 65)
print('【6】メンバー×ロール別  中盤キル関与率')
print('=' * 65)
mr_kp = df.groupby(['summonerName', 'role']).agg(
    試合数=('matchId', 'nunique'),
    平均KP=('kp_mid', 'mean'),
).reset_index()
mr_kp = mr_kp[mr_kp['試合数'] >= 3].sort_values(['summonerName', '平均KP'], ascending=[True, False])
mr_kp['平均KP'] = (mr_kp['平均KP'] * 100).round(1).astype(str) + '%'
mr_kp.columns = ['メンバー', 'ロール', '試合数', '平均KP']
print(mr_kp.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 【7】位置データからの寄り分析 (中盤のマップ中央への移動傾向)
# ---------------------------------------------------------------------------
print('=' * 65)
print('【7】中盤のマップポジション分析 (中央寄り度)')
print('=' * 65)
mid_frames = frames[
    (frames['timestampMin'] >= MID_START) & (frames['timestampMin'] <= MID_END)
    & (frames['summonerName'].isin(MEMBERS))
].copy()

MAP_CENTER_X = 7500
MAP_CENTER_Y = 7500

mid_frames['dist_from_center'] = np.sqrt(
    (mid_frames['positionX'] - MAP_CENTER_X) ** 2
    + (mid_frames['positionY'] - MAP_CENTER_Y) ** 2
)

pos_analysis = mid_frames.groupby(['summonerName', 'role']).agg(
    データ数=('dist_from_center', 'count'),
    平均中央距離=('dist_from_center', 'mean'),
).reset_index()
pos_analysis = pos_analysis[pos_analysis['データ数'] >= 10]
pos_analysis = pos_analysis.sort_values('平均中央距離', ascending=True)
pos_analysis['平均中央距離'] = pos_analysis['平均中央距離'].round(0).astype(int)
pos_analysis.columns = ['メンバー', 'ロール', 'フレーム数', '平均中央距離']
print(pos_analysis.to_string(index=False))
print()
print('※ 中央距離が小さいほどマップ中央に寄る傾向があり、')
print('  集団戦が起きやすいエリアにポジショニングしていることを示します。')
print()

# ---------------------------------------------------------------------------
# 【8】まとめ: 寄りの意識が最も勝敗に影響するメンバー
# ---------------------------------------------------------------------------
print('=' * 65)
print('【8】総合まとめ')
print('=' * 65)

corr_results = []
for name in sorted(df['summonerName'].unique()):
    mdf = df[df['summonerName'] == name]
    if len(mdf) < 10:
        continue
    corr = mdf['kp_mid'].corr(mdf['win'].astype(float))
    avg_kp = mdf['kp_mid'].mean() * 100
    corr_results.append({
        'メンバー': name,
        '平均中盤KP': f'{avg_kp:.1f}%',
        'KP-勝率相関': round(corr, 3),
        '寄り意識評価': '◎' if corr > 0.15 else ('○' if corr > 0.05 else ('△' if corr > -0.05 else '×')),
    })

corr_df = pd.DataFrame(corr_results).sort_values('KP-勝率相関', ascending=False)
print(corr_df.to_string(index=False))
print()
print('評価基準: ◎ 寄りが勝敗に強く直結 | ○ やや直結 | △ 影響薄い | × 寄りすぎが裏目')

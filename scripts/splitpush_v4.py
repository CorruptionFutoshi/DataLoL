"""
ソロスプリットプッシュ分析 v4
- フレームベースの「スプリットプッシュ滞在エピソード」検出
- 敵生存チェック（直前60秒で敵3人以上死亡 → フリープッシュとして除外）
- エピソード結果: タワー破壊 / ソロキル / 死亡 / 安全撤退
"""
import pandas as pd
import numpy as np
import yaml
import sys
import ast

sys.stdout.reconfigure(encoding='utf-8')

with open('config/settings.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in cfg['members']]

print("データ読み込み中...")
frames = pd.read_csv('data/processed/timeline_frames.csv')
events = pd.read_csv('data/processed/timeline_events.csv')
stats = pd.read_csv('data/processed/player_stats.csv')

# ============================================================
# パラメータ
# ============================================================
ALLY_RADIUS = 3000
DEPTH_THRESHOLD = 0.60
MIN_TIME = 25
ENEMY_DEATH_WINDOW = 1.0   # 直前60秒
MAX_RECENT_ENEMY_DEATHS = 2  # これ以下なら敵3人以上生存
TEAMFIGHT_TIME = 0.25       # 15秒
TEAMFIGHT_DIST = 5000
EPISODE_GAP = 1.5           # 1.5分以上離れたら別エピソード

# ============================================================
# 共通関数
# ============================================================
def classify_zone(x, y):
    in_mid = (abs(x - y) < 2800) and (3500 < x < 11500) and (3500 < y < 11500)
    if in_mid:
        return 'MID'
    in_top = (x < 6000 and y > 5500) or (y > 10000 and x < 10000)
    in_bot = (y < 6000 and x > 5500) or (x > 10000 and y < 10000)
    if in_top and in_bot:
        return 'OTHER'
    if in_top:
        return 'TOP'
    if in_bot:
        return 'BOT'
    return 'OTHER'


def get_depth(x, y, team_id):
    progress = (x + y) / (14500 * 2)
    if team_id == 200:
        progress = 1 - progress
    return progress


# ============================================================
# 前処理: インデックス構築
# ============================================================
team_lookup = frames[frames['summonerName'].isin(MEMBERS)].groupby(
    ['matchId', 'summonerName'])['teamId'].first().to_dict()

# 敵キル時刻のインデックス (高速lookup用)
all_kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()
enemy_death_idx = {}
for (mid, vtid), grp in all_kills.groupby(['matchId', 'victimTeamId']):
    enemy_death_idx[(mid, int(vtid))] = np.sort(grp['timestampMin'].values)


def count_recent_enemy_deaths(match_id, team_id, timestamp):
    enemy_team = 200 if team_id == 100 else 100
    key = (match_id, enemy_team)
    if key not in enemy_death_idx:
        return 0
    times = enemy_death_idx[key]
    return int(np.sum((times >= timestamp - ENEMY_DEATH_WINDOW) & (times <= timestamp)))


# フレームグループ (ソロ判定用)
frame_groups = frames[frames['timestampMin'] >= MIN_TIME].groupby(['matchId', 'timestampMin'])

# ============================================================
# Part 1: メンバーフレームにスプリットプッシュ条件を付与
# ============================================================
print("フレーム分析中...")

mf = frames[(frames['summonerName'].isin(MEMBERS)) & (frames['timestampMin'] >= MIN_TIME)].copy()
mf['zone'] = mf.apply(lambda r: classify_zone(r['positionX'], r['positionY']), axis=1)
mf['depth'] = mf.apply(lambda r: get_depth(r['positionX'], r['positionY'], r['teamId']), axis=1)
mf['is_side'] = mf['zone'].isin(['TOP', 'BOT'])
mf['is_deep'] = mf['depth'] >= DEPTH_THRESHOLD

# ソロ判定 (味方3000以内に0人)
def check_solo_frame(row):
    key = (row['matchId'], row['timestampMin'])
    if key not in frame_groups.groups:
        return False
    fr = frames.loc[frame_groups.groups[key]]
    allies = fr[(fr['teamId'] == row['teamId']) & (fr['summonerName'] != row['summonerName'])]
    if len(allies) == 0:
        return True
    dist = np.sqrt((allies['positionX'] - row['positionX'])**2 +
                   (allies['positionY'] - row['positionY'])**2)
    return (dist < ALLY_RADIUS).sum() == 0

mf['is_solo'] = mf.apply(check_solo_frame, axis=1)

# 敵生存チェック (直前60秒で2人以下しか死んでいない = 3人以上生存)
mf['recent_enemy_deaths'] = mf.apply(
    lambda r: count_recent_enemy_deaths(r['matchId'], r['teamId'], r['timestampMin']), axis=1)
mf['enemies_alive'] = mf['recent_enemy_deaths'] <= MAX_RECENT_ENEMY_DEATHS

# スプリットプッシュフレーム = 全条件を満たす
mf['is_splitpush'] = mf['is_side'] & mf['is_deep'] & mf['is_solo'] & mf['enemies_alive']

# 敵生存チェックなしのフレームも記録 (比較用)
mf['is_splitpush_no_alive_check'] = mf['is_side'] & mf['is_deep'] & mf['is_solo']

sp_frames = mf[mf['is_splitpush']].copy()
sp_frames_no_check = mf[mf['is_splitpush_no_alive_check']].copy()

print(f"  メンバーフレーム: {len(mf)}")
print(f"  スプリットプッシュフレーム (敵生存チェックなし): {len(sp_frames_no_check)}")
print(f"  スプリットプッシュフレーム (敵生存チェックあり): {len(sp_frames)}")
print(f"  → 敵生存チェックで除外: {len(sp_frames_no_check) - len(sp_frames)}フレーム")

# ============================================================
# Part 2: エピソード検出
# ============================================================
print("\nエピソード検出中...")

episodes = []
for (mid, name), grp in sp_frames.groupby(['matchId', 'summonerName']):
    grp = grp.sort_values('timestampMin')
    times = grp['timestampMin'].values
    zones = grp['zone'].values
    depths = grp['depth'].values
    xs = grp['positionX'].values
    ys = grp['positionY'].values

    ep_start = 0
    for i in range(1, len(times)):
        # 別エピソードの条件: 時間が離れている or ゾーンが変わった
        if times[i] - times[i-1] > EPISODE_GAP or zones[i] != zones[ep_start]:
            episodes.append({
                'matchId': mid, 'summonerName': name,
                'zone': zones[ep_start],
                'start_time': times[ep_start], 'end_time': times[i-1],
                'n_frames': i - ep_start,
                'avg_depth': np.mean(depths[ep_start:i]),
                'max_depth': np.max(depths[ep_start:i]),
            })
            ep_start = i
    episodes.append({
        'matchId': mid, 'summonerName': name,
        'zone': zones[ep_start],
        'start_time': times[ep_start], 'end_time': times[-1],
        'n_frames': len(times) - ep_start,
        'avg_depth': np.mean(depths[ep_start:]),
        'max_depth': np.max(depths[ep_start:]),
    })

ep_df = pd.DataFrame(episodes)
print(f"  検出エピソード数: {len(ep_df)}")

# ============================================================
# Part 3: エピソード結果判定
# ============================================================
print("エピソード結果判定中...")

def parse_assists(val):
    if pd.isna(val):
        return []
    try:
        r = ast.literal_eval(str(val))
        return r if isinstance(r, list) else []
    except:
        return []

kills_ev = events[events['eventType'] == 'CHAMPION_KILL'].copy()
kills_ev['num_assists'] = kills_ev['assistingParticipantIds'].apply(parse_assists).apply(len)

tower_ev = events[
    (events['eventType'] == 'BUILDING_KILL') &
    (events['buildingType'] == 'TOWER_BUILDING')
].copy()

champ_lookup = stats.set_index(['matchId', 'summonerName'])['championName'].to_dict()
role_lookup = stats.set_index(['matchId', 'summonerName'])['role'].to_dict()
win_lookup = stats.set_index(['matchId', 'summonerName'])['win'].to_dict()


def determine_outcome(ep):
    mid = ep['matchId']
    name = ep['summonerName']
    t_start = ep['start_time'] - 0.5
    t_end = ep['end_time'] + 0.5

    # デス
    deaths = kills_ev[
        (kills_ev['matchId'] == mid) &
        (kills_ev['victimName'] == name) &
        (kills_ev['timestampMin'] >= t_start) &
        (kills_ev['timestampMin'] <= t_end)
    ]

    # キル (アシスト0のソロキルのみ)
    player_kills = kills_ev[
        (kills_ev['matchId'] == mid) &
        (kills_ev['killerName'] == name) &
        (kills_ev['timestampMin'] >= t_start) &
        (kills_ev['timestampMin'] <= t_end) &
        (kills_ev['num_assists'] == 0)
    ]

    # さらに集団戦フィルタ: ±15秒以内に近くで他キルがあれば除外
    true_solo_kills = 0
    for _, k in player_kills.iterrows():
        other = kills_ev[
            (kills_ev['matchId'] == mid) &
            (kills_ev['eventType'] == 'CHAMPION_KILL') &
            (kills_ev['timestampMin'] >= k['timestampMin'] - TEAMFIGHT_TIME) &
            (kills_ev['timestampMin'] <= k['timestampMin'] + TEAMFIGHT_TIME) &
            (kills_ev.index != k.name)
        ]
        if len(other) > 0:
            dist = np.sqrt((other['positionX'] - k['positionX'])**2 +
                           (other['positionY'] - k['positionY'])**2)
            if (dist < TEAMFIGHT_DIST).any():
                continue
        true_solo_kills += 1

    # タワー破壊
    towers = tower_ev[
        (tower_ev['matchId'] == mid) &
        (tower_ev['killerName'] == name) &
        (tower_ev['timestampMin'] >= t_start) &
        (tower_ev['timestampMin'] <= t_end)
    ]

    # デスの分類
    died = len(deaths) > 0
    death_type = None
    if died:
        d = deaths.iloc[0]
        na = d['num_assists']
        if na == 0:
            death_type = '1v1'
        elif na == 1:
            death_type = '1v2'
        else:
            death_type = '1v3+'

    # 結果文字列
    parts = []
    if len(towers) > 0:
        parts.append(f"タワー{len(towers)}")
    if true_solo_kills > 0:
        parts.append(f"ソロキル{true_solo_kills}")
    if died:
        parts.append(f"死亡({death_type})")
    if not parts:
        parts.append("安全撤退")

    return {
        'died': died, 'death_type': death_type,
        'n_solo_kills': true_solo_kills, 'n_towers': len(towers),
        'safe_exit': not died and len(towers) == 0 and true_solo_kills == 0,
        'outcome': ' + '.join(parts),
    }


results = [determine_outcome(row) for _, row in ep_df.iterrows()]
for col in ['died', 'death_type', 'n_solo_kills', 'n_towers', 'safe_exit', 'outcome']:
    ep_df[col] = [r[col] for r in results]

ep_df['champ'] = ep_df.apply(lambda r: champ_lookup.get((r['matchId'], r['summonerName']), '?'), axis=1)
ep_df['role'] = ep_df.apply(lambda r: role_lookup.get((r['matchId'], r['summonerName']), '?'), axis=1)
ep_df['win'] = ep_df.apply(lambda r: win_lookup.get((r['matchId'], r['summonerName']), False), axis=1)

# ============================================================
# Part 4: 出力
# ============================================================
matches_per_member = frames[frames['summonerName'].isin(MEMBERS)].groupby('summonerName')['matchId'].nunique().to_dict()

print()
print("=" * 78)
print("  ソロスプリットプッシュ分析 v4")
print(f"  条件: {MIN_TIME}分以降 / サイドレーン / depth≥{DEPTH_THRESHOLD} / 味方不在")
print(f"  敵生存: 直前{ENEMY_DEATH_WINDOW:.0f}分で敵デス{MAX_RECENT_ENEMY_DEATHS}人以下(3人以上生存)")
print("=" * 78)

# --- セクション1: スプリットプッシュ行動頻度 ---
print("\n【スプリットプッシュ行動頻度】")
print("  エピソード = サイドレーン敵陣で1人でプッシュし続けた連続期間")
print("-" * 78)

freq_rows = []
for name in MEMBERS:
    ep = ep_df[ep_df['summonerName'] == name]
    n_matches = matches_per_member.get(name, 1)
    total_frames = len(ep['n_frames'])
    total_time = ep['n_frames'].sum()
    avg_dur = ep['n_frames'].mean() if len(ep) > 0 else 0
    freq_rows.append({
        'name': name, 'matches': n_matches,
        'episodes': len(ep), 'total_min': total_time,
        'avg_dur': avg_dur,
        'per_match': len(ep) / n_matches,
    })

freq = pd.DataFrame(freq_rows).sort_values('episodes', ascending=False)
print(f"  {'':>14} {'試合':>4} {'エピソード':>8} {'1試合平均':>7} {'総滞在':>5} {'平均滞在':>6}")
print("-" * 78)
for _, r in freq.iterrows():
    print(f"  {r['name']:>14} {int(r['matches']):>4} {int(r['episodes']):>8}回"
          f"  {r['per_match']:>6.2f}回  {int(r['total_min']):>4}分  {r['avg_dur']:>5.1f}分")

# --- セクション2: エピソード結果内訳 ---
print()
print("【エピソード結果内訳】")
print("  各エピソードで何が起きたか")
print("-" * 78)

outcome_rows = []
for name in MEMBERS:
    ep = ep_df[ep_df['summonerName'] == name]
    if len(ep) == 0:
        continue
    n_tower = len(ep[ep['n_towers'] > 0])
    n_kill = len(ep[ep['n_solo_kills'] > 0])
    n_death = len(ep[ep['died']])
    n_safe = len(ep[ep['safe_exit']])
    n_total = len(ep)
    n_positive = n_tower + n_kill
    n_death_only = len(ep[ep['died'] & (ep['n_towers'] == 0) & (ep['n_solo_kills'] == 0)])
    outcome_rows.append({
        'name': name, 'total': n_total,
        'tower': n_tower, 'kill': n_kill,
        'death': n_death, 'death_only': n_death_only,
        'safe': n_safe,
    })

odf = pd.DataFrame(outcome_rows).sort_values('total', ascending=False)
print(f"  {'':>14} {'計':>3} {'タワー破壊':>8} {'ソロキル':>6} {'死亡':>4} {'安全撤退':>6}")
print("-" * 78)
for _, r in odf.iterrows():
    print(f"  {r['name']:>14} {int(r['total']):>3}"
          f"  {int(r['tower']):>5}回"
          f"  {int(r['kill']):>5}回"
          f"  {int(r['death']):>4}回"
          f"  {int(r['safe']):>5}回")

# --- セクション3: 滞在時間分布 ---
print()
print("【滞在時間分布】")
print("  1フレーム=約1分のスナップショット")
print("-" * 78)

dur_labels = ['1分(短)', '2分', '3分', '4分+']
for name in MEMBERS:
    ep = ep_df[ep_df['summonerName'] == name]
    if len(ep) == 0:
        continue
    d1 = len(ep[ep['n_frames'] == 1])
    d2 = len(ep[ep['n_frames'] == 2])
    d3 = len(ep[ep['n_frames'] == 3])
    d4 = len(ep[ep['n_frames'] >= 4])
    total = len(ep)
    print(f"  {name:>14}: ", end='')
    for label, cnt in zip(dur_labels, [d1, d2, d3, d4]):
        pct = cnt / total * 100 if total > 0 else 0
        bar = '■' * max(1, int(pct / 5)) if cnt > 0 else ''
        print(f"{label}{cnt}({pct:.0f}%) ", end='')
    print()

# --- セクション4: 滞在時間と結果の関係 ---
print()
print("【滞在時間 × 結果】")
print("  長くいるほど何が起きやすいか")
print("-" * 78)

for dur_name, dur_filter in [("1分(短期)", lambda e: e['n_frames'] == 1),
                              ("2分以上(持続)", lambda e: e['n_frames'] >= 2)]:
    subset = ep_df[dur_filter(ep_df)]
    if len(subset) == 0:
        continue
    n = len(subset)
    tw = len(subset[subset['n_towers'] > 0])
    sk = len(subset[subset['n_solo_kills'] > 0])
    dt = len(subset[subset['died']])
    sf = len(subset[subset['safe_exit']])
    print(f"  {dur_name:>14}: {n}回 → "
          f"タワー{tw}({tw/n*100:.0f}%) "
          f"キル{sk}({sk/n*100:.0f}%) "
          f"死亡{dt}({dt/n*100:.0f}%) "
          f"撤退{sf}({sf/n*100:.0f}%)")

# --- セクション5: 死亡パターン (エピソード中の死亡のみ) ---
print()
print("【死亡パターン (スプリットプッシュ中の死亡)】")
print("-" * 78)

death_eps = ep_df[ep_df['died']].copy()
for name in MEMBERS:
    de = death_eps[death_eps['summonerName'] == name]
    if len(de) == 0:
        continue
    d1v1 = len(de[de['death_type'] == '1v1'])
    d1v2 = len(de[de['death_type'] == '1v2'])
    d1v3 = len(de[de['death_type'] == '1v3+'])
    total = len(de)
    print(f"  {name:>14}: 計{total}回  1v1:{d1v1}  1v2:{d1v2}  1v3+:{d1v3}  ", end='')
    if total > 0:
        bl = 30
        b1 = int(round(d1v1 / total * bl))
        b2 = int(round(d1v2 / total * bl))
        b3 = bl - b1 - b2
        print(f"[{'█' * b1}{'▓' * b2}{'░' * b3}]", end='')
    print()

# --- セクション6: トレード分析 (死亡後にオブジェクト獲得) ---
print()
print("【死亡後トレード】")
print("  スプリットプッシュ中に死亡 → 90秒以内に味方がオブジェクト獲得")
print("-" * 78)

obj_ev = events[
    (events['eventType'].isin(['BUILDING_KILL', 'ELITE_MONSTER_KILL'])) &
    (events['timestampMin'] >= MIN_TIME)
].copy()
obj_ev['obj_zone'] = obj_ev.apply(lambda r: classify_zone(r['positionX'], r['positionY']), axis=1)


def check_trade(ep_row):
    if not ep_row['died']:
        return None, None
    mid = ep_row['matchId']
    name = ep_row['summonerName']
    dz = ep_row['zone']
    tid = team_lookup.get((mid, name))
    if tid is None:
        return None, None
    enemy_team = 200 if tid == 100 else 100
    opp = {'TOP': 'BOT', 'BOT': 'TOP'}.get(dz)
    if opp is None:
        return None, None

    death_time = ep_row['end_time'] + 0.5
    death_ev = kills_ev[
        (kills_ev['matchId'] == mid) &
        (kills_ev['victimName'] == name) &
        (kills_ev['timestampMin'] >= ep_row['start_time'] - 0.5) &
        (kills_ev['timestampMin'] <= death_time)
    ]
    if len(death_ev) == 0:
        return None, None
    dt = death_ev.iloc[-1]['timestampMin']

    window = obj_ev[(obj_ev['matchId'] == mid) &
                    (obj_ev['timestampMin'] > dt) &
                    (obj_ev['timestampMin'] <= dt + 1.5)]
    if len(window) == 0:
        return None, None

    team_buildings = window[
        (window['eventType'] == 'BUILDING_KILL') & (window['teamId'] == enemy_team)]
    team_monsters = window[
        (window['eventType'] == 'ELITE_MONSTER_KILL') & (window['killerTeamId'] == tid)]
    team_obj = pd.concat([team_buildings, team_monsters]).drop_duplicates()
    if len(team_obj) == 0:
        return None, None

    opp_obj = team_obj[team_obj['obj_zone'] == opp]
    if len(opp_obj) > 0:
        return '逆サイド', len(team_obj)
    return '同サイド/他', len(team_obj)


trade_results = [check_trade(row) for _, row in ep_df.iterrows()]
ep_df['trade_side'] = [t[0] for t in trade_results]
ep_df['trade_n'] = [t[1] for t in trade_results]

for name in MEMBERS:
    de = ep_df[(ep_df['summonerName'] == name) & (ep_df['died'])]
    if len(de) == 0:
        continue
    traded = de[de['trade_side'].notna()]
    opp_t = len(de[de['trade_side'] == '逆サイド'])
    same_t = len(de[de['trade_side'] == '同サイド/他'])
    rate = len(traded) / len(de) * 100 if len(de) > 0 else 0
    print(f"  {name:>14}: {len(traded)}/{len(de)}回でトレード成立 ({rate:.0f}%)  "
          f"逆サイド{opp_t} / 同サイド{same_t}")

# --- セクション7: タワー破壊 ---
print()
print("【ソロタワー破壊 (スプリットプッシュ中)】")
print("-" * 78)

tower_eps = ep_df[ep_df['n_towers'] > 0].copy()
for name in MEMBERS:
    te = tower_eps[tower_eps['summonerName'] == name]
    if len(te) == 0:
        continue
    total_tw = te['n_towers'].sum()
    n_matches = matches_per_member.get(name, 1)
    print(f"  {name:>14}: {int(total_tw)}本 (1試合あたり {total_tw/n_matches:.2f}本)  "
          f"エピソード{len(te)}回中")

# --- セクション8: 1v1収支 ---
print()
print("【1v1 ソロキル収支 (スプリットプッシュ中)】")
print("-" * 78)

for name in MEMBERS:
    ep = ep_df[ep_df['summonerName'] == name]
    kills_total = ep['n_solo_kills'].sum()
    deaths_1v1 = len(ep[ep['death_type'] == '1v1'])
    if kills_total == 0 and deaths_1v1 == 0:
        continue
    net = kills_total - deaths_1v1
    pfx = "+" if net > 0 else ""
    wr = kills_total / (kills_total + deaths_1v1) * 100 if (kills_total + deaths_1v1) > 0 else 0
    print(f"  {name:>14}: 勝ち{int(kills_total)}  負け{int(deaths_1v1)}  差分{pfx}{int(net)}  1v1勝率{wr:.0f}%")

# --- セクション9: 総合ランキング ---
print()
print("=" * 78)
print("【スプリットプッシュ総合ランキング】")
print("=" * 78)

summary_rows = []
for name in MEMBERS:
    ep = ep_df[ep_df['summonerName'] == name]
    n_matches = matches_per_member.get(name, 1)
    n_ep = len(ep)
    total_frames = ep['n_frames'].sum()
    n_towers = ep['n_towers'].sum()
    n_kills = ep['n_solo_kills'].sum()
    n_deaths = len(ep[ep['died']])
    n_safe = len(ep[ep['safe_exit']])
    n_death_only = len(ep[ep['died'] & (ep['n_towers'] == 0) & (ep['n_solo_kills'] == 0)])

    positive = n_towers + n_kills + len(ep[ep['trade_side'].notna()])
    total_events = positive + n_death_only
    success_rate = positive / total_events * 100 if total_events > 0 else 0

    summary_rows.append({
        'name': name, 'matches': n_matches,
        'episodes': n_ep, 'ep_per_match': n_ep / n_matches,
        'total_min': total_frames,
        'towers': n_towers, 'kills': n_kills,
        'deaths': n_deaths, 'death_only': n_death_only,
        'safe': n_safe, 'positive': positive,
        'success_rate': success_rate,
    })

sdf = pd.DataFrame(summary_rows).sort_values('ep_per_match', ascending=False)

print(f"\n  {'':>14} {'試合':>4} {'SP回':>4} {'/試合':>5} {'滞在':>4}"
      f" {'タワー':>5} {'ソロキル':>6} {'死亡':>4} {'撤退':>4} {'成功率':>5}")
print("-" * 78)
for _, r in sdf.iterrows():
    print(f"  {r['name']:>14} {int(r['matches']):>4} {int(r['episodes']):>4} {r['ep_per_match']:>5.2f}"
          f" {int(r['total_min']):>3}分"
          f" {int(r['towers']):>5} {int(r['kills']):>6} {int(r['deaths']):>4} {int(r['safe']):>4}"
          f" {r['success_rate']:>5.1f}%")

# --- セクション10: 個別評価コメント ---
print()
for _, r in sdf.iterrows():
    name = r['name']
    if r['episodes'] == 0:
        continue

    ep = ep_df[ep_df['summonerName'] == name]
    avg_depth = ep['avg_depth'].mean()
    long_eps = len(ep[ep['n_frames'] >= 3])

    print(f"  ▼ {name} ({r['ep_per_match']:.2f}回/試合 | 平均depth {avg_depth:.2f})")

    notes = []
    if r['ep_per_match'] >= 0.8:
        notes.append(f"✓ 積極的にスプリットプッシュを実行 ({r['ep_per_match']:.1f}回/試合)")
    elif r['ep_per_match'] >= 0.3:
        notes.append(f"  適度にスプリットプッシュを活用 ({r['ep_per_match']:.1f}回/試合)")
    else:
        notes.append(f"  スプリットプッシュ頻度は低め ({r['ep_per_match']:.1f}回/試合)")

    if long_eps > 0:
        notes.append(f"  長期滞在(3分+): {long_eps}回")

    if r['towers'] > 0:
        notes.append(f"✓ タワー破壊{int(r['towers'])}本")
    if r['kills'] > 0:
        notes.append(f"✓ ソロキル{int(r['kills'])}回")

    death_eps = ep[ep['died']]
    if len(death_eps) > 0:
        d1v3 = len(death_eps[death_eps['death_type'] == '1v3+'])
        if d1v3 / len(death_eps) > 0.5:
            notes.append(f"⚠ 死亡{len(death_eps)}回中{d1v3}回が1v3+: 引き際/ワードの改善余地")
        elif r['death_only'] > r['positive']:
            notes.append(f"⚠ 成果なし死亡({int(r['death_only'])})が成果({int(r['positive'])})より多い")

    if r['safe'] > 0 and r['episodes'] > 0:
        safe_pct = r['safe'] / r['episodes'] * 100
        if safe_pct >= 60:
            notes.append(f"  安全撤退率{safe_pct:.0f}%: 慎重なプッシュスタイル")

    for n in notes:
        print(f"    {n}")
    print()

print("=" * 78)
print("  ※ 「敵生存チェック」により集団戦勝利後のフリープッシュは除外されています")
print("  ※ 敵が3人以上生きている状態でのサイドプッシュのみが対象です")
print("=" * 78)

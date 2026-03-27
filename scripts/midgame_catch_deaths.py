"""中盤以降のキャッチデス分析

14分以降に「集団戦ではなく孤立した状態で倒されたデス」を特定し、
メンバー別にランキング表示する。

キャッチ判定:
  - 14分以降のCHAMPION_KILL
  - 前後15秒以内の総キル数が2以下（集団戦ではない）
  - 味方メンバーが犠牲者
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MID_GAME_START_MIN = 14.0
CLUSTER_WINDOW_SEC = 15
MAX_NEARBY_KILLS = 2


def load_data():
    with open(os.path.join(PROJECT_ROOT, 'config', 'settings.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members = [m['game_name'] for m in cfg['members']]

    events = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'timeline_events.csv'))
    player_stats = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'player_stats.csv'))
    matches = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'matches.csv'))

    return members, events, player_stats, matches


def classify_map_zone(x, y):
    if x < 4000 and y < 4000:
        return "ブルー側ベース"
    if x > 11000 and y > 11000:
        return "レッド側ベース"
    if abs(x - y) < 2500 and 3000 < x < 12000:
        return "ミッド付近"
    mid = 7500
    if x < mid and y > mid:
        return "トップサイド"
    if x > mid and y < mid:
        return "ボットサイド"
    if x < mid and y < mid:
        return "ブルー側JG"
    return "レッド側JG"


def find_catch_deaths(members, events, player_stats, matches):
    member_set = set(members)
    kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()
    kills = kills[kills['timestampMin'] >= MID_GAME_START_MIN]

    our_team_info = {}
    for _, row in player_stats.iterrows():
        if row['summonerName'] in member_set:
            mid = row['matchId']
            if mid not in our_team_info:
                our_team_info[mid] = {'teamId': row['teamId'], 'win': row['win']}

    member_champs = {}
    member_roles = {}
    for _, row in player_stats.iterrows():
        if row['summonerName'] in member_set:
            key = (row['matchId'], row['summonerName'])
            member_champs[key] = row['championName']
            member_roles[key] = row.get('role', '')

    match_durations = dict(zip(matches['matchId'], matches.get('gameDurationMin', pd.Series(dtype=float))))

    window_min = CLUSTER_WINDOW_SEC / 60.0
    catches = []

    for match_id, match_kills in kills.groupby('matchId'):
        if match_id not in our_team_info:
            continue

        our_team_id = our_team_info[match_id]['teamId']
        win = our_team_info[match_id]['win']
        match_kills_sorted = match_kills.sort_values('timestampMin')
        timestamps = match_kills_sorted['timestampMin'].values

        for idx, (_, kill) in enumerate(match_kills_sorted.iterrows()):
            if kill['victimName'] not in member_set:
                continue

            t = kill['timestampMin']
            nearby_mask = np.abs(timestamps - t) <= window_min
            nearby_count = int(nearby_mask.sum())

            if nearby_count > MAX_NEARBY_KILLS:
                continue

            victim = kill['victimName']
            zone = classify_map_zone(kill['positionX'], kill['positionY'])
            champ = member_champs.get((match_id, victim), '?')
            role = member_roles.get((match_id, victim), '?')

            catches.append({
                'matchId': match_id,
                'timestampMin': round(t, 1),
                'victim': victim,
                'champion': champ,
                'role': role,
                'killedBy': kill['killerName'],
                'zone': zone,
                'win': win,
                'gameDurationMin': match_durations.get(match_id, 0),
            })

    return pd.DataFrame(catches)


def main():
    members, events, player_stats, matches = load_data()
    df = find_catch_deaths(members, events, player_stats, matches)

    total_matches = player_stats[player_stats['summonerName'].isin(set(members))]['matchId'].nunique()

    print("=" * 70)
    print("  中盤以降のキャッチデス ランキング")
    print(f"  （{MID_GAME_START_MIN:.0f}分以降、前後{CLUSTER_WINDOW_SEC}秒以内のキル{MAX_NEARBY_KILLS}以下 = 孤立デス）")
    print("=" * 70)
    print(f"\n対象試合数: {total_matches}  |  キャッチデス総数: {len(df)}")

    if df.empty:
        print("キャッチデスが見つかりませんでした。")
        return

    # ── 1. メンバー別キャッチデス回数ランキング ──
    print(f"\n{'─' * 70}")
    print("■ メンバー別キャッチデス回数ランキング")
    print(f"{'─' * 70}")

    member_games = player_stats[player_stats['summonerName'].isin(set(members))].groupby('summonerName')['matchId'].nunique()
    victim_counts = df['victim'].value_counts()

    ranking = []
    for name in victim_counts.index:
        count = victim_counts[name]
        games = member_games.get(name, 1)
        per_game = count / games
        ranking.append({'name': name, 'count': count, 'games': games, 'per_game': per_game})

    ranking_df = pd.DataFrame(ranking).sort_values('per_game', ascending=False)

    max_count = ranking_df['count'].max() if len(ranking_df) > 0 else 1
    for i, (_, row) in enumerate(ranking_df.iterrows()):
        bar_len = int(row['count'] / max_count * 20)
        bar = "█" * bar_len
        print(f"  {i+1}. {row['name']:<14} {row['count']:>3}回 "
              f"({row['games']}試合, 1試合あたり {row['per_game']:.2f}回)  {bar}")

    # ── 2. 勝敗別の影響 ──
    print(f"\n{'─' * 70}")
    print("■ キャッチデスの勝敗への影響")
    print(f"{'─' * 70}")

    for name in ranking_df['name']:
        subset = df[df['victim'] == name]
        win_catches = len(subset[subset['win'] == True])
        lose_catches = len(subset[subset['win'] == False])
        total = len(subset)
        lose_pct = lose_catches / total * 100 if total > 0 else 0
        print(f"  {name:<14}  勝ち試合: {win_catches:>3}回  負け試合: {lose_catches:>3}回  "
              f"(敗北時率 {lose_pct:.0f}%)")

    # ── 3. キャッチされる場所 ──
    print(f"\n{'─' * 70}")
    print("■ メンバー別キャッチされやすい場所")
    print(f"{'─' * 70}")

    for name in ranking_df['name']:
        subset = df[df['victim'] == name]
        zones = subset['zone'].value_counts()
        zone_str = ', '.join([f"{z}({n})" for z, n in zones.head(3).items()])
        print(f"  {name:<14}  {zone_str}")

    # ── 4. キャッチされやすい時間帯 ──
    print(f"\n{'─' * 70}")
    print("■ メンバー別キャッチされやすい時間帯")
    print(f"{'─' * 70}")

    for name in ranking_df['name']:
        subset = df[df['victim'] == name]
        avg_time = subset['timestampMin'].mean()
        early_mid = len(subset[(subset['timestampMin'] >= 14) & (subset['timestampMin'] < 20)])
        mid = len(subset[(subset['timestampMin'] >= 20) & (subset['timestampMin'] < 28)])
        late = len(subset[subset['timestampMin'] >= 28])
        print(f"  {name:<14}  平均 {avg_time:.1f}分  "
              f"(14-20分: {early_mid}回, 20-28分: {mid}回, 28分以降: {late}回)")

    # ── 5. チャンピオン別キャッチデス ──
    print(f"\n{'─' * 70}")
    print("■ メンバー別 キャッチされやすいチャンピオン TOP3")
    print(f"{'─' * 70}")

    for name in ranking_df['name']:
        subset = df[df['victim'] == name]
        champs = subset['champion'].value_counts()
        champ_str = ', '.join([f"{c}({n}回)" for c, n in champs.head(3).items()])
        print(f"  {name:<14}  {champ_str}")

    # ── 6. キャッチしてきた敵チャンピオン ──
    print(f"\n{'─' * 70}")
    print("■ キャッチしてきた敵（全体TOP10）")
    print(f"{'─' * 70}")

    enemy_counts = df['killedBy'].value_counts().head(10)
    for killer, count in enemy_counts.items():
        print(f"  {killer:<18}  {count}回")

    # ── 7. ロール別キャッチ率 ──
    print(f"\n{'─' * 70}")
    print("■ ロール別キャッチデス")
    print(f"{'─' * 70}")

    role_map = {'TOP': 'トップ', 'JUNGLE': 'ジャングル', 'MIDDLE': 'ミッド',
                'BOTTOM': 'ボット', 'UTILITY': 'サポート'}
    role_counts = df['role'].value_counts()
    for role, count in role_counts.items():
        label = role_map.get(role, role)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<12}  {count:>3}回 ({pct:>5.1f}%)  {bar}")

    # ── 考察 ──
    print(f"\n{'=' * 70}")
    print("💡 考察:")
    top = ranking_df.iloc[0]
    print(f"  • 1試合あたり最もキャッチされているのは {top['name']} "
          f"({top['per_game']:.2f}回/試合)")

    worst_zone = df['zone'].value_counts().index[0]
    print(f"  • チーム全体で最もキャッチされやすいエリアは「{worst_zone}」")

    lose_catches = df[df['win'] == False]
    if len(lose_catches) > 0:
        lose_pct = len(lose_catches) / len(df) * 100
        print(f"  • キャッチデスの {lose_pct:.0f}% は敗北試合で発生")

    late_deaths = df[df['timestampMin'] >= 28]
    if len(late_deaths) > 0:
        late_pct = len(late_deaths) / len(df) * 100
        print(f"  • 28分以降の終盤キャッチが全体の {late_pct:.0f}% を占める")

    print(f"  • キャッチデスが多いメンバーはサイドレーンの深追いや"
          f"視界のない場所での単独行動に注意")


if __name__ == '__main__':
    main()

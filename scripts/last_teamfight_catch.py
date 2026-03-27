"""敗北試合の最後の集団戦 - 誰がキャッチされたか分析

敗北したゲームにおける「ゲームを決定づけた最後の集団戦」を特定し、
味方チームで最初に死んだメンバー（＝キャッチされた人）を分析する。
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMFIGHT_WINDOW_SEC = 30
MIN_KILLS = 3


def load_data():
    with open(os.path.join(PROJECT_ROOT, 'config', 'settings.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members = [m['game_name'] for m in cfg['members']]

    events = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'timeline_events.csv'))
    player_stats = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'player_stats.csv'))
    matches = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'matches.csv'))

    return members, events, player_stats, matches


def detect_teamfights(match_kills, window_sec=TEAMFIGHT_WINDOW_SEC, min_kills=MIN_KILLS):
    if len(match_kills) == 0:
        return []

    kills = match_kills.sort_values('timestampMin').reset_index(drop=True)
    fights = []
    cluster = [kills.iloc[0]]

    for i in range(1, len(kills)):
        row = kills.iloc[i]
        if (row['timestampMin'] - cluster[-1]['timestampMin']) * 60 <= window_sec:
            cluster.append(row)
        else:
            if len(cluster) >= min_kills:
                fights.append(pd.DataFrame(cluster))
            cluster = [row]

    if len(cluster) >= min_kills:
        fights.append(pd.DataFrame(cluster))

    return fights


def classify_map_zone(x, y):
    mid = 7500
    if x < 4000 and y < 4000:
        return "ブルー側ベース"
    if x > 11000 and y > 11000:
        return "レッド側ベース"
    if abs(x - y) < 2500 and 3000 < x < 12000:
        return "ミッド付近"
    if x < mid and y > mid:
        return "トップサイド"
    if x > mid and y < mid:
        return "ボットサイド"
    if x < mid and y < mid:
        return "ブルー側JG"
    return "レッド側JG"


def main():
    members, events, player_stats, matches = load_data()

    kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()

    member_set = set(members)

    our_team_info = {}
    for _, row in player_stats.iterrows():
        if row['summonerName'] in member_set:
            mid = row['matchId']
            if mid not in our_team_info:
                our_team_info[mid] = {
                    'teamId': row['teamId'],
                    'win': row['win'],
                }

    lost_matches = {mid for mid, info in our_team_info.items() if not info['win']}
    print("=" * 70)
    print("  敗北試合の最後の集団戦 - 誰がキャッチされたか")
    print("=" * 70)
    print(f"\n敗北試合数: {len(lost_matches)}")

    match_game_dur = {}
    for _, row in matches.iterrows():
        match_game_dur[row['matchId']] = row.get('gameDurationMin', 0)

    results = []

    for match_id in lost_matches:
        match_kills = kills[kills['matchId'] == match_id]
        if match_kills.empty:
            continue

        our_team_id = our_team_info[match_id]['teamId']
        fights = detect_teamfights(match_kills)

        if not fights:
            continue

        last_fight = fights[-1]
        last_fight_sorted = last_fight.sort_values('timestampMin')

        our_deaths = last_fight_sorted[last_fight_sorted['victimName'].isin(member_set)]

        if our_deaths.empty:
            continue

        first_death = our_deaths.iloc[0]
        first_victim = first_death['victimName']
        first_killer = first_death['killerName']
        death_time = first_death['timestampMin']

        all_deaths_in_fight = last_fight_sorted[last_fight_sorted['victimTeamId'] == our_team_id]
        our_kill_count = len(last_fight_sorted[last_fight_sorted['killerTeamId'] == our_team_id])
        our_death_count = len(all_deaths_in_fight)

        zone = classify_map_zone(first_death['positionX'], first_death['positionY'])
        game_dur = match_game_dur.get(match_id, 0)

        member_champs = {}
        match_ps = player_stats[player_stats['matchId'] == match_id]
        for _, row in match_ps.iterrows():
            if row['summonerName'] in member_set:
                member_champs[row['summonerName']] = row['championName']

        results.append({
            'matchId': match_id,
            'game_duration': round(game_dur, 1),
            'fight_time': round(death_time, 1),
            'first_victim': first_victim,
            'victim_champion': member_champs.get(first_victim, '?'),
            'killed_by': first_killer,
            'death_zone': zone,
            'our_kills': our_kill_count,
            'our_deaths': our_death_count,
            'fight_total_kills': len(last_fight_sorted),
        })

    if not results:
        print("分析対象となる集団戦データがありませんでした。")
        return

    df = pd.DataFrame(results)

    # ── 1. 誰が一番キャッチされているか ──
    print(f"\n最後の集団戦が検出された敗北試合: {len(df)} 試合")

    print(f"\n{'─' * 70}")
    print("■ メンバー別「最後の集団戦で最初に死んだ回数」ランキング")
    print(f"{'─' * 70}")
    victim_counts = df['first_victim'].value_counts()
    for name, count in victim_counts.items():
        pct = count / len(df) * 100
        champs = df[df['first_victim'] == name]['victim_champion'].value_counts()
        champ_str = ', '.join([f"{c}({n}回)" for c, n in champs.head(3).items()])
        print(f"  {name}: {count}回 ({pct:.1f}%)  チャンプ: {champ_str}")

    # ── 2. キャッチされた場所 ──
    print(f"\n{'─' * 70}")
    print("■ キャッチされた場所の分布")
    print(f"{'─' * 70}")
    zone_counts = df['death_zone'].value_counts()
    for zone, count in zone_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {zone:<14}: {count:>3}回 ({pct:>5.1f}%) {bar}")

    # ── 3. キャッチ後の集団戦結果 ──
    print(f"\n{'─' * 70}")
    print("■ 最後の集団戦でのキル交換")
    print(f"{'─' * 70}")
    avg_our_kills = df['our_kills'].mean()
    avg_our_deaths = df['our_deaths'].mean()
    print(f"  平均味方キル: {avg_our_kills:.1f}  平均味方デス: {avg_our_deaths:.1f}")
    print(f"  → 平均キル差: {avg_our_kills - avg_our_deaths:+.1f}")

    wipeouts = df[df['our_deaths'] >= 3]
    print(f"  味方3デス以上の壊滅的集団戦: {len(wipeouts)}回 ({len(wipeouts)/len(df)*100:.1f}%)")

    # ── 4. 各メンバーの詳細 ──
    print(f"\n{'─' * 70}")
    print("■ メンバー別の詳細分析")
    print(f"{'─' * 70}")
    for name in victim_counts.index:
        subset = df[df['first_victim'] == name]
        print(f"\n  【{name}】 最初に死亡: {len(subset)}回")

        zone_detail = subset['death_zone'].value_counts()
        zone_str = ', '.join([f"{z}({n})" for z, n in zone_detail.items()])
        print(f"    死亡場所: {zone_str}")

        champ_detail = subset['victim_champion'].value_counts()
        champ_str = ', '.join([f"{c}({n})" for c, n in champ_detail.items()])
        print(f"    使用チャンプ: {champ_str}")

        killed_by_detail = subset['killed_by'].value_counts()
        killed_by_str = ', '.join([f"{k}({n})" for k, n in killed_by_detail.head(3).items()])
        print(f"    倒された相手: {killed_by_str}")

        avg_time = subset['fight_time'].mean()
        print(f"    キャッチされた平均時刻: {avg_time:.1f}分")

    # ── 5. 試合ごとの詳細リスト ──
    print(f"\n{'─' * 70}")
    print("■ 敗北試合の最後の集団戦 - 試合別詳細")
    print(f"{'─' * 70}")
    df_sorted = df.sort_values('game_duration', ascending=False)
    print(f"\n{'試合ID':<20} {'試合時間':>6} {'集団戦':>5} {'最初の犠牲者':<14} {'チャンプ':<12} {'倒した敵':<14} {'場所':<14} {'キル交換'}")
    print("─" * 110)
    for _, row in df_sorted.iterrows():
        exchange = f"{row['our_kills']}K / {row['our_deaths']}D"
        print(f"{row['matchId']:<20} {row['game_duration']:>5.1f}分 {row['fight_time']:>4.1f}分 "
              f"{row['first_victim']:<14} {row['victim_champion']:<12} {row['killed_by']:<14} "
              f"{row['death_zone']:<14} {exchange}")

    print(f"\n{'=' * 70}")
    print("💡 考察ポイント:")
    top_victim = victim_counts.index[0]
    top_count = victim_counts.iloc[0]
    top_pct = top_count / len(df) * 100
    print(f"  • 敗北試合の最後の集団戦で最もキャッチされているのは {top_victim} ({top_count}回, {top_pct:.1f}%)")
    top_zone = zone_counts.index[0]
    print(f"  • 最もキャッチされやすい場所は「{top_zone}」")
    print(f"  • 最後の集団戦はキル交換 平均{avg_our_kills:.1f} vs {avg_our_deaths:.1f} で負けている")
    if len(victim_counts) > 1:
        print(f"  • 後衛がキャッチされている場合はポジション取りの改善、")
        print(f"    前衛がキャッチされている場合は単独行動や視界不足が原因の可能性")


if __name__ == '__main__':
    main()

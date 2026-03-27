"""集団戦イニシエート分析スクリプト（多要素スコアリング版）

5つのシグナルの多数決でイニシエート側を判定する:
  1. ファーストキル取得チーム
  2. 戦闘発生位置（敵陣地かどうか）
  3. ファーストキルのアシスト連携度
  4. キル連鎖パターン（最初の2キルが同一チームか）
  5. エンゲージチャンピオンの関与
"""
import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 集団戦クラスタリング設定 ─────────────────────
TEAMFIGHT_TIME_WINDOW = 0.5   # 分（30秒以内のキルは同一集団戦）
MIN_KILLS_FOR_TEAMFIGHT = 3

# ── マップ座標設定（サモナーズリフト）────────────────
# Blue(100)=左下, Red(200)=右上, 中心≒7500
MAP_CENTER = 7500
TERRITORY_THRESHOLD = 1000

# ── エンゲージ系チャンピオン定義 ─────────────────────
ENGAGE_CHAMPIONS = {
    # Engage Supports
    "Leona", "Nautilus", "Thresh", "Alistar", "Rakan", "Rell",
    "Blitzcrank", "Pyke", "Maokai", "Braum", "Rell",
    # Engage Tanks / Vanguards (Top / Jungle)
    "Malphite", "Ornn", "Sion", "Kled", "Gragas", "Zac", "Amumu",
    "JarvanIV", "Vi", "MonkeyKing", "Hecarim", "Nocturne",
    "Sejuani", "RekSai", "Skarner", "Poppy", "Rammus",
    # Engage Bruisers / Divers
    "Camille", "Aatrox", "Gnar", "KSante", "Yone", "Sett",
    "XinZhao", "Volibear", "TahmKench", "Renekton",
    # Engage Mids / AP Initiators
    "Diana", "Galio", "Annie", "Lissandra", "Neeko", "Kennen",
    "FiddleSticks",
}


# ── データ読み込み ─────────────────────────────
def load_data():
    with open(os.path.join(PROJECT_ROOT, 'config', 'settings.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members = [m['game_name'] for m in cfg['members']]

    events = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'timeline_events.csv'))
    kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()
    kills = kills.sort_values(['matchId', 'timestampMin']).reset_index(drop=True)

    player_stats = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'player_stats.csv'))

    champion_map = {}
    for _, row in player_stats.iterrows():
        champion_map[(row['matchId'], row['summonerName'])] = row['championName']

    return members, kills, player_stats, champion_map


def identify_our_team(match_id, player_stats, members):
    match_ps = player_stats[player_stats['matchId'] == match_id]
    for _, row in match_ps.iterrows():
        if row['summonerName'] in members:
            return row['teamId'], row['win']
    return None, None


# ── 集団戦クラスタリング ───────────────────────────
def cluster_teamfights(match_kills):
    teamfights = []
    if len(match_kills) == 0:
        return teamfights

    current_cluster = [match_kills.iloc[0]]
    for i in range(1, len(match_kills)):
        row = match_kills.iloc[i]
        prev = current_cluster[-1]
        if row['timestampMin'] - prev['timestampMin'] <= TEAMFIGHT_TIME_WINDOW:
            current_cluster.append(row)
        else:
            if len(current_cluster) >= MIN_KILLS_FOR_TEAMFIGHT:
                teamfights.append(pd.DataFrame(current_cluster))
            current_cluster = [row]

    if len(current_cluster) >= MIN_KILLS_FOR_TEAMFIGHT:
        teamfights.append(pd.DataFrame(current_cluster))

    return teamfights


# ── ヘルパー ──────────────────────────────────
def _parse_assists(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw, list):
        return raw
    return []


# ── 多要素イニシエート判定 ──────────────────────────
def score_initiation(tf_df, our_team_id, champion_map):
    """5つのシグナルの合計スコアでイニシエート側を判定する。

    Returns:
        we_initiated (bool), confidence ('high'/'medium'/'low'),
        raw_score (int, 正=味方, 負=敵), factors (dict)
    """
    first_kill = tf_df.iloc[0]
    match_id = first_kill['matchId']
    score = 0
    factors = {}

    # ── Signal 1: ファーストキル取得チーム ──
    fk_is_ours = (first_kill['killerTeamId'] == our_team_id)
    score += 1 if fk_is_ours else -1
    factors['first_kill'] = 'ally' if fk_is_ours else 'enemy'

    # ── Signal 2: 戦闘発生位置 ──
    valid_pos = tf_df[(tf_df['positionX'] > 0) & (tf_df['positionY'] > 0)]
    if not valid_pos.empty:
        avg_x = valid_pos['positionX'].mean()
        avg_y = valid_pos['positionY'].mean()
        if our_team_id == 100:
            territory = (avg_x + avg_y) / 2 - MAP_CENTER
        else:
            territory = MAP_CENTER - (avg_x + avg_y) / 2

        if territory > TERRITORY_THRESHOLD:
            score += 1
            factors['position'] = 'enemy_territory'
        elif territory < -TERRITORY_THRESHOLD:
            score -= 1
            factors['position'] = 'our_territory'
        else:
            factors['position'] = 'neutral'
    else:
        factors['position'] = 'no_data'

    # ── Signal 3: ファーストキルのアシスト連携度 ──
    assists = _parse_assists(first_kill.get('assistingParticipantIds', '[]'))
    factors['first_kill_assists'] = len(assists)
    if len(assists) >= 2:
        score += 1 if fk_is_ours else -1
        factors['coordinated'] = True
    else:
        factors['coordinated'] = False

    # ── Signal 4: キル連鎖パターン（最初の2キル） ──
    if len(tf_df) >= 2:
        t1 = tf_df.iloc[0]['killerTeamId']
        t2 = tf_df.iloc[1]['killerTeamId']
        if t1 == t2:
            score += 1 if (t1 == our_team_id) else -1
            factors['kill_sequence'] = 'dominant'
        else:
            factors['kill_sequence'] = 'traded'
    else:
        factors['kill_sequence'] = 'n/a'

    # ── Signal 5: エンゲージチャンピオンの関与 ──
    killer_name = first_kill['killerName']
    victim_name = first_kill['victimName']
    killer_champ = champion_map.get((match_id, killer_name), '')
    victim_champ = champion_map.get((match_id, victim_name), '')
    factors['killer_champion'] = killer_champ
    factors['victim_champion'] = victim_champ

    killer_is_engage = killer_champ in ENGAGE_CHAMPIONS
    victim_is_engage = victim_champ in ENGAGE_CHAMPIONS

    if killer_is_engage and not victim_is_engage:
        score += 1 if fk_is_ours else -1
        factors['engage_signal'] = f'{killer_champ}(killer)_engaged'
    elif victim_is_engage and not killer_is_engage:
        victim_is_ours = (first_kill['victimTeamId'] == our_team_id)
        score += 1 if victim_is_ours else -1
        factors['engage_signal'] = f'{victim_champ}(victim)_failed_engage'
    elif killer_is_engage and victim_is_engage:
        factors['engage_signal'] = 'both_engage'
    else:
        factors['engage_signal'] = 'none'

    # ── 最終判定 ──
    we_initiated = score > 0
    if score == 0:
        we_initiated = fk_is_ours

    if abs(score) >= 4:
        confidence = 'high'
    elif abs(score) >= 2:
        confidence = 'medium'
    else:
        confidence = 'low'

    return we_initiated, confidence, score, factors


# ── 1回の集団戦を分析 ──────────────────────────
def analyze_teamfight(tf_df, our_team_id, champion_map):
    first_kill = tf_df.iloc[0]

    team_kills = {}
    for team in [100, 200]:
        team_kills[team] = len(tf_df[tf_df['killerTeamId'] == team])

    if team_kills.get(100, 0) > team_kills.get(200, 0):
        winning_team = 100
    elif team_kills.get(200, 0) > team_kills.get(100, 0):
        winning_team = 200
    else:
        winning_team = 0

    we_won_fight = (winning_team == our_team_id)
    we_initiated, confidence, raw_score, factors = score_initiation(
        tf_df, our_team_id, champion_map
    )

    return {
        'timestamp_min': first_kill['timestampMin'],
        'total_kills': len(tf_df),
        'our_kills': team_kills.get(our_team_id, 0),
        'enemy_kills': team_kills.get(100 if our_team_id == 200 else 200, 0),
        'we_initiated': we_initiated,
        'initiation_confidence': confidence,
        'initiation_score': raw_score,
        'we_won_fight': we_won_fight,
        'fight_result': 'draw' if winning_team == 0 else ('win' if we_won_fight else 'loss'),
        'first_killer': first_kill['killerName'],
        'first_victim': first_kill['victimName'],
        'killer_champion': factors.get('killer_champion', ''),
        'victim_champion': factors.get('victim_champion', ''),
        'engage_signal': factors.get('engage_signal', ''),
        'position_signal': factors.get('position', ''),
        'coordinated': factors.get('coordinated', False),
    }


# ── メイン ──────────────────────────────────
def main():
    members, kills, player_stats, champion_map = load_data()
    match_ids = kills['matchId'].unique()

    all_fights = []
    for mid in match_ids:
        our_team_id, match_win = identify_our_team(mid, player_stats, members)
        if our_team_id is None:
            continue

        match_kills = kills[kills['matchId'] == mid].copy()
        teamfights = cluster_teamfights(match_kills)

        for tf_df in teamfights:
            fight = analyze_teamfight(tf_df, our_team_id, champion_map)
            fight['matchId'] = mid
            fight['match_win'] = match_win
            all_fights.append(fight)

    if not all_fights:
        print("集団戦データが見つかりませんでした。")
        return

    df = pd.DataFrame(all_fights)

    total = len(df)
    initiated = df[df['we_initiated']]
    responded = df[~df['we_initiated']]

    # ================================================================
    # 全体サマリー
    # ================================================================
    print("=" * 70)
    print("  集団戦イニシエート分析（多要素スコアリング版）")
    print("=" * 70)
    print("\n判定シグナル:")
    print("  1) ファーストキル取得チーム")
    print("  2) 戦闘発生位置（敵陣地 or 自陣地）")
    print("  3) ファーストキルのアシスト連携度（≥2人で組織的エンゲージ）")
    print("  4) キル連鎖パターン（最初の2キルが同一チームか）")
    print("  5) エンゲージ系チャンピオンの関与")

    print(f"\n■ 集団戦の総数: {total} 回")
    print(f"  - 味方がイニシエート: {len(initiated)} 回 ({len(initiated)/total*100:.1f}%)")
    print(f"  - 敵がイニシエート:   {len(responded)} 回 ({len(responded)/total*100:.1f}%)")

    print(f"\n  判定信頼度の内訳:")
    for conf in ['high', 'medium', 'low']:
        cnt = len(df[df['initiation_confidence'] == conf])
        if cnt > 0:
            print(f"    {conf:6s}: {cnt} 回 ({cnt/total*100:.1f}%)")

    # ================================================================
    # 集団戦の勝率
    # ================================================================
    print(f"\n■ 集団戦の勝率（キル数で判定）")
    for label, subset in [("味方イニシエート", initiated), ("敵イニシエート", responded)]:
        if len(subset) == 0:
            continue
        wins = len(subset[subset['fight_result'] == 'win'])
        losses = len(subset[subset['fight_result'] == 'loss'])
        draws = len(subset[subset['fight_result'] == 'draw'])
        print(f"  {label}時:  勝ち {wins} / 負け {losses} / 引分 {draws}"
              f"  → 勝率 {wins/len(subset)*100:.1f}%")

    # ================================================================
    # 信頼度別の集団戦勝率
    # ================================================================
    print(f"\n■ 判定信頼度別の集団戦勝率")
    for conf in ['high', 'medium', 'low']:
        parts = []
        for flag, label in [(True, "味方イニシ"), (False, "敵イニシ")]:
            sub = df[(df['initiation_confidence'] == conf) & (df['we_initiated'] == flag)]
            if len(sub) > 0:
                wr = len(sub[sub['fight_result'] == 'win']) / len(sub) * 100
                parts.append(f"{label} {len(sub)}回→勝率{wr:.0f}%")
        if parts:
            print(f"  {conf:6s}: {' / '.join(parts)}")

    # ================================================================
    # イニシエートと試合勝敗の関係
    # ================================================================
    print(f"\n■ イニシエートと試合勝敗の関係")
    match_summary = df.groupby('matchId').agg(
        total_fights=('we_initiated', 'count'),
        we_initiated_count=('we_initiated', 'sum'),
        fights_won=('fight_result', lambda x: (x == 'win').sum()),
        fights_lost=('fight_result', lambda x: (x == 'loss').sum()),
        match_win=('match_win', 'first')
    ).reset_index()
    match_summary['initiate_ratio'] = (
        match_summary['we_initiated_count'] / match_summary['total_fights']
    )

    high_init = match_summary[match_summary['initiate_ratio'] >= 0.5]
    low_init = match_summary[match_summary['initiate_ratio'] < 0.5]

    if len(high_init) > 0:
        print(f"  イニシエート率 ≥ 50% の試合: {len(high_init)} 試合"
              f" → 試合勝率 {high_init['match_win'].mean()*100:.1f}%")
    if len(low_init) > 0:
        print(f"  イニシエート率 < 50% の試合:  {len(low_init)} 試合"
              f" → 試合勝率 {low_init['match_win'].mean()*100:.1f}%")

    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels_bin = ['0~25%', '25~50%', '50~75%', '75~100%']
    match_summary['init_bin'] = pd.cut(
        match_summary['initiate_ratio'], bins=bins, labels=labels_bin, right=False
    )
    init_bin_wr = match_summary.groupby('init_bin', observed=True).agg(
        試合数=('match_win', 'count'),
        試合勝率=('match_win', 'mean'),
    )
    init_bin_wr['試合勝率%'] = (init_bin_wr['試合勝率'] * 100).round(1)
    if not init_bin_wr.empty:
        print(f"\n  イニシエート率帯別の試合勝率:")
        for bin_label, row in init_bin_wr.iterrows():
            print(f"    {bin_label}: {int(row['試合数'])} 試合 → 勝率 {row['試合勝率%']}%")

    # ================================================================
    # 時間帯別
    # ================================================================
    print(f"\n■ 時間帯別の集団戦イニシエート傾向")
    df['phase'] = pd.cut(
        df['timestamp_min'],
        bins=[0, 15, 25, 60],
        labels=['序盤(~15分)', '中盤(15~25分)', '終盤(25分~)']
    )
    phase_stats = df.groupby('phase', observed=True).agg(
        count=('we_initiated', 'count'),
        we_initiated=('we_initiated', 'sum'),
        fight_win=('fight_result', lambda x: (x == 'win').sum())
    )
    for phase, row in phase_stats.iterrows():
        init_pct = row['we_initiated'] / row['count'] * 100 if row['count'] > 0 else 0
        win_pct = row['fight_win'] / row['count'] * 100 if row['count'] > 0 else 0
        print(f"  {phase}: 集団戦 {int(row['count'])}回 | "
              f"イニシエート率 {init_pct:.1f}% | 集団戦勝率 {win_pct:.1f}%")

    # ================================================================
    # キル差
    # ================================================================
    print(f"\n■ 集団戦のキル差")
    for label, subset in [("味方イニシエート", initiated), ("敵イニシエート", responded)]:
        if len(subset) == 0:
            continue
        sub_diff = subset['our_kills'] - subset['enemy_kills']
        print(f"  {label}時の平均キル差: {sub_diff.mean():+.2f} (味方キル - 敵キル)")

    # ================================================================
    # エンゲージチャンピオン別イニシエート成績
    # ================================================================
    print(f"\n■ エンゲージチャンピオン別イニシエート成績（≥3回）")

    engage_initiated = df[
        df['engage_signal'].str.contains('engaged', na=False)
    ]
    if len(engage_initiated) > 0:
        engage_stats = engage_initiated.groupby('killer_champion').agg(
            回数=('fight_result', 'count'),
            集団戦勝利=('fight_result', lambda x: (x == 'win').sum()),
        )
        engage_stats['勝率%'] = (
            engage_stats['集団戦勝利'] / engage_stats['回数'] * 100
        ).round(1)
        engage_stats = engage_stats[engage_stats['回数'] >= 3].sort_values(
            '回数', ascending=False
        )
        if not engage_stats.empty:
            for champ, row in engage_stats.iterrows():
                print(f"  {champ:<16s}: {int(row['回数']):>3}回 → 集団戦勝率 {row['勝率%']}%")
        else:
            print("  3回以上イニシエートしたエンゲージチャンピオンなし")
    else:
        print("  該当データなし")

    failed_engage = df[
        df['engage_signal'].str.contains('failed', na=False)
    ]
    if len(failed_engage) > 0:
        print(f"\n  エンゲージ失敗（エンゲージチャンプが先に倒された）: {len(failed_engage)} 回")
        failed_stats = failed_engage['victim_champion'].value_counts()
        for champ, cnt in failed_stats.head(5).items():
            print(f"    {champ}: {cnt}回")

    # ================================================================
    # メンバー別イニシエート回数
    # ================================================================
    print(f"\n■ メンバー別イニシエート回数（ファーストキル取得者）")
    if len(initiated) > 0:
        initiator_counts = initiated['first_killer'].value_counts()
        for name, count in initiator_counts.items():
            if name not in members:
                continue
            pct = count / len(initiated) * 100
            wins = len(initiated[
                (initiated['first_killer'] == name) &
                (initiated['fight_result'] == 'win')
            ])
            win_pct = wins / count * 100 if count > 0 else 0
            champs = initiated[
                initiated['first_killer'] == name
            ]['killer_champion'].value_counts()
            top_champs = ', '.join(
                f'{c}({n})' for c, n in champs.head(3).items()
            )
            print(f"  {name}: {count}回 ({pct:.1f}%)"
                  f" → 集団戦勝率 {win_pct:.1f}%  [{top_champs}]")

    # ================================================================
    # 敵イニシエート時に最初にデスしたメンバー
    # ================================================================
    print(f"\n■ 敵イニシエート時に最初にデスしたメンバー")
    if len(responded) > 0:
        member_victims = responded[responded['first_victim'].isin(members)]
        if len(member_victims) > 0:
            first_victims = member_victims['first_victim'].value_counts()
            for name, count in first_victims.items():
                pct = count / len(responded) * 100
                champs = member_victims[
                    member_victims['first_victim'] == name
                ]['victim_champion'].value_counts()
                top_champs = ', '.join(
                    f'{c}({n})' for c, n in champs.head(3).items()
                )
                print(f"  {name}: {count}回 ({pct:.1f}%)  [{top_champs}]")
        else:
            print("  メンバーのファーストデスデータなし")

    # ================================================================
    # 集団戦の規模別分析
    # ================================================================
    print(f"\n■ 集団戦の規模別分析")
    df['fight_size'] = pd.cut(
        df['total_kills'],
        bins=[2, 4, 6, 8, 20],
        labels=['小規模(3-4)', '中規模(5-6)', '大規模(7-8)', '超大規模(9+)']
    )
    size_stats = df.groupby('fight_size', observed=True).agg(
        count=('we_initiated', 'count'),
        we_init=('we_initiated', 'sum'),
        we_win=('fight_result', lambda x: (x == 'win').sum())
    )
    for size, row in size_stats.iterrows():
        if row['count'] == 0:
            continue
        print(f"  {size}: {int(row['count'])}回 | "
              f"イニシエート率 {row['we_init']/row['count']*100:.1f}% | "
              f"集団戦勝率 {row['we_win']/row['count']*100:.1f}%")

    # ================================================================
    # フッター
    # ================================================================
    print("\n" + "=" * 70)
    print("※ 判定方法: 5つのシグナル（ファーストキル / 戦闘位置 / 連携度 /")
    print("  キル連鎖 / エンゲージチャンピオン）の多数決でイニシエート側を決定。")
    print("  score > 0 → 味方イニシ, score < 0 → 敵イニシ, 0 → ファーストキルで判定。")
    print("=" * 70)


if __name__ == '__main__':
    main()

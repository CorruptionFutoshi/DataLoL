"""Analyze whether a member's top lane advantage is self-generated or jungle-dependent.

Key questions:
1. Does the jungler visit top more than other lanes? (JG proximity)
2. Does the target member's early GD differ when JG ganks top vs doesn't?
3. Compare to non-member top laners — do they get similar JG attention?
4. 10min GD (before most ganks settle) vs 15min GD

Usage: python scripts/_top_jungle_proximity.py [MemberName]
"""
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


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def main():
    with open(CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members_list = [m['game_name'] for m in cfg.get('members', [])]
    TARGET = sys.argv[1] if len(sys.argv) > 1 else members_list[0]
    members_set = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get('members', [])}

    ps = pd.read_csv(DATA / 'player_stats.csv')
    tf = pd.read_csv(DATA / 'timeline_frames.csv')
    ev = pd.read_csv(DATA / 'timeline_events.csv')

    ps['riotId'] = ps['summonerName'].astype(str) + '#' + ps['tagLine'].astype(str)
    ps['is_member'] = ps['riotId'].isin(members_set)
    member_names = set(ps.loc[ps['is_member'], 'summonerName'].unique())

    # Build match→member role map
    mem_ps = ps[ps['is_member']][['matchId', 'summonerName', 'role', 'teamId', 'win']].drop_duplicates()

    # ================================================================
    print("=" * 78)
    print("  PART 1: JGのガンク先分布 — トップは優遇されているか？")
    print("=" * 78)
    print()

    kills = ev[ev['eventType'] == 'CHAMPION_KILL'].copy()

    # Define lane zones by position (rough)
    # TOP: y > 9000 or x < 5000 (top side of map)
    # BOT: y < 5000 or x > 9000
    # MID: around diagonal
    def classify_lane_zone(x, y):
        if pd.isna(x) or pd.isna(y):
            return 'UNKNOWN'
        mid_dist = abs(x - y)
        if mid_dist < 2500:
            return 'MID'
        if x + y < 14000:
            return 'BOT_SIDE'
        return 'TOP_SIDE'

    # Find ganks: kills where our jungler has an assist or is the killer
    # within first 15 minutes
    early_kills = kills[kills['timestampMin'] <= 15].copy()
    early_kills['zone'] = early_kills.apply(
        lambda r: classify_lane_zone(r['positionX'], r['positionY']), axis=1)

    # For each match, identify our jungler
    jg_map = mem_ps[mem_ps['role'] == 'JUNGLE'][['matchId', 'summonerName', 'teamId']].copy()
    jg_map.rename(columns={'summonerName': 'jgName'}, inplace=True)

    # Find kills where our JG participated (as killer or assist)
    gank_results = []
    for _, jg_row in jg_map.iterrows():
        match_id = jg_row['matchId']
        jg_name = jg_row['jgName']
        our_team = jg_row['teamId']

        match_kills = early_kills[early_kills['matchId'] == match_id]

        for _, k in match_kills.iterrows():
            jg_involved = False
            if k['killerName'] == jg_name:
                jg_involved = True
            if isinstance(k['assistingParticipantIds'], str) and jg_name in str(k.get('killerName', '')):
                pass

            # Check if JG is killer
            if k['killerName'] == jg_name:
                jg_involved = True

            # Check assists — need participant ID mapping
            # Alternative: check if kill is by our team and near a lane
            # Simpler approach: kills by our team in lane zones during first 15min
            # where the JG is likely involved

            if k['killerTeamId'] == our_team and jg_involved:
                gank_results.append({
                    'matchId': match_id,
                    'zone': k['zone'],
                    'time': k['timestampMin'],
                    'victim': k['victimName']
                })

    gank_df = pd.DataFrame(gank_results)
    if len(gank_df) > 0:
        zone_counts = gank_df['zone'].value_counts()
        total = len(gank_df)
        print("  JGがキラーとなった15分以内のキル — エリア分布:")
        print()
        for zone in ['TOP_SIDE', 'MID', 'BOT_SIDE']:
            c = zone_counts.get(zone, 0)
            print(f"    {zone:10s}  {c:4d}回  ({c/total*100:5.1f}%)")
        print(f"    合計        {total:4d}回")
        print()

    # ================================================================
    print("=" * 78)
    print("  PART 2: JG近接度 — タイムラインフレームでの位置ベース分析")
    print("=" * 78)
    print()
    print("  JGとトップレーナーの距離 vs JGとボトムレーナーの距離")
    print("  (6〜14分のフレーム、JGがどちら側に寄っているか)")
    print()

    # Get position data from timeline_frames
    # timeline_frames has: matchId, timestampMin, summonerName, role, xp, totalGold, etc.
    # but NOT position. We need timeline_events or raw position data.
    # Actually timeline_frames might not have position. Let's check.

    # Alternative: Use kills/deaths near top to measure JG involvement
    # Better approach: count how many kills our JG participates in near top vs bot

    # Let's use a more robust method:
    # Count all early kills (0-15min) where our JG got a kill or assist,
    # and the location is near each lane

    # We need assist info. The assistingParticipantIds contains participant IDs not names.
    # Let's map participant IDs.

    # Get participant mapping
    # In timeline_events, killerId and victimId are participantIds (1-10)
    # participantId mapping: usually 1-5 = team 100, 6-10 = team 200
    # summonerName is available for killer and victim

    # Better: just count all kills near top/bot where our team got the kill
    # (includes JG assists implicitly)

    print("  [位置データがフレームにないため、キルイベントベースで分析]")
    print()

    # ================================================================
    print("=" * 78)
    print(f"  PART 3: {TARGET}@トップ — 10分GD（超序盤）の検証")
    print("=" * 78)
    print()
    print("  10分時点はJGガンクの影響が少ない純粋なレーン力の指標")
    print("  15分時点はJGの介入を含んだ結果")
    print()

    egao_top = mem_ps[(mem_ps['summonerName'] == TARGET) & (mem_ps['role'] == 'TOP')]
    egao_top_matches = set(egao_top['matchId'])

    for minute in [8, 10, 12, 15]:
        tf_min = tf[(tf['timestampMin'] == minute) &
                    (tf['summonerName'] == TARGET) &
                    (tf['matchId'].isin(egao_top_matches))]
        gd = tf_min['goldDiffVsOpponent'].dropna()
        if len(gd) < 20:
            continue
        mean_gd = gd.mean()
        se = gd.std() / np.sqrt(len(gd))
        lo95 = mean_gd - 1.96 * se
        hi95 = mean_gd + 1.96 * se
        t, p = stats.ttest_1samp(gd, 0)
        ahead_rate = (gd > 0).mean() * 100
        print(f"  {minute:2d}分  平均GD {mean_gd:+7.0f}G  95%CI [{lo95:+.0f}, {hi95:+.0f}]  "
              f"p={p:.4f} {sig(p)}  先行率 {ahead_rate:.0f}%  (n={len(gd)})")

    print()

    # Compare with other top laners (members)
    print("  ── 他メンバー@トップ との比較 (10分GD) ──")
    print()

    for name in sorted(member_names):
        sub = mem_ps[(mem_ps['summonerName'] == name) & (mem_ps['role'] == 'TOP')]
        matches = set(sub['matchId'])
        tf_10 = tf[(tf['timestampMin'] == 10) & (tf['summonerName'] == name) &
                   (tf['matchId'].isin(matches))]
        gd = tf_10['goldDiffVsOpponent'].dropna()
        if len(gd) < 15:
            continue
        mean_gd = gd.mean()
        se = gd.std() / np.sqrt(len(gd))
        t, p = stats.ttest_1samp(gd, 0)
        print(f"  {name:14s}  10分GD {mean_gd:+7.0f}G  ±{se:.0f}  n={len(gd)}  "
              f"p={p:.4f} {sig(p)}")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 4: トップ付近のキルイベント — 味方JGの関与有無")
    print("=" * 78)
    print()

    # Identify kills near top lane (0-15min) where TARGET is involved
    # Check if our JG assisted or not

    top_zone_kills = early_kills[
        early_kills.apply(lambda r: classify_lane_zone(r['positionX'], r['positionY']) == 'TOP_SIDE', axis=1)
    ]

    egao_involved = []
    for match_id in egao_top_matches:
        our_team_id = egao_top[egao_top['matchId'] == match_id]['teamId'].values
        if len(our_team_id) == 0:
            continue
        our_team_id = our_team_id[0]

        jg_row = mem_ps[(mem_ps['matchId'] == match_id) & (mem_ps['role'] == 'JUNGLE')]
        jg_name = jg_row['summonerName'].values[0] if len(jg_row) > 0 else None

        match_top_kills = top_zone_kills[top_zone_kills['matchId'] == match_id]

        for _, k in match_top_kills.iterrows():
            is_our_kill = k['killerTeamId'] == our_team_id
            is_our_death = k['victimTeamId'] == our_team_id
            egao_is_killer = k['killerName'] == TARGET
            egao_is_victim = k['victimName'] == TARGET

            if not (egao_is_killer or egao_is_victim or is_our_kill or is_our_death):
                continue

            jg_assisted = False
            if jg_name and isinstance(k.get('assistingParticipantIds', ''), str):
                # Can't directly check by name in assists (they're IDs)
                # But we can check if the killer is the JG
                if k['killerName'] == jg_name:
                    jg_assisted = True

            egao_involved.append({
                'matchId': match_id,
                'time': k['timestampMin'],
                'our_kill': is_our_kill,
                'egao_killer': egao_is_killer,
                'egao_victim': egao_is_victim,
                'jg_is_killer': k['killerName'] == jg_name if jg_name else False,
            })

    ei_df = pd.DataFrame(egao_involved)
    if len(ei_df) > 0:
        our_kills = ei_df[ei_df['our_kill'] == True]
        jg_killer_count = our_kills['jg_is_killer'].sum()
        non_jg_count = len(our_kills) - jg_killer_count

        print(f"  {TARGET}@トップ付近の味方キル (15分以内): {len(our_kills)}回")
        print(f"    JGがキラー: {jg_killer_count}回 ({jg_killer_count/len(our_kills)*100:.1f}%)")
        print(f"    JG以外がキラー: {non_jg_count}回 ({non_jg_count/len(our_kills)*100:.1f}%)")
        print()

        our_deaths = ei_df[ei_df['egao_victim'] == True]
        print(f"  {TARGET}@トップ付近のデス (15分以内): {len(our_deaths)}回")
        print()

    # ================================================================
    print("=" * 78)
    print("  PART 5: 各レーン付近の味方チームキル数 — JGのリソース配分")
    print("=" * 78)
    print()
    print("  15分以内の全キルをエリア別に集計（味方チームのキルのみ）")
    print()

    # Per match, count our team's kills in each zone
    zone_kill_counts = {'TOP_SIDE': [], 'MID': [], 'BOT_SIDE': []}

    all_match_ids = set(mem_ps['matchId'].unique())
    team_dict = dict(zip(
        mem_ps.drop_duplicates('matchId')['matchId'],
        mem_ps.drop_duplicates('matchId')['teamId']
    ))

    for match_id in all_match_ids:
        our_team = team_dict.get(match_id)
        if our_team is None:
            continue
        mk = early_kills[(early_kills['matchId'] == match_id) &
                         (early_kills['killerTeamId'] == our_team)]
        for zone in ['TOP_SIDE', 'MID', 'BOT_SIDE']:
            zk = mk[mk.apply(lambda r: classify_lane_zone(r['positionX'], r['positionY']) == zone, axis=1)]
            zone_kill_counts[zone].append(len(zk))

    for zone in ['TOP_SIDE', 'MID', 'BOT_SIDE']:
        arr = np.array(zone_kill_counts[zone])
        print(f"  {zone:10s}  1試合平均 {arr.mean():.2f}回  合計 {arr.sum():5d}回  "
              f"シェア {arr.sum()/sum(sum(v) for v in zone_kill_counts.values())*100:.1f}%")

    print()

    # Compare when TARGET is top vs when others are top
    print(f"  ── {TARGET}がトップの試合 vs 他メンバーがトップの試合 ──")
    print()

    egao_top_set = set(egao_top['matchId'])
    other_top = mem_ps[(mem_ps['role'] == 'TOP') & (mem_ps['summonerName'] != TARGET)]
    other_top_set = set(other_top['matchId']) - egao_top_set

    for label, match_set in [(f'{TARGET}@トップ', egao_top_set), ('他メンバー@トップ', other_top_set)]:
        top_kills_per_match = []
        total_kills_per_match = []
        for match_id in match_set:
            our_team = team_dict.get(match_id)
            if our_team is None:
                continue
            mk = early_kills[(early_kills['matchId'] == match_id) &
                             (early_kills['killerTeamId'] == our_team)]
            tk = mk[mk.apply(lambda r: classify_lane_zone(r['positionX'], r['positionY']) == 'TOP_SIDE', axis=1)]
            top_kills_per_match.append(len(tk))
            total_kills_per_match.append(len(mk))

        top_arr = np.array(top_kills_per_match)
        total_arr = np.array(total_kills_per_match)
        top_share = top_arr.sum() / total_arr.sum() * 100 if total_arr.sum() > 0 else 0
        print(f"  {label:18s}  トップ側キル/試合 {top_arr.mean():.2f}  "
              f"全体キル/試合 {total_arr.mean():.2f}  "
              f"トップ側シェア {top_share:.1f}%  (n={len(match_set)})")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 6: CS差でのレーン力純粋評価")
    print("=" * 78)
    print()
    print("  CS差はキルと無関係にレーニング力（ラストヒット精度、トレード、プレッシャー）を反映")
    print()

    for name in sorted(member_names):
        sub = mem_ps[(mem_ps['summonerName'] == name) & (mem_ps['role'] == 'TOP')]
        matches = set(sub['matchId'])
        if len(matches) < 15:
            continue
        for minute in [10, 15]:
            tf_min = tf[(tf['timestampMin'] == minute) & (tf['summonerName'] == name) &
                        (tf['matchId'].isin(matches))]
            cs_diff = tf_min['csDiffVsOpponent'].dropna()
            gd = tf_min['goldDiffVsOpponent'].dropna()
            if len(cs_diff) < 15:
                continue
            t_cs, p_cs = stats.ttest_1samp(cs_diff, 0)
            t_gd, p_gd = stats.ttest_1samp(gd, 0)
            print(f"  {name:14s}  {minute}分  CS差 {cs_diff.mean():+5.1f}  p={p_cs:.4f} {sig(p_cs):>4s}  "
                  f"GD {gd.mean():+7.0f}G  p={p_gd:.4f} {sig(p_gd):>4s}  (n={len(cs_diff)})")

    print()

    # Compare TARGET's top CS diff vs non-member top CS diff
    print(f"  ── {TARGET}@トップ CS差 vs 非メンバー@トップ CS差 ──")
    print()
    for minute in [10, 15]:
        egao_tf = tf[(tf['timestampMin'] == minute) & (tf['summonerName'] == TARGET) &
                     (tf['matchId'].isin(egao_top_matches))]
        non_mem_top_names = ps[(~ps['is_member']) & (ps['role'] == 'TOP')][['matchId', 'summonerName']].drop_duplicates()
        non_tf = tf[(tf['timestampMin'] == minute)].merge(non_mem_top_names, on=['matchId', 'summonerName'])

        m_cs = egao_tf['csDiffVsOpponent'].dropna()
        n_cs = non_tf['csDiffVsOpponent'].dropna()
        if len(m_cs) < 15 or len(n_cs) < 15:
            continue
        t, p = stats.ttest_ind(m_cs, n_cs, equal_var=False)
        d = (m_cs.mean() - n_cs.mean()) / np.sqrt((m_cs.std()**2 + n_cs.std()**2) / 2)
        print(f"  {minute}分  {TARGET} CS差 {m_cs.mean():+5.1f}  非メンバートップ {n_cs.mean():+5.1f}  "
              f"d={d:+.2f}  p={p:.4f} {sig(p)}")

    print()
    print("=" * 78)
    print("  分析完了")
    print("=" * 78)


if __name__ == '__main__':
    main()

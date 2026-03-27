"""Void Grub fight comprehensive analysis.

Features:
  - Grub kills clustered into spawn waves (gap > 2 min = new wave)
  - Only CHAMPION_KILLs within ~3000 units of Grub pit
  - Wave 1 vs Wave 2 breakdown
  - Role-level our team vs enemy team (= 同ランク帯平均) comparison
  - Per-member grub fight stats
  - Disaster analysis (2+ allied deaths)

Usage:
    python scripts/grub_disaster.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

GRUB_PIT_X = 4850
GRUB_PIT_Y = 10250
GRUB_RADIUS = 3000

WAVE_GAP_THRESHOLD = 2.0   # gap > this = separate spawn wave
WINDOW_BEFORE = 1.0
WINDOW_AFTER = 1.0
DISASTER_THRESHOLD = 2

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'JG', 'MIDDLE': 'ミッド',
           'BOTTOM': 'ボトム', 'UTILITY': 'サポ'}


def load_data():
    events = pd.read_csv(DATA / 'timeline_events.csv')
    obj = pd.read_csv(DATA / 'objectives.csv')
    players = pd.read_csv(DATA / 'player_stats.csv')

    with open(CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get('members', [])}
    member_names = {m["game_name"] for m in cfg.get("members", [])}

    players['riotId'] = (players['summonerName'].astype(str) + '#'
                         + players['tagLine'].astype(str))
    mt = players[players['riotId'].isin(members)][
        ['matchId', 'teamId']].drop_duplicates()

    kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()
    hordes = obj[obj['objectiveType'] == 'HORDE']

    role_map = {}
    for _, r in players[['matchId', 'summonerName', 'role']].iterrows():
        role_map[(r['matchId'], r['summonerName'])] = r['role']

    return hordes, kills, players, mt, members, member_names, role_map


def cluster_grub_waves(grub_times):
    if not grub_times:
        return []
    times = sorted(grub_times)
    waves = [[times[0]]]
    for t in times[1:]:
        if t - waves[-1][-1] > WAVE_GAP_THRESHOLD:
            waves.append([t])
        else:
            waves[-1].append(t)
    return waves


def is_near_grub_pit(x, y):
    return np.sqrt((x - GRUB_PIT_X) ** 2 + (y - GRUB_PIT_Y) ** 2) <= GRUB_RADIUS


def collect_grub_fights():
    hordes, kills, players, mt, members, member_names, role_map = load_data()

    all_fights = []

    for match_id in hordes['matchId'].unique():
        our_row = mt[mt['matchId'] == match_id]
        if our_row.empty:
            continue
        our_tid = int(our_row['teamId'].values[0])

        mgrubs = hordes[hordes['matchId'] == match_id]
        waves = cluster_grub_waves(mgrubs['timestampMin'].tolist())
        grub_teams = mgrubs[['timestampMin', 'teamId']]

        win_rows = players[(players['matchId'] == match_id)
                           & (players['riotId'].isin(members))]
        win = bool(win_rows['win'].values[0]) if len(win_rows) > 0 else None

        for wi, wave in enumerate(waves):
            t0 = wave[0] - WINDOW_BEFORE
            t1 = wave[-1] + WINDOW_AFTER

            wk = kills[(kills['matchId'] == match_id)
                       & (kills['timestampMin'] >= t0)
                       & (kills['timestampMin'] <= t1)]

            if wk.empty:
                near = pd.DataFrame()
            else:
                near = wk[wk.apply(
                    lambda r: is_near_grub_pit(r['positionX'], r['positionY']),
                    axis=1)]

            wg = grub_teams[(grub_teams['timestampMin'] >= wave[0])
                            & (grub_teams['timestampMin'] <= wave[-1])]
            our_grubs = int((wg['teamId'] == our_tid).sum())
            enemy_grubs = int((wg['teamId'] != our_tid).sum())

            our_deaths, enemy_deaths = [], []
            if not near.empty:
                for _, k in near.iterrows():
                    v_role = role_map.get((match_id, k['victimName']), '?')
                    k_role = role_map.get((match_id, k['killerName']), '?')
                    entry_v = {'name': k['victimName'], 'role': v_role}
                    entry_k = {'name': k['killerName'], 'role': k_role}
                    if k['victimTeamId'] == our_tid:
                        our_deaths.append(entry_v)
                    else:
                        enemy_deaths.append(entry_v)

            all_fights.append({
                'matchId': match_id,
                'wave_num': wi + 1,
                'wave_start': wave[0],
                'wave_end': wave[-1],
                'our_deaths': our_deaths,
                'enemy_deaths': enemy_deaths,
                'our_death_count': len(our_deaths),
                'enemy_death_count': len(enemy_deaths),
                'our_grubs': our_grubs,
                'enemy_grubs': enemy_grubs,
                'win': win,
                'has_fight': len(our_deaths) + len(enemy_deaths) > 0,
            })

    return all_fights, member_names


# ═══════════════════════════════════════════════════════════════
#  PART 1 — 全体像
# ═══════════════════════════════════════════════════════════════
def print_overview(fights):
    print("=" * 70)
    print("  PART 1: グラブファイト全体像")
    print("=" * 70)

    N = len(fights)
    fought = sum(f['has_fight'] for f in fights)
    od_total = sum(f['our_death_count'] for f in fights)
    ed_total = sum(f['enemy_death_count'] for f in fights)
    og = sum(f['our_grubs'] for f in fights)
    eg = sum(f['enemy_grubs'] for f in fights)

    print(f"\n  グラブウェーブ総数: {N}")
    print(f"  キル発生ウェーブ:  {fought} ({fought/N*100:.1f}%)")
    print(f"\n  グラブ獲得: 味方 {og}体  vs  敵 {eg}体")

    print(f"\n  ■ グラブ付近のキル/デス (半径{GRUB_RADIUS}以内)")
    print(f"    味方デス合計: {od_total}  (平均 {od_total/N:.2f} / wave)")
    print(f"    敵デス合計:   {ed_total}  (平均 {ed_total/N:.2f} / wave)")
    net = ed_total - od_total
    sign = '+' if net > 0 else ''
    print(f"    差引: {sign}{net}  {'← 味方有利' if net > 0 else '← 敵有利' if net < 0 else '互角'}")

    print(f"\n  ■ グラブファイト結果 → 試合勝率")
    buckets = [
        ('味方0デス (安全)',    lambda f: f['our_death_count'] == 0),
        ('味方1デス',           lambda f: f['our_death_count'] == 1),
        ('味方2デス+ (大事故)', lambda f: f['our_death_count'] >= 2),
    ]
    for label, pred in buckets:
        g = [f for f in fights if pred(f)]
        if g:
            w = sum(f['win'] for f in g)
            print(f"    {label:<22} {len(g):>4}回  勝率 {w/len(g)*100:.1f}%")

    print(f"\n  ■ グラブ獲得数 → 試合勝率")
    grub_buckets = [
        ('味方が多く獲得', lambda f: f['our_grubs'] > f['enemy_grubs']),
        ('同数',           lambda f: f['our_grubs'] == f['enemy_grubs']),
        ('敵が多く獲得',   lambda f: f['our_grubs'] < f['enemy_grubs']),
    ]
    for label, pred in grub_buckets:
        g = [f for f in fights if pred(f)]
        if g:
            w = sum(f['win'] for f in g)
            print(f"    {label:<18} {len(g):>4}回  勝率 {w/len(g)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  PART 2 — 第一波 vs 第二波
# ═══════════════════════════════════════════════════════════════
def print_wave_comparison(fights):
    print()
    print("=" * 70)
    print("  PART 2: 第一波 vs 第二波")
    print("=" * 70)

    for label, pred in [('第一波', lambda f: f['wave_num'] == 1),
                        ('第二波', lambda f: f['wave_num'] >= 2)]:
        g = [f for f in fights if pred(f)]
        if not g:
            continue
        n = len(g)
        fc = sum(f['has_fight'] for f in g)
        od = sum(f['our_death_count'] for f in g)
        ed = sum(f['enemy_death_count'] for f in g)
        og = sum(f['our_grubs'] for f in g)
        eg = sum(f['enemy_grubs'] for f in g)
        dis = sum(f['our_death_count'] >= 2 for f in g)
        wr = sum(f['win'] for f in g) / n * 100

        print(f"\n  ■ {label}  ({n}回)")
        print(f"    キル発生: {fc}回 ({fc/n*100:.1f}%)")
        print(f"    味方デス: {od} (平均 {od/n:.2f})   敵デス: {ed} (平均 {ed/n:.2f})")
        print(f"    グラブ獲得: 味方 {og}  vs  敵 {eg}")
        print(f"    大事故(2デス+): {dis}回 ({dis/n*100:.1f}%)")
        print(f"    試合勝率: {wr:.1f}%")

        times = [f['wave_start'] for f in g]
        print(f"    発生時間: 平均 {np.mean(times):.1f}分  "
              f"中央値 {np.median(times):.1f}分  "
              f"範囲 {np.min(times):.1f}~{np.max(times):.1f}分")


# ═══════════════════════════════════════════════════════════════
#  PART 3 — ロール別比較 (味方 vs 敵 = 同ランク帯平均)
# ═══════════════════════════════════════════════════════════════
def print_role_comparison(fights):
    print()
    print("=" * 70)
    print("  PART 3: ロール別グラブファイト — 味方 vs 敵 (同ランク帯平均)")
    print("=" * 70)
    print()
    print("  ※ 敵チームの数値 = 同ランク帯の平均的なプレイヤーの振る舞い")
    print("  ※ 味方の数値が敵より悪ければ、平均より下手にグラブを戦っている")

    N = len(fights)
    our_rd = Counter()
    enemy_rd = Counter()

    for f in fights:
        for d in f['our_deaths']:
            if d['role'] in ROLES:
                our_rd[d['role']] += 1
        for d in f['enemy_deaths']:
            if d['role'] in ROLES:
                enemy_rd[d['role']] += 1

    print(f"\n  {'ロール':<8}  {'味方デス':>8} {'(/wave)':>8}  "
          f"{'敵デス':>8} {'(/wave)':>8}  {'差':>6}  評価")
    print("  " + "-" * 72)

    for role in ROLES:
        rj = ROLE_JP[role]
        od = our_rd[role]
        ed = enemy_rd[role]
        or_ = od / N
        er_ = ed / N
        diff = od - ed
        sign = '+' if diff > 0 else ''
        if diff > 3:
            grade = '⚠ 多すぎ'
        elif diff > 0:
            grade = '△ やや多い'
        elif diff < -3:
            grade = '◎ 優秀'
        elif diff < 0:
            grade = '○ 良好'
        else:
            grade = '— 互角'
        print(f"  {rj:<8}  {od:>8} {or_:>7.2f}   {ed:>8} {er_:>7.2f}   "
              f"{sign}{diff:>5}  {grade}")

    tod = sum(our_rd.values())
    ted = sum(enemy_rd.values())
    print("  " + "-" * 72)
    net = tod - ted
    sign = '+' if net > 0 else ''
    print(f"  {'合計':<8}  {tod:>8} {tod/N:>7.2f}   {ted:>8} {ted/N:>7.2f}   "
          f"{sign}{net:>5}")

    # Wave 1 vs Wave 2 role breakdown
    for wlabel, wpred in [('第一波', lambda f: f['wave_num'] == 1),
                          ('第二波', lambda f: f['wave_num'] >= 2)]:
        wg = [f for f in fights if wpred(f)]
        if not wg:
            continue
        wn = len(wg)
        w_our = Counter()
        w_enemy = Counter()
        for f in wg:
            for d in f['our_deaths']:
                if d['role'] in ROLES:
                    w_our[d['role']] += 1
            for d in f['enemy_deaths']:
                if d['role'] in ROLES:
                    w_enemy[d['role']] += 1

        print(f"\n  ■ {wlabel}のみ ({wn}回)")
        print(f"    {'ロール':<8} {'味方デス':>8} {'敵デス':>8} {'差':>6}")
        print("    " + "-" * 36)
        for role in ROLES:
            rj = ROLE_JP[role]
            od = w_our[role]
            ed = w_enemy[role]
            diff = od - ed
            s = '+' if diff > 0 else ''
            print(f"    {rj:<8} {od:>8} {ed:>8} {s}{diff:>5}")


# ═══════════════════════════════════════════════════════════════
#  PART 4 — メンバー別
# ═══════════════════════════════════════════════════════════════
def print_member_stats(fights, member_names):
    print()
    print("=" * 70)
    print("  PART 4: メンバー別グラブファイト成績")
    print("=" * 70)

    N = len(fights)
    m_deaths = Counter()
    m_deaths_w1 = Counter()
    m_deaths_w2 = Counter()

    for f in fights:
        for d in f['our_deaths']:
            name = d['name']
            if name in member_names:
                m_deaths[name] += 1
                if f['wave_num'] == 1:
                    m_deaths_w1[name] += 1
                else:
                    m_deaths_w2[name] += 1

    all_names = sorted(m_deaths.keys(), key=lambda n: -m_deaths[n])

    print(f"\n  {'メンバー':<14} {'総デス':>6} {'(/wave)':>8} "
          f"{'第一波':>6} {'第二波':>6}")
    print("  " + "-" * 50)
    for name in all_names:
        d = m_deaths[name]
        d1 = m_deaths_w1[name]
        d2 = m_deaths_w2[name]
        print(f"  {name:<14} {d:>6} {d/N:>7.3f}  {d1:>6} {d2:>6}")

    # Disaster involvement
    disasters = [f for f in fights if f['our_death_count'] >= 2]
    if disasters:
        print(f"\n  ■ 大事故 (2デス+) への関与回数:")
        inv = Counter()
        for f in disasters:
            for d in f['our_deaths']:
                if d['name'] in member_names:
                    inv[d['name']] += 1
        for name, cnt in inv.most_common():
            pct = cnt / len(disasters) * 100
            print(f"    {name:<14} {cnt}回 / {len(disasters)}件 ({pct:.0f}%)")


# ═══════════════════════════════════════════════════════════════
#  PART 5 — 大事故詳細
# ═══════════════════════════════════════════════════════════════
def print_disasters(fights):
    print()
    print("=" * 70)
    print("  PART 5: グラブ大事故の詳細 (味方2デス以上)")
    print("=" * 70)

    disasters = [f for f in fights if f['our_death_count'] >= DISASTER_THRESHOLD]
    N = len(fights)

    if not disasters:
        print("\n  大事故は見つかりませんでした。")
        return

    total = len(disasters)
    losses = sum(not d['win'] for d in disasters)

    print(f"\n  ウェーブ総数: {N}   大事故: {total}件 ({total/N*100:.1f}%)")
    print(f"  大事故後の戦績: {total - losses}勝 {losses}敗 "
          f"(勝率 {(total - losses)/total*100:.1f}%)")

    dc_groups = Counter(f['our_death_count'] for f in disasters)
    print(f"\n  ■ デス数別:")
    for dc in sorted(dc_groups):
        g = [f for f in disasters if f['our_death_count'] == dc]
        w = sum(f['win'] for f in g)
        print(f"    {dc}デス: {len(g)}件  勝率 {w/len(g)*100:.1f}%")

    print(f"\n  ■ 大事故時のグラブ獲得:")
    cats = Counter()
    for f in disasters:
        if f['our_grubs'] > f['enemy_grubs']:
            cats['味方が多く獲得'] += 1
        elif f['enemy_grubs'] > f['our_grubs']:
            cats['敵が多く獲得'] += 1
        else:
            cats['同数 or ゼロ'] += 1
    for k, v in cats.most_common():
        print(f"    {k}: {v}件")

    print(f"\n  ■ 一覧 (デス数降順):")
    for i, f in enumerate(sorted(disasters, key=lambda x: -x['our_death_count']), 1):
        wl = f"第{f['wave_num']}波"
        victims = ', '.join(
            f"{d['name']}({ROLE_JP.get(d['role'], d['role'])})"
            for d in f['our_deaths'])
        print(f"\n    [{i}] {f['matchId']}  {wl}")
        print(f"        {f['wave_start']:.1f}~{f['wave_end']:.1f}分  "
              f"味方{f['our_death_count']}デス vs 敵{f['enemy_death_count']}デス")
        print(f"        グラブ: 味方{f['our_grubs']} vs 敵{f['enemy_grubs']}  "
              f"→ {'WIN' if f['win'] else 'LOSE'}")
        print(f"        死亡者: {victims}")

    severe = [f for f in disasters if f['our_death_count'] >= 3]
    if severe:
        print(f"\n  ■ 3デス以上の重大事故: {len(severe)}件")
        for f in sorted(severe, key=lambda x: -x['our_death_count']):
            victims = ', '.join(d['name'] for d in f['our_deaths'])
            print(f"    {f['matchId']}  {f['wave_start']:.1f}分  "
                  f"{f['our_death_count']}デス  → {'WIN' if f['win'] else 'LOSE'}  "
                  f"({victims})")


def main():
    fights, member_names = collect_grub_fights()
    if not fights:
        print("グラブデータが見つかりませんでした。")
        return

    print_overview(fights)
    print_wave_comparison(fights)
    print_role_comparison(fights)
    print_member_stats(fights, member_names)
    print_disasters(fights)


if __name__ == '__main__':
    main()

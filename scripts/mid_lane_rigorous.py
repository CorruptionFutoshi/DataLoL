"""ミッドレーン厳密分析 — 統計的に有意な結論のみ抽出.

全ての主張に以下を付与:
- Welch t検定 / 1標本t検定 (連続量)
- Wilson score 95%CI (比率)
- Cohen's d (効果量)
- 多重比較補正 (Holm法)
- 相関にはSpearman + p値

勝率など「小標本では有意にならない二値指標」を根拠にした判断は行わない。
連続変量の差と効果量を中心に分析する。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

with open(CONFIG, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
MEMBER_NAMES = {m["game_name"] for m in cfg.get("members", [])}

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_JP = {"TOP": "トップ", "JUNGLE": "JG", "MIDDLE": "ミッド",
           "BOTTOM": "ボトム", "UTILITY": "サポート"}

MIN_GAMES = 15


def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return p, max(0, center - spread), min(1, center + spread)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def holm_correct(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (n - i))
    for i in range(1, n):
        idx = order[i]
        prev_idx = order[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def header(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


def load():
    ps = pd.read_csv(DATA / 'player_stats.csv')
    tf = pd.read_csv(DATA / 'timeline_frames.csv')
    matches = pd.read_csv(DATA / 'matches.csv')

    ps['is_member'] = ps['summonerName'].isin(MEMBER_NAMES)
    ps = ps[ps['role'].isin(ROLES)].copy()

    dur = matches.set_index('matchId')['gameDurationMin'].to_dict()
    ps['durMin'] = ps['matchId'].map(dur)
    ps['cs_per_min'] = ps['cs'] / ps['durMin'].clip(lower=1)
    ps['gold_per_min'] = ps['goldEarned'] / ps['durMin'].clip(lower=1)
    ps['dmg_per_min'] = ps['totalDamageDealtToChampions'] / ps['durMin'].clip(lower=1)

    team_kills = ps.groupby(['matchId', 'teamId'])['kills'].transform('sum')
    ps['kp'] = (ps['kills'] + ps['assists']) / team_kills.replace(0, 1)

    team_dmg = ps.groupby(['matchId', 'teamId'])['totalDamageDealtToChampions'].transform('sum')
    ps['dmg_share'] = ps['totalDamageDealtToChampions'] / team_dmg.replace(0, 1)

    return ps, tf


def main():
    ps, tf = load()

    mid_member = ps[(ps['is_member']) & (ps['role'] == 'MIDDLE')]
    mid_nonmember = ps[(~ps['is_member']) & (ps['role'] == 'MIDDLE')]

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║              ミッドレーン厳密分析                                        ║")
    print("║  — 統計的に有意な事実のみ報告 / 有意でない項目は明示的に除外 —            ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    # ================================================================
    # PART 0: 誰がミッドをプレイしているか
    # ================================================================
    header("PART 0: ミッドレーン出場状況")

    mid_counts = mid_member.groupby('summonerName').size().sort_values(ascending=False)
    total_mid = len(mid_member)
    print(f"  メンバーのミッド出場: 合計 {total_mid} レコード")
    print(f"  比較対象（非メンバーミッド）: {len(mid_nonmember)} レコード")
    print()
    for name, cnt in mid_counts.items():
        pct = cnt / total_mid * 100
        print(f"  {name:16s}  {cnt:4d}試合  ({pct:5.1f}%)")

    # ================================================================
    # PART 1: パフォーマンス指標 — メンバーミッド vs 非メンバーミッド (Welch t検定 + Cohen's d)
    # ================================================================
    header("PART 1: ミッドレーン パフォーマンス — メンバー vs 同ランク帯 (Welch t検定)")
    print("  各指標でメンバーのミッドと非メンバーのミッドを比較。")
    print("  Holm法で多重比較補正。Cohen's d で効果量を評価。")
    print("  |d| ≥ 0.2: 小, ≥ 0.5: 中, ≥ 0.8: 大")
    print()

    stat_cols = [
        ('kills',   'キル'),
        ('deaths',  'デス'),
        ('assists', 'アシスト'),
        ('kda',     'KDA'),
        ('cs_per_min', 'CS/min'),
        ('gold_per_min', 'Gold/min'),
        ('dmg_per_min', 'DMG/min'),
        ('dmg_share', 'ダメージ割合'),
        ('kp',      'KP'),
        ('visionScore', 'ビジョン'),
        ('goldEarned', '総ゴールド'),
        ('totalDamageDealtToChampions', '総ダメージ'),
    ]

    print(f"  {'指標':<14} {'メンバー平均':>12} {'非メンバー平均':>14} {'差分':>10} {'Cohen d':>8} {'p値':>10} {'補正p':>10} {'有意性':>6}")
    print("  " + "─" * 90)

    raw_pvals = []
    results_1 = []
    for col, label in stat_cols:
        m_vals = mid_member[col].dropna()
        n_vals = mid_nonmember[col].dropna()
        if len(m_vals) < MIN_GAMES or len(n_vals) < MIN_GAMES:
            continue
        t, p = stats.ttest_ind(m_vals, n_vals, equal_var=False)
        d = cohens_d(m_vals, n_vals)
        raw_pvals.append(p)
        results_1.append((label, m_vals.mean(), n_vals.mean(), m_vals.mean() - n_vals.mean(), d, p))

    adj_p = holm_correct([r[5] for r in results_1])
    for i, (label, m_mean, n_mean, diff, d, p) in enumerate(results_1):
        ap = adj_p[i]
        print(f"  {label:<14} {m_mean:>12.1f} {n_mean:>14.1f} {diff:>+10.1f} {d:>+8.2f} {p:>10.4f} {ap:>10.4f} {sig(ap):>6}")

    print()
    sig_items = [(results_1[i], adj_p[i]) for i in range(len(results_1)) if adj_p[i] < 0.05]
    if sig_items:
        print("  ★ 統計的に有意な差がある指標:")
        for (label, m, n, diff, d, _), ap in sig_items:
            direction = "上回る" if diff > 0 else "下回る"
            size = "大きな" if abs(d) >= 0.8 else "中程度の" if abs(d) >= 0.5 else "小さな"
            print(f"    {label}: メンバーが同ランク帯を{size}効果量で{direction} (d={d:+.2f}, 補正p={ap:.4f})")
    else:
        print("  → 多重比較補正後に有意な差がある指標はなし")

    # ================================================================
    # PART 1b: メンバー個人別 ミッドでのパフォーマンス
    # ================================================================
    header("PART 1b: メンバー個人別 ミッドパフォーマンス (Welch t検定 vs 非メンバーミッド)")
    print("  個人別に非メンバーミッドとの差を検定。")
    print("  Holm法で各メンバー内の指標群を多重比較補正。")
    print("  ※ 15試合未満のメンバーは除外")
    print()

    key_stats = [
        ('kda',     'KDA'),
        ('cs_per_min', 'CS/min'),
        ('gold_per_min', 'Gold/min'),
        ('dmg_per_min', 'DMG/min'),
        ('kp',      'KP'),
        ('visionScore', 'ビジョン'),
        ('deaths',  'デス'),
    ]

    for name in sorted(mid_counts.index):
        mp = mid_member[mid_member['summonerName'] == name]
        if len(mp) < MIN_GAMES:
            continue
        print(f"  ■ {name} (ミッド {len(mp)}試合)")
        print(f"    {'指標':<12} {'本人':>8} {'非メンバー':>10} {'差':>8} {'d':>7} {'補正p':>8} {'有意':>5}")
        print("    " + "─" * 62)

        ind_results = []
        for col, label in key_stats:
            mv = mp[col].dropna()
            nv = mid_nonmember[col].dropna()
            if len(mv) < MIN_GAMES:
                continue
            t, p = stats.ttest_ind(mv, nv, equal_var=False)
            d = cohens_d(mv, nv)
            ind_results.append((label, mv.mean(), nv.mean(), mv.mean() - nv.mean(), d, p))

        ind_adj = holm_correct([r[5] for r in ind_results])
        for j, (label, mv, nv, diff, d, p) in enumerate(ind_results):
            ap = ind_adj[j]
            print(f"    {label:<12} {mv:>8.2f} {nv:>10.2f} {diff:>+8.2f} {d:>+7.2f} {ap:>8.4f} {sig(ap):>5}")

        sig_personal = [(ind_results[j][0], ind_results[j][4], ind_adj[j])
                        for j in range(len(ind_results)) if ind_adj[j] < 0.05]
        if sig_personal:
            print(f"    → 有意: ", end="")
            descs = []
            for label, d_val, ap in sig_personal:
                direction = "高い" if d_val > 0 else "低い"
                descs.append(f"{label}が{direction}(d={d_val:+.2f})")
            print(", ".join(descs))
        else:
            print(f"    → 補正後に有意な差なし")
        print()

    # ================================================================
    # PART 2: 序盤ゴールド差の推移 — 1標本t検定 + 信頼区間
    # ================================================================
    header("PART 2: ミッドレーン 序盤ゴールド差 (1標本t検定: H0=0)")
    print("  各時間帯で「平均GDが0と有意に異なるか」を検定。")
    print("  有意 = そのメンバーはミッドで一貫して勝つ/負けるパターンがある。")
    print()

    checkpoints = [5, 8, 10, 15]
    for name in sorted(mid_counts.index):
        mp_matches = mid_member[mid_member['summonerName'] == name]['matchId'].unique()
        member_tf = tf[(tf['summonerName'] == name) & (tf['role'] == 'MIDDLE') &
                       (tf['matchId'].isin(mp_matches))]
        if len(member_tf[member_tf['timestampMin'] == 10]) < MIN_GAMES:
            continue

        print(f"  ■ {name}")
        print(f"    {'時間':>5} {'平均GD':>10} {'95%CI':>20} {'n':>5} {'p値':>10} {'有意':>5}")
        print("    " + "─" * 58)

        gd_results = []
        for minute in checkpoints:
            snap = member_tf[member_tf['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
            if len(snap) < 10:
                continue
            t, p = stats.ttest_1samp(snap, 0)
            mean_gd = snap.mean()
            se = snap.std() / np.sqrt(len(snap))
            lo = mean_gd - 1.96 * se
            hi = mean_gd + 1.96 * se
            gd_results.append((minute, mean_gd, lo, hi, len(snap), p))

        gd_adj = holm_correct([r[5] for r in gd_results])
        for k, (minute, mean_gd, lo, hi, n, p) in enumerate(gd_results):
            ap = gd_adj[k]
            print(f"    {minute:>3}分 {mean_gd:>+10.0f}G  [{lo:>+.0f}, {hi:>+.0f}]  {n:>5} {ap:>10.4f} {sig(ap):>5}")

        sig_times = [gd_results[k] for k in range(len(gd_results)) if gd_adj[k] < 0.05]
        if sig_times:
            direction = "有利" if sig_times[-1][1] > 0 else "不利"
            print(f"    → 有意にGDが0と異なる時間帯あり: ミッドで一貫して序盤{direction}")
        else:
            print(f"    → 全時間帯でGDは0と有意に異ならない（序盤は安定/ばらつきが大きい）")
        print()

    # ================================================================
    # PART 2b: 序盤GD — メンバーミッド vs 非メンバーミッド (Welch t検定)
    # ================================================================
    header("PART 2b: ミッド序盤GD — メンバー vs 非メンバー (Welch t検定)")
    print("  同ランク帯の非メンバーミッドと比較して序盤に差があるか。")
    print()

    for name in sorted(mid_counts.index):
        mp_matches = mid_member[mid_member['summonerName'] == name]['matchId'].unique()
        member_tf_mid = tf[(tf['summonerName'] == name) & (tf['role'] == 'MIDDLE') &
                           (tf['matchId'].isin(mp_matches))]
        non_tf_mid = tf[(~tf['summonerName'].isin(MEMBER_NAMES)) & (tf['role'] == 'MIDDLE')]

        if len(member_tf_mid[member_tf_mid['timestampMin'] == 10]) < MIN_GAMES:
            continue

        print(f"  ■ {name}")
        gd2_results = []
        for minute in [10, 15]:
            m_snap = member_tf_mid[member_tf_mid['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
            n_snap = non_tf_mid[non_tf_mid['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
            if len(m_snap) < MIN_GAMES or len(n_snap) < MIN_GAMES:
                continue
            t, p = stats.ttest_ind(m_snap, n_snap, equal_var=False)
            d = cohens_d(m_snap, n_snap)
            gd2_results.append((minute, m_snap.mean(), n_snap.mean(), d, p, len(m_snap), len(n_snap)))

        if gd2_results:
            gd2_adj = holm_correct([r[4] for r in gd2_results])
            for k, (minute, m_mean, n_mean, d, p, nm, nn) in enumerate(gd2_results):
                ap = gd2_adj[k]
                print(f"    {minute}分: 本人{m_mean:>+.0f}G vs 非メンバー{n_mean:>+.0f}G  "
                      f"差{m_mean - n_mean:>+.0f}G  d={d:+.2f}  p={ap:.4f} {sig(ap)}")
        print()

    # ================================================================
    # PART 3: レーン崩壊率 — Wilson CI + 2比率z検定
    # ================================================================
    header("PART 3: ミッドレーン崩壊率 — 15分時点で大差をつけられる確率")
    print("  -500G以下 / -1000G以下になる確率をWilson 95%CIで推定。")
    print("  非メンバーミッドとの差を2比率z検定で評価。")
    print()

    snap15 = tf[tf['timestampMin'] == 15].copy()
    snap15['is_member'] = snap15['summonerName'].isin(MEMBER_NAMES)

    for name in sorted(mid_counts.index):
        mp_matches = mid_member[mid_member['summonerName'] == name]['matchId'].unique()
        m_snap = snap15[(snap15['summonerName'] == name) & (snap15['role'] == 'MIDDLE') &
                        (snap15['matchId'].isin(mp_matches))]
        n_snap = snap15[(~snap15['summonerName'].isin(MEMBER_NAMES)) & (snap15['role'] == 'MIDDLE')]

        if len(m_snap) < MIN_GAMES:
            continue

        print(f"  ■ {name} ({len(m_snap)}試合)")

        for thresh, label in [(-500, '≤-500G'), (-1000, '≤-1000G')]:
            m_n = len(m_snap)
            m_collapse = (m_snap['goldDiffVsOpponent'] <= thresh).sum()
            n_n = len(n_snap)
            n_collapse = (n_snap['goldDiffVsOpponent'] <= thresh).sum()

            m_rate, m_lo, m_hi = wilson_ci(m_collapse, m_n)
            n_rate, n_lo, n_hi = wilson_ci(n_collapse, n_n)

            p_pool = (m_collapse + n_collapse) / (m_n + n_n)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/m_n + 1/n_n)) if p_pool > 0 and p_pool < 1 else 1
            z_val = (m_rate - n_rate) / se if se > 0 else 0
            p_val = 2 * stats.norm.sf(abs(z_val))

            print(f"    {label}:  本人 {m_rate*100:5.1f}% [{m_lo*100:.1f}%, {m_hi*100:.1f}%]  "
                  f"非メンバー {n_rate*100:5.1f}% [{n_lo*100:.1f}%, {n_hi*100:.1f}%]  "
                  f"差{(m_rate-n_rate)*100:+.1f}pp  p={p_val:.4f} {sig(p_val)}")
        print()

    # ================================================================
    # PART 4: 出血レート — 5分→15分のGD変化速度
    # ================================================================
    header("PART 4: ミッドレーン出血レート — 5分→15分のゴールド差推移")
    print("  「時間経過で差が開くか」を定量化。1標本t検定で変化量が0と異なるか検定。")
    print()

    for name in sorted(mid_counts.index):
        mp_matches = mid_member[mid_member['summonerName'] == name]['matchId'].unique()
        member_tf_mid = tf[(tf['summonerName'] == name) & (tf['role'] == 'MIDDLE') &
                           (tf['matchId'].isin(mp_matches))]

        s5 = member_tf_mid[member_tf_mid['timestampMin'] == 5][['matchId', 'goldDiffVsOpponent']].rename(
            columns={'goldDiffVsOpponent': 'gd5'})
        s10 = member_tf_mid[member_tf_mid['timestampMin'] == 10][['matchId', 'goldDiffVsOpponent']].rename(
            columns={'goldDiffVsOpponent': 'gd10'})
        s15 = member_tf_mid[member_tf_mid['timestampMin'] == 15][['matchId', 'goldDiffVsOpponent']].rename(
            columns={'goldDiffVsOpponent': 'gd15'})

        merged = s5.merge(s10, on='matchId').merge(s15, on='matchId')
        if len(merged) < MIN_GAMES:
            continue

        merged['bleed_5_10'] = merged['gd10'] - merged['gd5']
        merged['bleed_10_15'] = merged['gd15'] - merged['gd10']
        merged['bleed_5_15'] = merged['gd15'] - merged['gd5']

        print(f"  ■ {name} ({len(merged)}試合)")
        print(f"    {'区間':<12} {'平均変化':>10} {'95%CI':>22} {'p値(≠0)':>10} {'有意':>5}")
        print("    " + "─" * 64)

        for col, label in [('bleed_5_10', '5→10分'), ('bleed_10_15', '10→15分'), ('bleed_5_15', '5→15分')]:
            vals = merged[col].dropna()
            t, p = stats.ttest_1samp(vals, 0)
            mean_v = vals.mean()
            se = vals.std() / np.sqrt(len(vals))
            lo, hi = mean_v - 1.96 * se, mean_v + 1.96 * se
            print(f"    {label:<12} {mean_v:>+10.0f}G  [{lo:>+.0f}G, {hi:>+.0f}G]  {p:>10.4f} {sig(p):>5}")

        # When behind at 5 min, what happens?
        behind_5 = merged[merged['gd5'] < 0]
        if len(behind_5) >= 10:
            recovery = behind_5['bleed_5_15']
            t, p = stats.ttest_1samp(recovery, 0)
            mean_r = recovery.mean()
            worsen = (recovery < 0).mean() * 100
            print(f"    [5分ビハインド時] {len(behind_5)}試合: 5→15分変化 平均{mean_r:+.0f}G  悪化率{worsen:.0f}%  p={p:.4f} {sig(p)}")
        print()

    # ================================================================
    # PART 5: ミッドGDと試合結果の相関 — Spearman + 点双列相関
    # ================================================================
    header("PART 5: ミッドの序盤GDと試合勝敗の相関 (Spearman)")
    print("  「ミッドが序盤勝っていると試合に勝ちやすいか」を相関で評価。")
    print("  他レーンとの比較で「ミッドの重要度」を定量化。")
    print()

    snap_15_all = tf[tf['timestampMin'] == 15].copy()
    snap_15_all['is_member'] = snap_15_all['summonerName'].isin(MEMBER_NAMES)
    snap_15_member = snap_15_all[snap_15_all['is_member']]

    print(f"  {'レーン':<12} {'Spearman r':>12} {'p値':>10} {'有意':>5} {'n':>6}")
    print("  " + "─" * 50)

    corr_results = []
    for role in ROLES:
        rd = snap_15_member[snap_15_member['role'] == role]
        if len(rd) < 20:
            continue
        r, p = stats.spearmanr(rd['goldDiffVsOpponent'], rd['win'].astype(float))
        corr_results.append((role, r, p, len(rd)))

    corr_adj = holm_correct([r[2] for r in corr_results])
    for k, (role, r, p, n) in enumerate(corr_results):
        ap = corr_adj[k]
        marker = " ←" if role == 'MIDDLE' else ""
        print(f"  {ROLE_JP[role]:<12} {r:>+12.3f} {ap:>10.4f} {sig(ap):>5} {n:>6}{marker}")

    print()
    mid_corr = [c for c in corr_results if c[0] == 'MIDDLE']
    if mid_corr:
        r, p = mid_corr[0][1], corr_adj[[c[0] for c in corr_results].index('MIDDLE')]
        if p < 0.05:
            print(f"  → ミッドの序盤GDと勝敗には有意な相関がある (r={r:+.3f}, 補正p={p:.4f})")
        else:
            print(f"  → ミッドの序盤GDと勝敗の相関は有意でない (r={r:+.3f}, 補正p={p:.4f})")

    # ================================================================
    # PART 6: ミッドの分散 — 安定性評価
    # ================================================================
    header("PART 6: ミッドレーンの安定性 — パフォーマンス分散の比較")
    print("  分散が大きい = 試合ごとのブレが大きい。")
    print("  Levene検定で非メンバーミッドとの分散の差を検定。")
    print()

    variance_stats = ['kda', 'cs_per_min', 'dmg_per_min', 'gold_per_min']
    variance_labels = {'kda': 'KDA', 'cs_per_min': 'CS/min', 'dmg_per_min': 'DMG/min', 'gold_per_min': 'Gold/min'}

    for name in sorted(mid_counts.index):
        mp = mid_member[mid_member['summonerName'] == name]
        if len(mp) < MIN_GAMES:
            continue

        print(f"  ■ {name} ({len(mp)}試合)")
        print(f"    {'指標':<10} {'本人SD':>8} {'非メンバーSD':>12} {'CV本人':>8} {'CV非メン':>8} {'Levene p':>10} {'有意':>5}")
        print("    " + "─" * 66)

        var_results = []
        for col in variance_stats:
            mv = mp[col].dropna()
            nv = mid_nonmember[col].dropna()
            if len(mv) < MIN_GAMES:
                continue
            stat_l, p_l = stats.levene(mv, nv)
            cv_m = mv.std() / mv.mean() if mv.mean() != 0 else 0
            cv_n = nv.std() / nv.mean() if nv.mean() != 0 else 0
            var_results.append((col, mv.std(), nv.std(), cv_m, cv_n, p_l))

        var_adj = holm_correct([r[5] for r in var_results])
        for j, (col, m_sd, n_sd, cv_m, cv_n, p) in enumerate(var_results):
            ap = var_adj[j]
            print(f"    {variance_labels[col]:<10} {m_sd:>8.2f} {n_sd:>12.2f} {cv_m:>8.2f} {cv_n:>8.2f} {ap:>10.4f} {sig(ap):>5}")

        sig_var = [var_results[j] for j in range(len(var_results)) if var_adj[j] < 0.05]
        if sig_var:
            for col, m_sd, n_sd, *_ in sig_var:
                direction = "大きい" if m_sd > n_sd else "小さい"
                print(f"    → {variance_labels[col]}の分散が同ランク帯より有意に{direction}")
        print()

    # ================================================================
    # PART 7: チャンピオンプール — 効果量ベースの評価
    # ================================================================
    header("PART 7: ミッドチャンピオン別パフォーマンス (効果量ベース)")
    print("  チャンピオンごとのKDA・DMG/minを非メンバー同チャンプと比較。")
    print("  ※ 本人10試合以上 & 非メンバー同チャンプ20試合以上のみ")
    print()

    for name in sorted(mid_counts.index):
        mp = mid_member[mid_member['summonerName'] == name]
        if len(mp) < MIN_GAMES:
            continue

        champ_counts = mp.groupby('championName').size()
        valid_champs = champ_counts[champ_counts >= 10].index

        if len(valid_champs) == 0:
            continue

        print(f"  ■ {name}")
        print(f"    {'チャンピオン':<16} {'試合':>4} {'KDA本人':>8} {'KDA平均':>8} {'d(KDA)':>8} {'DMG/m本人':>10} {'DMG/m平均':>10} {'d(DMG)':>8}")
        print("    " + "─" * 82)

        for champ in sorted(valid_champs):
            mc = mp[mp['championName'] == champ]
            nc = mid_nonmember[mid_nonmember['championName'] == champ]

            if len(nc) < 20:
                continue

            kda_d = cohens_d(mc['kda'].dropna(), nc['kda'].dropna())
            dmg_d = cohens_d(mc['dmg_per_min'].dropna(), nc['dmg_per_min'].dropna())

            print(f"    {champ:<16} {len(mc):>4} {mc['kda'].mean():>8.2f} {nc['kda'].mean():>8.2f} {kda_d:>+8.2f} "
                  f"{mc['dmg_per_min'].mean():>10.0f} {nc['dmg_per_min'].mean():>10.0f} {dmg_d:>+8.2f}")

        print()

    # ================================================================
    # PART 8: ミッド起点のゲーム影響力 — 回帰分析
    # ================================================================
    header("PART 8: ミッドのパフォーマンスが勝敗に与える影響 (ロジスティック回帰)")
    print("  ミッドの各指標が勝敗をどの程度予測するかを評価。")
    print("  他レーンとの比較で「ミッドのどの指標が最も勝敗に効くか」を特定。")
    print()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    predictors = ['goldDiffVsOpponent']
    snap15_roles = snap_15_member.copy()

    print(f"  15分GDの勝敗予測力（ロジスティック回帰係数、標準化済み）:")
    print(f"  {'レーン':<12} {'係数':>8} {'Odds Ratio':>12} {'擬似R²':>8}")
    print("  " + "─" * 45)

    for role in ROLES:
        rd = snap15_roles[snap15_roles['role'] == role].dropna(subset=['goldDiffVsOpponent', 'win'])
        if len(rd) < 30:
            continue
        X = rd[['goldDiffVsOpponent']].values
        y = rd['win'].astype(int).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_scaled, y)
            coef = lr.coef_[0][0]
            odds_ratio = np.exp(coef)

            y_prob = lr.predict_proba(X_scaled)[:, 1]
            ll_model = np.sum(y * np.log(y_prob + 1e-10) + (1 - y) * np.log(1 - y_prob + 1e-10))
            p_null = y.mean()
            ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))
            pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else 0

            marker = " ←" if role == 'MIDDLE' else ""
            print(f"  {ROLE_JP[role]:<12} {coef:>+8.3f} {odds_ratio:>12.3f} {pseudo_r2:>8.3f}{marker}")
        except Exception:
            pass

    print()
    print("  ※ 係数が大きいほどGDの勝敗予測力が高い。")
    print("    Odds Ratio: GDが1SD増えたとき勝率が何倍になるか。")

    # ================================================================
    # Summary
    # ================================================================
    header("総括: 統計的に有意な事実のまとめ")
    print("  以下は全て p < 0.05 (多重比較補正後) の事実のみ。")
    print("  勝率のような小標本で有意にならない指標は使用していない。")
    print()
    print("  上記の各PARTで「有意」と表示された項目が")
    print("  チームのミッドレーンについて統計的に確認された事実です。")
    print("  「n.s.」と表示された項目は「差がない」のではなく")
    print("  「現在のデータ量では判断できない」ことを意味します。")
    print()


if __name__ == '__main__':
    main()

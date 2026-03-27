"""ミッドレーン チャンピオン別深掘り分析 — 統計的に有意な事実のみ.

各メンバーのミッドで一定試合数以上使われたチャンピオンについて:
- 同チャンプの非メンバーミッドとの比較 (Welch t検定 + Cohen's d)
- 本人の全ミッド平均との比較 (対応のないt検定)
- チャンピオン別の序盤GD推移 (1標本t検定)
- チャンピオン別の出血・崩壊パターン
- 全て Holm法で多重比較補正

7試合以上のチャンピオンを分析対象とし、
非メンバー比較は同チャンプ15試合以上ある場合のみ実施。
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

CHAMP_MIN = 7       # player minimum games on champion
BENCH_MIN = 15      # non-member minimum games on same champion for benchmark


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def cohens_d_one(a, mu):
    if len(a) < 2 or a.std() == 0:
        return 0.0
    return (a.mean() - mu) / a.std()


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def holm_correct(pvals):
    n = len(pvals)
    if n == 0:
        return np.array([])
    pvals = np.array(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (n - i))
    for i in range(1, n):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted


def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - spread), min(1, center + spread)


def header(title, char="=", width=80):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)
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
    ps['dmg_taken_per_min'] = ps['totalDamageTaken'] / ps['durMin'].clip(lower=1)

    team_kills = ps.groupby(['matchId', 'teamId'])['kills'].transform('sum')
    ps['kp'] = (ps['kills'] + ps['assists']) / team_kills.replace(0, 1)

    team_dmg = ps.groupby(['matchId', 'teamId'])['totalDamageDealtToChampions'].transform('sum')
    ps['dmg_share'] = ps['totalDamageDealtToChampions'] / team_dmg.replace(0, 1)

    return ps, tf


def analyze_champion(name, champ, mp_champ, mp_all_mid, nm_champ, tf_champ, tf_nm_champ,
                     champ_idx, total_champs):
    """Analyze a single champion for a single player. Returns list of significant findings."""
    n_games = len(mp_champ)
    findings = []

    print(f"  ── {champ} ({n_games}試合) ──")
    print()

    # ----------------------------------------------------------------
    # A) Performance vs non-member same champ (if enough data)
    # ----------------------------------------------------------------
    bench_available = len(nm_champ) >= BENCH_MIN

    stat_cols = [
        ('kills',          'キル',       False),
        ('deaths',         'デス',       True),
        ('assists',        'アシスト',    False),
        ('kda',            'KDA',        False),
        ('cs_per_min',     'CS/min',     False),
        ('gold_per_min',   'Gold/min',   False),
        ('dmg_per_min',    'DMG/min',    False),
        ('dmg_taken_per_min', '被DMG/min', True),
        ('kp',             'KP',         False),
        ('visionScore',    'ビジョン',    False),
        ('dmg_share',      'ダメ割合',    False),
    ]

    if bench_available:
        print(f"    [A] vs 非メンバー同チャンプミッド ({len(nm_champ)}試合)")
        print(f"    {'指標':<12} {'本人':>8} {'非メンバー':>10} {'差':>8} {'d':>7} {'補正p':>8} {'有意':>5}")
        print("    " + "─" * 64)

        bench_results = []
        for col, label, invert in stat_cols:
            mv = mp_champ[col].dropna()
            nv = nm_champ[col].dropna()
            if len(mv) < 3 or len(nv) < 5:
                continue
            t, p = stats.ttest_ind(mv, nv, equal_var=False)
            d = cohens_d(mv, nv)
            bench_results.append((label, mv.mean(), nv.mean(), d, p, invert))

        if bench_results:
            bench_adj = holm_correct([r[4] for r in bench_results])
            for j, (label, mv, nv, d, p, invert) in enumerate(bench_results):
                ap = bench_adj[j]
                print(f"    {label:<12} {mv:>8.2f} {nv:>10.2f} {mv - nv:>+8.2f} {d:>+7.2f} {ap:>8.4f} {sig(ap):>5}")
                if ap < 0.05:
                    direction = ("低い" if d > 0 else "高い") if invert else ("高い" if d > 0 else "低い")
                    size = "大" if abs(d) >= 0.8 else "中" if abs(d) >= 0.5 else "小"
                    findings.append(f"{champ}/{label}: 非メンバーより{direction} (d={d:+.2f}, {size}効果量)")

            sig_bench = [(bench_results[j], bench_adj[j]) for j in range(len(bench_results)) if bench_adj[j] < 0.05]
            if sig_bench:
                print(f"    → 有意な差: {len(sig_bench)}項目")
            else:
                print(f"    → 補正後に有意な差なし")
        print()

    # ----------------------------------------------------------------
    # B) Performance vs own all-mid average
    # ----------------------------------------------------------------
    other_mid = mp_all_mid[mp_all_mid['championName'] != champ]
    if len(other_mid) >= 10:
        print(f"    [B] vs 本人の他チャンプミッド平均 ({len(other_mid)}試合)")
        print(f"    {'指標':<12} {'本チャンプ':>10} {'他チャンプ':>10} {'差':>8} {'d':>7} {'補正p':>8} {'有意':>5}")
        print("    " + "─" * 66)

        self_results = []
        for col, label, invert in stat_cols:
            mv = mp_champ[col].dropna()
            ov = other_mid[col].dropna()
            if len(mv) < 3 or len(ov) < 5:
                continue
            t, p = stats.ttest_ind(mv, ov, equal_var=False)
            d = cohens_d(mv, ov)
            self_results.append((label, mv.mean(), ov.mean(), d, p, invert))

        if self_results:
            self_adj = holm_correct([r[4] for r in self_results])
            for j, (label, mv, ov, d, p, invert) in enumerate(self_results):
                ap = self_adj[j]
                print(f"    {label:<12} {mv:>10.2f} {ov:>10.2f} {mv - ov:>+8.2f} {d:>+7.2f} {ap:>8.4f} {sig(ap):>5}")
                if ap < 0.05:
                    direction = ("低い" if d > 0 else "高い") if invert else ("高い" if d > 0 else "低い")
                    findings.append(f"{champ}/{label}: 自分の他チャンプ平均より{direction} (d={d:+.2f})")

            sig_self = [(self_results[j], self_adj[j]) for j in range(len(self_results)) if self_adj[j] < 0.05]
            if sig_self:
                print(f"    → 有意な差: {len(sig_self)}項目")
            else:
                print(f"    → 補正後に有意な差なし")
        print()

    # ----------------------------------------------------------------
    # C) Early game GD trajectory for this champion
    # ----------------------------------------------------------------
    print(f"    [C] 序盤GD推移 (1標本t検定: H0 = 0)")

    checkpoints = [5, 8, 10, 15]
    gd_results = []
    for minute in checkpoints:
        snap = tf_champ[tf_champ['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
        if len(snap) < 5:
            continue
        t, p = stats.ttest_1samp(snap, 0)
        mean_gd = snap.mean()
        se = snap.std() / np.sqrt(len(snap))
        lo, hi = mean_gd - 1.96 * se, mean_gd + 1.96 * se
        gd_results.append((minute, mean_gd, lo, hi, len(snap), p))

    if gd_results:
        gd_adj = holm_correct([r[5] for r in gd_results])
        print(f"    {'時間':>5} {'平均GD':>10} {'95%CI':>22} {'n':>4} {'補正p':>8} {'有意':>5}")
        print("    " + "─" * 58)
        for k, (minute, mg, lo, hi, n, p) in enumerate(gd_results):
            ap = gd_adj[k]
            print(f"    {minute:>3}分 {mg:>+10.0f}G  [{lo:>+.0f}, {hi:>+.0f}]  {n:>4} {ap:>8.4f} {sig(ap):>5}")
            if ap < 0.05:
                direction = "有利" if mg > 0 else "不利"
                findings.append(f"{champ}/{minute}分GD: 有意に{direction} ({mg:+.0f}G)")

        sig_gd = [gd_results[k] for k in range(len(gd_results)) if gd_adj[k] < 0.05]
        if not sig_gd:
            print(f"    → 補正後に有意なGD偏りなし")
    else:
        print(f"    → タイムラインデータ不足")
    print()

    # ----------------------------------------------------------------
    # D) vs non-member same champ early GD comparison
    # ----------------------------------------------------------------
    if len(tf_nm_champ) > 0:
        print(f"    [D] 序盤GD — vs 非メンバー同チャンプ比較 (Welch t)")

        gd_bench = []
        for minute in [10, 15]:
            m_snap = tf_champ[tf_champ['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
            n_snap = tf_nm_champ[tf_nm_champ['timestampMin'] == minute]['goldDiffVsOpponent'].dropna()
            if len(m_snap) < 5 or len(n_snap) < 10:
                continue
            t, p = stats.ttest_ind(m_snap, n_snap, equal_var=False)
            d = cohens_d(m_snap, n_snap)
            gd_bench.append((minute, m_snap.mean(), n_snap.mean(), d, p, len(m_snap), len(n_snap)))

        if gd_bench:
            gd_bench_adj = holm_correct([r[4] for r in gd_bench])
            for k, (minute, mm, nm, d, p, mn, nn) in enumerate(gd_bench):
                ap = gd_bench_adj[k]
                print(f"    {minute}分: 本人{mm:>+.0f}G vs 非メンバー{nm:>+.0f}G  "
                      f"差{mm - nm:>+.0f}G  d={d:+.2f}  p={ap:.4f} {sig(ap)}")
                if ap < 0.05:
                    direction = "上回る" if d > 0 else "下回る"
                    findings.append(f"{champ}/{minute}分GD: 非メンバー同チャンプを{direction} (d={d:+.2f})")
        else:
            print(f"    → 比較データ不足")
        print()

    # ----------------------------------------------------------------
    # E) Bleed pattern: 5→15 min change
    # ----------------------------------------------------------------
    s5 = tf_champ[tf_champ['timestampMin'] == 5][['matchId', 'goldDiffVsOpponent']].rename(
        columns={'goldDiffVsOpponent': 'gd5'})
    s15 = tf_champ[tf_champ['timestampMin'] == 15][['matchId', 'goldDiffVsOpponent']].rename(
        columns={'goldDiffVsOpponent': 'gd15'})
    merged = s5.merge(s15, on='matchId')

    if len(merged) >= 5:
        merged['bleed'] = merged['gd15'] - merged['gd5']
        bleed_vals = merged['bleed']
        t, p = stats.ttest_1samp(bleed_vals, 0)
        mean_b = bleed_vals.mean()
        se = bleed_vals.std() / np.sqrt(len(bleed_vals))
        lo, hi = mean_b - 1.96 * se, mean_b + 1.96 * se

        print(f"    [E] 出血レート (5→15分GD変化)")
        print(f"    平均変化: {mean_b:+.0f}G  95%CI [{lo:+.0f}, {hi:+.0f}]  n={len(merged)}  p={p:.4f} {sig(p)}")

        if p < 0.05:
            direction = "広がる" if mean_b < 0 else "改善する"
            findings.append(f"{champ}/出血: 5→15分で差が{direction} ({mean_b:+.0f}G, p={p:.4f})")

        # When behind at 5min
        behind = merged[merged['gd5'] < 0]
        if len(behind) >= 5:
            behind_bleed = behind['bleed']
            t2, p2 = stats.ttest_1samp(behind_bleed, 0)
            worsen = (behind_bleed < 0).mean() * 100
            print(f"    [5分ビハインド時] {len(behind)}試合: 変化 {behind_bleed.mean():+.0f}G  悪化率{worsen:.0f}%  p={p2:.4f} {sig(p2)}")
            if p2 < 0.05 and behind_bleed.mean() < 0:
                findings.append(f"{champ}/ビハインド回復: 5分で負けるとさらに悪化 (悪化率{worsen:.0f}%)")
        print()

    # ----------------------------------------------------------------
    # F) Collapse rate for this champion
    # ----------------------------------------------------------------
    snap15_champ = tf_champ[tf_champ['timestampMin'] == 15]['goldDiffVsOpponent'].dropna()
    if len(snap15_champ) >= 5:
        n_total = len(snap15_champ)
        n_500 = (snap15_champ <= -500).sum()
        n_1000 = (snap15_champ <= -1000).sum()
        r500, lo500, hi500 = wilson_ci(n_500, n_total)
        r1000, lo1000, hi1000 = wilson_ci(n_1000, n_total)

        print(f"    [F] 崩壊率 (15分時点)")
        print(f"    ≤-500G:  {r500*100:5.1f}%  95%CI [{lo500*100:.1f}%, {hi500*100:.1f}%]  ({n_500}/{n_total})")
        print(f"    ≤-1000G: {r1000*100:5.1f}%  95%CI [{lo1000*100:.1f}%, {hi1000*100:.1f}%]  ({n_1000}/{n_total})")

        # Compare vs non-member same champ
        nm_snap15 = tf_nm_champ[tf_nm_champ['timestampMin'] == 15]['goldDiffVsOpponent'].dropna()
        if len(nm_snap15) >= BENCH_MIN:
            nm_n = len(nm_snap15)
            nm_500 = (nm_snap15 <= -500).sum()
            nm_1000 = (nm_snap15 <= -1000).sum()

            for thresh_n, nm_thresh_n, nm_total, label in [
                (n_500, nm_500, nm_n, '≤-500G'),
                (n_1000, nm_1000, nm_n, '≤-1000G'),
            ]:
                p_pool = (thresh_n + nm_thresh_n) / (n_total + nm_total)
                if 0 < p_pool < 1:
                    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_total + 1/nm_total))
                    z = ((thresh_n/n_total) - (nm_thresh_n/nm_total)) / se if se > 0 else 0
                    pv = 2 * stats.norm.sf(abs(z))
                    diff_pp = (thresh_n/n_total - nm_thresh_n/nm_total) * 100
                    if pv < 0.05:
                        direction = "高い" if diff_pp > 0 else "低い"
                        print(f"    {label}: 非メンバー比 {diff_pp:+.1f}pp (p={pv:.4f} {sig(pv)}) → 崩壊率が有意に{direction}")
                        findings.append(f"{champ}/崩壊率{label}: 非メンバーより{direction} ({diff_pp:+.1f}pp)")

        print()

    return findings


def main():
    ps, tf_raw = load()

    mid_member = ps[(ps['is_member']) & (ps['role'] == 'MIDDLE')]
    mid_nonmember = ps[(~ps['is_member']) & (ps['role'] == 'MIDDLE')]

    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║         ミッドレーン チャンピオン別深掘り分析                                  ║")
    print("║  — 各チャンピオンの統計的に有意な特徴のみ報告 —                                ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  分析基準: メンバーが{CHAMP_MIN}試合以上使用したチャンピオン")
    print(f"  ベンチマーク比較: 非メンバーが同チャンプを{BENCH_MIN}試合以上使用している場合のみ")
    print(f"  全検定に Holm法多重比較補正を適用")

    mid_counts = mid_member.groupby('summonerName').size().sort_values(ascending=False)
    all_findings = {}

    for name in mid_counts.index:
        mp_all_mid = mid_member[mid_member['summonerName'] == name]
        if len(mp_all_mid) < 15:
            continue

        champ_counts = mp_all_mid.groupby('championName').size().sort_values(ascending=False)
        valid_champs = champ_counts[champ_counts >= CHAMP_MIN]

        if len(valid_champs) == 0:
            continue

        header(f"{name} — ミッド {len(mp_all_mid)}試合, {len(valid_champs)}チャンプ分析対象")

        print(f"  チャンピオンプール概要:")
        for champ, cnt in valid_champs.items():
            pct = cnt / len(mp_all_mid) * 100
            print(f"    {champ:<16} {cnt:>3}試合  ({pct:5.1f}%)")
        print()

        player_findings = []

        for idx, (champ, cnt) in enumerate(valid_champs.items()):
            mp_champ = mp_all_mid[mp_all_mid['championName'] == champ]
            nm_champ = mid_nonmember[mid_nonmember['championName'] == champ]

            mp_match_ids = mp_champ['matchId'].unique()
            tf_champ = tf_raw[
                (tf_raw['summonerName'] == name) &
                (tf_raw['role'] == 'MIDDLE') &
                (tf_raw['matchId'].isin(mp_match_ids))
            ]
            tf_nm_champ = tf_raw[
                (~tf_raw['summonerName'].isin(MEMBER_NAMES)) &
                (tf_raw['role'] == 'MIDDLE') &
                (tf_raw['championName'] == champ)
            ]

            champ_findings = analyze_champion(
                name, champ, mp_champ, mp_all_mid, nm_champ,
                tf_champ, tf_nm_champ, idx, len(valid_champs)
            )
            player_findings.extend(champ_findings)

        all_findings[name] = player_findings

        # Per-player summary
        print("  " + "─" * 76)
        print(f"  ■ {name} まとめ — 有意な事実:")
        if player_findings:
            for f in player_findings:
                print(f"    ✓ {f}")
        else:
            print(f"    → 補正後に有意な差のあるチャンピオン特有パターンはなし")
        print()

    # ================================================================
    # Cross-player champion comparison
    # ================================================================
    header("クロスプレイヤー: 同チャンピオンを複数メンバーが使用した場合の比較")

    champ_player_map = {}
    for name in mid_counts.index:
        mp = mid_member[mid_member['summonerName'] == name]
        if len(mp) < 15:
            continue
        for champ in mp['championName'].unique():
            cnt = len(mp[mp['championName'] == champ])
            if cnt >= CHAMP_MIN:
                champ_player_map.setdefault(champ, []).append((name, cnt))

    shared_champs = {c: players for c, players in champ_player_map.items() if len(players) >= 2}

    if shared_champs:
        for champ, players in sorted(shared_champs.items()):
            print(f"  ── {champ} ──")
            player_names = [p[0] for p in players]

            key_cols = [
                ('kda',        'KDA',     False),
                ('cs_per_min', 'CS/min',  False),
                ('dmg_per_min','DMG/min', False),
                ('gold_per_min','Gold/min',False),
                ('deaths',     'デス',    True),
            ]

            print(f"    {'プレイヤー':<16} {'試合':>4}", end="")
            for _, label, _ in key_cols:
                print(f" {label:>8}", end="")
            print()
            print("    " + "─" * 60)

            player_data = {}
            for pname, cnt in players:
                pdata = mid_member[(mid_member['summonerName'] == pname) & (mid_member['championName'] == champ)]
                player_data[pname] = pdata
                line = f"    {pname:<16} {cnt:>4}"
                for col, _, _ in key_cols:
                    line += f" {pdata[col].mean():>8.2f}"
                print(line)

            # Pairwise comparison if exactly 2 players
            if len(players) == 2:
                p1, p2 = player_names
                print()
                print(f"    {p1} vs {p2}:")
                cross_results = []
                for col, label, invert in key_cols:
                    v1 = player_data[p1][col].dropna()
                    v2 = player_data[p2][col].dropna()
                    if len(v1) < 5 or len(v2) < 5:
                        continue
                    t, p = stats.ttest_ind(v1, v2, equal_var=False)
                    d = cohens_d(v1, v2)
                    cross_results.append((label, v1.mean(), v2.mean(), d, p))

                if cross_results:
                    cross_adj = holm_correct([r[4] for r in cross_results])
                    for j, (label, m1, m2, d, p) in enumerate(cross_results):
                        ap = cross_adj[j]
                        if ap < 0.05:
                            better = p1 if d > 0 else p2
                            print(f"      {label}: {p1} {m1:.2f} vs {p2} {m2:.2f}  d={d:+.2f}  p={ap:.4f} {sig(ap)} → {better}が有意に上")
                    if all(cross_adj[j] >= 0.05 for j in range(len(cross_results))):
                        print(f"      → 補正後に有意な差なし")
            print()
    else:
        print("  → 複数メンバーが{CHAMP_MIN}試合以上使ったミッドチャンピオンなし")

    # ================================================================
    # Final summary
    # ================================================================
    header("総合サマリー: 全メンバーの有意なチャンピオン特性")

    any_findings = False
    for name, findings in all_findings.items():
        if findings:
            any_findings = True
            print(f"  ■ {name}:")
            for f in findings:
                print(f"    ✓ {f}")
            print()

    if not any_findings:
        print("  → 全メンバーで、補正後に有意なチャンピオン特有パターンは検出されなかった。")
        print("    各チャンピオン7試合程度ではサンプル不足の可能性が高い。")
    print()


if __name__ == '__main__':
    main()

"""Carry vs Weak-side analysis.

Who converts resources into wins? Who functions without resources?

Metrics:
1. Gold efficiency: Damage per Gold (DPG)
2. Snowball ability: Win rate when ahead at 15min
3. Comeback ability: Win rate when behind at 15min
4. Resource independence: Performance delta (ahead vs behind)
5. Carry correlation: correlation between individual gold lead and team win
6. Weak-side survival: deaths when behind, KP when behind
7. Statistical tests on all key comparisons
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


def wilson_ci(s, n, alpha=0.05):
    if n == 0:
        return 0, 0, 1
    z = stats.norm.ppf(1 - alpha / 2)
    p = s / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    w = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return p, max(0, c - w), min(1, c + w)


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def main():
    with open(CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members_set = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get('members', [])}

    ps = pd.read_csv(DATA / 'player_stats.csv')
    tf = pd.read_csv(DATA / 'timeline_frames.csv')

    ps['riotId'] = ps['summonerName'].astype(str) + '#' + ps['tagLine'].astype(str)
    ps['is_member'] = ps['riotId'].isin(members_set)
    member_names = sorted(ps.loc[ps['is_member'], 'summonerName'].unique())

    # Merge 15min GD into player_stats
    tf15 = tf[tf['timestampMin'] == 15][['matchId', 'summonerName', 'goldDiffVsOpponent']].copy()
    tf15.rename(columns={'goldDiffVsOpponent': 'gd15'}, inplace=True)
    ps = ps.merge(tf15, on=['matchId', 'summonerName'], how='left')

    mem = ps[ps['is_member']].copy()
    mem['dpg'] = mem['totalDamageDealtToChampions'] / mem['goldEarned'].clip(lower=1)
    mem['ahead15'] = mem['gd15'] > 0
    mem['behind15'] = mem['gd15'] < 0
    total_team_kills = mem.groupby('matchId')['kills'].transform('sum').clip(lower=1)
    mem['kp'] = (mem['kills'] + mem['assists']) / total_team_kills
    mem['dmg_share'] = mem['totalDamageDealtToChampions'] / \
        mem.groupby('matchId')['totalDamageDealtToChampions'].transform('sum').clip(lower=1)
    mem['gold_share'] = mem['goldEarned'] / \
        mem.groupby('matchId')['goldEarned'].transform('sum').clip(lower=1)
    mem['dmg_gold_efficiency'] = mem['dmg_share'] / mem['gold_share'].clip(lower=0.01)

    # ================================================================
    print("=" * 78)
    print("  PART 1: リソース効率 — 「ゴールドをダメージに変換する力」")
    print("=" * 78)
    print()
    print("  DPG = チャンピオンダメージ / 獲得ゴールド (高い = ゴールド効率が良い)")
    print("  DMG/Gold効率 = ダメージシェア ÷ ゴールドシェア (1.0超 = もらった以上に出す)")
    print()

    dpg_rows = []
    for name in member_names:
        sub = mem[mem['summonerName'] == name]
        n = len(sub)
        dpg_mean = sub['dpg'].mean()
        dpg_se = sub['dpg'].std() / np.sqrt(n)
        eff_mean = sub['dmg_gold_efficiency'].mean()
        eff_se = sub['dmg_gold_efficiency'].std() / np.sqrt(n)
        dmg_share = sub['dmg_share'].mean()
        gold_share = sub['gold_share'].mean()
        dpg_rows.append((name, n, dpg_mean, dpg_se, eff_mean, eff_se, dmg_share, gold_share))

    dpg_rows.sort(key=lambda x: -x[4])
    print(f"  {'メンバー':14s}  {'試合':>5s}  {'DPG':>6s}  {'DMG/G効率':>9s}  {'ダメシェア':>8s}  {'Gシェア':>7s}  差分")
    print("  " + "-" * 75)
    for name, n, dpg, dpg_se, eff, eff_se, ds, gs in dpg_rows:
        diff = ds - gs
        print(f"  {name:14s}  {n:5d}  {dpg:6.2f}  {eff:8.2f}±{eff_se:.2f}  {ds*100:7.1f}%  {gs*100:6.1f}%  {diff*100:+5.1f}pp")

    print()
    print("  ※ DMG/G効率 > 1.0: もらったゴールド以上のダメージを出している")
    print("  ※ DMG/G効率 < 1.0: もらったゴールドの割にダメージが少ない")
    print()

    # Pairwise significance for DMG/Gold efficiency
    print("  ── DMG/Gold効率の有意差ペア (Welch t検定, p < 0.05) ──")
    print()
    tested = []
    for i in range(len(dpg_rows)):
        for j in range(i + 1, len(dpg_rows)):
            n1, n2 = dpg_rows[i][0], dpg_rows[j][0]
            v1 = mem[mem['summonerName'] == n1]['dmg_gold_efficiency']
            v2 = mem[mem['summonerName'] == n2]['dmg_gold_efficiency']
            t, p = stats.ttest_ind(v1, v2, equal_var=False)
            d = (v1.mean() - v2.mean()) / np.sqrt((v1.std()**2 + v2.std()**2) / 2)
            if p < 0.05 and abs(d) >= 0.2:
                tested.append((n1, n2, v1.mean(), v2.mean(), d, p))

    if tested:
        for n1, n2, m1, m2, d, p in sorted(tested, key=lambda x: -abs(x[4])):
            print(f"    {n1} ({m1:.2f}) vs {n2} ({m2:.2f})  d={d:+.2f}  p={p:.4f} {sig(p)}")
    else:
        print("    有意差のあるペアなし")
    print()

    # ================================================================
    print("=" * 78)
    print("  PART 2: スノーボール力 vs 逆境耐性")
    print("=" * 78)
    print()
    print("  15分有利時勝率 = キャリー力 (リソースを勝利に変換)")
    print("  15分不利時勝率 = ウィークサイド耐性 (少ないリソースで耐える)")
    print()

    print(f"  {'メンバー':14s}  {'有利時WR':>9s}  {'(n)':>5s}  {'不利時WR':>9s}  {'(n)':>5s}  "
          f"{'スイング':>8s}  {'有利時p':>8s}  {'不利時p':>8s}")
    print("  " + "-" * 85)

    swing_data = []
    for name in member_names:
        sub = mem[mem['summonerName'] == name]
        ahead = sub[sub['ahead15'] == True]
        behind = sub[sub['behind15'] == True]
        if len(ahead) < 15 or len(behind) < 15:
            continue

        aw = int(ahead['win'].sum())
        an = len(ahead)
        bw = int(behind['win'].sum())
        bn = len(behind)

        awr, alo, ahi = wilson_ci(aw, an)
        bwr, blo, bhi = wilson_ci(bw, bn)

        ap = stats.binomtest(aw, an, 0.5).pvalue
        bp = stats.binomtest(bw, bn, 0.5).pvalue

        swing = awr - bwr
        swing_data.append((name, awr, an, bwr, bn, swing, ap, bp))

    swing_data.sort(key=lambda x: -x[1])
    for name, awr, an, bwr, bn, swing, ap, bp in swing_data:
        print(f"  {name:14s}  {awr*100:8.1f}%  ({an:3d})  {bwr*100:8.1f}%  ({bn:3d})  "
              f"{swing*100:+7.1f}pp  {ap:.4f}{sig(ap):>4s}  {bp:.4f}{sig(bp):>4s}")

    print()
    print("  ※ 有利時p: 有利時WRが50%と有意に異なるか")
    print("  ※ 不利時p: 不利時WRが50%と有意に異なるか")
    print()

    # ================================================================
    print("=" * 78)
    print("  PART 3: 個人GDとチーム勝敗の相関 — 「誰に投資すべきか」")
    print("=" * 78)
    print()
    print("  ポイントバイシリアル相関: 15分時点のGDと試合勝敗(0/1)の相関")
    print("  高い = この人のGDが大きいほどチームが勝つ → リソースを回す価値がある")
    print()

    corr_data = []
    for name in member_names:
        sub = mem[mem['summonerName'] == name].dropna(subset=['gd15'])
        if len(sub) < 30:
            continue
        r, p = stats.pointbiserialr(sub['win'].astype(int), sub['gd15'])
        n = len(sub)

        # Fisher's z transform for CI
        z_r = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        z_lo = z_r - 1.96 * se_z
        z_hi = z_r + 1.96 * se_z
        r_lo = np.tanh(z_lo)
        r_hi = np.tanh(z_hi)

        corr_data.append((name, r, r_lo, r_hi, p, n))

    corr_data.sort(key=lambda x: -x[1])
    print(f"  {'メンバー':14s}  {'相関r':>7s}  {'95%CI':>16s}  {'p値':>10s}  {'n':>5s}  判定")
    print("  " + "-" * 70)
    for name, r, rlo, rhi, p, n in corr_data:
        label = ""
        if p < 0.05 and r > 0.15:
            label = "← 投資価値あり"
        elif p < 0.05 and r < 0.05:
            label = "← GDと勝敗が弱い"
        print(f"  {name:14s}  {r:+6.3f}  [{rlo:+.3f}, {rhi:+.3f}]  p={p:.4f} {sig(p):>4s}  {n:5d}  {label}")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 4: 不利時のパフォーマンス — ウィークサイド適性")
    print("=" * 78)
    print()
    print("  15分GD < 0 の試合でのスタッツ")
    print()

    print(f"  {'メンバー':14s}  {'不利時デス':>8s}  {'不利時KP':>7s}  {'不利時ダメ':>9s}  {'不利時Vision':>11s}  {'不利時KDA':>8s}")
    print("  " + "-" * 72)

    ws_data = []
    for name in member_names:
        sub = mem[(mem['summonerName'] == name) & (mem['behind15'] == True)]
        if len(sub) < 20:
            continue
        d = sub['deaths'].mean()
        kp = sub['kp'].mean()
        dmg = sub['totalDamageDealtToChampions'].mean()
        vis = sub['visionScore'].mean()
        kda = sub['kda'].mean()
        ws_data.append((name, len(sub), d, kp, dmg, vis, kda))

    ws_data.sort(key=lambda x: x[3], reverse=True)
    for name, n, d, kp, dmg, vis, kda in ws_data:
        print(f"  {name:14s}  {d:8.1f}  {kp*100:6.1f}%  {dmg:9.0f}  {vis:11.1f}  {kda:8.2f}  (n={n})")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 5: 有利時のパフォーマンス — キャリー適性")
    print("=" * 78)
    print()
    print("  15分GD > 0 の試合でのスタッツ")
    print()

    print(f"  {'メンバー':14s}  {'有利時キル':>8s}  {'有利時KP':>7s}  {'有利時ダメ':>9s}  {'有利時ダメシェア':>13s}  {'有利時KDA':>8s}")
    print("  " + "-" * 72)

    cs_data = []
    for name in member_names:
        sub = mem[(mem['summonerName'] == name) & (mem['ahead15'] == True)]
        if len(sub) < 20:
            continue
        k = sub['kills'].mean()
        kp = sub['kp'].mean()
        dmg = sub['totalDamageDealtToChampions'].mean()
        ds = sub['dmg_share'].mean()
        kda = sub['kda'].mean()
        cs_data.append((name, len(sub), k, kp, dmg, ds, kda))

    cs_data.sort(key=lambda x: x[5], reverse=True)
    for name, n, k, kp, dmg, ds, kda in cs_data:
        print(f"  {name:14s}  {k:8.1f}  {kp*100:6.1f}%  {dmg:9.0f}  {ds*100:12.1f}%  {kda:8.2f}  (n={n})")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 6: キャリー/ウィークサイド総合スコア")
    print("=" * 78)
    print()
    print("  各指標をチーム内z-scoreに変換し、方向性を統合")
    print("  キャリースコア: 有利時WR + GD-勝敗相関 + DMG/G効率 + 有利時ダメシェア")
    print("  ウィークサイドスコア: 不利時WR + 不利時KDA + 不利時Vision + (1-デス率)")
    print()

    all_scores = {}

    # Collect raw values for z-scoring
    carry_raw = {}
    ws_raw = {}

    for name in member_names:
        sub = mem[mem['summonerName'] == name]
        ahead = sub[sub['ahead15'] == True]
        behind = sub[sub['behind15'] == True]
        if len(ahead) < 15 or len(behind) < 15:
            continue

        # Carry indicators
        ahead_wr = ahead['win'].mean()
        gd_corr = [x[1] for x in corr_data if x[0] == name]
        gd_corr = gd_corr[0] if gd_corr else 0
        dpg_eff = sub['dmg_gold_efficiency'].mean()
        ahead_ds = ahead['dmg_share'].mean()

        # Weakside indicators
        behind_wr = behind['win'].mean()
        behind_kda = behind['kda'].mean()
        behind_vis = behind['visionScore'].mean()
        behind_deaths = behind['deaths'].mean()

        carry_raw[name] = [ahead_wr, gd_corr, dpg_eff, ahead_ds]
        ws_raw[name] = [behind_wr, behind_kda, behind_vis, -behind_deaths]

    # Z-score within team
    names_with_data = sorted(carry_raw.keys())
    if len(names_with_data) >= 3:
        carry_arr = np.array([carry_raw[n] for n in names_with_data])
        ws_arr = np.array([ws_raw[n] for n in names_with_data])

        carry_z = (carry_arr - carry_arr.mean(axis=0)) / carry_arr.std(axis=0).clip(min=0.001)
        ws_z = (ws_arr - ws_arr.mean(axis=0)) / ws_arr.std(axis=0).clip(min=0.001)

        carry_scores = carry_z.mean(axis=1)
        ws_scores = ws_z.mean(axis=1)

        results = list(zip(names_with_data, carry_scores, ws_scores))

        print(f"  {'メンバー':14s}  {'キャリーScore':>12s}  {'WS Score':>10s}  {'適性判定':>10s}")
        print("  " + "-" * 55)

        for name, cs, ws in sorted(results, key=lambda x: -x[1]):
            if cs > 0.3 and cs > ws:
                role_label = "★ キャリー向き"
            elif ws > 0.3 and ws > cs:
                role_label = "◆ WS向き"
            elif cs > 0.3 and ws > 0.3:
                role_label = "● 万能"
            elif cs < -0.3 and ws < -0.3:
                role_label = "▽ 要改善"
            else:
                role_label = "─ 中間"
            print(f"  {name:14s}  {cs:+11.2f}   {ws:+9.2f}   {role_label}")

        print()

        # Detail breakdown
        carry_labels = ['有利時WR', 'GD-勝敗r', 'DMG/G効率', '有利時DMGシェア']
        ws_labels = ['不利時WR', '不利時KDA', '不利時Vision', '不利時生存']

        print("  ── キャリースコア内訳 (チーム内z-score) ──")
        print()
        print(f"  {'メンバー':14s}  ", end="")
        for l in carry_labels:
            print(f"{l:>12s}  ", end="")
        print("合計")
        print("  " + "-" * 75)
        for i, name in enumerate(names_with_data):
            print(f"  {name:14s}  ", end="")
            for j in range(4):
                print(f"{carry_z[i, j]:+11.2f}   ", end="")
            print(f"{carry_scores[i]:+.2f}")

        print()
        print("  ── ウィークサイドスコア内訳 ──")
        print()
        print(f"  {'メンバー':14s}  ", end="")
        for l in ws_labels:
            print(f"{l:>12s}  ", end="")
        print("合計")
        print("  " + "-" * 75)
        for i, name in enumerate(names_with_data):
            print(f"  {name:14s}  ", end="")
            for j in range(4):
                print(f"{ws_z[i, j]:+11.2f}   ", end="")
            print(f"{ws_scores[i]:+.2f}")

    print()

    # ================================================================
    print("=" * 78)
    print("  PART 7: 15分GDごとの勝率曲線 — 誰の差が最もチームに効くか")
    print("=" * 78)
    print()
    print("  GD区間ごとの勝率（各メンバー）")
    print()

    gd_bins = [(-9999, -500, '< -500G'), (-500, -100, '-500~-100G'),
               (-100, 100, '-100~+100G'), (100, 500, '+100~+500G'),
               (500, 9999, '> +500G')]

    header = f"  {'メンバー':14s}"
    for _, _, label in gd_bins:
        header += f"  {label:>12s}"
    print(header)
    print("  " + "-" * 80)

    for name in member_names:
        sub = mem[mem['summonerName'] == name].dropna(subset=['gd15'])
        if len(sub) < 30:
            continue
        row = f"  {name:14s}"
        for lo, hi, label in gd_bins:
            bucket = sub[(sub['gd15'] >= lo) & (sub['gd15'] < hi)]
            if len(bucket) >= 5:
                wr = bucket['win'].mean() * 100
                row += f"  {wr:5.0f}%({len(bucket):3d})"
            else:
                row += f"     -  ({len(bucket):3d})"
        print(row)

    print()
    print("=" * 78)
    print("  分析完了")
    print("=" * 78)


if __name__ == '__main__':
    main()

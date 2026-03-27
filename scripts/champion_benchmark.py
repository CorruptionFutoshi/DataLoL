"""Champion-specific benchmark: compare members vs Emerald/Diamond-rank players on the SAME champion.

Eliminates champion bias — e.g. "PlayerX wins lane because Jinx is strong"
becomes "PlayerX on Jinx vs Emerald/Diamond Jinx average."

Role-aware: compares member's champion-in-role against benchmark's same champion-in-role.
Falls back to all-role comparison when role-specific data is insufficient.

Requires: benchmark_stats.csv (run collect_benchmark.py + process.py first)

Usage:
    python scripts/champion_benchmark.py                # full team report
    python scripts/champion_benchmark.py --member X     # deep dive on one member
"""

import argparse
import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
MEMBER_NAMES = {m["game_name"] for m in cfg.get("members", [])}

ROLE_JP = {
    "TOP": "トップ", "JUNGLE": "ジャングル", "MIDDLE": "ミッド",
    "BOTTOM": "ボトム", "UTILITY": "サポート",
}
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

MIN_BENCHMARK_GAMES = 10
MIN_BENCHMARK_ROLE_GAMES = 8
SMALL_SAMPLE_WARN = 30
MIN_MEMBER_GAMES = 5

COMPARE_STATS = [
    ("kda",           "KDA",       "raw",  False),
    ("cs_per_min",    "CS/m",      "raw",  False),
    ("gold_per_min",  "Gold/m",    "raw",  False),
    ("dmg_per_min",   "DMG/m",     "raw",  False),
    ("visionScore",   "Vision",    "raw",  False),
    ("deaths",        "Deaths",    "raw",  True),
    ("kp",            "KP",        "pct",  False),
    ("dmg_share",     "DMG%",      "pct",  False),
]


def _add_duration(df: pd.DataFrame) -> pd.Series:
    """Get gameDurationMin, joining with matches.csv if needed."""
    if "gameDurationMin" in df.columns:
        return df["gameDurationMin"]
    matches = pd.read_csv(DATA / "matches.csv")
    dur_map = matches.set_index("matchId")["gameDurationMin"].to_dict()
    return df["matchId"].map(dur_map)


def load_data():
    ps = pd.read_csv(DATA / "player_stats.csv")
    bm_path = DATA / "benchmark_stats.csv"
    if not bm_path.exists():
        print("\n  エラー: benchmark_stats.csv が見つかりません")
        print("  先に以下を実行してください:")
        print("    python src/collect_benchmark.py")
        print("    python src/process.py")
        sys.exit(1)

    bm = pd.read_csv(bm_path)

    for df in [ps, bm]:
        dur = _add_duration(df).clip(lower=1)
        df["cs_per_min"] = df["cs"] / dur
        df["gold_per_min"] = df["goldEarned"] / dur
        df["dmg_per_min"] = df["totalDamageDealtToChampions"] / dur

        team_kills = df.groupby(["matchId", "teamId"])["kills"].transform("sum")
        df["kp"] = (df["kills"] + df["assists"]) / team_kills.replace(0, 1)

        team_dmg = df.groupby(["matchId", "teamId"])["totalDamageDealtToChampions"].transform("sum")
        df["dmg_share"] = df["totalDamageDealtToChampions"] / team_dmg.replace(0, 1)

    bm_tl_path = DATA / "benchmark_timeline_frames.csv"
    bm_tl = pd.read_csv(bm_tl_path) if bm_tl_path.exists() else None

    tl_path = DATA / "timeline_frames.csv"
    tl = pd.read_csv(tl_path) if tl_path.exists() else None

    return ps, bm, tl, bm_tl


def fmt(val, style):
    if pd.isna(val):
        return "   -"
    if style == "pct":
        return f"{val:.0%}"
    return f"{val:.1f}"


def fmt_diff(val, style):
    if pd.isna(val):
        return "   -"
    sign = "+" if val >= 0 else ""
    if style == "pct":
        return f"{sign}{val:.0%}"
    return f"{sign}{val:.1f}"


def section(title, char="=", width=78):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def bar_visual(diff, max_diff=1.5, width=20):
    """Visual bar centered at 0. Positive = right, negative = left."""
    if pd.isna(diff):
        return " " * width
    clamped = max(-max_diff, min(max_diff, diff))
    ratio = clamped / max_diff
    center = width // 2
    if ratio >= 0:
        filled = int(ratio * center)
        return "░" * center + "█" * filled + "░" * (center - filled)
    else:
        filled = int(-ratio * center)
        return "░" * (center - filled) + "█" * filled + "░" * center


# ======================================================================
#  1. Coverage summary
# ======================================================================

def _select_benchmark(bm, champ, member_role):
    """Select benchmark data for a champion, preferring role-matched data.

    Returns (bm_subset, match_mode) where match_mode is:
      'role'     - exact role match with sufficient data
      'all'      - fell back to all-role (role data insufficient)
      'none'     - no benchmark data available
    """
    bm_champ = bm[bm["championName"] == champ]
    if len(bm_champ) < MIN_BENCHMARK_GAMES:
        return bm_champ, "none"

    bm_role = bm_champ[bm_champ["role"] == member_role]
    if len(bm_role) >= MIN_BENCHMARK_ROLE_GAMES:
        return bm_role, "role"

    return bm_champ, "all"


def coverage_summary(ps, bm):
    section("ベンチマークカバレッジ")

    ps_m = ps[ps["summonerName"].isin(MEMBER_NAMES)]
    bm_champs = bm.groupby("championName").size()

    print()
    print(f"  ベンチマークデータ: {len(bm)}レコード / {bm['matchId'].nunique()}試合")
    print(f"  カバーされたチャンピオン: {len(bm_champs)} ({(bm_champs >= MIN_BENCHMARK_GAMES).sum()}"
          f"チャンプがベンチマーク閾値{MIN_BENCHMARK_GAMES}試合以上)")
    print()

    for name in sorted(MEMBER_NAMES):
        mp = ps_m[ps_m["summonerName"] == name]
        champ_counts = mp.groupby("championName").size().sort_values(ascending=False)
        top_champs = champ_counts[champ_counts >= MIN_MEMBER_GAMES]

        role_matched = 0
        all_role = 0
        for champ in top_champs.index:
            m_role = mp[mp["championName"] == champ]["role"].value_counts().idxmax()
            _, mode = _select_benchmark(bm, champ, m_role)
            if mode == "role":
                role_matched += 1
            elif mode == "all":
                all_role += 1

        total_covered = role_matched + all_role
        print(f"  {name}: {len(top_champs)}チャンプ中 {total_covered}カバー済み "
              f"(ロール一致{role_matched} / 全ロール{all_role})")


# ======================================================================
#  2. Per-member champion comparison
# ======================================================================

def member_champion_comparison(ps, bm, tl, bm_tl, target_member=None):
    ps_m = ps[ps["summonerName"].isin(MEMBER_NAMES)]
    names = [target_member] if target_member else sorted(MEMBER_NAMES)

    for name in names:
        mp = ps_m[ps_m["summonerName"] == name]
        if mp.empty:
            continue

        main_role = mp["role"].value_counts().idxmax()
        wr = mp["win"].mean() * 100
        section(f"{name} — チャンピオン別ベンチマーク比較 ({ROLE_JP.get(main_role, main_role)}, WR {wr:.0f}%)")

        champ_counts = mp.groupby("championName").size().sort_values(ascending=False)
        top_champs = champ_counts[champ_counts >= MIN_MEMBER_GAMES].index.tolist()

        if not top_champs:
            print(f"\n  {MIN_MEMBER_GAMES}試合以上のチャンピオンなし")
            continue

        all_diffs = []

        for champ in top_champs:
            mp_champ = mp[mp["championName"] == champ]
            m_role = mp_champ["role"].value_counts().idxmax()

            bm_compare, match_mode = _select_benchmark(bm, champ, m_role)
            if match_mode == "none":
                continue

            n_self = len(mp_champ)
            n_bm = len(bm_compare)
            m_wr = mp_champ["win"].mean() * 100
            bm_wr = bm_compare["win"].mean() * 100

            role_label = ROLE_JP.get(m_role, m_role)
            if match_mode == "role":
                bm_label = f"BM@{role_label}{n_bm}戦"
            else:
                bm_roles = bm_compare["role"].value_counts()
                top_bm_role = bm_roles.idxmax()
                bm_role_pct = bm_roles.iloc[0] / n_bm * 100
                bm_label = f"BM全ロール{n_bm}戦({ROLE_JP.get(top_bm_role, top_bm_role)}{bm_role_pct:.0f}%)"

            warn = "⚠少量" if n_bm < SMALL_SAMPLE_WARN else ""

            print()
            print(f"  ── {champ} ({role_label}) ──  "
                  f"本人{n_self}戦(WR{m_wr:.0f}%)  vs  {bm_label}(WR{bm_wr:.0f}%) {warn}")
            if match_mode == "all":
                print(f"    ※ {role_label}のBMデータ不足のため全ロールで比較")
            print()

            header = f"    {'指標':<10}"
            for _, label, _, _ in COMPARE_STATS:
                header += f" {label:>8}"
            print(header)
            print("    " + "─" * (len(header) - 4))

            m_line = f"    {'本人':<10}"
            bm_line = f"    {'ベンチマーク':<10}"
            diff_line = f"    {'差分':<10}"
            verdict_line = f"    {'判定':<10}"

            champ_diffs = {}
            for col, label, style, lower_is_better in COMPARE_STATS:
                mv = mp_champ[col].mean()
                bv = bm_compare[col].mean()
                d = mv - bv

                m_line += f" {fmt(mv, style):>8}"
                bm_line += f" {fmt(bv, style):>8}"
                diff_line += f" {fmt_diff(d, style):>8}"

                if lower_is_better:
                    v = "◎" if d < -0.1 else "△" if d > 0.1 else "─"
                else:
                    v = "◎" if d > 0.1 else "△" if d < -0.1 else "─"
                verdict_line += f" {v:>8}"
                champ_diffs[col] = d

            print(m_line)
            print(bm_line)
            print(diff_line)
            print(verdict_line)

            if tl is not None and bm_tl is not None:
                _show_early_game(tl, bm_tl, name, champ, m_role, match_mode)

            all_diffs.append({"champion": champ, "role": m_role, "games": n_self,
                              "bm_games": n_bm, "match_mode": match_mode, **champ_diffs})

        if all_diffs:
            _show_champion_adjusted_summary(name, all_diffs)


def _show_early_game(tl, bm_tl, name, champ, member_role, match_mode):
    """Show early game gold diff comparison for a specific champion."""
    for minute in [10, 15]:
        m_snap = tl[(tl["summonerName"] == name) &
                    (tl["championName"] == champ) &
                    (tl["timestampMin"] == minute)]

        bm_snap = bm_tl[(bm_tl["championName"] == champ) &
                         (bm_tl["timestampMin"] == minute)]
        if match_mode == "role":
            bm_snap = bm_snap[bm_snap["role"] == member_role]

        if m_snap.empty or bm_snap.empty:
            continue

        m_gd = m_snap["goldDiffVsOpponent"].mean()
        bm_gd = bm_snap["goldDiffVsOpponent"].mean()

        if pd.notna(m_gd) and pd.notna(bm_gd):
            diff = m_gd - bm_gd
            verdict = "◎" if diff > 50 else "△" if diff < -50 else "─"
            print(f"    {minute}分GD: 本人{m_gd:+.0f}  BM{bm_gd:+.0f}  差{diff:+.0f}  {verdict}")


def _show_champion_adjusted_summary(name, all_diffs):
    """Weighted average across champions → overall adjusted evaluation."""
    df = pd.DataFrame(all_diffs)
    weights = df["games"]
    total_games = weights.sum()

    print()
    print(f"  ── {name} チャンピオン補正済み総合評価 ──")
    print()

    strengths = []
    weaknesses = []

    for col, label, style, lower_is_better in COMPARE_STATS:
        if col not in df.columns:
            continue
        weighted_diff = (df[col] * weights).sum() / total_games

        if lower_is_better:
            if weighted_diff < -0.15:
                strengths.append(f"{label}({fmt_diff(weighted_diff, style)})")
            elif weighted_diff > 0.15:
                weaknesses.append(f"{label}({fmt_diff(weighted_diff, style)})")
        else:
            if weighted_diff > 0.15:
                strengths.append(f"{label}({fmt_diff(weighted_diff, style)})")
            elif weighted_diff < -0.15:
                weaknesses.append(f"{label}({fmt_diff(weighted_diff, style)})")

    if strengths:
        print(f"    ◎ エメラルド平均を上回る: {', '.join(strengths)}")
    if weaknesses:
        print(f"    △ エメラルド平均を下回る: {', '.join(weaknesses)}")
    if not strengths and not weaknesses:
        print(f"    ─ 同チャンピオンのエメラルド帯と概ね同水準")


# ======================================================================
#  3. Team-wide champion-adjusted summary
# ======================================================================

def team_summary(ps, bm):
    section("チーム全体 — チャンピオン+ロール補正済みサマリー")
    print()
    print("  ※ 各メンバーの頻出チャンプごとに同ロールのベンチマーク平均と比較し、")
    print("    試合数で加重平均した総合評価（ロール一致データ優先）")
    print()

    ps_m = ps[ps["summonerName"].isin(MEMBER_NAMES)]

    results = []
    for name in sorted(MEMBER_NAMES):
        mp = ps_m[ps_m["summonerName"] == name]
        if mp.empty:
            continue
        main_role = mp["role"].value_counts().idxmax()

        champ_counts = mp.groupby("championName").size().sort_values(ascending=False)
        top_champs = champ_counts[champ_counts >= MIN_MEMBER_GAMES].index.tolist()

        weighted_diffs = {col: 0.0 for col, _, _, _ in COMPARE_STATS}
        total_w = 0
        role_matched = 0
        all_role = 0

        for champ in top_champs:
            mp_c = mp[mp["championName"] == champ]
            m_role = mp_c["role"].value_counts().idxmax()
            bm_c, mode = _select_benchmark(bm, champ, m_role)
            if mode == "none":
                continue
            if mode == "role":
                role_matched += 1
            else:
                all_role += 1
            w = len(mp_c)
            total_w += w
            for col, _, _, _ in COMPARE_STATS:
                d = mp_c[col].mean() - bm_c[col].mean()
                weighted_diffs[col] += d * w

        if total_w == 0:
            continue

        for col in weighted_diffs:
            weighted_diffs[col] /= total_w

        results.append({
            "name": name,
            "role": ROLE_JP.get(main_role, main_role),
            "role_matched": role_matched,
            "all_role": all_role,
            "games_covered": total_w,
            **weighted_diffs,
        })

    if not results:
        print("  ベンチマークデータが不足しています")
        return

    header = f"  {'メンバー':<14} {'ロール':<10}"
    for _, label, _, _ in COMPARE_STATS:
        header += f" {label:>8}"
    header += "   比較方法"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in results:
        line = f"  {r['name']:<14} {r['role']:<10}"
        for col, _, style, lower_is_better in COMPARE_STATS:
            d = r[col]
            if lower_is_better:
                mark = "◎" if d < -0.1 else "△" if d > 0.1 else ""
            else:
                mark = "◎" if d > 0.1 else "△" if d < -0.1 else ""
            line += f" {fmt_diff(d, style):>6}{mark:>2}"
        line += f"   R{r['role_matched']}/A{r['all_role']}"
        print(line)

    print()
    print(f"  （R=ロール一致 / A=全ロール代用）")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="チャンピオン別エメラルドベンチマーク比較")
    parser.add_argument("--member", type=str, default=None,
                        help="特定メンバーの詳細を表示")
    args = parser.parse_args()

    ps, bm, tl, bm_tl = load_data()

    n_bm_matches = bm["matchId"].nunique()
    n_bm_champs = bm["championName"].nunique()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     チャンピオン+ロール補正済み ベンチマーク比較 (Em+Dia)               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  ベンチマーク: {n_bm_matches}試合 / {len(bm)}レコード / {n_bm_champs}チャンピオン")
    print(f"  タイムライン: {'あり' if bm_tl is not None else 'なし'}")
    print(f"  比較方式: ロール一致優先（不足時は全ロール代用, ⚠少量={SMALL_SAMPLE_WARN}試合未満）")

    if args.member:
        if args.member not in MEMBER_NAMES:
            matches = [n for n in MEMBER_NAMES if args.member.lower() in n.lower()]
            if len(matches) == 1:
                args.member = matches[0]
            else:
                print(f"\n  エラー: '{args.member}' が見つかりません")
                print(f"  有効なメンバー: {', '.join(sorted(MEMBER_NAMES))}")
                return
        coverage_summary(ps, bm)
        member_champion_comparison(ps, bm, tl, bm_tl, target_member=args.member)
    else:
        coverage_summary(ps, bm)
        team_summary(ps, bm)
        member_champion_comparison(ps, bm, tl, bm_tl)

    print()


if __name__ == "__main__":
    main()

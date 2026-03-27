"""Compare team members against Emerald-rank benchmark players.

Uses player_stats.csv for members + benchmark_stats.csv / benchmark_timeline_frames.csv
for comparison. Requires running collect_benchmark.py + process.py first.

Usage:
    python scripts/benchmark_comparison.py              # full team report
    python scripts/benchmark_comparison.py --member X   # deep dive on one member
"""

import argparse
import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from scipy import stats as sp_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
MEMBER_NAMES = {m["game_name"] for m in cfg.get("members", [])}

ROLE_JP = {
    "TOP": "トップ",
    "JUNGLE": "ジャングル",
    "MIDDLE": "ミッド",
    "BOTTOM": "ボトム",
    "UTILITY": "サポート",
}
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

STAT_DEFS = [
    ("kills",                        "キル",       "raw",  False),
    ("deaths",                       "デス",       "raw",  True),
    ("assists",                      "アシスト",    "raw",  False),
    ("kda",                          "KDA",        "raw",  False),
    ("cs_per_min",                   "CS/min",     "raw",  False),
    ("gold_per_min",                 "Gold/min",   "raw",  False),
    ("dmg_per_min",                  "DMG/min",    "raw",  False),
    ("totalDamageTaken",             "被ダメージ",  "comma", False),
    ("visionScore",                  "ビジョン",    "raw",  False),
    ("wardsPlaced",                  "ワード設置",  "raw",  False),
    ("wardsKilled",                  "ワード除去",  "raw",  False),
    ("kp",                           "KP",         "pct",  False),
    ("dmg_share",                    "ダメ割合",    "pct",  False),
    ("gold_share",                   "Gold割合",   "pct",  False),
    ("firstBloodKill",               "FB率",       "pct",  False),
]

EARLY_MINUTES = [10, 15]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived per-minute and share columns."""
    dur = df["gameDurationMin"].clip(lower=1)
    df["cs_per_min"] = df["cs"] / dur
    df["gold_per_min"] = df["goldEarned"] / dur
    df["dmg_per_min"] = df["totalDamageDealtToChampions"] / dur

    team_kills = df.groupby(["matchId", "teamId"])["kills"].transform("sum")
    df["kp"] = (df["kills"] + df["assists"]) / team_kills.replace(0, 1)

    team_dmg = df.groupby(["matchId", "teamId"])["totalDamageDealtToChampions"].transform("sum")
    df["dmg_share"] = df["totalDamageDealtToChampions"] / team_dmg.replace(0, 1)

    team_gold = df.groupby(["matchId", "teamId"])["goldEarned"].transform("sum")
    df["gold_share"] = df["goldEarned"] / team_gold.replace(0, 1)

    return df


def load_data():
    ps = pd.read_csv(DATA / "player_stats.csv")
    matches = pd.read_csv(DATA / "matches.csv")

    ps["is_member"] = ps["summonerName"].isin(MEMBER_NAMES)
    ps = ps[ps["role"].isin(ROLES)].copy()

    duration_map = matches.set_index("matchId")["gameDurationMin"].to_dict()
    ps["gameDurationMin"] = ps["matchId"].map(duration_map)
    ps = _enrich(ps)

    bm_path = DATA / "benchmark_stats.csv"
    if not bm_path.exists():
        print("\n  エラー: benchmark_stats.csv が見つかりません")
        print("  先に以下を実行してください:")
        print("    python src/collect_benchmark.py")
        print("    python src/process.py")
        sys.exit(1)

    bm = pd.read_csv(bm_path)
    bm = bm[bm["role"].isin(ROLES)].copy()
    bm = _enrich(bm)

    tf = pd.read_csv(DATA / "timeline_frames.csv")

    bm_tl_path = DATA / "benchmark_timeline_frames.csv"
    bm_tl = pd.read_csv(bm_tl_path) if bm_tl_path.exists() else None

    return ps, bm, tf, bm_tl


def fmt_val(val, style):
    if pd.isna(val):
        return "   -"
    if style == "pct":
        return f"{val:.0%}"
    if style == "comma":
        return f"{val:,.0f}"
    return f"{val:.1f}"


def fmt_diff(diff, style):
    if pd.isna(diff):
        return "   -"
    sign = "+" if diff >= 0 else ""
    if style == "pct":
        return f"{sign}{diff:.0%}"
    if style == "comma":
        return f"{sign}{diff:,.0f}"
    return f"{sign}{diff:.1f}"


def percentile_rank(member_val, all_vals):
    """What percentile the member value falls at among all_vals."""
    if pd.isna(member_val) or len(all_vals) == 0:
        return np.nan
    return sp_stats.percentileofscore(all_vals.dropna(), member_val, kind="rank")


def bar_visual(pct, width=20):
    """Render a bar from 0–100% percentile."""
    if pd.isna(pct):
        return " " * width
    filled = int(round(pct / 100 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def section(title, char="=", width=78):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


# ======================================================================
#  1. Team-wide overview
# ======================================================================

def team_overview(ps, bm):
    section("チーム全体 vs エメラルド帯ベンチマーク")
    print()
    print(f"  ※ 別途収集したエメラルド帯フレックスランクのプレイヤーデータをベンチマークに使用")
    print(f"  ※ ベンチマーク: {len(bm)}レコード / {bm['matchId'].nunique()}試合")
    print()

    m = ps[ps["is_member"]]

    header = f"  {'指標':<12} {'自チーム':>10} {'ベンチマーク':>12} {'差分':>10} {'判定':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for col, label, style, lower_is_better in STAT_DEFS:
        m_val = m[col].mean()
        bm_val = bm[col].mean()
        diff = m_val - bm_val

        if lower_is_better:
            verdict = "◎" if diff < 0 else "△"
        else:
            verdict = "◎" if diff > 0 else "△"
        if abs(diff) < 0.05 * max(abs(bm_val), 1e-9):
            verdict = "─"

        print(f"  {label:<12} {fmt_val(m_val, style):>10} {fmt_val(bm_val, style):>12}"
              f" {fmt_diff(diff, style):>10} {verdict:>6}")


# ======================================================================
#  2. Role-by-role comparison
# ======================================================================

def role_comparison(ps, bm):
    section("ロール別 メンバー vs エメラルド帯ベンチマーク")

    key_stats = [
        ("kda",         "KDA",      "raw",  False),
        ("cs_per_min",  "CS/m",     "raw",  False),
        ("gold_per_min","Gold/m",   "raw",  False),
        ("dmg_per_min", "DMG/m",    "raw",  False),
        ("kp",          "KP",       "pct",  False),
        ("visionScore", "Vision",   "raw",  False),
        ("deaths",      "Deaths",   "raw",  True),
    ]

    for role in ROLES:
        print()
        print(f"  ── {ROLE_JP[role]} ──")
        m = ps[(ps["is_member"]) & (ps["role"] == role)]
        bm_role = bm[bm["role"] == role]
        if m.empty:
            print("    （該当データなし）")
            continue

        print(f"    メンバー試合数: {len(m)},  ベンチマーク: {len(bm_role)}")
        header = f"    {'指標':<10}"
        for _, label, _, _ in key_stats:
            header += f" {label:>8}"
        print(header)
        print("    " + "─" * (len(header) - 4))

        m_line = f"    {'自チーム':<10}"
        bm_line = f"    {'ベンチマーク':<10}"
        diff_line = f"    {'差分':<10}"
        for col, _, style, lower_is_better in key_stats:
            mv = m[col].mean()
            bv = bm_role[col].mean()
            d = mv - bv
            m_line += f" {fmt_val(mv, style):>8}"
            bm_line += f" {fmt_val(bv, style):>8}"
            diff_line += f" {fmt_diff(d, style):>8}"
        print(m_line)
        print(bm_line)
        print(diff_line)


# ======================================================================
#  3. Per-member percentile dashboard
# ======================================================================

def member_percentiles(ps, bm, target_member=None):
    if target_member:
        section(f"{target_member} — エメラルド帯パーセンタイル")
    else:
        section("メンバー別 エメラルド帯パーセンタイル（上位何%か）")

    print()
    print("  ※ 50% = エメラルド帯の中央,  75%↑ = 上位25%,  25%↓ = 下位25%")

    pct_stats = [
        ("kda",         "KDA",      False),
        ("cs_per_min",  "CS/m",     False),
        ("dmg_per_min", "DMG/m",    False),
        ("gold_per_min","Gold/m",   False),
        ("kp",          "KP",       False),
        ("visionScore", "Vision",   False),
        ("deaths",      "Deaths",   True),
    ]

    names = [target_member] if target_member else sorted(
        ps[ps["is_member"]]["summonerName"].unique(),
        key=lambda n: ps[(ps["summonerName"] == n)]["kda"].mean(),
        reverse=True,
    )

    for name in names:
        mp = ps[ps["summonerName"] == name]
        if mp.empty:
            continue
        main_role = mp["role"].value_counts().idxmax()
        n_games = len(mp)
        wr = mp["win"].mean() * 100

        print()
        print(f"  ■ {name}  ({ROLE_JP.get(main_role, main_role)},  {n_games}試合  WR {wr:.0f}%)")
        print()

        bm_role = bm[bm["role"] == main_role]

        for col, label, invert in pct_stats:
            benchmark_pool = bm_role[col]
            member_val = mp[mp["role"] == main_role][col].mean()
            if invert:
                pct = 100 - percentile_rank(member_val, benchmark_pool)
            else:
                pct = percentile_rank(member_val, benchmark_pool)

            marker = "◎" if pct >= 65 else "△" if pct <= 35 else "─"
            print(f"    {label:<10} {bar_visual(pct)} {pct:5.0f}%tile  (avg {fmt_val(member_val, 'raw')})  {marker}")

        if target_member:
            _member_deep_dive(ps, bm, mp, name, main_role)


def _member_deep_dive(ps, bm, mp, name, main_role):
    """Extra detail when --member is specified."""
    bm_role = bm[bm["role"] == main_role]

    print()
    print(f"    ── 全指標詳細（{ROLE_JP.get(main_role, main_role)}として） ──")
    print()
    header = f"    {'指標':<12} {'本人平均':>10} {'ベンチマーク':>12} {'差分':>10} {'%tile':>8}"
    print(header)
    print("    " + "─" * (len(header) - 4))

    for col, label, style, lower_is_better in STAT_DEFS:
        mv = mp[mp["role"] == main_role][col].mean()
        bv = bm_role[col].mean()
        diff = mv - bv
        benchmark_pool = bm_role[col]
        if lower_is_better:
            pct = 100 - percentile_rank(mv, benchmark_pool)
        else:
            pct = percentile_rank(mv, benchmark_pool)
        print(f"    {label:<12} {fmt_val(mv, style):>10} {fmt_val(bv, style):>12}"
              f" {fmt_diff(diff, style):>10} {pct:>6.0f}%")

    champs = mp[mp["role"] == main_role].groupby("championName").agg(
        games=("win", "count"),
        wins=("win", "sum"),
        kda=("kda", "mean"),
        cs_per_min=("cs_per_min", "mean"),
        dmg_per_min=("dmg_per_min", "mean"),
    ).reset_index()
    champs["wr"] = champs["wins"] / champs["games"] * 100
    champs = champs.sort_values("games", ascending=False).head(8)

    if not champs.empty:
        print()
        print(f"    ── チャンピオン別（エメラルド帯同チャンプ平均との差） ──")
        print()
        header2 = f"    {'チャンピオン':<16} {'試合':>4} {'WR':>6} {'KDA':>6} {'CS/m':>6} {'DMG/m':>8}"
        print(header2)
        print("    " + "─" * (len(header2) - 4))
        for _, cr in champs.iterrows():
            champ_pool = bm_role[bm_role["championName"] == cr["championName"]]
            kda_diff = ""
            if len(champ_pool) >= 3:
                pool_kda = champ_pool["kda"].mean()
                d = cr["kda"] - pool_kda
                kda_diff = f"({d:+.1f})"
            print(f"    {cr['championName']:<16} {int(cr['games']):>4} {cr['wr']:>5.0f}%"
                  f" {cr['kda']:>5.1f}{kda_diff}  {cr['cs_per_min']:>5.1f} {cr['dmg_per_min']:>7.0f}")


# ======================================================================
#  4. Early game benchmark
# ======================================================================

def early_game_benchmark(tf, bm_tl):
    section("序盤ゴールド差ベンチマーク（メンバー vs エメラルド帯）")

    if bm_tl is None:
        print("\n  ※ benchmark_timeline_frames.csv がないためスキップ")
        return

    for minute in EARLY_MINUTES:
        m_snap = tf[(tf["timestampMin"] == minute) & (tf["summonerName"].isin(MEMBER_NAMES))].copy()
        bm_snap = bm_tl[bm_tl["timestampMin"] == minute].copy()

        if m_snap.empty or bm_snap.empty:
            continue

        print()
        print(f"  ── {minute}分時点 ──")
        print(f"  {'ロール':<12} {'メンバー平均GD':>14} {'ベンチマーク平均GD':>18} {'序盤勝ち率(M)':>14} {'序盤勝ち率(BM)':>14}")
        print(f"  {'─'*12} {'─'*14} {'─'*18} {'─'*14} {'─'*14}")

        for role in ROLES:
            m_role = m_snap[m_snap["role"] == role]
            bm_role = bm_snap[bm_snap["role"] == role]

            if m_role.empty or bm_role.empty:
                continue

            m_gd = m_role["goldDiffVsOpponent"].mean()
            bm_gd = bm_role["goldDiffVsOpponent"].mean()
            m_ahead = (m_role["goldDiffVsOpponent"] > 0).mean() * 100
            bm_ahead = (bm_role["goldDiffVsOpponent"] > 0).mean() * 100

            print(f"  {ROLE_JP[role]:<12} {m_gd:>+12.0f}G {bm_gd:>+16.0f}G"
                  f" {m_ahead:>12.0f}% {bm_ahead:>12.0f}%")


# ======================================================================
#  5. Strengths / weaknesses auto-summary
# ======================================================================

def auto_summary(ps, bm):
    section("強み・課題の自動判定")
    print()
    print("  ※ エメラルド帯ベンチマークの中で偏差値ベースで判定")
    print("    ◎ 60以上（上位16%相当）  △ 40以下（下位16%相当）")

    check_stats = [
        ("kda",         "KDA",        False),
        ("cs_per_min",  "CS/min",     False),
        ("dmg_per_min", "DMG/min",    False),
        ("visionScore", "ビジョン",    False),
        ("kp",          "KP",         False),
        ("deaths",      "デス",        True),
    ]

    for name in sorted(MEMBER_NAMES):
        mp = ps[ps["summonerName"] == name]
        if mp.empty:
            continue
        main_role = mp["role"].value_counts().idxmax()
        bm_role = bm[bm["role"] == main_role]

        strengths = []
        weaknesses = []

        for col, label, invert in check_stats:
            pool_mean = bm_role[col].mean()
            pool_std = bm_role[col].std()
            m_val = mp[mp["role"] == main_role][col].mean()
            if pd.isna(m_val) or pool_std == 0:
                continue
            z = (m_val - pool_mean) / pool_std
            dev = 50 + z * 10

            if invert:
                dev = 100 - dev

            if dev >= 60:
                strengths.append(f"{label}(偏差値{dev:.0f})")
            elif dev <= 40:
                weaknesses.append(f"{label}(偏差値{dev:.0f})")

        print()
        wr = mp["win"].mean() * 100
        print(f"  ■ {name} ({ROLE_JP.get(main_role, main_role)}, WR {wr:.0f}%)")
        if strengths:
            print(f"    ◎ 強み: {', '.join(strengths)}")
        if weaknesses:
            print(f"    △ 課題: {', '.join(weaknesses)}")
        if not strengths and not weaknesses:
            print(f"    ─ エメラルド帯の平均的な水準")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="エメラルド帯ベンチマーク比較")
    parser.add_argument("--member", type=str, default=None,
                        help="特定メンバーの詳細を表示")
    args = parser.parse_args()

    ps, bm, tf, bm_tl = load_data()

    n_matches = ps["matchId"].nunique()
    n_member_rows = ps[ps["is_member"]].shape[0]
    n_bm_matches = bm["matchId"].nunique()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          エメラルド帯ベンチマーク比較                                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  メンバー対象試合数: {n_matches}")
    print(f"  メンバーデータ: {n_member_rows} レコード")
    print(f"  ベンチマーク: {len(bm)} レコード / {n_bm_matches} 試合（エメラルド帯フレックス）")
    print(f"  メンバー: {', '.join(sorted(MEMBER_NAMES))}")

    if args.member:
        if args.member not in MEMBER_NAMES:
            matches = [n for n in MEMBER_NAMES if args.member.lower() in n.lower()]
            if len(matches) == 1:
                args.member = matches[0]
            else:
                print(f"\n  エラー: '{args.member}' が見つかりません")
                print(f"  有効なメンバー: {', '.join(sorted(MEMBER_NAMES))}")
                return
        member_percentiles(ps, bm, target_member=args.member)
    else:
        team_overview(ps, bm)
        role_comparison(ps, bm)
        member_percentiles(ps, bm)
        early_game_benchmark(tf, bm_tl)
        auto_summary(ps, bm)

    print()


if __name__ == "__main__":
    main()

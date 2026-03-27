"""ダメージキャリー人数 vs 勝率分析

チャンピオンの設計上の役割（ダメージ型 vs タンク/ユーティリティ型）で分類し、
チーム内のダメージキャリー人数と勝率の関係を分析する。
"""

import pandas as pd
import yaml
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

NON_CARRY = {
    # Tanks / Frontline Juggernauts
    "Alistar", "Amumu", "Braum", "Chogath", "DrMundo", "Galio", "Garen",
    "Gnar", "Gragas", "JarvanIV", "KSante", "Leona", "Malphite", "Maokai",
    "Nasus", "Nautilus", "Nunu", "Ornn", "Poppy", "Rammus", "Sejuani",
    "Shen", "Singed", "Sion", "Skarner", "TahmKench", "Trundle", "Udyr",
    "Volibear", "Warwick", "Zac",
    # Enchanters / Utility Supports / Utility Fighters
    "Bard", "Blitzcrank", "Ivern", "Janna", "Karma", "LeeSin", "Lulu",
    "Milio", "Morgana", "Nami", "Rakan", "Rell", "Renata", "Seraphine",
    "Sona", "Soraka", "Taric", "Thresh", "Yuumi", "Zilean",
}


def main():
    df = pd.read_csv(DATA / "player_stats.csv")
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    member_names = [m["game_name"] for m in cfg["members"]]
    members = df[df["summonerName"].isin(member_names)].copy()
    members["is_carry"] = ~members["championName"].isin(NON_CARRY)

    match_carry = (
        members.groupby("matchId")
        .agg(
            carry_count=("is_carry", "sum"),
            win=("win", "first"),
            n_members=("summonerName", "count"),
        )
        .reset_index()
    )

    match5 = match_carry[match_carry["n_members"] == 5].copy()
    total = len(match5)
    total_wr = match5["win"].mean() * 100

    print()
    print("=" * 65)
    print("  ダメージキャリー人数 vs 勝率分析（チャンピオン分類ベース）")
    print("=" * 65)
    print(f"  対象: 5人参加試合 {total}試合 | 全体勝率 {total_wr:.1f}%")
    print()

    # ── Main table ──
    print("─" * 65)
    print("  キャリー人数 │ 勝率      │ 戦績          │ ")
    print("─" * 65)

    best_n, best_wr, best_games = 0, 0.0, 0
    for n in sorted(match5["carry_count"].unique()):
        n_int = int(n)
        sub = match5[match5["carry_count"] == n]
        wr = sub["win"].mean() * 100
        games = len(sub)
        wins = int(sub["win"].sum())
        losses = games - wins
        bar_len = int(wr / 2)
        bar = "#" * bar_len + "." * (50 - bar_len)
        marker = ""
        if games >= 10 and wr > best_wr:
            best_n, best_wr, best_games = n_int, wr, games
        print(f"    {n_int}人       │ {wr:5.1f}%    │ {wins:3d}勝 {losses:3d}敗 ({games:3d}試合) │ {bar}")

    print("─" * 65)
    print()

    if best_games > 0:
        print(f"  ★ 最高勝率: キャリー{best_n}人 → {best_wr:.1f}% ({best_games}試合)")
    print()

    # ── Chi-squared test ──
    valid_counts = sorted(
        [n for n in match5["carry_count"].unique() if len(match5[match5["carry_count"] == n]) >= 5]
    )
    if len(valid_counts) >= 2:
        chi2_data = []
        for n in valid_counts:
            sub = match5[match5["carry_count"] == n]
            chi2_data.append([int(sub["win"].sum()), len(sub) - int(sub["win"].sum())])
        chi2, p, dof, _ = stats.chi2_contingency(chi2_data)
        sig = "有意差あり (p<0.05)" if p < 0.05 else "有意差なし"
        print(f"  カイ二乗検定: χ²={chi2:.2f}, p={p:.4f} → {sig}")
        print()

    # ── Best pairs: 2-carry vs 3-carry comparison ──
    for a, b in [(2, 3), (3, 4), (2, 4)]:
        ga = match5[match5["carry_count"] == a]
        gb = match5[match5["carry_count"] == b]
        if len(ga) >= 10 and len(gb) >= 10:
            wa = ga["win"].mean() * 100
            wb = gb["win"].mean() * 100
            _, p2 = stats.fisher_exact(
                [[int(ga["win"].sum()), len(ga) - int(ga["win"].sum())],
                 [int(gb["win"].sum()), len(gb) - int(gb["win"].sum())]]
            )
            print(f"  {a}人 vs {b}人: {wa:.1f}% vs {wb:.1f}% (差 {wa-wb:+.1f}pp, p={p2:.3f})")

    print()

    # ── Role breakdown ──
    print("=" * 65)
    print("  ロール別: キャリー vs 非キャリー時の勝率")
    print("=" * 65)
    for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
        role_df = members[members["role"] == role]
        if len(role_df) == 0:
            continue
        c = role_df[role_df["is_carry"]]
        nc = role_df[~role_df["is_carry"]]
        c_wr = c["win"].mean() * 100 if len(c) > 0 else 0
        nc_wr = nc["win"].mean() * 100 if len(nc) > 0 else 0
        diff = c_wr - nc_wr if len(c) > 0 and len(nc) > 0 else float("nan")
        diff_s = f"{diff:+.1f}pp" if not pd.isna(diff) else "N/A"
        print(f"  {role:8s}: キャリー {c_wr:5.1f}% ({len(c):3d}試合) │ 非キャリー {nc_wr:5.1f}% ({len(nc):3d}試合) │ 差 {diff_s}")
    print()

    # ── Which role's carry pick matters most? ──
    print("=" * 65)
    print("  どのロールでキャリーを出すと勝率が変わるか？")
    print("=" * 65)

    role_impact = []
    for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
        role_df = members[members["role"] == role]
        if len(role_df) == 0:
            continue
        carry_matches = set(role_df[role_df["is_carry"]]["matchId"])
        noncarry_matches = set(role_df[~role_df["is_carry"]]["matchId"])

        c_games = match5[match5["matchId"].isin(carry_matches)]
        nc_games = match5[match5["matchId"].isin(noncarry_matches)]

        if len(c_games) >= 5 and len(nc_games) >= 5:
            c_wr = c_games["win"].mean() * 100
            nc_wr = nc_games["win"].mean() * 100
            diff = c_wr - nc_wr
            role_impact.append((role, diff, c_wr, nc_wr, len(c_games), len(nc_games)))

    role_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    for role, diff, c_wr, nc_wr, c_n, nc_n in role_impact:
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {role:8s}: キャリー時 {c_wr:.1f}%({c_n}試合) vs 非キャリー時 {nc_wr:.1f}%({nc_n}試合) → {diff:+.1f}pp {arrow}")
    print()

    # ── Top champions by carry classification ──
    print("=" * 65)
    print("  ロール別 よく使うキャリー / 非キャリーチャンプ")
    print("=" * 65)
    for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
        role_df = members[members["role"] == role]
        if len(role_df) == 0:
            continue
        print(f"\n  【{role}】")
        c_top = role_df[role_df["is_carry"]].groupby("championName").agg(
            games=("win", "count"), wr=("win", "mean")
        ).sort_values("games", ascending=False).head(5)
        nc_top = role_df[~role_df["is_carry"]].groupby("championName").agg(
            games=("win", "count"), wr=("win", "mean")
        ).sort_values("games", ascending=False).head(5)

        if len(c_top) > 0:
            parts = [f"{ch} {int(r.games)}試合({r.wr*100:.0f}%)" for ch, r in c_top.iterrows()]
            print(f"    キャリー  : {', '.join(parts)}")
        if len(nc_top) > 0:
            parts = [f"{ch} {int(r.games)}試合({r.wr*100:.0f}%)" for ch, r in nc_top.iterrows()]
            print(f"    非キャリー: {', '.join(parts)}")
    print()

    # ── Carry count by composition pattern ──
    print("=" * 65)
    print("  よくある構成パターン（TOP/JG/MID/BOT/SUP のキャリー有無）")
    print("=" * 65)

    pattern_data = []
    for match_id, grp in members[members["matchId"].isin(match5["matchId"])].groupby("matchId"):
        if len(grp) != 5:
            continue
        pattern = {}
        win = grp["win"].iloc[0]
        for _, row in grp.iterrows():
            if pd.notna(row["role"]):
                pattern[row["role"]] = "C" if row["is_carry"] else "T"
        if len(pattern) == 5:
            pat_str = f"{pattern.get('TOP','?')}/{pattern.get('JUNGLE','?')}/{pattern.get('MIDDLE','?')}/{pattern.get('BOTTOM','?')}/{pattern.get('UTILITY','?')}"
            pattern_data.append({"pattern": pat_str, "win": win})

    if pattern_data:
        pat_df = pd.DataFrame(pattern_data)
        pat_summary = pat_df.groupby("pattern").agg(
            games=("win", "count"), wins=("win", "sum"), wr=("win", "mean")
        ).sort_values("games", ascending=False)

        for pat, r in pat_summary.head(15).iterrows():
            carry_n = pat.count("C")
            wr = r.wr * 100
            print(f"  {pat}  (キャリー{carry_n}人): {wr:5.1f}% ({int(r.wins)}勝{int(r.games - r.wins)}敗 / {int(r.games)}試合)")

    print()
    print("  C=キャリー, T=タンク/ユーティリティ")
    print("  (順序: TOP/JG/MID/BOT/SUP)")
    print()


if __name__ == "__main__":
    main()

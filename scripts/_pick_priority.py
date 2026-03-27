"""チャンピオン ピック優先度分析 — 勝率順ランキング"""

import pandas as pd
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("config/settings.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
members = {m["game_name"] for m in cfg["members"]}

df = pd.read_csv("data/processed/player_stats.csv")
df_m = df[df["summonerName"].isin(members)].copy()

total_matches = df_m["matchId"].nunique()
print(f"=== チーム全体：勝率の高いチャンピオン ピック優先順 ===")
print(f"(メンバー {len(members)}人, 全 {total_matches} 試合)")
print()

# ── PART 1: チーム全体 チャンピオン勝率ランキング ──
SEP = "━" * 70
print(SEP)
print("PART 1: チーム全体 チャンピオン勝率ランキング (3試合以上)")
print(SEP)
champ = df_m.groupby("championName").agg(
    games=("win", "count"),
    wins=("win", "sum"),
    avg_kda=("kda", "mean"),
    avg_dmg=("totalDamageDealtToChampions", "mean"),
    avg_gold=("goldEarned", "mean"),
).reset_index()
champ["winrate"] = (champ["wins"] / champ["games"] * 100).round(1)
champ = champ[champ["games"] >= 3].sort_values("winrate", ascending=False)

header = f"{'チャンピオン':<16} {'試合':>4} {'勝率':>6} {'KDA':>6} {'DMG':>7} {'Gold':>7}"
print(header)
print("-" * 56)
for _, r in champ.head(25).iterrows():
    print(
        f"{r['championName']:<16} {r['games']:>4} {r['winrate']:>5.1f}% "
        f"{r['avg_kda']:>6.2f} {r['avg_dmg']:>7.0f} {r['avg_gold']:>7.0f}"
    )
print(f"\n... 全 {len(champ)} チャンピオン (3試合以上)\n")

# ── PART 2: ロール別 チャンピオン勝率ランキング ──
print(SEP)
print("PART 2: ロール別 チャンピオン勝率ランキング (3試合以上)")
print(SEP)
ROLE_JP = {
    "TOP": "トップ",
    "JUNGLE": "ジャングル",
    "MIDDLE": "ミッド",
    "BOTTOM": "ボット",
    "UTILITY": "サポート",
}
for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
    role_df = df_m[df_m["role"] == role]
    rs = role_df.groupby("championName").agg(
        games=("win", "count"),
        wins=("win", "sum"),
        avg_kda=("kda", "mean"),
        avg_dmg=("totalDamageDealtToChampions", "mean"),
    ).reset_index()
    rs["winrate"] = (rs["wins"] / rs["games"] * 100).round(1)
    rs = rs[rs["games"] >= 3].sort_values("winrate", ascending=False)

    print(f"\n▼ {ROLE_JP.get(role, role)} ({len(role_df)} 試合)")
    print(f"  {'チャンピオン':<16} {'試合':>4} {'勝率':>6} {'KDA':>6} {'DMG':>7}")
    print("  " + "-" * 47)
    for _, r in rs.head(10).iterrows():
        print(
            f"  {r['championName']:<16} {r['games']:>4} {r['winrate']:>5.1f}% "
            f"{r['avg_kda']:>6.2f} {r['avg_dmg']:>7.0f}"
        )
    if len(rs) == 0:
        print("  (3試合以上のチャンピオンなし)")
print()

# ── PART 3: メンバー別 得意チャンピオン Top5 ──
print(SEP)
print("PART 3: メンバー別 得意チャンピオン Top5 (3試合以上, 勝率順)")
print(SEP)
for member in sorted(members):
    mem_df = df_m[df_m["summonerName"] == member]
    if len(mem_df) == 0:
        continue
    ms = mem_df.groupby(["championName", "role"]).agg(
        games=("win", "count"),
        wins=("win", "sum"),
        avg_kda=("kda", "mean"),
    ).reset_index()
    ms["winrate"] = (ms["wins"] / ms["games"] * 100).round(1)
    ms = ms[ms["games"] >= 3].sort_values("winrate", ascending=False)

    total_wr = mem_df["win"].mean() * 100
    print(f"\n▼ {member} (全{len(mem_df)}試合, 総合勝率{total_wr:.1f}%)")
    if len(ms) == 0:
        print("  (3試合以上のチャンピオンなし)")
        continue
    for _, r in ms.head(5).iterrows():
        print(
            f"  {r['championName']:<16} @ {r['role']:<8} "
            f"{r['games']:>3}試合 {r['winrate']:>5.1f}%  KDA {r['avg_kda']:.2f}"
        )
print()

# ── PART 4: 安定ピック (5試合以上 & 勝率55%以上) ──
print(SEP)
print("PART 4: 安定ピック (5試合以上 & 勝率55%以上)")
print(SEP)
stable = df_m.groupby(["summonerName", "championName", "role"]).agg(
    games=("win", "count"),
    wins=("win", "sum"),
    avg_kda=("kda", "mean"),
    avg_dmg=("totalDamageDealtToChampions", "mean"),
).reset_index()
stable["winrate"] = (stable["wins"] / stable["games"] * 100).round(1)
stable = stable[(stable["games"] >= 5) & (stable["winrate"] >= 55)].sort_values(
    "winrate", ascending=False
)

print(
    f"{'メンバー':<14} {'チャンピオン':<16} {'ロール':<8} "
    f"{'試合':>4} {'勝率':>6} {'KDA':>6}"
)
print("-" * 62)
for _, r in stable.iterrows():
    print(
        f"{r['summonerName']:<14} {r['championName']:<16} {r['role']:<8} "
        f"{r['games']:>4} {r['winrate']:>5.1f}% {r['avg_kda']:>6.2f}"
    )
if len(stable) == 0:
    print("(条件を満たすピックなし)")
print()

# ── PART 5: 要注意 低勝率ピック (5試合以上 & 勝率40%以下) ──
print(SEP)
print("PART 5: 要注意 低勝率ピック (5試合以上 & 勝率40%以下)")
print(SEP)
bad = df_m.groupby(["summonerName", "championName", "role"]).agg(
    games=("win", "count"),
    wins=("win", "sum"),
    avg_kda=("kda", "mean"),
).reset_index()
bad["winrate"] = (bad["wins"] / bad["games"] * 100).round(1)
bad = bad[(bad["games"] >= 5) & (bad["winrate"] <= 40)].sort_values(
    "winrate", ascending=True
)

print(
    f"{'メンバー':<14} {'チャンピオン':<16} {'ロール':<8} "
    f"{'試合':>4} {'勝率':>6} {'KDA':>6}"
)
print("-" * 62)
for _, r in bad.iterrows():
    print(
        f"{r['summonerName']:<14} {r['championName']:<16} {r['role']:<8} "
        f"{r['games']:>4} {r['winrate']:>5.1f}% {r['avg_kda']:>6.2f}"
    )
if len(bad) == 0:
    print("(条件を満たすピックなし)")

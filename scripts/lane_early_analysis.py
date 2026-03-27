"""Analyze which lane's early game advantage/disadvantage correlates most with winning/losing."""
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
members = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get("members", [])}

tf = pd.read_csv(DATA / "timeline_frames.csv")
tf["riotId"] = tf["summonerName"].astype(str) + "#" + tf["tagLine"].astype(str)
tf = tf[tf["riotId"].isin(members)]
tf = tf[tf["role"].notna()]

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_JP = {"TOP": "トップ", "JUNGLE": "ジャングル", "MIDDLE": "ミッド", "BOTTOM": "ボット(ADC)", "UTILITY": "サポート"}

# Load player_stats to map members to roles
ps = pd.read_csv(DATA / "player_stats.csv")
ps["riotId"] = ps["summonerName"].astype(str) + "#" + ps["tagLine"].astype(str)
ps_members = ps[ps["riotId"].isin(members)]

for minute in [10, 15]:
    print()
    print("=" * 60)
    print(f"  {minute}分時点：レーン別 序盤パフォーマンス vs 試合勝率")
    print("=" * 60)

    snap = tf[tf["timestampMin"] == minute].copy()
    if snap.empty:
        print(f"  {minute}分のデータなし")
        continue

    print()
    print("【序盤勝ち(ゴールド差 > 0)のときの試合勝率】")
    print(f"  {'レーン':<14} {'勝率':>8} {'該当試合数':>10} {'全試合数':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    results_ahead = []
    for role in ROLES:
        role_data = snap[snap["role"] == role]
        ahead = role_data[role_data["goldDiffVsOpponent"] > 0]
        if len(ahead) > 0:
            wr = ahead["win"].mean() * 100
            results_ahead.append((role, wr, len(ahead), len(role_data)))
    results_ahead.sort(key=lambda x: x[1], reverse=True)
    for role, wr, n, total in results_ahead:
        print(f"  {ROLE_JP[role]:<14} {wr:>6.1f}% {n:>10} {total:>10}")

    print()
    print("【序盤負け(ゴールド差 < 0)のときの試合勝率】")
    print(f"  {'レーン':<14} {'勝率':>8} {'該当試合数':>10} {'全試合数':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    results_behind = []
    for role in ROLES:
        role_data = snap[snap["role"] == role]
        behind = role_data[role_data["goldDiffVsOpponent"] < 0]
        if len(behind) > 0:
            wr = behind["win"].mean() * 100
            results_behind.append((role, wr, len(behind), len(role_data)))
    results_behind.sort(key=lambda x: x[1])
    for role, wr, n, total in results_behind:
        print(f"  {ROLE_JP[role]:<14} {wr:>6.1f}% {n:>10} {total:>10}")

    print()
    print("【ゴールド差と勝敗の相関係数（高いほど序盤の差が勝敗に直結）】")
    print(f"  {'レーン':<14} {'相関係数':>10} {'影響度':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*8}")
    corr_results = []
    for role in ROLES:
        role_data = snap[snap["role"] == role]
        if len(role_data) > 2:
            corr = role_data["goldDiffVsOpponent"].corr(role_data["win"].astype(float))
            corr_results.append((role, corr))
    corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
    for role, corr in corr_results:
        stars = "★★★" if abs(corr) > 0.35 else "★★" if abs(corr) > 0.25 else "★"
        print(f"  {ROLE_JP[role]:<14} {corr:>+8.3f}   {stars}")

    # Additional: average gold diff when winning vs losing
    print()
    print("【勝ち試合 vs 負け試合の平均ゴールド差】")
    print(f"  {'レーン':<14} {'勝ち時の平均差':>14} {'負け時の平均差':>14} {'差分':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    for role in ROLES:
        role_data = snap[snap["role"] == role]
        win_avg = role_data[role_data["win"] == True]["goldDiffVsOpponent"].mean()
        lose_avg = role_data[role_data["win"] == False]["goldDiffVsOpponent"].mean()
        diff = win_avg - lose_avg
        print(f"  {ROLE_JP[role]:<14} {win_avg:>+12.0f}G {lose_avg:>+12.0f}G {diff:>+8.0f}G")

    # Lane ahead rate (probability of being ahead at this minute)
    print()
    print("【レーン別 序盤勝ち確率（ゴールド差 > 0 の割合）】")
    print(f"  {'レーン':<14} {'序盤勝ち率':>10} {'勝ち':>6} {'負け':>6} {'イーブン':>8} {'合計':>6}")
    print(f"  {'-'*14} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    lane_rate_results = []
    for role in ROLES:
        role_data = snap[snap["role"] == role]
        total = len(role_data)
        ahead = len(role_data[role_data["goldDiffVsOpponent"] > 0])
        behind = len(role_data[role_data["goldDiffVsOpponent"] < 0])
        even = len(role_data[role_data["goldDiffVsOpponent"] == 0])
        rate = ahead / total * 100 if total > 0 else 0
        lane_rate_results.append((role, rate, ahead, behind, even, total))
    lane_rate_results.sort(key=lambda x: x[1], reverse=True)
    for role, rate, ahead, behind, even, total in lane_rate_results:
        print(f"  {ROLE_JP[role]:<14} {rate:>8.1f}% {ahead:>6} {behind:>6} {even:>8} {total:>6}")

    # Per-member lane ahead rate
    print()
    print("【メンバー×レーン別 序盤勝ち確率】")
    member_role_data = snap.copy()
    member_role_data = member_role_data[member_role_data["riotId"].isin(members)]
    pivot = member_role_data.groupby(["summonerName", "role"]).apply(
        lambda g: pd.Series({
            "序盤勝ち率": f"{(g['goldDiffVsOpponent'] > 0).mean() * 100:.1f}%",
            "試合数": len(g),
            "平均差": f"{g['goldDiffVsOpponent'].mean():+.0f}G",
        })
    ).reset_index()
    if not pivot.empty:
        for name in sorted(member_role_data["summonerName"].unique()):
            member_data = pivot[pivot["summonerName"] == name]
            if member_data.empty:
                continue
            print(f"  {name}:")
            for _, row in member_data.iterrows():
                role_name = ROLE_JP.get(row["role"], row["role"])
                print(f"    {role_name:<14} 序盤勝ち率: {row['序盤勝ち率']:>7}  平均差: {row['平均差']:>8}  ({row['試合数']}試合)")

print()
print("=" * 60)
print("  まとめ")
print("=" * 60)
print()

# Final summary using 15-min data
snap15 = tf[tf["timestampMin"] == 15].copy()
if not snap15.empty:
    corr_final = []
    for role in ROLES:
        role_data = snap15[snap15["role"] == role]
        if len(role_data) > 2:
            corr = role_data["goldDiffVsOpponent"].corr(role_data["win"].astype(float))
            corr_final.append((role, corr))
    corr_final.sort(key=lambda x: abs(x[1]), reverse=True)

    best = corr_final[0]
    worst_impact = corr_final[0]
    print(f"  序盤に最も勝敗に影響するレーン: {ROLE_JP[best[0]]}（相関 {best[1]:+.3f}）")
    print(f"  序盤に最も勝敗に影響しにくいレーン: {ROLE_JP[corr_final[-1][0]]}（相関 {corr_final[-1][1]:+.3f}）")
    print()

    # Winrate when ahead vs behind for each role
    print("  レーン別「序盤勝ち時の勝率」と「序盤負け時の勝率」の差:")
    swing_results = []
    for role in ROLES:
        role_data = snap15[snap15["role"] == role]
        ahead = role_data[role_data["goldDiffVsOpponent"] > 0]
        behind = role_data[role_data["goldDiffVsOpponent"] < 0]
        if len(ahead) > 0 and len(behind) > 0:
            wr_ahead = ahead["win"].mean() * 100
            wr_behind = behind["win"].mean() * 100
            swing = wr_ahead - wr_behind
            swing_results.append((role, wr_ahead, wr_behind, swing))
    swing_results.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'レーン':<14} {'勝ち時勝率':>10} {'負け時勝率':>10} {'差(スイング)':>14}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*14}")
    for role, wa, wb, sw in swing_results:
        print(f"  {ROLE_JP[role]:<14} {wa:>8.1f}% {wb:>8.1f}% {sw:>+12.1f}pp")
print()

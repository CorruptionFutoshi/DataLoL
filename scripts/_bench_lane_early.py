"""Benchmark (Emerald) lane-by-lane early gold diff vs win rate analysis."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

tf = pd.read_csv(DATA / "benchmark_timeline_frames.csv")
bs = pd.read_csv(DATA / "benchmark_stats.csv")

ROLE_JP = {
    "TOP": "トップ",
    "JUNGLE": "ジャングル",
    "MIDDLE": "ミッド",
    "BOTTOM": "ボトム(ADC)",
    "UTILITY": "サポート",
}
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

print(f"  ベンチマークデータ: {bs['matchId'].nunique():,} 試合 / {len(bs):,} プレイヤーレコード")
print()

for minute in [10, 15]:
    print("=" * 70)
    print(f"  エメラルド帯: {minute}分時点 レーン別 序盤ゴールド差 vs 勝率")
    print("=" * 70)

    frame = tf[tf["timestampMin"] == minute].copy()
    if frame.empty:
        print(f"  ※ {minute}分のデータがありません")
        continue

    frame = frame.dropna(subset=["goldDiffVsOpponent"])

    # --- 1. Win rate when ahead / behind per role ---
    print()
    print("【優勢時(GD > 0)の勝率】")
    print(f"  {'レーン':<14} {'勝率':>8} {'該当数':>10} {'全試合数':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")

    results = []
    for role in ROLES:
        rd = frame[frame["role"] == role]
        total = len(rd)
        ahead = rd[rd["goldDiffVsOpponent"] > 0]
        behind = rd[rd["goldDiffVsOpponent"] < 0]
        ahead_wr = ahead["win"].mean() * 100 if len(ahead) > 0 else 0
        behind_wr = behind["win"].mean() * 100 if len(behind) > 0 else 0
        corr = rd[["goldDiffVsOpponent", "win"]].corr().iloc[0, 1]

        win_gd = rd[rd["win"] == True]["goldDiffVsOpponent"].mean()
        lose_gd = rd[rd["win"] == False]["goldDiffVsOpponent"].mean()

        results.append({
            "role": role,
            "total": total,
            "ahead_n": len(ahead),
            "behind_n": len(behind),
            "ahead_wr": ahead_wr,
            "behind_wr": behind_wr,
            "corr": corr,
            "win_gd": win_gd,
            "lose_gd": lose_gd,
        })

    results_sorted = sorted(results, key=lambda x: x["ahead_wr"], reverse=True)
    for r in results_sorted:
        print(f"  {ROLE_JP[r['role']]:<14} {r['ahead_wr']:>7.1f}% {r['ahead_n']:>10} {r['total']:>10}")

    print()
    print("【劣勢時(GD < 0)の勝率】")
    print(f"  {'レーン':<14} {'勝率':>8} {'該当数':>10} {'全試合数':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    results_sorted2 = sorted(results, key=lambda x: x["behind_wr"])
    for r in results_sorted2:
        print(f"  {ROLE_JP[r['role']]:<14} {r['behind_wr']:>7.1f}% {r['behind_n']:>10} {r['total']:>10}")

    print()
    print("【ゴールド差と勝敗の相関係数（大きいほど序盤の差が勝敗に直結）】")
    print(f"  {'レーン':<14} {'相関係数':>10} {'影響度':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*8}")
    results_sorted3 = sorted(results, key=lambda x: x["corr"], reverse=True)
    for r in results_sorted3:
        if r["corr"] >= 0.30:
            level = "最強"
        elif r["corr"] >= 0.20:
            level = "強"
        elif r["corr"] >= 0.15:
            level = "中"
        else:
            level = "弱"
        print(f"  {ROLE_JP[r['role']]:<14} {r['corr']:>+10.3f} {level:>8}")

    print()
    print("【勝ち試合 vs 負け試合の平均ゴールド差】")
    print(f"  {'レーン':<14} {'勝ち時の平均GD':>14} {'負け時の平均GD':>14} {'スイング':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    for r in sorted(results, key=lambda x: x["win_gd"] - x["lose_gd"], reverse=True):
        swing = r["win_gd"] - r["lose_gd"]
        print(f"  {ROLE_JP[r['role']]:<14} {r['win_gd']:>+13.0f}G {r['lose_gd']:>+13.0f}G {swing:>+9.0f}G")

    print()
    print("【優勢率（GD > 0 の割合）】")
    print(f"  {'レーン':<14} {'優勢率':>10} {'優勢':>6} {'劣勢':>6} {'合計':>6}")
    print(f"  {'-'*14} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")
    for r in sorted(results, key=lambda x: x["ahead_n"] / x["total"] if x["total"] > 0 else 0, reverse=True):
        rate = r["ahead_n"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {ROLE_JP[r['role']]:<14} {rate:>9.1f}% {r['ahead_n']:>6} {r['behind_n']:>6} {r['total']:>6}")

    # Swing summary
    print()
    print("【スイング（優勢時勝率 - 劣勢時勝率）】")
    print(f"  {'レーン':<14} {'優勢時勝率':>10} {'劣勢時勝率':>10} {'差(スイング)':>14}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*14}")
    for r in sorted(results, key=lambda x: x["ahead_wr"] - x["behind_wr"], reverse=True):
        swing = r["ahead_wr"] - r["behind_wr"]
        print(f"  {ROLE_JP[r['role']]:<14} {r['ahead_wr']:>9.1f}% {r['behind_wr']:>9.1f}% {swing:>+13.1f}pp")

print()
print("=" * 70)
print("  自チーム vs エメラルド帯 比較サマリー（15分時点）")
print("=" * 70)

# Load member data for comparison
mtf = pd.read_csv(DATA / "timeline_frames.csv")
import yaml
with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
members = {m["game_name"] for m in cfg.get("members", [])}

mf15 = mtf[(mtf["timestampMin"] == 15) & (mtf["summonerName"].isin(members))].dropna(subset=["goldDiffVsOpponent"])
bf15 = tf[tf["timestampMin"] == 15].dropna(subset=["goldDiffVsOpponent"])

print()
print(f"  {'レーン':<14} {'チーム相関':>10} {'エメラルド相関':>14} {'差':>8}  {'チームスイング':>14} {'エメラルドスイング':>18}")
print(f"  {'-'*14} {'-'*10} {'-'*14} {'-'*8}  {'-'*14} {'-'*18}")

for role in ROLES:
    mr = mf15[mf15["role"] == role]
    br = bf15[bf15["role"] == role]

    m_corr = mr[["goldDiffVsOpponent", "win"]].corr().iloc[0, 1] if len(mr) > 10 else float("nan")
    b_corr = br[["goldDiffVsOpponent", "win"]].corr().iloc[0, 1] if len(br) > 10 else float("nan")

    m_ahead_wr = mr[mr["goldDiffVsOpponent"] > 0]["win"].mean() * 100 if len(mr[mr["goldDiffVsOpponent"] > 0]) > 0 else 0
    m_behind_wr = mr[mr["goldDiffVsOpponent"] < 0]["win"].mean() * 100 if len(mr[mr["goldDiffVsOpponent"] < 0]) > 0 else 0
    b_ahead_wr = br[br["goldDiffVsOpponent"] > 0]["win"].mean() * 100 if len(br[br["goldDiffVsOpponent"] > 0]) > 0 else 0
    b_behind_wr = br[br["goldDiffVsOpponent"] < 0]["win"].mean() * 100 if len(br[br["goldDiffVsOpponent"] < 0]) > 0 else 0

    m_swing = m_ahead_wr - m_behind_wr
    b_swing = b_ahead_wr - b_behind_wr
    diff = m_corr - b_corr

    print(f"  {ROLE_JP[role]:<14} {m_corr:>+10.3f} {b_corr:>+14.3f} {diff:>+8.3f}  {m_swing:>+13.1f}pp {b_swing:>+17.1f}pp")

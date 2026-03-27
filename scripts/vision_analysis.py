"""視界（ビジョン）の包括的分析。

チームの「視界がない」「すぐ消される」という感覚を
データで検証する。

分析項目:
  1. チーム vs 敵チームのビジョンスコア / ワード設置数 / ワード除去数
  2. メンバー別のビジョン貢献
  3. ロール別のビジョン比較 (チーム vs 敵)
  4. ワード生存分析（タイムライン: 設置→破壊の時間差）
  5. ビジョンスコアと勝率の相関
  6. エメラルドベンチマークとの比較

Usage:
    python scripts/vision_analysis.py
"""

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
    "TOP": "トップ",
    "JUNGLE": "ジャングル",
    "MIDDLE": "ミッド",
    "BOTTOM": "ボトム",
    "UTILITY": "サポート",
}
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def load():
    ps = pd.read_csv(DATA / "player_stats.csv")
    matches = pd.read_csv(DATA / "matches.csv")
    ev = pd.read_csv(DATA / "timeline_events.csv")

    ps = ps[ps["role"].isin(ROLES)].copy()
    ps["is_member"] = ps["summonerName"].isin(MEMBER_NAMES)

    dur = matches.set_index("matchId")["gameDurationMin"].to_dict()
    ps["gameDurationMin"] = ps["matchId"].map(dur)
    ps["vision_per_min"] = ps["visionScore"] / ps["gameDurationMin"].clip(lower=1)
    ps["wards_placed_per_min"] = ps["wardsPlaced"] / ps["gameDurationMin"].clip(lower=1)
    ps["wards_killed_per_min"] = ps["wardsKilled"] / ps["gameDurationMin"].clip(lower=1)

    member_matches = ps.loc[ps["is_member"], "matchId"].unique()
    ps = ps[ps["matchId"].isin(member_matches)].copy()

    member_team = (
        ps[ps["is_member"]]
        .groupby("matchId")["teamId"]
        .first()
        .to_dict()
    )
    ps["is_ally_team"] = ps.apply(
        lambda r: r["teamId"] == member_team.get(r["matchId"]), axis=1
    )

    bm = None
    bm_path = DATA / "benchmark_stats.csv"
    if bm_path.exists():
        bm = pd.read_csv(bm_path)
        bm = bm[bm["role"].isin(ROLES)].copy()
        bm["vision_per_min"] = bm["visionScore"] / bm["gameDurationMin"].clip(lower=1)
        bm["wards_placed_per_min"] = bm["wardsPlaced"] / bm["gameDurationMin"].clip(lower=1)
        bm["wards_killed_per_min"] = bm["wardsKilled"] / bm["gameDurationMin"].clip(lower=1)

    return ps, ev, matches, bm


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def part1_team_vs_enemy(ps):
    section("1. チーム vs 敵チーム — ビジョン概要")

    team_agg = (
        ps.groupby(["matchId", "is_ally_team"])
        .agg(
            vision=("visionScore", "sum"),
            placed=("wardsPlaced", "sum"),
            killed=("wardsKilled", "sum"),
            dur=("gameDurationMin", "first"),
        )
        .reset_index()
    )
    team_agg["vision_pm"] = team_agg["vision"] / team_agg["dur"].clip(lower=1)
    team_agg["placed_pm"] = team_agg["placed"] / team_agg["dur"].clip(lower=1)
    team_agg["killed_pm"] = team_agg["killed"] / team_agg["dur"].clip(lower=1)

    ally = team_agg[team_agg["is_ally_team"]]
    enemy = team_agg[~team_agg["is_ally_team"]]

    metrics = [
        ("ビジョンスコア/min", "vision_pm"),
        ("ワード設置数/min", "placed_pm"),
        ("ワード除去数/min", "killed_pm"),
        ("ビジョンスコア合計", "vision"),
        ("ワード設置数合計", "placed"),
        ("ワード除去数合計", "killed"),
    ]

    print(f"\n  対象試合数: {ally['matchId'].nunique()}")
    print(f"\n  {'指標':<20} {'味方チーム':>12} {'敵チーム':>12} {'差分':>10} {'判定':>6}")
    print(f"  {'-'*62}")
    for label, col in metrics:
        a_val = ally[col].mean()
        e_val = enemy[col].mean()
        diff = a_val - e_val
        verdict = "✓" if diff >= 0 else "✗"
        fmt = ".2f" if "pm" in col or "min" in col else ".1f"
        print(f"  {label:<20} {a_val:>12{fmt}} {e_val:>12{fmt}} {diff:>+10{fmt}}  {verdict}")

    print("\n  ※ ✗ = 味方チームが劣っている指標")

    wins = ps[ps["is_member"]].drop_duplicates("matchId")
    total = len(wins)
    w = wins["win"].sum()
    print(f"\n  チーム勝率: {w}/{total} ({w/total*100:.1f}%)")


def part2_member_vision(ps):
    section("2. メンバー別ビジョン貢献")

    mem = ps[ps["is_member"]].copy()

    agg = (
        mem.groupby("summonerName")
        .agg(
            games=("matchId", "nunique"),
            role_mode=("role", lambda x: x.mode().iloc[0] if len(x) > 0 else "?"),
            vision_avg=("visionScore", "mean"),
            vision_pm=("vision_per_min", "mean"),
            placed_avg=("wardsPlaced", "mean"),
            placed_pm=("wards_placed_per_min", "mean"),
            killed_avg=("wardsKilled", "mean"),
            killed_pm=("wards_killed_per_min", "mean"),
        )
        .sort_values("vision_pm", ascending=False)
        .reset_index()
    )

    print(f"\n  {'メンバー':<16} {'主ロール':<10} {'試合':>4} "
          f"{'VS':>5} {'VS/m':>6} {'設置':>5} {'設/m':>6} {'除去':>5} {'除/m':>6}")
    print(f"  {'-'*72}")
    for _, r in agg.iterrows():
        rj = ROLE_JP.get(r["role_mode"], r["role_mode"])
        print(f"  {r['summonerName']:<16} {rj:<10} {r['games']:>4} "
              f"{r['vision_avg']:>5.1f} {r['vision_pm']:>6.2f} "
              f"{r['placed_avg']:>5.1f} {r['placed_pm']:>6.2f} "
              f"{r['killed_avg']:>5.1f} {r['killed_pm']:>6.2f}")

    print("\n  VS=ビジョンスコア, 設置=ワード設置数, 除去=ワード除去数, /m=毎分")


def part3_role_comparison(ps):
    section("3. ロール別 — 味方 vs 敵 ビジョン比較")

    for col, label in [
        ("vision_per_min", "ビジョンスコア/min"),
        ("wards_placed_per_min", "ワード設置/min"),
        ("wards_killed_per_min", "ワード除去/min"),
    ]:
        print(f"\n  【{label}】")
        print(f"  {'ロール':<12} {'味方':>8} {'敵':>8} {'差分':>8} {'判定':>6}")
        print(f"  {'-'*44}")
        for role in ROLES:
            ally_vals = ps[(ps["is_ally_team"]) & (ps["role"] == role)][col]
            enemy_vals = ps[(~ps["is_ally_team"]) & (ps["role"] == role)][col]
            a = ally_vals.mean()
            e = enemy_vals.mean()
            d = a - e
            v = "✓" if d >= 0 else "✗"
            rj = ROLE_JP.get(role, role)
            print(f"  {rj:<12} {a:>8.2f} {e:>8.2f} {d:>+8.2f}  {v}")


def part4_ward_survival(ev, ps):
    section("4. ワード生存分析（設置→破壊の時間差）")

    member_team_map = (
        ps[ps["is_member"]]
        .groupby("matchId")["teamId"]
        .first()
        .to_dict()
    )

    placed = ev[ev["eventType"] == "WARD_PLACED"].copy()
    killed = ev[ev["eventType"] == "WARD_KILL"].copy()

    placed = placed[placed["wardType"].isin(["YELLOW_TRINKET", "SIGHT_WARD", "CONTROL_WARD", "BLUE_TRINKET"])].copy()
    killed = killed[killed["wardType"].isin(["YELLOW_TRINKET", "SIGHT_WARD", "CONTROL_WARD", "BLUE_TRINKET"])].copy()

    placed["placer_team"] = placed["killerTeamId"]
    placed = placed[placed["matchId"].isin(member_team_map)].copy()
    placed["is_ally_ward"] = placed.apply(
        lambda r: r["placer_team"] == member_team_map.get(r["matchId"]), axis=1
    )

    killed = killed[killed["matchId"].isin(member_team_map)].copy()
    killed["killer_team"] = killed["killerTeamId"]
    killed["ward_was_ally"] = killed.apply(
        lambda r: r["killer_team"] != member_team_map.get(r["matchId"]), axis=1
    )

    ally_placed = placed[placed["is_ally_ward"]]
    enemy_placed = placed[~placed["is_ally_ward"]]

    ally_wards_destroyed = killed[killed["ward_was_ally"]]
    enemy_wards_destroyed = killed[~killed["ward_was_ally"]]

    ally_total_placed = len(ally_placed)
    enemy_total_placed = len(enemy_placed)

    ally_destroyed_count = len(ally_wards_destroyed)
    enemy_destroyed_count = len(enemy_wards_destroyed)

    ally_destroy_rate = ally_destroyed_count / max(ally_total_placed, 1) * 100
    enemy_destroy_rate = enemy_destroyed_count / max(enemy_total_placed, 1) * 100

    print(f"\n  {'指標':<30} {'味方ワード':>12} {'敵ワード':>12}")
    print(f"  {'-'*56}")
    print(f"  {'設置数（タイムライン）':<30} {ally_total_placed:>12,} {enemy_total_placed:>12,}")
    print(f"  {'破壊された数':<30} {ally_destroyed_count:>12,} {enemy_destroyed_count:>12,}")
    print(f"  {'被破壊率':<30} {ally_destroy_rate:>11.1f}% {enemy_destroy_rate:>11.1f}%")

    if ally_destroy_rate > enemy_destroy_rate:
        diff = ally_destroy_rate - enemy_destroy_rate
        print(f"\n  → 味方ワードは敵ワードより {diff:.1f}pp 多く壊されている（事実）")
    else:
        diff = enemy_destroy_rate - ally_destroy_rate
        print(f"\n  → 敵ワードの方が {diff:.1f}pp 多く壊されている（味方の方が除去上手）")

    print("\n  【ワードタイプ別の被破壊数】")
    print(f"  {'ワードタイプ':<20} {'味方被破壊':>10} {'敵被破壊':>10}")
    print(f"  {'-'*42}")
    for wt in ["YELLOW_TRINKET", "SIGHT_WARD", "CONTROL_WARD", "BLUE_TRINKET"]:
        wt_labels = {
            "YELLOW_TRINKET": "黄トリンケット",
            "SIGHT_WARD": "ステルスワード",
            "CONTROL_WARD": "コントロールワード",
            "BLUE_TRINKET": "青トリンケット",
        }
        a = len(ally_wards_destroyed[ally_wards_destroyed["wardType"] == wt])
        e = len(enemy_wards_destroyed[enemy_wards_destroyed["wardType"] == wt])
        print(f"  {wt_labels.get(wt, wt):<20} {a:>10} {e:>10}")


def part5_vision_winrate(ps):
    section("5. ビジョンスコアと勝率の相関")

    team_vision = (
        ps[ps["is_ally_team"]]
        .groupby("matchId")
        .agg(
            vision_total=("visionScore", "sum"),
            vision_pm=("vision_per_min", "sum"),
            placed=("wardsPlaced", "sum"),
            killed=("wardsKilled", "sum"),
            win=("win", "first"),
            dur=("gameDurationMin", "first"),
        )
        .reset_index()
    )

    enemy_vision = (
        ps[~ps["is_ally_team"]]
        .groupby("matchId")
        .agg(enemy_vision=("visionScore", "sum"))
        .reset_index()
    )
    team_vision = team_vision.merge(enemy_vision, on="matchId")
    team_vision["vision_diff"] = team_vision["vision_total"] - team_vision["enemy_vision"]

    team_vision["vision_advantage"] = team_vision["vision_diff"] > 0

    adv_wr = team_vision[team_vision["vision_advantage"]]["win"].mean() * 100
    disadv_wr = team_vision[~team_vision["vision_advantage"]]["win"].mean() * 100
    adv_n = team_vision["vision_advantage"].sum()
    disadv_n = (~team_vision["vision_advantage"]).sum()

    print(f"\n  ビジョン優勢（味方VS合計 > 敵VS合計）時の勝率:")
    print(f"    優勢: {adv_wr:.1f}% ({adv_n}試合)")
    print(f"    劣勢: {disadv_wr:.1f}% ({disadv_n}試合)")
    print(f"    差: {adv_wr - disadv_wr:+.1f}pp")

    q_labels = ["下位25%", "25-50%", "50-75%", "上位25%"]
    team_vision["vision_q"] = pd.qcut(
        team_vision["vision_diff"], 4, labels=q_labels, duplicates="drop"
    )
    print(f"\n  ビジョン差分の四分位と勝率:")
    print(f"  {'区分':<12} {'勝率':>8} {'試合数':>8} {'平均VS差':>10}")
    print(f"  {'-'*40}")
    for q in q_labels:
        sub = team_vision[team_vision["vision_q"] == q]
        if len(sub) == 0:
            continue
        wr = sub["win"].mean() * 100
        n = len(sub)
        vd = sub["vision_diff"].mean()
        print(f"  {q:<12} {wr:>7.1f}% {n:>8} {vd:>+10.1f}")

    corr = team_vision["vision_diff"].corr(team_vision["win"].astype(float))
    print(f"\n  ビジョン差分 ↔ 勝利の相関係数: r = {corr:.3f}")


def part6_benchmark(ps, bm):
    section("6. エメラルドベンチマークとの比較")

    if bm is None:
        print("\n  ベンチマークデータなし（benchmark_stats.csv が見つかりません）")
        return

    mem = ps[ps["is_member"]].copy()

    for col, label in [
        ("vision_per_min", "ビジョンスコア/min"),
        ("wards_placed_per_min", "ワード設置/min"),
        ("wards_killed_per_min", "ワード除去/min"),
    ]:
        print(f"\n  【{label}】")
        print(f"  {'ロール':<12} {'メンバー平均':>12} {'エメラルド平均':>14} {'差分':>8} {'パーセンタイル':>14}")
        print(f"  {'-'*64}")
        for role in ROLES:
            m_vals = mem[mem["role"] == role][col]
            b_vals = bm[bm["role"] == role][col]
            if len(m_vals) < 5 or len(b_vals) < 5:
                continue
            m_avg = m_vals.mean()
            b_avg = b_vals.mean()
            diff = m_avg - b_avg
            pctile = (b_vals < m_avg).mean() * 100
            rj = ROLE_JP.get(role, role)
            print(f"  {rj:<12} {m_avg:>12.2f} {b_avg:>14.2f} {diff:>+8.2f} {pctile:>13.0f}%")

    print(f"\n  【メンバー別 — ビジョンスコア/min パーセンタイル（ロール別）】")
    print(f"  {'メンバー':<16} {'ロール':<10} {'本人':>6} {'エメラルド':>10} {'pctile':>8}")
    print(f"  {'-'*52}")
    for name in sorted(MEMBER_NAMES):
        m_data = mem[mem["summonerName"] == name]
        if len(m_data) < 3:
            continue
        main_role = m_data["role"].mode().iloc[0]
        m_avg = m_data[m_data["role"] == main_role]["vision_per_min"].mean()
        b_vals = bm[bm["role"] == main_role]["vision_per_min"]
        if len(b_vals) < 5:
            continue
        pctile = (b_vals < m_avg).mean() * 100
        b_avg = b_vals.mean()
        rj = ROLE_JP.get(main_role, main_role)
        print(f"  {name:<16} {rj:<10} {m_avg:>6.2f} {b_avg:>10.2f} {pctile:>7.0f}%")


def part7_member_vs_direct_opponent(ps):
    section("7. メンバー vs 直接の対面 — ビジョン勝率")

    mem = ps[ps["is_member"]].copy()
    enemy_same_role = ps[~ps["is_ally_team"]].copy()

    merged = mem.merge(
        enemy_same_role[["matchId", "role", "vision_per_min", "wards_placed_per_min", "wards_killed_per_min"]],
        on=["matchId", "role"],
        suffixes=("", "_enemy"),
    )

    merged["vision_won"] = merged["vision_per_min"] > merged["vision_per_min_enemy"]
    merged["placed_won"] = merged["wards_placed_per_min"] > merged["wards_placed_per_min_enemy"]
    merged["killed_won"] = merged["wards_killed_per_min"] > merged["wards_killed_per_min_enemy"]

    print(f"\n  対面とのビジョン比較で勝っている割合（%）")
    print(f"  {'メンバー':<16} {'試合':>4} {'VS/m勝率':>10} {'設置/m勝率':>10} {'除去/m勝率':>10}")
    print(f"  {'-'*54}")
    for name in sorted(MEMBER_NAMES):
        sub = merged[merged["summonerName"] == name]
        if len(sub) < 3:
            continue
        n = len(sub)
        vw = sub["vision_won"].mean() * 100
        pw = sub["placed_won"].mean() * 100
        kw = sub["killed_won"].mean() * 100
        print(f"  {name:<16} {n:>4} {vw:>9.1f}% {pw:>9.1f}% {kw:>9.1f}%")

    overall_v = merged["vision_won"].mean() * 100
    overall_p = merged["placed_won"].mean() * 100
    overall_k = merged["killed_won"].mean() * 100
    print(f"  {'チーム全体':<16} {len(merged):>4} {overall_v:>9.1f}% {overall_p:>9.1f}% {overall_k:>9.1f}%")


def part8_conclusion(ps, ev):
    section("8. 総合診断")

    team_agg = (
        ps.groupby(["matchId", "is_ally_team"])
        .agg(vision=("visionScore", "sum"), placed=("wardsPlaced", "sum"), killed=("wardsKilled", "sum"))
        .reset_index()
    )
    ally = team_agg[team_agg["is_ally_team"]]
    enemy = team_agg[~team_agg["is_ally_team"]]

    v_diff = ally["vision"].mean() - enemy["vision"].mean()
    p_diff = ally["placed"].mean() - enemy["placed"].mean()
    k_diff = ally["killed"].mean() - enemy["killed"].mean()

    findings = []

    if v_diff < -3:
        findings.append(f"  ✗ ビジョンスコアが敵より平均 {abs(v_diff):.1f} 低い → 視界で負けている")
    elif v_diff < 0:
        findings.append(f"  △ ビジョンスコアが敵よりやや低い（差: {v_diff:.1f}）→ ほぼ互角")
    else:
        findings.append(f"  ✓ ビジョンスコアは敵以上（差: {v_diff:+.1f}）")

    if p_diff < -2:
        findings.append(f"  ✗ ワード設置数が敵より平均 {abs(p_diff):.1f} 少ない → 設置不足")
    elif p_diff < 0:
        findings.append(f"  △ ワード設置数がやや少ない（差: {p_diff:.1f}）")
    else:
        findings.append(f"  ✓ ワード設置数は敵以上（差: {p_diff:+.1f}）")

    if k_diff < -2:
        findings.append(f"  ✗ ワード除去数が敵より平均 {abs(k_diff):.1f} 少ない → 除去不足")
    elif k_diff < 0:
        findings.append(f"  △ ワード除去数がやや少ない（差: {k_diff:.1f}）")
    else:
        findings.append(f"  ✓ ワード除去数は敵以上（差: {k_diff:+.1f}）")

    print()
    for f in findings:
        print(f)

    sup = ps[(ps["is_member"]) & (ps["role"] == "UTILITY")]
    jg = ps[(ps["is_member"]) & (ps["role"] == "JUNGLE")]
    sup_enemy = ps[(~ps["is_ally_team"]) & (ps["role"] == "UTILITY")]
    jg_enemy = ps[(~ps["is_ally_team"]) & (ps["role"] == "JUNGLE")]

    print(f"\n  サポートのビジョン/min: 味方 {sup['vision_per_min'].mean():.2f} vs 敵 {sup_enemy['vision_per_min'].mean():.2f}")
    print(f"  ジャングルのビジョン/min: 味方 {jg['vision_per_min'].mean():.2f} vs 敵 {jg_enemy['vision_per_min'].mean():.2f}")


def main():
    ps, ev, matches, bm = load()

    print("=" * 60)
    print("  LoL フレックス ビジョン（視界）包括分析")
    print("=" * 60)

    part1_team_vs_enemy(ps)
    part2_member_vision(ps)
    part3_role_comparison(ps)
    part4_ward_survival(ev, ps)
    part5_vision_winrate(ps)
    part6_benchmark(ps, bm)
    part7_member_vs_direct_opponent(ps)
    part8_conclusion(ps, ev)


if __name__ == "__main__":
    main()

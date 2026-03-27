"""On-demand analysis CLI for LoL Flex Rank data.

Usage:
    python src/analyze.py overview
    python src/analyze.py champion [--member NAME]
    python src/analyze.py synergy
    python src/analyze.py trends
    python src/analyze.py early [--minute 15]
    python src/analyze.py objectives
    python src/analyze.py tempo
    python src/analyze.py member NAME
    python src/analyze.py match MATCH_ID
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"


def _load_members() -> set[str]:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get("members", [])}


def _load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        print(f"ERROR: {path} not found. Run process.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def _member_filter(df: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["riotId"] = df["summonerName"].astype(str) + "#" + df["tagLine"].astype(str)
    return df[df["riotId"].isin(members)]


# ------------------------------------------------------------------
# overview
# ------------------------------------------------------------------
def cmd_overview(_args):
    members = _load_members()
    df = _load("player_stats.csv")
    dm = _member_filter(df, members)

    print("=" * 60)
    print("  FLEX RANK 全体サマリー")
    print("=" * 60)

    total_matches = dm["matchId"].nunique()
    wins = dm.drop_duplicates("matchId")["win"].sum()
    print(f"\n総試合数: {total_matches}  勝利: {int(wins)}  敗北: {total_matches - int(wins)}")
    print(f"チーム全体勝率: {wins / total_matches * 100:.1f}%\n")

    summary = dm.groupby("summonerName").agg(
        試合数=("win", "count"),
        勝利=("win", "sum"),
        勝率=("win", "mean"),
        平均KDA=("kda", "mean"),
        平均キル=("kills", "mean"),
        平均デス=("deaths", "mean"),
        平均アシスト=("assists", "mean"),
    ).round(2)
    summary["勝率"] = (summary["勝率"] * 100).round(1)
    summary.sort_values("勝率", ascending=False, inplace=True)
    print(summary.to_string())


# ------------------------------------------------------------------
# champion
# ------------------------------------------------------------------
def cmd_champion(args):
    members = _load_members()
    df = _load("player_stats.csv")
    dm = _member_filter(df, members)

    if args.member:
        dm = dm[dm["summonerName"] == args.member]
        if dm.empty:
            print(f"Member '{args.member}' not found.")
            return

    print("=" * 60)
    print("  チャンピオン別成績")
    print("=" * 60)

    champ = dm.groupby(["summonerName", "championName"]).agg(
        試合数=("win", "count"),
        勝利=("win", "sum"),
        勝率=("win", "mean"),
        平均KDA=("kda", "mean"),
        平均CS=("cs", "mean"),
        平均ダメージ=("totalDamageDealtToChampions", "mean"),
    ).round(1)
    champ["勝率"] = (champ["勝率"] * 100).round(1)
    champ = champ[champ["試合数"] >= 2].sort_values(
        ["summonerName", "勝率"], ascending=[True, False]
    )
    print()

    for name, grp in champ.groupby(level=0):
        print(f"\n--- {name} ---")
        print(grp.droplevel(0).head(10).to_string())


# ------------------------------------------------------------------
# synergy
# ------------------------------------------------------------------
def cmd_synergy(_args):
    members = _load_members()
    df = _load("player_stats.csv")
    dm = _member_filter(df, members)
    from itertools import combinations

    match_members = dm.groupby("matchId").agg(
        members=("summonerName", lambda x: frozenset(x)),
        win=("win", "first"),
    ).reset_index()

    all_names = sorted(dm["summonerName"].unique())

    print("=" * 60)
    print("  デュオ組み合わせ勝率 (≥3試合)")
    print("=" * 60)

    rows = []
    for a, b in combinations(all_names, 2):
        mask = match_members["members"].apply(lambda s: a in s and b in s)
        sub = match_members[mask]
        if len(sub) >= 3:
            rows.append({"Duo": f"{a} + {b}", "試合数": len(sub),
                         "勝率%": round(sub["win"].mean() * 100, 1)})
    if rows:
        print(pd.DataFrame(rows).sort_values("勝率%", ascending=False).to_string(index=False))
    else:
        print("3試合以上のデュオデータなし")


# ------------------------------------------------------------------
# trends
# ------------------------------------------------------------------
def cmd_trends(_args):
    members = _load_members()
    df = _load("player_stats.csv")
    df["gameCreation"] = pd.to_datetime(df["gameCreation"])
    dm = _member_filter(df, members).sort_values("gameCreation")

    print("=" * 60)
    print("  勝率トレンド")
    print("=" * 60)

    dm["week"] = dm["gameCreation"].dt.strftime("%Y-W%U")
    weekly = dm.drop_duplicates("matchId").groupby("week").agg(
        試合数=("win", "count"),
        勝率=("win", "mean"),
    )
    weekly["勝率%"] = (weekly["勝率"] * 100).round(1)
    print("\n週別勝率:")
    print(weekly[["試合数", "勝率%"]].to_string())

    dm["hour"] = dm["gameCreation"].dt.hour
    hourly = dm.groupby("hour").agg(試合数=("win", "count"), 勝率=("win", "mean"))
    hourly["勝率%"] = (hourly["勝率"] * 100).round(1)
    print("\n時間帯別勝率:")
    print(hourly[["試合数", "勝率%"]].to_string())


# ------------------------------------------------------------------
# early game
# ------------------------------------------------------------------
def cmd_early(args):
    minute = args.minute
    members = _load_members()
    frames = _load("timeline_frames.csv")
    matches = _load("matches.csv")

    print("=" * 60)
    print(f"  アーリーゲーム分析 ({minute}分時点)")
    print("=" * 60)

    f_at = frames[frames["timestampMin"] == float(minute)]
    if f_at.empty:
        print(f"{minute}分のデータがありません")
        return

    team_gold = f_at.groupby(["matchId", "teamId"])["totalGold"].sum().unstack(fill_value=0)
    team_gold.columns = ["gold_100", "gold_200"]
    team_gold["goldDiff"] = team_gold["gold_100"] - team_gold["gold_200"]
    team_gold = team_gold.reset_index().merge(
        matches[["matchId", "team100Win"]], on="matchId", how="left"
    )

    bins = [-np.inf, -3000, -1000, 0, 1000, 3000, np.inf]
    labels = ["<-3k", "-3k~-1k", "-1k~0", "0~1k", "1k~3k", ">3k"]
    team_gold["bin"] = pd.cut(team_gold["goldDiff"], bins=bins, labels=labels)
    bin_wr = team_gold.groupby("bin", observed=False).agg(
        試合数=("team100Win", "count"), 勝率=("team100Win", "mean")
    )
    bin_wr["勝率%"] = (bin_wr["勝率"] * 100).round(1)
    print(f"\n{minute}分ゴールド差ビン別 勝率:")
    print(bin_wr[["試合数", "勝率%"]].to_string())

    f_at_copy = f_at.copy()
    f_at_copy["riotId"] = f_at_copy["summonerName"].astype(str) + "#" + f_at_copy["tagLine"].astype(str)
    member_at = f_at_copy[f_at_copy["riotId"].isin(members)]
    if "goldDiffVsOpponent" in member_at.columns:
        med = member_at.groupby("summonerName")["goldDiffVsOpponent"].median().round(0)
        print(f"\nメンバー別 {minute}分ゴールド差中央値 (vs 対面):")
        print(med.sort_values(ascending=False).to_string())


# ------------------------------------------------------------------
# objectives
# ------------------------------------------------------------------
def cmd_objectives(_args):
    members = _load_members()
    obj = _load("objectives.csv")
    matches = _load("matches.csv")
    players = _load("player_stats.csv")
    players["riotId"] = players["summonerName"].astype(str) + "#" + players["tagLine"].astype(str)
    mt = players[players["riotId"].isin(members)][["matchId", "teamId"]].drop_duplicates()

    print("=" * 60)
    print("  オブジェクト分析")
    print("=" * 60)

    dragons = obj[obj["objectiveType"] == "DRAGON"]
    dc = dragons.groupby(["matchId", "teamId"]).size().reset_index(name="count")
    dc = dc.merge(mt, on=["matchId", "teamId"], how="inner")
    dc = dc.merge(matches[["matchId", "team100Win"]], on="matchId", how="left")
    dc["win"] = ((dc["teamId"] == 100) & dc["team100Win"]) | (
        (dc["teamId"] == 200) & ~dc["team100Win"]
    )
    dragon_wr = dc.groupby("count").agg(試合数=("win", "count"), 勝率=("win", "mean"))
    dragon_wr["勝率%"] = (dragon_wr["勝率"] * 100).round(1)
    print("\nドラゴン獲得数別 勝率:")
    print(dragon_wr[["試合数", "勝率%"]].to_string())

    for otype in ["DRAGON", "BARON"]:
        label = "ファーストドラゴン" if otype == "DRAGON" else "ファーストバロン"
        firsts = obj[(obj["objectiveType"] == otype) & (obj["isFirst"] == True)]
        firsts = firsts.merge(mt, on=["matchId", "teamId"], how="inner")
        firsts = firsts.merge(matches[["matchId", "team100Win"]], on="matchId", how="left")
        firsts["win"] = ((firsts["teamId"] == 100) & firsts["team100Win"]) | (
            (firsts["teamId"] == 200) & ~firsts["team100Win"]
        )
        if len(firsts) > 0:
            wr = firsts["win"].mean() * 100
            print(f"\n{label}取得時の勝率: {wr:.1f}% ({len(firsts)}試合)")


# ------------------------------------------------------------------
# tempo
# ------------------------------------------------------------------
def cmd_tempo(_args):
    members = _load_members()
    matches = _load("matches.csv")
    players = _load("player_stats.csv")
    frames = _load("timeline_frames.csv")

    players["riotId"] = players["summonerName"].astype(str) + "#" + players["tagLine"].astype(str)
    mt = players[players["riotId"].isin(members)][["matchId", "teamId"]].drop_duplicates()

    print("=" * 60)
    print("  ゲームテンポ分析")
    print("=" * 60)

    mwt = matches.merge(mt, on="matchId", how="inner")
    mwt["win"] = ((mwt["teamId"] == 100) & mwt["team100Win"]) | (
        (mwt["teamId"] == 200) & mwt["team200Win"]
    )

    dur_bins = [0, 20, 25, 30, 35, 40, 60]
    dur_labels = ["<20分", "20-25", "25-30", "30-35", "35-40", "40分+"]
    mwt["durBin"] = pd.cut(mwt["gameDurationMin"], bins=dur_bins, labels=dur_labels)
    dur_wr = mwt.groupby("durBin", observed=False).agg(
        試合数=("win", "count"), 勝率=("win", "mean")
    )
    dur_wr["勝率%"] = (dur_wr["勝率"] * 100).round(1)
    print("\n試合時間帯別 勝率:")
    print(dur_wr[["試合数", "勝率%"]].to_string())

    tg = frames.groupby(["matchId", "timestampMin", "teamId"])["totalGold"].sum().reset_index()
    piv = tg.pivot_table(index=["matchId", "timestampMin"], columns="teamId", values="totalGold").reset_index()
    piv.columns = ["matchId", "timestampMin", "g100", "g200"]
    piv["goldDiff"] = piv["g100"] - piv["g200"]
    piv = piv.merge(mt, on="matchId", how="inner")
    piv["lead"] = np.where(piv["teamId"] == 100, piv["goldDiff"], -piv["goldDiff"])
    piv = piv.merge(matches[["matchId", "team100Win"]], on="matchId", how="left")
    piv["win"] = ((piv["teamId"] == 100) & piv["team100Win"]) | (
        (piv["teamId"] == 200) & ~piv["team100Win"]
    )

    at15 = piv[piv["timestampMin"] == 15.0]
    if not at15.empty:
        at15 = at15.copy()
        at15["leadAt15"] = at15["lead"] > 0
        total_ahead = at15[at15["leadAt15"]].shape[0]
        snowball = at15[at15["leadAt15"] & at15["win"]].shape[0]
        total_behind = at15[~at15["leadAt15"]].shape[0]
        comeback = at15[~at15["leadAt15"] & at15["win"]].shape[0]
        print(f"\nスノーボール率 (15分有利→勝利): {snowball}/{total_ahead} = {snowball/max(total_ahead,1)*100:.1f}%")
        print(f"逆転勝利率 (15分不利→勝利): {comeback}/{total_behind} = {comeback/max(total_behind,1)*100:.1f}%")


# ------------------------------------------------------------------
# member (single member deep dive)
# ------------------------------------------------------------------
def cmd_member(args):
    name = args.name
    members = _load_members()
    df = _load("player_stats.csv")
    dm = _member_filter(df, members)
    dm = dm[dm["summonerName"] == name]

    if dm.empty:
        print(f"'{name}' not found. Available: {sorted(_member_filter(_load('player_stats.csv'), members)['summonerName'].unique())}")
        return

    print("=" * 60)
    print(f"  {name} 個人分析")
    print("=" * 60)

    total = len(dm)
    wins = dm["win"].sum()
    print(f"\n試合数: {total}  勝率: {wins/total*100:.1f}%")
    print(f"平均KDA: {dm['kda'].mean():.2f}  K/D/A: {dm['kills'].mean():.1f}/{dm['deaths'].mean():.1f}/{dm['assists'].mean():.1f}")
    print(f"平均CS: {dm['cs'].mean():.0f}  平均ダメージ: {dm['totalDamageDealtToChampions'].mean():.0f}")

    print("\nロール別成績:")
    role_stats = dm.groupby("role").agg(
        試合数=("win", "count"), 勝率=("win", "mean"), 平均KDA=("kda", "mean")
    ).round(2)
    role_stats["勝率%"] = (role_stats["勝率"] * 100).round(1)
    print(role_stats[["試合数", "勝率%", "平均KDA"]].to_string())

    print("\nチャンピオン別 (≥2試合):")
    champ = dm.groupby("championName").agg(
        試合数=("win", "count"), 勝率=("win", "mean"), 平均KDA=("kda", "mean")
    ).round(2)
    champ["勝率%"] = (champ["勝率"] * 100).round(1)
    champ = champ[champ["試合数"] >= 2].sort_values("勝率%", ascending=False)
    print(champ[["試合数", "勝率%", "平均KDA"]].to_string())


# ------------------------------------------------------------------
# match (single match detail)
# ------------------------------------------------------------------
def cmd_match(args):
    mid = args.match_id
    raw_path = ROOT / "data" / "raw" / "matches" / f"{mid}.json"

    if raw_path.exists():
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        info = data["info"]
        print("=" * 60)
        print(f"  試合詳細: {mid}")
        print("=" * 60)
        dur = info["gameDuration"]
        print(f"試合時間: {dur//60}分{dur%60}秒")
        print(f"\n{'チーム':<6} {'チャンピオン':<14} {'K/D/A':<10} {'CS':<6} {'ゴールド':<8} {'ダメージ':<8} {'勝敗'}")
        print("-" * 70)
        for p in sorted(info["participants"], key=lambda x: (x["teamId"], x.get("teamPosition", ""))):
            team = "Blue" if p["teamId"] == 100 else "Red"
            kda = f"{p['kills']}/{p['deaths']}/{p['assists']}"
            cs = p["totalMinionsKilled"] + p.get("neutralMinionsKilled", 0)
            wd = "WIN" if p["win"] else "LOSE"
            name = p.get("riotIdGameName", p.get("summonerName", ""))
            print(f"{team:<6} {p['championName']:<14} {kda:<10} {cs:<6} {p['goldEarned']:<8} {p['totalDamageDealtToChampions']:<8} {wd}  ({name})")
    else:
        df = _load("player_stats.csv")
        match_rows = df[df["matchId"] == mid]
        if match_rows.empty:
            print(f"Match {mid} not found.")
            return
        print(match_rows[["summonerName", "championName", "kills", "deaths", "assists", "cs", "goldEarned", "win"]].to_string(index=False))


# ------------------------------------------------------------------
# teamfight – detect teamfights and analyse first-death tendencies
# ------------------------------------------------------------------
def _detect_teamfights(kills_df: pd.DataFrame, window_sec: int = 30, min_kills: int = 3) -> list[pd.DataFrame]:
    """Group CHAMPION_KILL events into teamfights.

    A teamfight is a cluster of ≥ min_kills kills where each consecutive
    kill is within window_sec seconds of the previous one.
    """
    if kills_df.empty:
        return []

    kills = kills_df.sort_values("timestampMin").copy()
    kills["ts_sec"] = kills["timestampMin"] * 60

    fights: list[pd.DataFrame] = []
    cluster = [kills.iloc[0]]

    for i in range(1, len(kills)):
        row = kills.iloc[i]
        if row["ts_sec"] - cluster[-1]["ts_sec"] <= window_sec:
            cluster.append(row)
        else:
            if len(cluster) >= min_kills:
                fights.append(pd.DataFrame(cluster))
            cluster = [row]

    if len(cluster) >= min_kills:
        fights.append(pd.DataFrame(cluster))

    return fights


def cmd_teamfight(args):
    members = _load_members()
    events = _load("timeline_events.csv")
    players = _load("player_stats.csv")

    kills = events[events["eventType"] == "CHAMPION_KILL"].copy()

    players_m = players.copy()
    players_m["riotId"] = players_m["summonerName"].astype(str) + "#" + players_m["tagLine"].astype(str)
    member_names = set(players_m[players_m["riotId"].isin(members)]["summonerName"].unique())

    pid_to_name: dict[str, dict[int, str]] = {}
    for _, row in players_m.iterrows():
        mid = row["matchId"]
        pid = row["participantId"]
        if mid not in pid_to_name:
            pid_to_name[mid] = {}
        pid_to_name[mid][pid] = row["summonerName"]

    member_team: dict[str, int] = {}
    for _, row in players_m[players_m["riotId"].isin(members)].iterrows():
        member_team[f'{row["matchId"]}_{row["summonerName"]}'] = row["teamId"]

    print("=" * 70)
    print("  集団戦分析 - 最初の死亡者 & 死亡傾向")
    print("=" * 70)

    window = args.window
    min_kills = args.min_kills
    print(f"\n判定条件: {window}秒以内に{min_kills}キル以上 = 集団戦")

    first_death_counts: dict[str, int] = {}
    teamfight_participation: dict[str, int] = {}
    total_deaths_in_tf: dict[str, int] = {}
    first_death_win: dict[str, list[bool]] = {}
    total_teamfights = 0

    for match_id, match_kills in kills.groupby("matchId"):
        fights = _detect_teamfights(match_kills, window_sec=window, min_kills=min_kills)
        pmap = pid_to_name.get(match_id, {})

        for fight in fights:
            total_teamfights += 1
            fight_sorted = fight.sort_values("timestampMin")

            involved_members = set()
            for _, ev in fight_sorted.iterrows():
                vname = ev["victimName"]
                if vname in member_names:
                    involved_members.add(vname)
                kname = ev["killerName"]
                if kname in member_names:
                    involved_members.add(kname)
                assist_ids = ev.get("assistingParticipantIds", "[]")
                if isinstance(assist_ids, str):
                    try:
                        assist_ids = json.loads(assist_ids)
                    except (json.JSONDecodeError, TypeError):
                        assist_ids = []
                for aid in assist_ids:
                    aname = pmap.get(int(aid), "")
                    if aname in member_names:
                        involved_members.add(aname)

            for m in involved_members:
                teamfight_participation[m] = teamfight_participation.get(m, 0) + 1

            for _, ev in fight_sorted.iterrows():
                vname = ev["victimName"]
                if vname in member_names:
                    total_deaths_in_tf[vname] = total_deaths_in_tf.get(vname, 0) + 1

            member_deaths = fight_sorted[fight_sorted["victimName"].isin(member_names)]
            if member_deaths.empty:
                continue

            first_victim = member_deaths.iloc[0]["victimName"]
            first_death_counts[first_victim] = first_death_counts.get(first_victim, 0) + 1

            team_key = f"{match_id}_{first_victim}"
            my_team = member_team.get(team_key)
            if my_team is not None:
                match_win_rows = players_m[(players_m["matchId"] == match_id) & (players_m["summonerName"] == first_victim)]
                if not match_win_rows.empty:
                    win = bool(match_win_rows.iloc[0]["win"])
                    if first_victim not in first_death_win:
                        first_death_win[first_victim] = []
                    first_death_win[first_victim].append(win)

    print(f"検出された集団戦: {total_teamfights} 回\n")

    if not first_death_counts:
        print("メンバーが関わる集団戦データが見つかりませんでした")
        return

    rows = []
    for name in sorted(member_names):
        tf_count = teamfight_participation.get(name, 0)
        fd_count = first_death_counts.get(name, 0)
        deaths_tf = total_deaths_in_tf.get(name, 0)
        fd_rate = fd_count / tf_count * 100 if tf_count > 0 else 0
        fd_wins = first_death_win.get(name, [])
        fd_winrate = sum(fd_wins) / len(fd_wins) * 100 if fd_wins else float("nan")

        rows.append({
            "メンバー": name,
            "集団戦参加": tf_count,
            "集団戦デス": deaths_tf,
            "最初に死亡": fd_count,
            "最初死亡率%": round(fd_rate, 1),
            "最初死亡時勝率%": round(fd_winrate, 1) if fd_wins else "-",
        })

    result = pd.DataFrame(rows).sort_values("最初死亡率%", ascending=False)
    print("【メンバー別 集団戦 最初の死亡者ランキング】")
    print(result.to_string(index=False))

    print("\n--- 解説 ---")
    print("最初死亡率%: そのメンバーが参加した集団戦のうち、味方で最初に倒された割合")
    print("最初死亡時勝率%: そのメンバーが最初に倒された集団戦の勝率 (低い=先に落ちると負けやすい)")

    if args.member:
        target = args.member
        if target not in member_names:
            print(f"\n'{target}' が見つかりません")
            return

        print(f"\n{'=' * 70}")
        print(f"  {target} の集団戦デス位置分析")
        print(f"{'=' * 70}")

        member_kills_as_victim = kills[kills["victimName"] == target].copy()
        if member_kills_as_victim.empty:
            print("デスデータなし")
            return

        member_kills_as_victim["mapZone"] = member_kills_as_victim.apply(
            lambda r: _classify_map_zone(r["positionX"], r["positionY"]), axis=1
        )
        zone_counts = member_kills_as_victim["mapZone"].value_counts()
        print(f"\n{target} の死亡ゾーン分布 (全デス):")
        for zone, cnt in zone_counts.items():
            pct = cnt / len(member_kills_as_victim) * 100
            print(f"  {zone}: {cnt}回 ({pct:.1f}%)")


def _classify_map_zone(x: int, y: int) -> str:
    """Classify a position into a coarse map zone (Summoner's Rift)."""
    mid = 7500
    if x < 4000 and y < 4000:
        return "ブルー側ベース付近"
    if x > 11000 and y > 11000:
        return "レッド側ベース付近"
    if abs(x - y) < 2500 and 3000 < x < 12000:
        return "ミッドレーン付近"
    if x < mid and y > mid:
        return "トップサイド"
    if x > mid and y < mid:
        return "ボットサイド"
    if x < mid and y < mid:
        return "ブルー側ジャングル"
    return "レッド側ジャングル"


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoL Flex Rank Analysis CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("overview", help="全体成績サマリー")

    p_champ = sub.add_parser("champion", help="チャンピオン別分析")
    p_champ.add_argument("--member", default=None, help="特定メンバーに絞る")

    sub.add_parser("synergy", help="デュオ組み合わせ勝率")
    sub.add_parser("trends", help="勝率トレンド")

    p_early = sub.add_parser("early", help="アーリーゲーム分析")
    p_early.add_argument("--minute", type=int, default=15, help="分析する分 (default: 15)")

    sub.add_parser("objectives", help="オブジェクト分析")
    sub.add_parser("tempo", help="ゲームテンポ分析")

    p_member = sub.add_parser("member", help="個人詳細分析")
    p_member.add_argument("name", help="サモナー名")

    p_match = sub.add_parser("match", help="試合詳細")
    p_match.add_argument("match_id", help="マッチID")

    p_tf = sub.add_parser("teamfight", help="集団戦分析（最初の死亡者等）")
    p_tf.add_argument("--member", default=None, help="特定メンバーに絞る")
    p_tf.add_argument("--window", type=int, default=30, help="集団戦判定の時間窓（秒, default: 30）")
    p_tf.add_argument("--min-kills", type=int, default=3, help="集団戦と見なす最小キル数 (default: 3)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "overview": cmd_overview, "champion": cmd_champion, "synergy": cmd_synergy,
        "trends": cmd_trends, "early": cmd_early, "objectives": cmd_objectives,
        "tempo": cmd_tempo, "member": cmd_member, "match": cmd_match,
        "teamfight": cmd_teamfight,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()

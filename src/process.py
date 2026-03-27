"""Transform raw JSON match/timeline files into analysis-ready CSVs.

Produces five CSVs in data/processed/:
  - matches.csv          : one row per match
  - player_stats.csv     : one row per participant per match
  - timeline_frames.csv  : one row per participant per minute
  - timeline_events.csv  : one row per event
  - objectives.csv       : one row per objective event
"""

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

JST = timezone(timedelta(hours=9))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MATCHES_DIR = ROOT / "data" / "raw" / "matches"
TIMELINES_DIR = ROOT / "data" / "raw" / "timelines"
OUTPUT_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_members() -> set[str]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get("members", [])}


def _build_participant_map(info: dict) -> dict[int, dict]:
    """Map participantId -> {puuid, summonerName, teamId, role, ...}."""
    pmap = {}
    for p in info["participants"]:
        pmap[p["participantId"]] = {
            "puuid": p["puuid"],
            "summonerName": p.get("riotIdGameName", p.get("summonerName", "")),
            "tagLine": p.get("riotIdTagline", ""),
            "teamId": p["teamId"],
            "role": p.get("teamPosition", p.get("individualPosition", "")),
            "championName": p["championName"],
            "win": p["win"],
        }
    return pmap


def _opponent_for(pid: int, pmap: dict) -> int | None:
    """Find the opponent in the same role on the other team."""
    me = pmap[pid]
    for other_pid, other in pmap.items():
        if other["teamId"] != me["teamId"] and other["role"] == me["role"] and me["role"]:
            return other_pid
    return None


# ======================================================================
# matches.csv
# ======================================================================

def process_matches() -> pd.DataFrame:
    rows = []
    for fp in sorted(MATCHES_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        info = data["info"]

        team100 = next((t for t in info["teams"] if t["teamId"] == 100), {})
        team200 = next((t for t in info["teams"] if t["teamId"] == 200), {})

        game_dt = datetime.fromtimestamp(info["gameCreation"] / 1000, tz=JST)

        rows.append({
            "matchId": data["metadata"]["matchId"],
            "gameCreation": game_dt.isoformat(),
            "gameDurationSec": info["gameDuration"],
            "gameDurationMin": round(info["gameDuration"] / 60, 1),
            "gameVersion": info.get("gameVersion", ""),
            "team100Win": team100.get("win", False),
            "team200Win": team200.get("win", False),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("gameCreation", ascending=False, inplace=True)
    return df


# ======================================================================
# player_stats.csv
# ======================================================================

def process_player_stats() -> pd.DataFrame:
    rows = []
    for fp in sorted(MATCHES_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        match_id = data["metadata"]["matchId"]
        info = data["info"]
        game_dt = datetime.fromtimestamp(info["gameCreation"] / 1000, tz=JST)

        for p in info["participants"]:
            rows.append({
                "matchId": match_id,
                "gameCreation": game_dt.isoformat(),
                "participantId": p["participantId"],
                "puuid": p["puuid"],
                "summonerName": p.get("riotIdGameName", p.get("summonerName", "")),
                "tagLine": p.get("riotIdTagline", ""),
                "teamId": p["teamId"],
                "role": p.get("teamPosition", p.get("individualPosition", "")),
                "championName": p["championName"],
                "championId": p["championId"],
                "win": p["win"],
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "kda": round((p["kills"] + p["assists"]) / max(p["deaths"], 1), 2),
                "totalMinionsKilled": p["totalMinionsKilled"],
                "neutralMinionsKilled": p.get("neutralMinionsKilled", 0),
                "cs": p["totalMinionsKilled"] + p.get("neutralMinionsKilled", 0),
                "goldEarned": p["goldEarned"],
                "totalDamageDealtToChampions": p["totalDamageDealtToChampions"],
                "totalDamageTaken": p["totalDamageTaken"],
                "visionScore": p.get("visionScore", 0),
                "wardsPlaced": p.get("wardsPlaced", 0),
                "wardsKilled": p.get("wardsKilled", 0),
                "turretKills": p.get("turretKills", 0),
                "dragonKills": p.get("dragonKills", 0),
                "baronKills": p.get("baronKills", 0),
                "firstBloodKill": p.get("firstBloodKill", False),
                "firstTowerKill": p.get("firstTowerKill", False),
                "doubleKills": p.get("doubleKills", 0),
                "tripleKills": p.get("tripleKills", 0),
                "quadraKills": p.get("quadraKills", 0),
                "pentaKills": p.get("pentaKills", 0),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["gameCreation", "participantId"], ascending=[False, True], inplace=True)
    return df


# ======================================================================
# timeline_frames.csv
# ======================================================================

def process_timeline_frames() -> pd.DataFrame:
    rows = []
    for fp in sorted(TIMELINES_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        match_id = data["metadata"]["matchId"]
        info = data.get("info", {})
        frames = info.get("frames", [])

        match_file = MATCHES_DIR / fp.name
        if not match_file.exists():
            continue
        match_data = json.loads(match_file.read_text(encoding="utf-8"))
        pmap = _build_participant_map(match_data["info"])
        opponent_map = {pid: _opponent_for(pid, pmap) for pid in pmap}

        for frame in frames:
            ts_ms = frame.get("timestamp", 0)
            ts_min = round(ts_ms / 60000, 1)
            pframes = frame.get("participantFrames", {})

            gold_by_pid = {}
            cs_by_pid = {}
            for pid_str, pf in pframes.items():
                pid = int(pid_str)
                gold_by_pid[pid] = pf.get("totalGold", 0)
                cs_by_pid[pid] = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)

            for pid_str, pf in pframes.items():
                pid = int(pid_str)
                pmeta = pmap.get(pid, {})
                opp_pid = opponent_map.get(pid)

                gold = pf.get("totalGold", 0)
                cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
                pos = pf.get("position", {})

                rows.append({
                    "matchId": match_id,
                    "timestampMin": ts_min,
                    "participantId": pid,
                    "summonerName": pmeta.get("summonerName", ""),
                    "tagLine": pmeta.get("tagLine", ""),
                    "teamId": pmeta.get("teamId", 0),
                    "role": pmeta.get("role", ""),
                    "championName": pmeta.get("championName", ""),
                    "win": pmeta.get("win", False),
                    "totalGold": gold,
                    "currentGold": pf.get("currentGold", 0),
                    "xp": pf.get("xp", 0),
                    "level": pf.get("level", 1),
                    "cs": cs,
                    "jungleMinionsKilled": pf.get("jungleMinionsKilled", 0),
                    "goldDiffVsOpponent": (gold - gold_by_pid[opp_pid]) if opp_pid else None,
                    "csDiffVsOpponent": (cs - cs_by_pid[opp_pid]) if opp_pid else None,
                    "positionX": pos.get("x", 0),
                    "positionY": pos.get("y", 0),
                })

    return pd.DataFrame(rows)


# ======================================================================
# timeline_events.csv
# ======================================================================

def process_timeline_events() -> pd.DataFrame:
    rows = []
    for fp in sorted(TIMELINES_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        match_id = data["metadata"]["matchId"]
        info = data.get("info", {})

        match_file = MATCHES_DIR / fp.name
        pmap = {}
        if match_file.exists():
            match_data = json.loads(match_file.read_text(encoding="utf-8"))
            pmap = _build_participant_map(match_data["info"])

        for frame in info.get("frames", []):
            for ev in frame.get("events", []):
                ts_ms = ev.get("timestamp", 0)
                etype = ev.get("type", "")

                killer_id = ev.get("killerId", ev.get("creatorId"))
                killer_meta = pmap.get(killer_id, {})
                victim_id = ev.get("victimId")
                victim_meta = pmap.get(victim_id, {})
                pos = ev.get("position", {})

                rows.append({
                    "matchId": match_id,
                    "timestampMin": round(ts_ms / 60000, 1),
                    "eventType": etype,
                    "killerId": killer_id,
                    "killerName": killer_meta.get("summonerName", ""),
                    "killerTeamId": killer_meta.get("teamId"),
                    "victimId": victim_id,
                    "victimName": victim_meta.get("summonerName", ""),
                    "victimTeamId": victim_meta.get("teamId"),
                    "assistingParticipantIds": json.dumps(ev.get("assistingParticipantIds", [])),
                    "wardType": ev.get("wardType", ""),
                    "buildingType": ev.get("buildingType", ""),
                    "towerType": ev.get("towerType", ""),
                    "laneType": ev.get("laneType", ""),
                    "monsterType": ev.get("monsterType", ""),
                    "monsterSubType": ev.get("monsterSubType", ""),
                    "teamId": ev.get("teamId"),
                    "itemId": ev.get("itemId"),
                    "skillSlot": ev.get("skillSlot"),
                    "positionX": pos.get("x", 0),
                    "positionY": pos.get("y", 0),
                })

    return pd.DataFrame(rows)


# ======================================================================
# objectives.csv
# ======================================================================

OBJECTIVE_EVENTS = {
    "ELITE_MONSTER_KILL",
    "BUILDING_KILL",
}

MONSTER_MAP = {
    "DRAGON": "DRAGON",
    "BARON_NASHOR": "BARON",
    "RIFTHERALD": "RIFT_HERALD",
}


def process_objectives() -> pd.DataFrame:
    rows = []
    for fp in sorted(TIMELINES_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        match_id = data["metadata"]["matchId"]
        info = data.get("info", {})

        seen_first = {}

        for frame in info.get("frames", []):
            for ev in frame.get("events", []):
                etype = ev.get("type", "")
                if etype not in OBJECTIVE_EVENTS:
                    continue

                ts_min = round(ev.get("timestamp", 0) / 60000, 1)
                team_id = ev.get("killerTeamId", ev.get("teamId"))

                if etype == "ELITE_MONSTER_KILL":
                    monster = ev.get("monsterType", "")
                    obj_type = MONSTER_MAP.get(monster, monster)
                    sub_type = ev.get("monsterSubType", "")

                    first_key = obj_type
                    is_first = first_key not in seen_first
                    seen_first[first_key] = True

                    rows.append({
                        "matchId": match_id,
                        "timestampMin": ts_min,
                        "objectiveType": obj_type,
                        "monsterSubType": sub_type,
                        "teamId": team_id,
                        "isFirst": is_first,
                    })

                elif etype == "BUILDING_KILL":
                    building = ev.get("buildingType", "")
                    if building == "TOWER_BUILDING":
                        obj_type = "TOWER"
                    elif building == "INHIBITOR_BUILDING":
                        obj_type = "INHIBITOR"
                    else:
                        obj_type = building

                    first_key = obj_type
                    is_first = first_key not in seen_first
                    seen_first[first_key] = True

                    rows.append({
                        "matchId": match_id,
                        "timestampMin": ts_min,
                        "objectiveType": obj_type,
                        "monsterSubType": "",
                        "teamId": team_id,
                        "isFirst": is_first,
                    })

    return pd.DataFrame(rows)


# ======================================================================
# Benchmark data (Emerald-rank)
# ======================================================================

BENCHMARK_DIR = ROOT / "data" / "raw" / "benchmark"
BM_MATCHES_DIR = BENCHMARK_DIR / "matches"
BM_TIMELINES_DIR = BENCHMARK_DIR / "timelines"


def process_benchmark_stats() -> pd.DataFrame:
    """Process benchmark match JSONs → same schema as player_stats."""
    if not BM_MATCHES_DIR.exists():
        return pd.DataFrame()

    match_files = list(BM_MATCHES_DIR.glob("*.json"))
    if not match_files:
        return pd.DataFrame()

    rows = []
    for fp in sorted(match_files):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        match_id = data["metadata"]["matchId"]
        info = data["info"]
        game_dt = datetime.fromtimestamp(info["gameCreation"] / 1000, tz=JST)

        for p in info["participants"]:
            rows.append({
                "matchId": match_id,
                "gameCreation": game_dt.isoformat(),
                "gameDurationSec": info["gameDuration"],
                "gameDurationMin": round(info["gameDuration"] / 60, 1),
                "participantId": p["participantId"],
                "puuid": p["puuid"],
                "summonerName": p.get("riotIdGameName", p.get("summonerName", "")),
                "tagLine": p.get("riotIdTagline", ""),
                "teamId": p["teamId"],
                "role": p.get("teamPosition", p.get("individualPosition", "")),
                "championName": p["championName"],
                "championId": p["championId"],
                "win": p["win"],
                "kills": p["kills"],
                "deaths": p["deaths"],
                "assists": p["assists"],
                "kda": round((p["kills"] + p["assists"]) / max(p["deaths"], 1), 2),
                "totalMinionsKilled": p["totalMinionsKilled"],
                "neutralMinionsKilled": p.get("neutralMinionsKilled", 0),
                "cs": p["totalMinionsKilled"] + p.get("neutralMinionsKilled", 0),
                "goldEarned": p["goldEarned"],
                "totalDamageDealtToChampions": p["totalDamageDealtToChampions"],
                "totalDamageTaken": p["totalDamageTaken"],
                "visionScore": p.get("visionScore", 0),
                "wardsPlaced": p.get("wardsPlaced", 0),
                "wardsKilled": p.get("wardsKilled", 0),
                "turretKills": p.get("turretKills", 0),
                "dragonKills": p.get("dragonKills", 0),
                "baronKills": p.get("baronKills", 0),
                "firstBloodKill": p.get("firstBloodKill", False),
                "firstTowerKill": p.get("firstTowerKill", False),
                "doubleKills": p.get("doubleKills", 0),
                "tripleKills": p.get("tripleKills", 0),
                "quadraKills": p.get("quadraKills", 0),
                "pentaKills": p.get("pentaKills", 0),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["gameCreation", "participantId"], ascending=[False, True], inplace=True)
    return df


def process_benchmark_timeline_frames() -> pd.DataFrame:
    """Process benchmark timeline JSONs → same schema as timeline_frames."""
    if not BM_TIMELINES_DIR.exists():
        return pd.DataFrame()

    tl_files = list(BM_TIMELINES_DIR.glob("*.json"))
    if not tl_files:
        return pd.DataFrame()

    rows = []
    for fp in sorted(tl_files):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        match_id = data["metadata"]["matchId"]
        info = data.get("info", {})
        frames = info.get("frames", [])

        match_file = BM_MATCHES_DIR / fp.name
        if not match_file.exists():
            continue
        try:
            match_data = json.loads(match_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        pmap = _build_participant_map(match_data["info"])
        opponent_map = {pid: _opponent_for(pid, pmap) for pid in pmap}

        for frame in frames:
            ts_ms = frame.get("timestamp", 0)
            ts_min = round(ts_ms / 60000, 1)
            pframes = frame.get("participantFrames") or {}

            gold_by_pid = {}
            cs_by_pid = {}
            for pid_str, pf in pframes.items():
                if pf is None:
                    continue
                pid = int(pid_str)
                gold_by_pid[pid] = pf.get("totalGold", 0)
                cs_by_pid[pid] = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)

            for pid_str, pf in pframes.items():
                if pf is None:
                    continue
                pid = int(pid_str)
                pmeta = pmap.get(pid, {})
                opp_pid = opponent_map.get(pid)

                gold = pf.get("totalGold", 0)
                cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)

                rows.append({
                    "matchId": match_id,
                    "timestampMin": ts_min,
                    "participantId": pid,
                    "summonerName": pmeta.get("summonerName", ""),
                    "tagLine": pmeta.get("tagLine", ""),
                    "teamId": pmeta.get("teamId", 0),
                    "role": pmeta.get("role", ""),
                    "championName": pmeta.get("championName", ""),
                    "win": pmeta.get("win", False),
                    "totalGold": gold,
                    "cs": cs,
                    "goldDiffVsOpponent": (gold - gold_by_pid[opp_pid]) if opp_pid else None,
                    "csDiffVsOpponent": (cs - cs_by_pid[opp_pid]) if opp_pid else None,
                    "level": pf.get("level", 1),
                    "xp": pf.get("xp", 0),
                })

    return pd.DataFrame(rows)


# ======================================================================
# Main
# ======================================================================

MIN_TEAM_MEMBERS = 3


def _find_excluded_matches(df_players: pd.DataFrame) -> set[str]:
    """Return matchIds where fewer than MIN_TEAM_MEMBERS configured members participated."""
    members = {riot_id.split("#")[0] for riot_id in _load_members()}
    member_counts = (
        df_players[df_players["summonerName"].isin(members)]
        .groupby("matchId")["summonerName"]
        .nunique()
    )
    all_ids = set(df_players["matchId"].unique())
    counted_ids = set(member_counts[member_counts >= MIN_TEAM_MEMBERS].index)
    return all_ids - counted_ids


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    match_files = list(MATCHES_DIR.glob("*.json"))
    timeline_files = list(TIMELINES_DIR.glob("*.json"))
    logger.info("Found %d match files, %d timeline files", len(match_files), len(timeline_files))

    if not match_files:
        logger.warning("No raw match data found. Run collect.py first.")
        return

    logger.info("Processing player_stats.csv ...")
    df_players = process_player_stats()

    excluded = _find_excluded_matches(df_players)
    if excluded:
        logger.info("Excluding %d matches with fewer than %d members: %s",
                     len(excluded), MIN_TEAM_MEMBERS, excluded)
        df_players = df_players[~df_players["matchId"].isin(excluded)]

    df_players.to_csv(OUTPUT_DIR / "player_stats.csv", index=False, encoding="utf-8-sig")
    logger.info("  -> %d rows", len(df_players))

    logger.info("Processing matches.csv ...")
    df_matches = process_matches()
    if excluded:
        df_matches = df_matches[~df_matches["matchId"].isin(excluded)]
    df_matches.to_csv(OUTPUT_DIR / "matches.csv", index=False, encoding="utf-8-sig")
    logger.info("  -> %d rows", len(df_matches))

    if timeline_files:
        logger.info("Processing timeline_frames.csv ...")
        df_frames = process_timeline_frames()
        if excluded:
            df_frames = df_frames[~df_frames["matchId"].isin(excluded)]
        df_frames.to_csv(OUTPUT_DIR / "timeline_frames.csv", index=False, encoding="utf-8-sig")
        logger.info("  -> %d rows", len(df_frames))

        logger.info("Processing timeline_events.csv ...")
        df_events = process_timeline_events()
        if excluded:
            df_events = df_events[~df_events["matchId"].isin(excluded)]
        df_events.to_csv(OUTPUT_DIR / "timeline_events.csv", index=False, encoding="utf-8-sig")
        logger.info("  -> %d rows", len(df_events))

        logger.info("Processing objectives.csv ...")
        df_obj = process_objectives()
        if excluded:
            df_obj = df_obj[~df_obj["matchId"].isin(excluded)]
        df_obj.to_csv(OUTPUT_DIR / "objectives.csv", index=False, encoding="utf-8-sig")
        logger.info("  -> %d rows", len(df_obj))
    else:
        logger.warning("No timeline files found – skipping timeline CSVs")

    # ------------------------------------------------------------------
    # Benchmark data (if collected)
    # ------------------------------------------------------------------
    bm_match_files = list(BM_MATCHES_DIR.glob("*.json")) if BM_MATCHES_DIR.exists() else []
    if bm_match_files:
        logger.info("Processing benchmark_stats.csv (%d match files) ...", len(bm_match_files))
        df_bm = process_benchmark_stats()
        df_bm.to_csv(OUTPUT_DIR / "benchmark_stats.csv", index=False, encoding="utf-8-sig")
        logger.info("  -> %d rows", len(df_bm))

        bm_tl_files = list(BM_TIMELINES_DIR.glob("*.json")) if BM_TIMELINES_DIR.exists() else []
        if bm_tl_files:
            logger.info("Processing benchmark_timeline_frames.csv (%d files) ...", len(bm_tl_files))
            df_bm_tl = process_benchmark_timeline_frames()
            df_bm_tl.to_csv(OUTPUT_DIR / "benchmark_timeline_frames.csv",
                            index=False, encoding="utf-8-sig")
            logger.info("  -> %d rows", len(df_bm_tl))
    else:
        logger.info("No benchmark data found – skipping benchmark CSVs")

    logger.info("All CSVs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

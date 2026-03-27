"""Collect Emerald-rank game data for champion-specific benchmarks.

Pipeline:
  1. Read player_stats.csv → extract each member's top champions
  2. Fetch Emerald league entries from League-V4
  3. Resolve PUUIDs via Summoner-V4
  4. Download their recent Flex matches (+ timelines)
  5. Store raw JSON in data/raw/benchmark/

Usage:
    python src/collect_benchmark.py                     # default: 80 players, 20 matches each
    python src/collect_benchmark.py --players 120       # sample more players
    python src/collect_benchmark.py --matches 30        # more matches per player
    python src/collect_benchmark.py --tier DIAMOND      # use a different tier
    python src/collect_benchmark.py --no-timelines      # skip timeline collection
    python src/collect_benchmark.py --min-games 8       # champion threshold
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.riot_api import RiotAPI, load_config

DATA_RAW = ROOT / "data" / "raw"
BENCHMARK_DIR = DATA_RAW / "benchmark"
BM_MATCHES_DIR = BENCHMARK_DIR / "matches"
BM_TIMELINES_DIR = BENCHMARK_DIR / "timelines"
STATE_FILE = BENCHMARK_DIR / "collection_state.json"
PLAYER_STATS = ROOT / "data" / "processed" / "player_stats.csv"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

DIVISIONS = ["I", "II", "III", "IV"]
FLEX_QUEUE = 440

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"collected_puuids": [], "match_ids": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ======================================================================
#  Step 1: Extract target champions from existing data
# ======================================================================

def extract_target_champions(min_games: int = 5) -> dict[str, list[dict]]:
    """Return {member_name: [{champion, games, role}, ...]} for frequently-played champs."""
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    member_names = {m["game_name"] for m in cfg.get("members", [])}

    ps = pd.read_csv(PLAYER_STATS)
    ps_m = ps[ps["summonerName"].isin(member_names)]

    result = {}
    all_champs = set()

    for name in sorted(member_names):
        player = ps_m[ps_m["summonerName"] == name]
        champ_stats = (
            player.groupby(["championName", "role"])
            .agg(games=("win", "count"), wins=("win", "sum"))
            .reset_index()
        )
        champ_stats["wr"] = champ_stats["wins"] / champ_stats["games"] * 100
        top = champ_stats[champ_stats["games"] >= min_games].sort_values("games", ascending=False)

        champs = []
        for _, row in top.iterrows():
            champs.append({
                "champion": row["championName"],
                "role": row["role"],
                "games": int(row["games"]),
                "wr": round(row["wr"], 1),
            })
            all_champs.add(row["championName"])
        result[name] = champs

    logger.info("Target champions across all members: %d unique", len(all_champs))
    for name, champs in result.items():
        champ_str = ", ".join(f'{c["champion"]}({c["games"]}戦)' for c in champs[:5])
        logger.info("  %s: %s%s", name, champ_str, " ..." if len(champs) > 5 else "")

    return result


# ======================================================================
#  Step 2-4: Collect Emerald-rank match data
# ======================================================================

def fetch_emerald_players(api: RiotAPI, tier: str, n_players: int) -> list[dict]:
    """Fetch summoner entries from the league ladder."""
    entries = []
    for div in DIVISIONS:
        page = 1
        while len(entries) < n_players * 2:
            batch = api.get_league_entries(
                queue="RANKED_FLEX_SR", tier=tier, division=div, page=page
            )
            if not batch:
                break
            entries.extend(batch)
            logger.info("  League %s %s page %d → %d entries (total %d)",
                        tier, div, page, len(batch), len(entries))
            page += 1
            if len(batch) < 200:
                break
        if len(entries) >= n_players * 2:
            break

    logger.info("Fetched %d league entries total", len(entries))
    return entries


def resolve_puuids(api: RiotAPI, entries: list[dict], n_players: int,
                   already_collected: set[str]) -> list[str]:
    """Extract PUUIDs from league entries. Falls back to Summoner-V4 if needed."""
    import random
    random.shuffle(entries)

    puuids = []
    skipped = 0
    api_lookups = 0
    errors = 0

    for entry in entries:
        if len(puuids) >= n_players:
            break

        puuid = entry.get("puuid")

        if not puuid:
            sid = entry.get("summonerId", "")
            try:
                summoner = api.get_summoner_by_id(sid)
                puuid = summoner.get("puuid")
                api_lookups += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning("  Summoner lookup failed: %s", e)
                continue

        if not puuid:
            continue
        if puuid in already_collected:
            skipped += 1
            continue

        puuids.append(puuid)
        if len(puuids) % 20 == 0:
            logger.info("  Resolved %d/%d PUUIDs (skipped %d, api lookups %d)",
                        len(puuids), n_players, skipped, api_lookups)

    logger.info("Resolved %d PUUIDs (skipped %d existing, %d api lookups, %d errors)",
                len(puuids), skipped, api_lookups, errors)
    return puuids


def collect_matches(api: RiotAPI, puuids: list[str], matches_per_player: int,
                    with_timelines: bool, existing_match_ids: set[str]) -> int:
    """Download match (+ timeline) data for a list of PUUIDs."""
    new_matches = 0
    total_players = len(puuids)

    for i, puuid in enumerate(puuids, 1):
        try:
            match_ids = api.get_match_ids_any_queue(
                puuid, count=matches_per_player, queue=FLEX_QUEUE
            )
        except Exception as e:
            logger.warning("  Failed to get matches for player %d: %s", i, e)
            continue

        for mid in match_ids:
            if mid in existing_match_ids:
                continue

            match_file = BM_MATCHES_DIR / f"{mid}.json"
            if match_file.exists():
                existing_match_ids.add(mid)
                continue

            try:
                match_data = api.get_match(mid)
                match_file.write_text(
                    json.dumps(match_data, ensure_ascii=False), encoding="utf-8"
                )

                if with_timelines:
                    tl_file = BM_TIMELINES_DIR / f"{mid}.json"
                    if not tl_file.exists():
                        tl_data = api.get_timeline(mid)
                        tl_file.write_text(
                            json.dumps(tl_data, ensure_ascii=False), encoding="utf-8"
                        )

                existing_match_ids.add(mid)
                new_matches += 1

            except Exception as e:
                logger.warning("  Failed to download %s: %s", mid, e)

        if i % 10 == 0 or i == total_players:
            total = len(existing_match_ids)
            logger.info("  Progress: %d/%d players, %d new matches (%d total on disk)",
                        i, total_players, new_matches, total)

    return new_matches


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect Emerald benchmark data")
    parser.add_argument("--players", type=int, default=80,
                        help="Number of Emerald players to sample (default: 80)")
    parser.add_argument("--matches", type=int, default=20,
                        help="Matches to fetch per player (default: 20)")
    parser.add_argument("--tier", type=str, nargs="+", default=["EMERALD", "DIAMOND"],
                        help="Tier(s) to sample from (default: EMERALD DIAMOND)")
    parser.add_argument("--no-timelines", action="store_true",
                        help="Skip timeline collection")
    parser.add_argument("--min-games", type=int, default=5,
                        help="Min games to consider a champion as 'frequently played' (default: 5)")
    args = parser.parse_args()

    BM_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_timelines:
        BM_TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

    if not PLAYER_STATS.exists():
        logger.error("player_stats.csv not found. Run process.py first.")
        return

    tiers = [t.upper() for t in args.tier]
    players_per_tier = max(1, args.players // len(tiers))

    logger.info("=" * 60)
    logger.info("  Benchmark Data Collection")
    logger.info("  Tiers: %s  |  Players/tier: %d  |  Matches/player: %d",
                "+".join(tiers), players_per_tier, args.matches)
    logger.info("  Timelines: %s", "Yes" if not args.no_timelines else "No")
    logger.info("=" * 60)

    # Step 1: Extract target champions
    logger.info("")
    logger.info("Step 1: Extracting target champions ...")
    target_champs = extract_target_champions(min_games=args.min_games)
    all_target = set()
    for champs in target_champs.values():
        for c in champs:
            all_target.add(c["champion"])
    logger.info("Will benchmark %d unique champions", len(all_target))

    # Step 2: Fetch league entries from all tiers
    config = load_config()
    api = RiotAPI(config)
    state = _load_state()
    already = set(state.get("collected_puuids", []))
    all_puuids: list[str] = []

    for tier in tiers:
        logger.info("")
        logger.info("Step 2: Fetching %s league entries ...", tier)
        entries = fetch_emerald_players(api, tier, players_per_tier)

        if not entries:
            logger.warning("No league entries found for %s Flex — skipping", tier)
            continue

        # Step 3: Resolve PUUIDs for this tier
        logger.info("Step 3: Resolving PUUIDs for %s ...", tier)
        puuids = resolve_puuids(api, entries, players_per_tier, already)
        already |= set(puuids)
        all_puuids.extend(puuids)
        logger.info("  %s: %d players resolved", tier, len(puuids))

    puuids = all_puuids
    if not puuids:
        logger.error("No players resolved. Check API key or try different tiers.")
        return

    # Step 4: Download matches
    logger.info("")
    logger.info("Step 4: Downloading matches ...")
    existing_ids = set(state.get("match_ids", []))
    on_disk = {f.stem for f in BM_MATCHES_DIR.glob("*.json")}
    existing_ids |= on_disk

    new_count = collect_matches(
        api, puuids, args.matches, not args.no_timelines, existing_ids
    )

    # Save state
    state["collected_puuids"] = list(set(state.get("collected_puuids", [])) | set(puuids))
    state["match_ids"] = list(existing_ids)
    state["tiers"] = tiers
    state["target_champions"] = sorted(all_target)
    _save_state(state)

    # Summary
    total_on_disk = len(list(BM_MATCHES_DIR.glob("*.json")))
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Collection Complete")
    logger.info("  New matches: %d  |  Total on disk: %d", new_count, total_on_disk)
    logger.info("  PUUIDs collected (cumulative): %d", len(state["collected_puuids"]))
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step: python src/process.py  (benchmark_stats.csv will be generated)")


if __name__ == "__main__":
    main()

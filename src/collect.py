"""Collect Flex Rank match data and timelines from the Riot API.

Usage:
    python src/collect.py            # collect new matches for all members
    python src/collect.py --full     # re-download everything (ignore cache)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.riot_api import RiotAPI, load_config

DATA_RAW = ROOT / "data" / "raw"
MATCHES_DIR = DATA_RAW / "matches"
TIMELINES_DIR = DATA_RAW / "timelines"
STATE_FILE = DATA_RAW / "collection_state.json"

API_PAGE_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_start_date(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to epoch seconds (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _fetch_all_match_ids(api: RiotAPI, puuid: str, start_time: int) -> list[str]:
    """Paginate through all match IDs since start_time."""
    all_ids: list[str] = []
    offset = 0
    while True:
        batch = api.get_match_ids(puuid, count=API_PAGE_SIZE, start=offset, start_time=start_time)
        all_ids.extend(batch)
        if len(batch) < API_PAGE_SIZE:
            break
        offset += API_PAGE_SIZE
    return all_ids


def collect_member(api: RiotAPI, game_name: str, tag_line: str,
                   config_start_epoch: int, state: dict, full: bool) -> list[str]:
    """Fetch match + timeline data for one member. Returns new match IDs."""

    member_key = f"{game_name}#{tag_line}"
    logger.info("=== Collecting for %s ===", member_key)

    puuid = api.get_puuid(game_name, tag_line)
    logger.info("PUUID: %s", puuid)

    if full:
        start_time = config_start_epoch
    elif member_key in state and state[member_key].get("last_game_epoch"):
        start_time = state[member_key]["last_game_epoch"]
    else:
        start_time = config_start_epoch

    match_ids = _fetch_all_match_ids(api, puuid, start_time)
    logger.info("Found %d match IDs since %s", len(match_ids),
                datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%Y-%m-%d"))

    new_ids = []
    for mid in match_ids:
        match_file = MATCHES_DIR / f"{mid}.json"
        timeline_file = TIMELINES_DIR / f"{mid}.json"

        if match_file.exists() and timeline_file.exists() and not full:
            continue

        if not match_file.exists() or full:
            logger.info("  Downloading match %s", mid)
            match_data = api.get_match(mid)
            match_file.write_text(json.dumps(match_data, ensure_ascii=False), encoding="utf-8")

        if not timeline_file.exists() or full:
            logger.info("  Downloading timeline %s", mid)
            timeline_data = api.get_timeline(mid)
            timeline_file.write_text(json.dumps(timeline_data, ensure_ascii=False), encoding="utf-8")

        new_ids.append(mid)

    if match_ids:
        first_match = json.loads((MATCHES_DIR / f"{match_ids[0]}.json").read_text(encoding="utf-8"))
        game_creation = first_match["info"]["gameCreation"] // 1000
        state[member_key] = {"puuid": puuid, "last_game_epoch": game_creation + 1}

    logger.info("  Collected %d new matches for %s", len(new_ids), member_key)
    return new_ids


def main():
    parser = argparse.ArgumentParser(description="Collect LoL Flex Rank data")
    parser.add_argument("--full", action="store_true", help="Ignore cache; re-download all")
    args = parser.parse_args()

    config = load_config()
    api = RiotAPI(config)
    members = config.get("members", [])
    col_cfg = config.get("collection", {})

    start_date_str = col_cfg.get("start_date", "2024-01-01")
    config_start_epoch = _parse_start_date(start_date_str)
    logger.info("Collection start date: %s", start_date_str)

    if not members:
        logger.error("No members configured in config/settings.yaml")
        return

    state = _load_state()
    total_new = 0

    for member in members:
        new_ids = collect_member(
            api,
            member["game_name"],
            member["tag_line"],
            config_start_epoch,
            state,
            full=args.full,
        )
        total_new += len(new_ids)

    _save_state(state)
    logger.info("Done. Total new matches collected: %d", total_new)


if __name__ == "__main__":
    main()

"""Riot Games API wrapper with rate limiting and retry logic."""

import time
import logging
import requests
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RiotAPI:
    """Thin wrapper around the Riot Games REST API.

    Handles:
      - Account V1 (PUUID lookup)
      - Match V5 (match list + match detail)
      - Match Timeline V5
      - Automatic rate-limit back-off and retries
    """

    ACCOUNT_HOST = "https://{routing}.api.riotgames.com"
    MATCH_HOST = "https://{routing}.api.riotgames.com"
    PLATFORM_HOST = "https://{region}.api.riotgames.com"

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        api_cfg = cfg["riot_api"]
        col_cfg = cfg.get("collection", {})

        self.api_key = api_cfg["api_key"]
        self.region = api_cfg["region"]
        self.routing = api_cfg["routing"]
        self.queue_id = cfg.get("queue_id", 440)
        self.rate_limit_sleep = col_cfg.get("rate_limit_sleep", 1.5)
        self.max_retries = col_cfg.get("max_retries", 3)

        self._session = requests.Session()
        self._session.headers.update({"X-Riot-Token": self.api_key})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, params: dict | None = None) -> dict | list:
        """Issue a GET request with retry / rate-limit handling."""
        for attempt in range(1, self.max_retries + 1):
            resp = self._session.get(url, params=params)

            if resp.status_code == 200:
                time.sleep(self.rate_limit_sleep)
                return resp.json()

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.warning("Rate limited – sleeping %ds (attempt %d)", retry_after, attempt)
                time.sleep(retry_after)
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = 2 ** attempt
                logger.warning("Server error %d – retrying in %ds (attempt %d)",
                               resp.status_code, wait, attempt)
                time.sleep(wait)
                continue

            resp.raise_for_status()

        raise RuntimeError(f"Max retries exceeded for {url}")

    # ------------------------------------------------------------------
    # Account V1
    # ------------------------------------------------------------------

    def get_puuid(self, game_name: str, tag_line: str) -> str:
        url = (
            self.ACCOUNT_HOST.format(routing=self.routing)
            + f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        )
        data = self._request(url)
        return data["puuid"]

    # ------------------------------------------------------------------
    # Match V5
    # ------------------------------------------------------------------

    def get_match_ids(
        self,
        puuid: str,
        count: int = 100,
        start: int = 0,
        start_time: int | None = None,
    ) -> list[str]:
        """Return a list of match IDs for Flex Ranked (queue 440)."""
        url = (
            self.MATCH_HOST.format(routing=self.routing)
            + f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        )
        params = {"queue": self.queue_id, "type": "ranked", "count": count, "start": start}
        if start_time is not None:
            params["startTime"] = start_time
        return self._request(url, params=params)

    def get_match(self, match_id: str) -> dict:
        url = (
            self.MATCH_HOST.format(routing=self.routing)
            + f"/lol/match/v5/matches/{match_id}"
        )
        return self._request(url)

    def get_timeline(self, match_id: str) -> dict:
        url = (
            self.MATCH_HOST.format(routing=self.routing)
            + f"/lol/match/v5/matches/{match_id}/timeline"
        )
        return self._request(url)

    # ------------------------------------------------------------------
    # League V4
    # ------------------------------------------------------------------

    def get_league_entries(
        self,
        queue: str = "RANKED_FLEX_SR",
        tier: str = "EMERALD",
        division: str = "I",
        page: int = 1,
    ) -> list[dict]:
        """Return league entries for a given queue/tier/division."""
        url = (
            self.PLATFORM_HOST.format(region=self.region)
            + f"/lol/league/v4/entries/{queue}/{tier}/{division}"
        )
        return self._request(url, params={"page": page})

    # ------------------------------------------------------------------
    # Summoner V4
    # ------------------------------------------------------------------

    def get_summoner_by_id(self, summoner_id: str) -> dict:
        """Look up summoner by encrypted summoner ID (returns puuid, etc.)."""
        url = (
            self.PLATFORM_HOST.format(region=self.region)
            + f"/lol/summoner/v4/summoners/{summoner_id}"
        )
        return self._request(url)

    def get_match_ids_any_queue(
        self,
        puuid: str,
        count: int = 20,
        start: int = 0,
        queue: int | None = None,
        start_time: int | None = None,
    ) -> list[str]:
        """Return match IDs with flexible queue filter (or no filter)."""
        url = (
            self.MATCH_HOST.format(routing=self.routing)
            + f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        )
        params: dict = {"count": count, "start": start}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        return self._request(url, params=params)

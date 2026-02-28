"""X (Twitter) API v2 client with rate limiting and retry logic."""

from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


class XApiNotConfigured(Exception):
    """Raised when X API credentials are not set."""


class XApiError(Exception):
    """Raised on API errors after retries are exhausted."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"X API error {status_code}: {message}")


class XClient:
    """
    X API v2 HTTP client.

    Bearer token is read from the ``bearer_token`` argument, or from the
    ``X_BEARER_TOKEN`` environment variable.
    """

    BASE_URL = "https://api.x.com/2"

    # Default tweet fields requested on every search / lookup
    TWEET_FIELDS = (
        "text,created_at,public_metrics,entities,author_id,"
        "lang,conversation_id,in_reply_to_user_id,attachments"
    )
    USER_FIELDS = "public_metrics,verified,created_at,description"
    EXPANSIONS = "author_id,attachments.media_keys"
    MEDIA_FIELDS = "type,url,preview_image_url"

    def __init__(self, bearer_token: str | None = None):
        self.bearer_token = bearer_token or os.environ.get("X_BEARER_TOKEN")
        self._session = requests.Session()
        self._rate_remaining: dict[str, int] = {}
        self._rate_reset: dict[str, float] = {}

    @property
    def is_available(self) -> bool:
        return bool(self.bearer_token)

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def search_recent(self, query: str, max_results: int = 10, next_token: str | None = None) -> dict | None:
        """GET /2/tweets/search/recent"""
        if not self.is_available:
            return None
        params: dict = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": self.TWEET_FIELDS,
            "user.fields": self.USER_FIELDS,
            "expansions": self.EXPANSIONS,
            "media.fields": self.MEDIA_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token
        return self._get("/tweets/search/recent", params)

    def get_post(self, post_id: str) -> dict | None:
        """GET /2/tweets/:id"""
        if not self.is_available:
            return None
        params = {
            "tweet.fields": self.TWEET_FIELDS,
            "user.fields": self.USER_FIELDS,
            "expansions": self.EXPANSIONS,
            "media.fields": self.MEDIA_FIELDS,
        }
        return self._get(f"/tweets/{post_id}", params)

    def get_user_timeline(self, user_id: str, max_results: int = 10, next_token: str | None = None) -> dict | None:
        """GET /2/users/:id/tweets"""
        if not self.is_available:
            return None
        params: dict = {
            "max_results": min(max_results, 100),
            "tweet.fields": self.TWEET_FIELDS,
            "user.fields": self.USER_FIELDS,
            "expansions": self.EXPANSIONS,
            "media.fields": self.MEDIA_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token
        return self._get(f"/users/{user_id}/tweets", params)

    def get_user_by_username(self, username: str) -> dict | None:
        """GET /2/users/by/username/:username"""
        if not self.is_available:
            return None
        username = username.lstrip("@")
        return self._get(f"/users/by/username/{username}", {"user.fields": self.USER_FIELDS})

    def get_trends(self, woeid: int = 1) -> dict | None:
        """
        GET trending topics.
        Note: X API v2 trends endpoint availability varies by access level.
        Falls back to v1.1 if v2 is unavailable.
        """
        if not self.is_available:
            return None
        # Try v1.1 endpoint (wider availability)
        try:
            resp = self._session.get(
                f"https://api.x.com/1.1/trends/place.json",
                params={"id": woeid},
                headers=self._headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal HTTP layer
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.bearer_token:
            raise XApiNotConfigured("X_BEARER_TOKEN not set")
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{self.BASE_URL}{path}"
        endpoint = path.split("/")[1] if "/" in path else path

        # Wait for rate limit reset if exhausted
        self._wait_for_rate_limit(endpoint)

        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                resp = self._session.get(url, params=params, headers=self._headers(), timeout=15)
            except requests.RequestException as e:
                logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise XApiError(0, str(e))

            # Update rate limit tracking
            self._update_rate_limits(endpoint, resp.headers)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                reset_at = float(resp.headers.get("x-rate-limit-reset", time.time() + 60))
                wait = max(reset_at - time.time(), 1.0)
                logger.warning("Rate limited on %s. Waiting %.1fs", endpoint, wait)
                time.sleep(min(wait, 60))
                continue

            if resp.status_code >= 500 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue

            logger.error("X API error %d: %s", resp.status_code, resp.text[:200])
            raise XApiError(resp.status_code, resp.text[:200])

        return None

    def _wait_for_rate_limit(self, endpoint: str):
        remaining = self._rate_remaining.get(endpoint)
        reset_at = self._rate_reset.get(endpoint, 0)
        if remaining is not None and remaining <= 0:
            wait = max(reset_at - time.time(), 0)
            if wait > 0:
                logger.info("Pre-waiting %.1fs for rate limit reset on %s", wait, endpoint)
                time.sleep(min(wait, 60))

    def _update_rate_limits(self, endpoint: str, headers):
        remaining = headers.get("x-rate-limit-remaining")
        reset = headers.get("x-rate-limit-reset")
        if remaining is not None:
            self._rate_remaining[endpoint] = int(remaining)
        if reset is not None:
            self._rate_reset[endpoint] = float(reset)

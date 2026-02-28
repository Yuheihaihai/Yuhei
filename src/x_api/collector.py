"""High-level data collector — fetches from X API and stores locally."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.core.virality_engine import MediaType, PostInput
from src.storage.database import Database
from src.storage.models import StoredPost
from src.x_api.client import XClient

logger = logging.getLogger(__name__)

# Map X API media type strings to our MediaType enum values
_MEDIA_TYPE_MAP = {
    "photo": "image",
    "video": "video",
    "animated_gif": "gif",
}


class XCollector:
    """Fetches posts from X API and stores them in the local database."""

    def __init__(self, client: XClient, store: Database):
        self.client = client
        self.store = store

    def collect_by_search(self, query: str, max_posts: int = 20) -> list[PostInput]:
        """Search X for recent posts and return as PostInput objects."""
        if not self.client.is_available:
            logger.warning("X API not configured — returning empty results")
            return []

        all_posts: list[PostInput] = []
        next_token = None
        remaining = max_posts

        while remaining > 0:
            batch_size = min(remaining, 100)
            resp = self.client.search_recent(query, max_results=batch_size, next_token=next_token)
            if not resp or "data" not in resp:
                break

            posts = self._parse_response(resp)
            all_posts.extend(posts)
            remaining -= len(posts)

            next_token = resp.get("meta", {}).get("next_token")
            if not next_token:
                break

        logger.info("Collected %d posts for query '%s'", len(all_posts), query)
        return all_posts

    def collect_user_timeline(self, username: str, max_posts: int = 20) -> list[PostInput]:
        """Fetch a user's recent posts."""
        if not self.client.is_available:
            logger.warning("X API not configured — returning empty results")
            return []

        # Resolve username → user_id
        user_resp = self.client.get_user_by_username(username)
        if not user_resp or "data" not in user_resp:
            logger.error("Could not resolve username: %s", username)
            return []

        user_id = user_resp["data"]["id"]
        user_data = user_resp["data"]

        all_posts: list[PostInput] = []
        next_token = None
        remaining = max_posts

        while remaining > 0:
            batch_size = min(remaining, 100)
            resp = self.client.get_user_timeline(user_id, max_results=batch_size, next_token=next_token)
            if not resp or "data" not in resp:
                break

            posts = self._parse_response(resp, default_user=user_data)
            all_posts.extend(posts)
            remaining -= len(posts)

            next_token = resp.get("meta", {}).get("next_token")
            if not next_token:
                break

        logger.info("Collected %d posts from @%s", len(all_posts), username)
        return all_posts

    def collect_trends(self, woeid: int = 1) -> list[dict]:
        """Fetch current trending topics and store them."""
        if not self.client.is_available:
            logger.warning("X API not configured — returning empty trends")
            return []

        resp = self.client.get_trends(woeid=woeid)
        if not resp:
            return []

        # v1.1 response format: [{"trends": [...], "locations": [...]}]
        trends_data = []
        if isinstance(resp, list) and resp:
            raw_trends = resp[0].get("trends", [])
            for t in raw_trends[:25]:
                entry = {
                    "name": t.get("name", ""),
                    "volume": t.get("tweet_volume") or 0,
                    "velocity_normalized": min((t.get("tweet_volume") or 0) / 500000, 1.0),
                    "embedding": [],  # Will be computed by NLP pipeline if needed
                    "category": "general",
                }
                trends_data.append(entry)

        if trends_data:
            self.store.save_trends(trends_data)
            logger.info("Stored %d trends", len(trends_data))

        return trends_data

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, resp: dict, default_user: dict | None = None) -> list[PostInput]:
        """Parse X API v2 response into PostInput objects and store them."""
        tweets = resp.get("data", [])
        includes = resp.get("includes", {})
        users = {u["id"]: u for u in includes.get("users", [])}
        media = {m["media_key"]: m for m in includes.get("media", [])}

        results: list[PostInput] = []
        for tweet in tweets:
            author_id = tweet.get("author_id", "")
            user = users.get(author_id, default_user or {})
            user_metrics = user.get("public_metrics", {})

            # Determine media type
            media_type = "none"
            media_keys = tweet.get("attachments", {}).get("media_keys", [])
            for mk in media_keys:
                if mk in media:
                    raw_type = media[mk].get("type", "")
                    media_type = _MEDIA_TYPE_MAP.get(raw_type, "none")
                    break

            # Extract entities
            entities = tweet.get("entities", {})
            hashtags = [h["tag"] for h in entities.get("hashtags", [])]
            mentions = [m["username"] for m in entities.get("mentions", [])]
            urls = [u.get("expanded_url", u.get("url", "")) for u in entities.get("urls", [])]

            # Engagement metrics
            metrics = tweet.get("public_metrics", {})
            follower_count = user_metrics.get("followers_count", 0)
            impressions = metrics.get("impression_count", 0)
            total_eng = (
                metrics.get("like_count", 0)
                + metrics.get("retweet_count", 0)
                + metrics.get("reply_count", 0)
                + metrics.get("quote_count", 0)
            )
            eng_rate = total_eng / max(impressions, 1)

            # Store in database
            stored = StoredPost(
                x_post_id=tweet["id"],
                author_x_id=author_id,
                author_username=user.get("username", "unknown"),
                text=tweet.get("text", ""),
                media_type=media_type,
                language=tweet.get("lang", "en"),
                posted_at=tweet.get("created_at", datetime.now(timezone.utc).isoformat()),
                impressions=impressions,
                likes=metrics.get("like_count", 0),
                reposts=metrics.get("retweet_count", 0),
                replies=metrics.get("reply_count", 0),
                quotes=metrics.get("quote_count", 0),
                bookmarks=metrics.get("bookmark_count", 0),
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                follower_count=follower_count,
                engagement_rate=eng_rate,
            )
            self.store.save_post(stored)

            # Convert to PostInput for the engine
            media_enum = {
                "none": MediaType.NONE,
                "image": MediaType.IMAGE,
                "video": MediaType.VIDEO,
                "gif": MediaType.GIF,
                "poll": MediaType.POLL,
            }.get(media_type, MediaType.NONE)

            post_input = PostInput(
                text=tweet.get("text", ""),
                media_type=media_enum,
                author_follower_count=follower_count,
                author_engagement_rate=eng_rate,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                posted_at=tweet.get("created_at", ""),
                language=tweet.get("lang", "en"),
            )
            results.append(post_input)

        return results

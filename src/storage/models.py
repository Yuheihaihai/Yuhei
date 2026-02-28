"""Data models for storage layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trend:
    """Trend record — matches interface expected by FeatureExtractor."""

    name: str
    embedding: list[float]
    velocity_normalized: float


@dataclass
class VirPattern:
    """Viral pattern cluster — matches interface expected by FeatureExtractor."""

    cluster_id: int
    centroid: list[float]
    avg_virality_rate: float


@dataclass
class StoredPost:
    """Full post record as stored in the database."""

    x_post_id: str
    author_x_id: str
    author_username: str
    text: str
    media_type: str = "none"
    language: str = "en"
    posted_at: str = ""
    impressions: int = 0
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    quotes: int = 0
    bookmarks: int = 0
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    follower_count: int = 0
    engagement_rate: float = 0.0

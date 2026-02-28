"""SQLite-based storage layer for the Virality Engine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from src.storage.models import StoredPost, Trend, VirPattern


class Database:
    """SQLite storage with TrendStore and PatternDB interfaces."""

    def __init__(self, db_path: str = "data/virality.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._seed_defaults()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x_user_id TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                display_name TEXT,
                follower_count INTEGER DEFAULT 0,
                following_count INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                bio TEXT,
                avg_engagement_rate REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x_post_id TEXT UNIQUE NOT NULL,
                author_id INTEGER REFERENCES authors(id),
                post_text TEXT NOT NULL,
                media_type TEXT DEFAULT 'none',
                media_urls TEXT DEFAULT '[]',
                language TEXT DEFAULT 'en',
                posted_at TEXT NOT NULL,
                impressions INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                reposts INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                quotes INTEGER DEFAULT 0,
                bookmarks INTEGER DEFAULT 0,
                hashtags TEXT DEFAULT '[]',
                mentions TEXT DEFAULT '[]',
                urls TEXT DEFAULT '[]',
                conversation_id TEXT,
                collected_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS post_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER UNIQUE REFERENCES posts(id),
                hook_strength REAL DEFAULT 0.0,
                emotional_intensity REAL DEFAULT 0.0,
                curiosity_gap REAL DEFAULT 0.0,
                controversy_index REAL DEFAULT 0.0,
                readability_score REAL DEFAULT 0.0,
                structure_score REAL DEFAULT 0.0,
                timing_score REAL DEFAULT 0.0,
                trend_alignment REAL DEFAULT 0.0,
                sentiment_compound REAL,
                emotion_scores TEXT,
                detected_topic TEXT,
                embedding TEXT,
                cluster_id INTEGER DEFAULT -1,
                pattern_similarity REAL DEFAULT 0.0,
                virality_score_raw REAL,
                virality_score_final REAL,
                extracted_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trend_name TEXT NOT NULL,
                volume INTEGER DEFAULT 0,
                velocity REAL DEFAULT 0.0,
                velocity_normalized REAL DEFAULT 0.0,
                embedding TEXT,
                category TEXT,
                recorded_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS viral_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER UNIQUE NOT NULL,
                centroid TEXT NOT NULL,
                post_count INTEGER DEFAULT 0,
                avg_virality_rate REAL DEFAULT 0.0,
                dominant_pattern TEXT,
                dominant_emotion TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS engagement_heatmap (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment TEXT DEFAULT 'global',
                day_of_week INTEGER NOT NULL,
                hour_utc INTEGER NOT NULL,
                avg_engagement_rate REAL NOT NULL,
                sample_count INTEGER DEFAULT 0,
                UNIQUE(segment, day_of_week, hour_utc)
            );

            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER REFERENCES posts(id),
                input_text TEXT NOT NULL,
                original_score REAL NOT NULL,
                optimized_score REAL,
                recommendations TEXT DEFAULT '[]',
                hook_variants TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_posts_posted_at ON posts(posted_at);
            CREATE INDEX IF NOT EXISTS idx_posts_impressions ON posts(impressions);
            CREATE INDEX IF NOT EXISTS idx_trends_recorded ON trends(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_heatmap_segment ON engagement_heatmap(segment);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Seed data (so engine works on first run with no API data)
    # ------------------------------------------------------------------

    def _seed_defaults(self):
        cur = self.conn.cursor()

        # Seed trends if empty
        cur.execute("SELECT COUNT(*) FROM trends")
        if cur.fetchone()[0] == 0:
            seed_trends = [
                ("AI agents", 0.85, "tech"),
                ("startup layoffs", 0.60, "business"),
                ("remote work debate", 0.45, "career"),
            ]
            for name, velocity, category in seed_trends:
                emb = self._hash_embedding(name)
                cur.execute(
                    "INSERT INTO trends (trend_name, velocity_normalized, embedding, category) VALUES (?, ?, ?, ?)",
                    (name, velocity, json.dumps(emb), category),
                )

        # Seed viral patterns if empty
        cur.execute("SELECT COUNT(*) FROM viral_patterns")
        if cur.fetchone()[0] == 0:
            seed_patterns = [
                (0, "listicle_viral", 0.75, "LISTICLE"),
                (1, "contrarian_viral", 0.60, "CONTRARIAN"),
                (2, "story_viral", 0.50, "STORY_OPEN"),
            ]
            for cid, seed, rate, pattern in seed_patterns:
                centroid = self._hash_embedding(seed)
                cur.execute(
                    "INSERT INTO viral_patterns (cluster_id, centroid, avg_virality_rate, dominant_pattern) VALUES (?, ?, ?, ?)",
                    (cid, json.dumps(centroid), rate, pattern),
                )

        # Seed engagement heatmap if empty
        cur.execute("SELECT COUNT(*) FROM engagement_heatmap")
        if cur.fetchone()[0] == 0:
            day_factor = {0: 0.60, 1: 0.85, 2: 0.90, 3: 0.85, 4: 0.70, 5: 0.40, 6: 0.35}
            for dow in range(7):
                for hour in range(24):
                    hour_peak = math.exp(-0.5 * ((hour - 14.5) / 3.0) ** 2)
                    rate = max(0.0, min(1.0, day_factor[dow] * hour_peak))
                    cur.execute(
                        "INSERT INTO engagement_heatmap (segment, day_of_week, hour_utc, avg_engagement_rate) VALUES (?, ?, ?, ?)",
                        ("global", dow, hour, rate),
                    )

        self.conn.commit()

    @staticmethod
    def _hash_embedding(seed: str, dim: int = 384) -> list[float]:
        """Deterministic pseudo-embedding from a seed string."""
        random.seed(int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**31))
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in vec) + 1e-10)
        return [v / norm for v in vec]

    # ------------------------------------------------------------------
    # TrendStore interface
    # ------------------------------------------------------------------

    def get_active(self, hours: int = 6) -> list[Trend]:
        """Alias matching the interface FeatureExtractor expects."""
        return self.get_active_trends(hours)

    def get_active_trends(self, hours: int = 6) -> list[Trend]:
        cur = self.conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur.execute(
            "SELECT trend_name, embedding, velocity_normalized FROM trends "
            "WHERE recorded_at >= ? ORDER BY velocity_normalized DESC LIMIT 25",
            (cutoff,),
        )
        rows = cur.fetchall()
        # If no recent trends, return all trends (seed data has no realistic timestamps)
        if not rows:
            cur.execute(
                "SELECT trend_name, embedding, velocity_normalized FROM trends "
                "ORDER BY velocity_normalized DESC LIMIT 25"
            )
            rows = cur.fetchall()
        return [
            Trend(
                name=r["trend_name"],
                embedding=json.loads(r["embedding"]) if r["embedding"] else [],
                velocity_normalized=r["velocity_normalized"],
            )
            for r in rows
        ]

    def engagement_heatmap_lookup(self, dow: int, hour: int) -> float:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT avg_engagement_rate FROM engagement_heatmap "
            "WHERE segment = 'global' AND day_of_week = ? AND hour_utc = ?",
            (dow, hour),
        )
        row = cur.fetchone()
        if row:
            return row["avg_engagement_rate"]
        # Fallback: Gaussian heuristic
        day_factor = {0: 0.60, 1: 0.85, 2: 0.90, 3: 0.85, 4: 0.70, 5: 0.40, 6: 0.35}
        hour_peak = math.exp(-0.5 * ((hour - 14.5) / 3.0) ** 2)
        return max(0.0, min(1.0, day_factor.get(dow, 0.5) * hour_peak))

    # ------------------------------------------------------------------
    # PatternDB interface
    # ------------------------------------------------------------------

    def get_viral_patterns(self) -> list[VirPattern]:
        cur = self.conn.cursor()
        cur.execute("SELECT cluster_id, centroid, avg_virality_rate FROM viral_patterns")
        return [
            VirPattern(
                cluster_id=r["cluster_id"],
                centroid=json.loads(r["centroid"]),
                avg_virality_rate=r["avg_virality_rate"],
            )
            for r in cur.fetchall()
        ]

    def nearest_cluster(self, emb: list[float]) -> int:
        patterns = self.get_viral_patterns()
        if not patterns:
            return -1
        best_id, best_sim = -1, -1.0
        for p in patterns:
            sim = self._cosine_similarity(emb, p.centroid)
            if sim > best_sim:
                best_id, best_sim = p.cluster_id, sim
        return best_id

    def topic_controversy_baseline(self, topic: str) -> float:
        baselines = {
            "tech": 0.30, "finance": 0.40, "career": 0.35,
            "health": 0.25, "marketing": 0.20, "politics": 0.55,
            "sports": 0.30, "entertainment": 0.20, "science": 0.25,
            "general": 0.15,
        }
        return baselines.get(topic, 0.15)

    # ------------------------------------------------------------------
    # Post CRUD
    # ------------------------------------------------------------------

    def save_author(self, x_user_id: str, username: str, **kwargs) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO authors (x_user_id, username, display_name, follower_count, following_count, is_verified, bio) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                x_user_id, username,
                kwargs.get("display_name", ""),
                kwargs.get("follower_count", 0),
                kwargs.get("following_count", 0),
                int(kwargs.get("is_verified", False)),
                kwargs.get("bio", ""),
            ),
        )
        self.conn.commit()
        cur.execute("SELECT id FROM authors WHERE x_user_id = ?", (x_user_id,))
        return cur.fetchone()["id"]

    def save_post(self, post: StoredPost) -> int:
        cur = self.conn.cursor()
        author_id = self.save_author(post.author_x_id, post.author_username,
                                     follower_count=post.follower_count)
        cur.execute(
            "INSERT OR IGNORE INTO posts "
            "(x_post_id, author_id, post_text, media_type, language, posted_at, "
            "impressions, likes, reposts, replies, quotes, bookmarks, "
            "hashtags, mentions, urls) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post.x_post_id, author_id, post.text, post.media_type,
                post.language, post.posted_at,
                post.impressions, post.likes, post.reposts, post.replies,
                post.quotes, post.bookmarks,
                json.dumps(post.hashtags), json.dumps(post.mentions), json.dumps(post.urls),
            ),
        )
        self.conn.commit()
        cur.execute("SELECT id FROM posts WHERE x_post_id = ?", (post.x_post_id,))
        row = cur.fetchone()
        return row["id"] if row else -1

    def save_trends(self, trends: list[dict]):
        cur = self.conn.cursor()
        for t in trends:
            cur.execute(
                "INSERT INTO trends (trend_name, volume, velocity_normalized, embedding, category) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    t["name"], t.get("volume", 0), t.get("velocity_normalized", 0.5),
                    json.dumps(t.get("embedding", [])), t.get("category", "general"),
                ),
            )
        self.conn.commit()

    def get_posts(self, limit: int = 100) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT p.*, a.username, a.follower_count FROM posts p "
            "JOIN authors a ON p.author_id = a.id "
            "ORDER BY p.posted_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

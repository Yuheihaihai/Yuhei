#!/usr/bin/env python3
"""X Virality Optimization Engine — Mobile Web UI"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps

import requests as http_requests
import yaml
from flask import Flask, request, jsonify, send_file, send_from_directory, session

from src.core.virality_engine import PostInput, ViralityEngine
from src.llm.analyzer import LLMAnalyzer

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_config = None


def get_config():
    global _config
    if _config is None:
        _config = {}
        if os.path.exists("config.yaml"):
            with open("config.yaml") as f:
                _config = yaml.safe_load(f) or {}
    return _config


def _init_app():
    config = get_config()
    secret = (
        config.get("auth", {}).get("secret_key")
        or os.environ.get("FLASK_SECRET_KEY")
        or os.urandom(32).hex()
    )
    app.secret_key = secret


_init_app()


def get_allowed_handles() -> list[str]:
    config = get_config()
    handles = list(config.get("auth", {}).get("allowed_handles", []))
    env_handles = os.environ.get("ALLOWED_HANDLES", "")
    if env_handles:
        handles.extend(h.strip().lower() for h in env_handles.split(",") if h.strip())
    return [h.lower() for h in handles]


def get_allowed_origins() -> list[str]:
    config = get_config()
    origins = list(config.get("allowed_origins") or [])
    env_origins = os.environ.get("ALLOWED_ORIGINS", "")
    if env_origins:
        origins.extend(o.strip() for o in env_origins.split(",") if o.strip())
    return origins


def get_x_bearer_token() -> str:
    config = get_config()
    return (
        config.get("x_api", {}).get("bearer_token")
        or os.environ.get("X_BEARER_TOKEN")
        or ""
    )


# ---------------------------------------------------------------------------
# CORS — restrict to configured origins (same-origin if empty)
# ---------------------------------------------------------------------------

@app.after_request
def apply_cors(response):
    allowed = get_allowed_origins()
    origin = request.headers.get("Origin", "")
    if allowed and origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Rate Limiting (in-memory, per-IP)
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, max_requests: int = 30, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        self._hits[key] = [t for t in self._hits[key] if now - t < self.window]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True


_limiter = _RateLimiter(max_requests=30, window=60)


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_ip = request.remote_addr or "unknown"
        if not _limiter.check(client_ip):
            return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Server-side Authentication
# ---------------------------------------------------------------------------

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Engine singletons
# ---------------------------------------------------------------------------

_engine = None
_llm = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = ViralityEngine(get_config())
    return _engine


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMAnalyzer()
    return _llm


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_BASE_DIR, "docs", "index.html")
_STATIC_DIR = os.path.join(_BASE_DIR, "static")

MAX_TEXT_LENGTH = 10000


def _validated_post_data(data: dict):
    text = (data.get("text") or "")[:MAX_TEXT_LENGTH]
    followers = max(0, min(int(data.get("followers", 10000)), 500_000_000))
    engagement_rate = max(0.0, min(float(data.get("engagement_rate", 0.02)), 1.0))
    return text, followers, engagement_rate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(_HTML_PATH)


@app.route("/sw.js")
def service_worker():
    return send_from_directory(_STATIC_DIR, "sw.js", mimetype="application/javascript")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(_STATIC_DIR, filename)


@app.route("/api/login", methods=["POST"])
@rate_limit
def api_login():
    data = request.get_json() or {}
    handle = (data.get("handle") or "").strip().lower().lstrip("@")
    if not handle:
        return jsonify({"error": "Handle is required"}), 400
    if handle not in get_allowed_handles():
        return jsonify({"error": "Not authorized"}), 403
    session["user"] = handle
    return jsonify({"user": handle})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.pop("user", None)
    return jsonify({"ok": True})


@app.route("/api/session", methods=["GET"])
def api_session():
    user = session.get("user")
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not logged in"}), 401


@app.route("/api/analyze", methods=["POST"])
@rate_limit
@require_auth
def api_analyze():
    data = request.get_json() or {}
    text, followers, engagement_rate = _validated_post_data(data)
    if not text:
        return jsonify({"error": "text is required"}), 400

    engine = get_engine()
    post = PostInput(
        text=text,
        author_follower_count=followers,
        author_engagement_rate=engagement_rate,
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    result = engine.analyze(post)

    return jsonify({
        "score": {
            "final_score": result.score.final_score,
            "raw_score": result.score.raw_score,
            "predicted_impressions_p50": result.score.predicted_impressions_p50,
            "predicted_impressions_p75": result.score.predicted_impressions_p75,
            "predicted_impressions_p90": result.score.predicted_impressions_p90,
            "feature_breakdown": result.score.feature_breakdown,
            "weaknesses": result.score.weaknesses,
            "category": result.score.category,
        },
        "recommendations": [
            {
                "type": r.rec_type,
                "detail": r.detail,
                "score_delta": r.score_delta,
            }
            for r in result.recommendations
        ],
        "hook_variants": result.hook_variants,
        "thread_expansion": result.thread_expansion,
        "media_recommendation": result.media_recommendation,
        "hashtag_strategy": result.hashtag_strategy,
        "engagement_bait_type": result.engagement_bait_type,
    })


@app.route("/api/llm-analyze", methods=["POST"])
@rate_limit
@require_auth
def api_llm_analyze():
    data = request.get_json() or {}
    text, followers, engagement_rate = _validated_post_data(data)
    if not text:
        return jsonify({"error": "text is required"}), 400

    llm = get_llm()
    if not llm.is_available:
        return jsonify({"error": "ANTHROPIC_API_KEY not set."}), 503

    engine = get_engine()
    post = PostInput(
        text=text,
        author_follower_count=followers,
        author_engagement_rate=engagement_rate,
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    result = engine.analyze(post)

    analysis = llm.analyze(
        post_text=text,
        followers=followers,
        feature_scores=result.score.feature_breakdown,
    )

    if analysis.error:
        return jsonify({"error": analysis.error}), 500

    return jsonify({
        "viral_score": analysis.viral_score,
        "diagnosis": analysis.diagnosis,
        "rewrites": analysis.rewrites,
        "hooks": analysis.hooks,
        "thread_strategy": analysis.thread_strategy,
        "posting_advice": analysis.posting_advice,
    })


@app.route("/api/past-posts-analyze", methods=["POST"])
@rate_limit
@require_auth
def api_past_posts_analyze():
    """Analyze a list of past posts using the LLM."""
    data = request.get_json() or {}
    posts = data.get("posts", [])
    username = data.get("username", session.get("user", "user"))

    if not posts or not isinstance(posts, list):
        return jsonify({"error": "posts array is required"}), 400

    posts = [str(p)[:1000] for p in posts[:50]]

    llm = get_llm()
    if not llm.is_available:
        return jsonify({"error": "ANTHROPIC_API_KEY not set."}), 503

    system_prompt = (
        "You are an expert X (Twitter) content strategist.\n"
        "You analyze a user's past posts to identify what content performs best.\n"
        "Your goal: find patterns in high-performing content and give actionable recommendations.\n\n"
        "You respond ONLY in valid JSON. No markdown fences. No explanation outside JSON.\n"
        "IMPORTANT: All text values in the JSON MUST be written in Japanese."
    )

    numbered = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(posts))
    user_prompt = (
        f"以下は @{username} の過去の投稿です。\n\n"
        f"投稿一覧:\n{numbered}\n\n"
        "この投稿データを分析して、以下のJSON形式で回答してください:\n"
        '{\n'
        '  "summary": "<ユーザーのコンテンツ全体の傾向を2-3文で要約>",\n'
        f'  "total_analyzed": {len(posts)},\n'
        '  "top_posts": [{"rank": 1, "text": "<投稿>", "why_good": "<理由>"}],\n'
        '  "winning_patterns": [{"pattern": "<名>", "description": "<説明>", "frequency": "<頻度>"}],\n'
        '  "content_recommendations": [{"title": "<タイプ>", "description": "<説明>", "example": "<例>"}],\n'
        '  "avoid": ["<避けるべき1>"],\n'
        '  "optimal_style": {"tone": "<>", "length": "<>", "format": "<>", "topics": ["<>"]}\n'
        '}\n\n'
        "top_postsは最大5件。winning_patternsは最大5件。content_recommendationsは最大3件。"
    )

    try:
        client = llm._get_client()
        response = client.messages.create(
            model=llm.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        parsed = llm._parse_json(raw)
        if not parsed:
            return jsonify({"error": "Failed to parse AI response"}), 500
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user-tweets", methods=["POST", "OPTIONS"])
@rate_limit
@require_auth
def api_user_tweets():
    """Fetch a user's recent tweets using server-side X API credentials."""
    if request.method == "OPTIONS":
        return app.make_default_options_response()

    data = request.get_json() or {}
    username = (data.get("username") or "").strip().lstrip("@")
    count = max(1, min(int(data.get("count", 30)), 100))

    if not username:
        return jsonify({"error": "username is required"}), 400

    bearer = get_x_bearer_token()
    if not bearer:
        return jsonify({"error": "X API not configured on server"}), 503

    headers = {"Authorization": f"Bearer {bearer}"}

    user_resp = http_requests.get(
        f"https://api.x.com/2/users/by/username/{username}",
        headers=headers,
        params={"user.fields": "public_metrics"},
        timeout=15,
    )
    if not user_resp.ok:
        return jsonify({"error": f"ユーザー取得に失敗しました ({user_resp.status_code})"}), 400

    user_json = user_resp.json()
    if "data" not in user_json:
        return jsonify({"error": "ユーザーが見つかりません"}), 404

    user_data = user_json["data"]
    user_id = user_data["id"]

    tweets_resp = http_requests.get(
        f"https://api.x.com/2/users/{user_id}/tweets",
        headers=headers,
        params={
            "max_results": count,
            "tweet.fields": "public_metrics,created_at,text",
            "exclude": "retweets,replies",
        },
        timeout=15,
    )
    if not tweets_resp.ok:
        return jsonify({"error": f"投稿の取得に失敗しました ({tweets_resp.status_code})"}), 400

    tweets_json = tweets_resp.json()
    tweets = tweets_json.get("data", [])

    return jsonify({
        "user": {
            "id": user_id,
            "username": user_data["username"],
            "name": user_data.get("name", ""),
            "metrics": user_data.get("public_metrics", {}),
        },
        "tweets": tweets,
        "count": len(tweets),
    })


@app.route("/api/auth/exchange", methods=["POST", "OPTIONS"])
@rate_limit
def api_auth_exchange():
    """Proxy X OAuth 2.0 token exchange."""
    if request.method == "OPTIONS":
        return app.make_default_options_response()

    data = request.get_json() or {}
    code = data.get("code")
    verifier = data.get("code_verifier")
    redirect_uri = data.get("redirect_uri")
    client_id = data.get("client_id")

    if not all([code, verifier, redirect_uri, client_id]):
        return jsonify({"error": "Missing required parameters"}), 400

    token_resp = http_requests.post(
        "https://api.x.com/2/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
        timeout=15,
    )

    if not token_resp.ok:
        return jsonify({"error": f"Token exchange failed: {token_resp.status_code}"}), 400

    access_token = token_resp.json().get("access_token")

    user_resp = http_requests.get(
        "https://api.x.com/2/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )

    if not user_resp.ok:
        return jsonify({"error": "Failed to fetch user profile"}), 400

    user_data = user_resp.json()["data"]
    username = user_data["username"].lower()

    if username not in get_allowed_handles():
        return jsonify({"error": "Not authorized"}), 403

    session["user"] = username
    return jsonify({
        "username": user_data["username"],
        "name": user_data.get("name", user_data["username"]),
    })


if __name__ == "__main__":
    print("WARNING: Flask dev server is not for production. Use gunicorn/waitress.", file=sys.stderr)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

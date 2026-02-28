#!/usr/bin/env python3
"""X Virality Optimization Engine — Mobile Web UI"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import requests as http_requests
import yaml
from flask import Flask, request, jsonify, send_file

from src.core.virality_engine import PostInput, ViralityEngine
from src.llm.analyzer import LLMAnalyzer

app = Flask(__name__)

_engine = None
_llm = None


def get_engine():
    global _engine
    if _engine is None:
        config = {}
        if os.path.exists("config.yaml"):
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
        _engine = ViralityEngine(config)
    return _engine


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMAnalyzer()
    return _llm


_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "index.html")


@app.route("/")
def index():
    return send_file(_HTML_PATH)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    engine = get_engine()
    post = PostInput(
        text=text,
        author_follower_count=data.get("followers", 10000),
        author_engagement_rate=data.get("engagement_rate", 0.02),
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
def api_llm_analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    llm = get_llm()
    if not llm.is_available:
        return jsonify({
            "error": "ANTHROPIC_API_KEY not set. Set the environment variable to enable AI analysis."
        })

    followers = data.get("followers", 10000)

    # Optionally run the standard engine first to get feature scores
    engine = get_engine()
    post = PostInput(
        text=text,
        author_follower_count=followers,
        author_engagement_rate=data.get("engagement_rate", 0.02),
        posted_at=datetime.now(timezone.utc).isoformat(),
    )
    result = engine.analyze(post)
    feature_scores = result.score.feature_breakdown

    analysis = llm.analyze(
        post_text=text,
        followers=followers,
        feature_scores=feature_scores,
    )

    if analysis.error:
        return jsonify({"error": analysis.error})

    return jsonify({
        "viral_score": analysis.viral_score,
        "diagnosis": analysis.diagnosis,
        "rewrites": analysis.rewrites,
        "hooks": analysis.hooks,
        "thread_strategy": analysis.thread_strategy,
        "posting_advice": analysis.posting_advice,
    })


@app.route("/api/user-tweets", methods=["POST", "OPTIONS"])
def api_user_tweets():
    """Proxy to fetch a user's recent tweets from X API v2 (avoids CORS)."""
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST"
        return resp

    data = request.get_json()
    username = data.get("username", "").strip().lstrip("@")
    bearer = data.get("bearer_token", "")
    count = min(int(data.get("count", 30)), 100)

    if not username or not bearer:
        resp = jsonify({"error": "username and bearer_token are required"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    headers = {"Authorization": f"Bearer {bearer}"}

    # Step 1: Look up user ID by username
    user_resp = http_requests.get(
        f"https://api.x.com/2/users/by/username/{username}",
        headers=headers,
        params={"user.fields": "public_metrics"},
    )
    if not user_resp.ok:
        resp = jsonify({"error": f"ユーザー取得に失敗しました ({user_resp.status_code})"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    user_json = user_resp.json()
    if "data" not in user_json:
        resp = jsonify({"error": "ユーザーが見つかりません"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 404

    user_data = user_json["data"]
    user_id = user_data["id"]

    # Step 2: Fetch the user's recent tweets with engagement metrics
    tweets_resp = http_requests.get(
        f"https://api.x.com/2/users/{user_id}/tweets",
        headers=headers,
        params={
            "max_results": count,
            "tweet.fields": "public_metrics,created_at,text",
            "exclude": "retweets,replies",
        },
    )
    if not tweets_resp.ok:
        resp = jsonify({"error": f"投稿の取得に失敗しました ({tweets_resp.status_code})"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    tweets_json = tweets_resp.json()
    tweets = tweets_json.get("data", [])

    resp = jsonify({
        "user": {
            "id": user_id,
            "username": user_data["username"],
            "name": user_data.get("name", ""),
            "metrics": user_data.get("public_metrics", {}),
        },
        "tweets": tweets,
        "count": len(tweets),
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/api/auth/exchange", methods=["POST", "OPTIONS"])
def api_auth_exchange():
    """Proxy X OAuth 2.0 token exchange to avoid CORS issues on static sites."""
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST"
        return resp

    data = request.get_json()
    code = data.get("code")
    verifier = data.get("code_verifier")
    redirect_uri = data.get("redirect_uri")
    client_id = data.get("client_id")

    if not all([code, verifier, redirect_uri, client_id]):
        resp = jsonify({"error": "Missing required parameters"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    # Exchange authorization code for access token (server-to-server, no CORS)
    token_resp = http_requests.post(
        "https://api.x.com/2/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )

    if not token_resp.ok:
        resp = jsonify({"error": f"Token exchange failed: {token_resp.status_code}"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    access_token = token_resp.json().get("access_token")

    # Fetch authenticated user profile
    user_resp = http_requests.get(
        "https://api.x.com/2/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    if not user_resp.ok:
        resp = jsonify({"error": "Failed to fetch user profile"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    user_data = user_resp.json()["data"]
    resp = jsonify({"username": user_data["username"], "name": user_data.get("name", user_data["username"])})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

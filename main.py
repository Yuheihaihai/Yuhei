#!/usr/bin/env python3
"""
X Virality Optimization Engine — CLI

Usage:
    python main.py analyze "Your post text here"
    python main.py search "AI tools" --max-results 20
    python main.py timeline @username --max-results 20
    python main.py trends
    python main.py demo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import yaml


def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 25) -> str:
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:.2f}"


def _sep(char: str = "─", width: int = 72):
    print(char * width)


def print_analysis(strategy_output, post_text: str = ""):
    from src.core.virality_engine import StrategyOutput
    s: StrategyOutput = strategy_output
    score = s.score

    if post_text:
        print()
        _sep("─")
        lines = post_text.split("\n")
        for line in lines[:6]:
            print(f"  {line}")
        if len(lines) > 6:
            print(f"  ... ({len(lines) - 6} more lines)")
        _sep("─")

    cat_icon = {"high_potential": "🟢", "medium_potential": "🟡", "moderate": "🟠", "low_potential": "🔴"}.get(score.category, "⚪")

    print()
    print("  Feature Breakdown:")
    for feat, val in score.feature_breakdown.items():
        print(f"    {feat:<22} {_bar(val)}")

    print()
    print(f"  Virality Score:       {_bar(score.final_score, 30)}")
    print(f"  Category:             {cat_icon} {score.category}")
    print(f"  Predicted Impressions: p50={score.predicted_impressions_p50:>10,}  "
          f"p75={score.predicted_impressions_p75:>10,}  "
          f"p90={score.predicted_impressions_p90:>10,}")

    if score.weaknesses:
        print()
        print("  Weaknesses:")
        for w in score.weaknesses:
            print(f"    ⚠  {w}")

    if s.recommendations:
        print()
        print("  Recommendations:")
        for rec in s.recommendations:
            print(f"    [{rec.rec_type.upper()}] score_delta: {rec.score_delta:+.3f}")
            if rec.rec_type == "rewrite" and "rewritten" in rec.detail:
                print(f"      Original:  \"{rec.detail['original']}\"")
                print(f"      Rewritten: \"{rec.detail['rewritten']}\"")
            elif rec.rec_type == "timing":
                for w in rec.detail.get("optimal_windows", [])[:3]:
                    print(f"      {w[0]} at {w[1]}:00 UTC (score: {w[2]:.2f})")
            elif rec.rec_type == "format":
                for sg in rec.detail.get("suggestions", []):
                    print(f"      → {sg['action']} (impact: +{sg['impact']:.2f})")
            elif rec.rec_type == "trigger":
                for t in rec.detail.get("triggers", []):
                    print(f"      → {t}")

    if s.hook_variants:
        print()
        print("  Hook Variants:")
        for i, hook in enumerate(s.hook_variants[:5], 1):
            print(f"    {i}. \"{hook}\"")

    print()
    print(f"  Media:    {s.media_recommendation}")
    print(f"  Hashtags: {json.dumps(s.hashtag_strategy)}")
    print(f"  CTA type: {s.engagement_bait_type}")
    thread = s.thread_expansion
    print(f"  Thread:   {'Recommended (' + str(thread.get('optimal_length', 0)) + ' posts)' if thread.get('recommended') else 'Not recommended'}")
    print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_analyze(args, config):
    """Analyze a single post text."""
    from datetime import datetime, timezone
    from src.core.virality_engine import PostInput, ViralityEngine

    engine = ViralityEngine(config)
    post = PostInput(
        text=args.text,
        author_follower_count=args.followers or 10000,
        author_engagement_rate=args.engagement_rate or 0.02,
        posted_at=args.posted_at or datetime.now(timezone.utc).isoformat(),
    )

    print()
    print("═" * 72)
    print("  X VIRALITY OPTIMIZATION ENGINE")
    print("═" * 72)

    result = engine.analyze(post)
    print_analysis(result, post_text=args.text)

    print("═" * 72)


def cmd_search(args, config):
    """Search X for posts, analyze them."""
    from src.x_api.client import XClient
    from src.x_api.collector import XCollector
    from src.storage.database import Database
    from src.core.virality_engine import ViralityEngine

    client = XClient(config.get("x_api", {}).get("bearer_token"))
    if not client.is_available:
        print("  X API credentials not found.")
        print("  Set X_BEARER_TOKEN environment variable or configure config.yaml")
        print()
        print("  Example: export X_BEARER_TOKEN=your_token_here")
        print("           python main.py search \"AI tools\"")
        return

    db_path = config.get("storage", {}).get("db_path", "data/virality.db")
    store = Database(db_path)
    collector = XCollector(client, store)
    engine = ViralityEngine(config)

    print()
    print("═" * 72)
    print(f"  Searching X for: \"{args.query}\" (max {args.max_results} posts)")
    print("═" * 72)

    posts = collector.collect_by_search(args.query, max_posts=args.max_results)
    if not posts:
        print("  No posts found.")
        return

    print(f"  Found {len(posts)} posts. Analyzing...\n")

    scored = []
    for post in posts:
        result = engine.analyze(post)
        scored.append((post, result))

    # Sort by virality score descending
    scored.sort(key=lambda x: x[1].score.final_score, reverse=True)

    # Print leaderboard
    print(f"  {'#':<4} {'Score':<8} {'Category':<18} {'p50 Imp.':<14} {'Post'}")
    _sep("─")
    for i, (post, result) in enumerate(scored[:20], 1):
        first_40 = post.text[:45].replace("\n", " ")
        s = result.score
        print(f"  {i:<4} {s.final_score:<8.3f} {s.category:<18} {s.predicted_impressions_p50:<14,} {first_40}...")

    # Print detailed analysis of top 3
    print()
    print("  TOP 3 DETAILED ANALYSIS")
    for post, result in scored[:3]:
        print_analysis(result, post_text=post.text)
        _sep("─")


def cmd_timeline(args, config):
    """Analyze a user's timeline."""
    from src.x_api.client import XClient
    from src.x_api.collector import XCollector
    from src.storage.database import Database
    from src.core.virality_engine import ViralityEngine

    client = XClient(config.get("x_api", {}).get("bearer_token"))
    if not client.is_available:
        print("  X API credentials not found.")
        print("  Set X_BEARER_TOKEN environment variable or configure config.yaml")
        return

    db_path = config.get("storage", {}).get("db_path", "data/virality.db")
    store = Database(db_path)
    collector = XCollector(client, store)
    engine = ViralityEngine(config)

    username = args.username.lstrip("@")
    print()
    print("═" * 72)
    print(f"  Analyzing timeline for @{username} (max {args.max_results} posts)")
    print("═" * 72)

    posts = collector.collect_user_timeline(username, max_posts=args.max_results)
    if not posts:
        print("  No posts found or user not accessible.")
        return

    scored = []
    for post in posts:
        result = engine.analyze(post)
        scored.append((post, result))

    scored.sort(key=lambda x: x[1].score.final_score, reverse=True)

    print(f"  {'#':<4} {'Score':<8} {'Category':<18} {'Post'}")
    _sep("─")
    for i, (post, result) in enumerate(scored[:20], 1):
        first_50 = post.text[:50].replace("\n", " ")
        print(f"  {i:<4} {result.score.final_score:<8.3f} {result.score.category:<18} {first_50}...")

    print()
    avg_score = sum(r.score.final_score for _, r in scored) / max(len(scored), 1)
    print(f"  Average virality score: {avg_score:.3f}")
    print(f"  Top score:              {scored[0][1].score.final_score:.3f}")
    print(f"  Posts analyzed:         {len(scored)}")
    print()


def cmd_trends(args, config):
    """Show current trending topics."""
    from src.x_api.client import XClient
    from src.x_api.collector import XCollector
    from src.storage.database import Database

    client = XClient(config.get("x_api", {}).get("bearer_token"))
    if not client.is_available:
        print("  X API credentials not found.")
        print("  Set X_BEARER_TOKEN environment variable or configure config.yaml")
        return

    db_path = config.get("storage", {}).get("db_path", "data/virality.db")
    store = Database(db_path)
    collector = XCollector(client, store)

    print()
    print("═" * 72)
    print("  CURRENT X TRENDS")
    print("═" * 72)

    trends = collector.collect_trends(woeid=args.woeid)
    if not trends:
        print("  Could not fetch trends.")
        return

    print(f"\n  {'#':<4} {'Trend':<30} {'Volume':<12} {'Velocity'}")
    _sep("─")
    for i, t in enumerate(trends[:25], 1):
        vol = f"{t['volume']:,}" if t['volume'] else "N/A"
        print(f"  {i:<4} {t['name']:<30} {vol:<12} {t['velocity_normalized']:.2f}")
    print()


def cmd_demo(args, config):
    """Run the demo with sample posts."""
    from demo import run_demo
    run_demo()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="X Virality Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze "5 AI tools that replaced my team:"
  python main.py analyze "Unpopular opinion: most startup advice is wrong." --followers 50000
  python main.py search "AI tools" --max-results 10
  python main.py timeline @elonmusk --max-results 20
  python main.py trends
  python main.py demo
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command")

    # analyze
    p_a = sub.add_parser("analyze", help="Analyze a single post text")
    p_a.add_argument("text", help="Post text to analyze")
    p_a.add_argument("--followers", type=int, help="Author follower count (default: 10000)")
    p_a.add_argument("--engagement-rate", type=float, help="Author avg engagement rate (default: 0.02)")
    p_a.add_argument("--posted-at", help="ISO-8601 timestamp (default: now)")

    # search
    p_s = sub.add_parser("search", help="Search X for posts and analyze")
    p_s.add_argument("query", help="Search query")
    p_s.add_argument("--max-results", type=int, default=20, help="Max posts to fetch")

    # timeline
    p_t = sub.add_parser("timeline", help="Analyze a user's timeline")
    p_t.add_argument("username", help="X username (with or without @)")
    p_t.add_argument("--max-results", type=int, default=20, help="Max posts to fetch")

    # trends
    p_tr = sub.add_parser("trends", help="Show current trending topics")
    p_tr.add_argument("--woeid", type=int, default=1, help="Yahoo WOEID (default: 1 = worldwide)")

    # demo
    sub.add_parser("demo", help="Run demo with sample posts (no API needed)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    commands = {
        "analyze": cmd_analyze,
        "search": cmd_search,
        "timeline": cmd_timeline,
        "trends": cmd_trends,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
arrwDB CLI - Command-line interface for arrwDB.

Usage:
    arrwdb health
    arrwdb libraries list
    arrwdb libraries create "My Library" --index-type hnsw
    arrwdb search <library_id> "query text" --k 10
    arrwdb temperature-search <library_id> "query" --temperature 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from arrwdb.client import ArrwDBClient, ArrwDBException


def get_client() -> ArrwDBClient:
    """Get client from environment or defaults."""
    base_url = os.environ.get("ARRWDB_URL", "http://localhost:8000")
    api_key = os.environ.get("ARRWDB_API_KEY")
    return ArrwDBClient(base_url=base_url, api_key=api_key)


def print_json(data: Any) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, default=str))


def print_table(data: List[Dict[str, Any]], columns: List[str]) -> None:
    """Print data as a simple table."""
    if not data:
        print("No results.")
        return

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            val = str(row.get(col, ""))[:50]  # Truncate long values
            widths[col] = max(widths[col], len(val))

    # Print header
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    # Print rows
    for row in data:
        line = " | ".join(
            str(row.get(col, ""))[:50].ljust(widths[col]) for col in columns
        )
        print(line)


# =============================================================================
# Commands
# =============================================================================


def cmd_health(args: argparse.Namespace) -> int:
    """Check server health."""
    client = get_client()
    try:
        health = client.health_check()
        print_json(health)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_libraries_list(args: argparse.Namespace) -> int:
    """List all libraries."""
    client = get_client()
    try:
        libraries = client.list_libraries()
        if args.json:
            print_json(libraries)
        else:
            print_table(libraries, ["id", "name", "index_type", "created_at"])
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_libraries_create(args: argparse.Namespace) -> int:
    """Create a new library."""
    client = get_client()
    try:
        library = client.create_library(
            name=args.name,
            description=args.description,
            index_type=args.index_type,
        )
        print_json(library)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_libraries_get(args: argparse.Namespace) -> int:
    """Get library details."""
    client = get_client()
    try:
        library = client.get_library(args.library_id)
        print_json(library)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_libraries_delete(args: argparse.Namespace) -> int:
    """Delete a library."""
    client = get_client()
    try:
        client.delete_library(args.library_id)
        print(f"Library {args.library_id} deleted.")
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_libraries_stats(args: argparse.Namespace) -> int:
    """Get library statistics."""
    client = get_client()
    try:
        stats = client.get_library_statistics(args.library_id)
        print_json(stats)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search a library."""
    client = get_client()
    try:
        results = client.search(
            library_id=args.library_id,
            query=args.query,
            k=args.k,
        )
        if args.json:
            print_json(results)
        else:
            for i, r in enumerate(results.get("results", []), 1):
                print(f"\n[{i}] Distance: {r.get('distance', 'N/A'):.4f}")
                print(f"    Text: {r.get('text', '')[:200]}...")
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_temperature_search(args: argparse.Namespace) -> int:
    """Temperature-controlled search."""
    client = get_client()
    try:
        results = client.temperature_search(
            corpus_id=args.library_id,
            query_text=args.query,
            k=args.k,
            temperature=args.temperature,
        )
        if args.json:
            print_json(results)
        else:
            print(f"Temperature: {args.temperature}")
            for i, r in enumerate(results.get("results", []), 1):
                print(f"\n[{i}] Score: {r.get('score', 'N/A'):.4f}")
                print(f"    Text: {r.get('text', '')[:200]}...")
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_index_oracle(args: argparse.Namespace) -> int:
    """Get index recommendation."""
    client = get_client()
    try:
        rec = client.get_index_recommendation(args.library_id)
        print_json(rec)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_embedding_health(args: argparse.Namespace) -> int:
    """Analyze embedding health."""
    client = get_client()
    try:
        health = client.analyze_embedding_health(args.library_id)
        print_json(health)
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_webhooks_list(args: argparse.Namespace) -> int:
    """List webhooks."""
    client = get_client()
    try:
        webhooks = client.list_webhooks()
        if args.json:
            print_json(webhooks)
        else:
            print_table(webhooks, ["id", "url", "status", "created_at"])
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_webhooks_create(args: argparse.Namespace) -> int:
    """Create a webhook."""
    client = get_client()
    try:
        webhook = client.create_webhook(
            url=args.url,
            events=args.events,
            description=args.description,
        )
        print_json(webhook)
        print(f"\n** Save the secret for HMAC verification: {webhook.get('secret')} **")
        return 0
    except ArrwDBException as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="arrwdb",
        description="arrwDB CLI - Production Vector Database",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Health
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.set_defaults(func=cmd_health)

    # Libraries
    lib_parser = subparsers.add_parser("libraries", help="Library operations")
    lib_sub = lib_parser.add_subparsers(dest="lib_command")

    lib_list = lib_sub.add_parser("list", help="List libraries")
    lib_list.add_argument("--json", action="store_true", help="Output JSON")
    lib_list.set_defaults(func=cmd_libraries_list)

    lib_create = lib_sub.add_parser("create", help="Create library")
    lib_create.add_argument("name", help="Library name")
    lib_create.add_argument("--description", help="Description")
    lib_create.add_argument(
        "--index-type",
        default="brute_force",
        choices=["brute_force", "kd_tree", "lsh", "hnsw", "ivf"],
    )
    lib_create.set_defaults(func=cmd_libraries_create)

    lib_get = lib_sub.add_parser("get", help="Get library")
    lib_get.add_argument("library_id", help="Library ID")
    lib_get.set_defaults(func=cmd_libraries_get)

    lib_delete = lib_sub.add_parser("delete", help="Delete library")
    lib_delete.add_argument("library_id", help="Library ID")
    lib_delete.set_defaults(func=cmd_libraries_delete)

    lib_stats = lib_sub.add_parser("stats", help="Library statistics")
    lib_stats.add_argument("library_id", help="Library ID")
    lib_stats.set_defaults(func=cmd_libraries_stats)

    # Search
    search_parser = subparsers.add_parser("search", help="Search a library")
    search_parser.add_argument("library_id", help="Library ID")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", type=int, default=10, help="Number of results")
    search_parser.add_argument("--json", action="store_true", help="Output JSON")
    search_parser.set_defaults(func=cmd_search)

    # Temperature Search
    temp_parser = subparsers.add_parser(
        "temperature-search", help="Temperature-controlled search"
    )
    temp_parser.add_argument("library_id", help="Library ID")
    temp_parser.add_argument("query", help="Search query")
    temp_parser.add_argument("-k", type=int, default=10, help="Number of results")
    temp_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Temperature (0=greedy, 2=exploratory)",
    )
    temp_parser.add_argument("--json", action="store_true", help="Output JSON")
    temp_parser.set_defaults(func=cmd_temperature_search)

    # Index Oracle
    oracle_parser = subparsers.add_parser(
        "index-oracle", help="Get index recommendation"
    )
    oracle_parser.add_argument("library_id", help="Library ID")
    oracle_parser.set_defaults(func=cmd_index_oracle)

    # Embedding Health
    health_parser = subparsers.add_parser(
        "embedding-health", help="Analyze embedding quality"
    )
    health_parser.add_argument("library_id", help="Library ID")
    health_parser.set_defaults(func=cmd_embedding_health)

    # Webhooks
    wh_parser = subparsers.add_parser("webhooks", help="Webhook operations")
    wh_sub = wh_parser.add_subparsers(dest="wh_command")

    wh_list = wh_sub.add_parser("list", help="List webhooks")
    wh_list.add_argument("--json", action="store_true", help="Output JSON")
    wh_list.set_defaults(func=cmd_webhooks_list)

    wh_create = wh_sub.add_parser("create", help="Create webhook")
    wh_create.add_argument("url", help="Webhook URL")
    wh_create.add_argument(
        "--events",
        nargs="+",
        default=["*"],
        help="Events to subscribe",
    )
    wh_create.add_argument("--description", help="Description")
    wh_create.set_defaults(func=cmd_webhooks_create)

    # Parse and execute
    args = parser.parse_args()

    if args.version:
        from arrwdb.version import __version__

        print(f"arrwdb {__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

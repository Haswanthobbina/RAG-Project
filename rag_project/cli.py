"""Developer CLI interface for GitContext repository intelligence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Safe terminal encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_project.core import INDEX
from rag_project.evaluator import CodebaseEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GitContext: End-to-End Codebase Intelligence Copilot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest a repository")
    ingest_p.add_argument("repo", nargs="?", default=".", help="Path to repository (default: current directory)")

    # Query command (raw retrieval)
    query_p = subparsers.add_parser("query", help="Retrieve relevant code chunks")
    query_p.add_argument("question", help="Question, symbol, or error message to query")
    query_p.add_argument("--repo", default=".", help="Path to repository")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")

    # Ask command (copilot answer with citations)
    ask_p = subparsers.add_parser("ask", help="Ask copilot for code explanation & citations")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument("--repo", default=".", help="Path to repository")
    ask_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to consider")

    # Sync command
    sync_p = subparsers.add_parser("sync", help="Incrementally re-index stale or modified files")
    sync_p.add_argument("--repo", default=".", help="Path to repository")

    # Evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Run evaluation benchmark (Recall@K, Latency, Precision)")
    eval_p.add_argument("--repo", default=".", help="Path to repository")
    eval_p.add_argument("--top-k", type=int, default=5, help="Top-K for recall evaluation")

    # Backward compatibility: positional repo and question
    if len(sys.argv) >= 3 and sys.argv[1] not in ["ingest", "query", "ask", "sync", "evaluate", "-h", "--help"]:
        repo_arg = sys.argv[1]
        question_arg = sys.argv[2]
        INDEX.ingest_repository(repo_arg)
        results = INDEX.query(question_arg, top_k=5)
        print(json.dumps({"results": results}, indent=2))
        return

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"[INGEST] Ingesting repository at: {args.repo} ...")
        res = INDEX.ingest_repository(args.repo)
        print(f"[OK] Ingestion complete. Indexed {len(INDEX.chunks)} chunks.")
        print(json.dumps(res, indent=2))

    elif args.command == "query":
        INDEX.ingest_repository(args.repo)
        results = INDEX.query(args.question, top_k=args.top_k)
        print(f"\n[QUERY] Results for: '{args.question}'\n")
        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            print(f"[{i}] {meta.get('file_path')}:{meta.get('start_line')}-{meta.get('end_line')} "
                  f"(score: {r.get('score')}) Author: {meta.get('author')}")
            print(f"    Name: {meta.get('name')} | Type: {meta.get('chunk_type')}")
            snippet = r.get("text", "").replace("\n", " ")[:100]
            print(f"    Snippet: {snippet}...\n")

    elif args.command == "ask":
        INDEX.ingest_repository(args.repo)
        answer_data = INDEX.ask(args.question, top_k=args.top_k)
        print("\n" + "=" * 60)
        print(f"[ANSWER] GitContext Copilot: {args.question}")
        print("=" * 60)
        print(answer_data.get("answer"))
        print("\n[CITATIONS]:")
        for cit in answer_data.get("citations", []):
            print(f"  - [{cit.get('file_path')}:{cit.get('lines')}] ({cit.get('chunk_type')}: {cit.get('name')}) by {cit.get('author')}")
        print("=" * 60)

    elif args.command == "sync":
        INDEX.ingest_repository(args.repo)
        sync_res = INDEX.sync()
        print("[SYNC] Incremental Sync Result:")
        print(json.dumps(sync_res, indent=2))

    elif args.command == "evaluate":
        INDEX.ingest_repository(args.repo)
        evaluator = CodebaseEvaluator(INDEX)
        benchmark_res = evaluator.run_benchmark(top_k=args.top_k)
        print("\n[EVALUATION] Benchmark Results:")
        print(json.dumps(benchmark_res, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

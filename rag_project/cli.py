from __future__ import annotations

import argparse
import json

from .core import INDEX


def main() -> None:
    parser = argparse.ArgumentParser(description="Query repository intelligence index")
    parser.add_argument("repo", help="Path to repository to index")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    INDEX.ingest_repository(args.repo)
    results = INDEX.query(args.question, top_k=args.top_k)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
